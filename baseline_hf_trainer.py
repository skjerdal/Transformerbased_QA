import os
import numpy as np
import tensorflow as tf # Still potentially needed for TF Datasets compatibility or GPU checks
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForQuestionAnswering, # Use the TF version for consistency if preferred
    DefaultDataCollator, # Use default collator for TF
    create_optimizer,
    pipeline, # For easier prediction later if needed
    # TFTrainingArguments, # Import TFTrainingArguments - REMOVED
    # TFTrainer # Import TFTrainer - REMOVED
)
from evaluate import load as load_metric
import logging
import transformers 

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased" # Faster pre-trained model for baseline
MAX_SAMPLES = 3000 # Use the same number of train samples as previous tests
VAL_SAMPLES = MAX_SAMPLES // 5 # Keep validation split consistent
MAX_SEQ_LENGTH = 384  # Max sequence length (from previous config)
DOC_STRIDE = 128      # Stride for handling long contexts
BATCH_SIZE = 16       # Batch size (from previous config)
LEARNING_RATE = 2e-5  # Common default fine-tuning LR
NUM_EPOCHS = 3        # Train for a few epochs for baseline
OUTPUT_DIR = "baseline_qa_model"
LOGGING_STEPS = 50

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allow memory growth for GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.error(f"Memory growth must be set before GPUs have been initialized: {e}")

# --- Load Dataset ---
logger.info("Loading SQuAD dataset...")
squad_dataset = load_dataset("squad")

# --- Select Subset ---
logger.info(f"Selecting {MAX_SAMPLES} train and {VAL_SAMPLES} validation samples...")
train_dataset_raw = squad_dataset["train"].select(range(MAX_SAMPLES))
val_dataset_raw = squad_dataset["validation"].select(range(VAL_SAMPLES))

# --- Load Tokenizer ---
logger.info(f"Loading tokenizer: {MODEL_CHECKPOINT}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast), "Requires a fast tokenizer"

# --- Preprocessing Functions ---
# Based on Hugging Face QA example: https://github.com/huggingface/transformers/blob/main/examples/tensorflow/question-answering/run_qa.py
# And adapted from utils/hf_squad_preprocessing.py

pad_on_right = tokenizer.padding_side == "right"

def prepare_train_features(examples):
    # Tokenize questions and contexts, handle truncation/padding
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if not answers["text"]:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

# --- Apply Preprocessing ---
logger.info("Preprocessing training data...")
train_dataset = train_dataset_raw.map(
    prepare_train_features,
    batched=True,
    remove_columns=train_dataset_raw.column_names,
    desc="Running tokenizer on train dataset",
)
logger.info(f"Number of features in training set: {len(train_dataset)}")


# Prepare validation features (slightly different - don't need start/end pos, need example_id and offset mapping)
def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # Add example_id for postprocessing
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set offset mapping to None for non-context tokens
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

logger.info("Preprocessing validation data...")
val_dataset_processed = val_dataset_raw.map(
    prepare_validation_features,
    batched=True,
    remove_columns=val_dataset_raw.column_names,
    desc="Running tokenizer on validation dataset",
)
logger.info(f"Number of features in validation set: {len(val_dataset_processed)}")


# --- Load Model ---
logger.info(f"Loading pre-trained model: {MODEL_CHECKPOINT}...")
# Use TF version for Keras compile/fit
model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

# --- Compile Model (Keras way) ---
# Need optimizer and loss
num_train_steps = len(train_dataset) // BATCH_SIZE * NUM_EPOCHS
optimizer, schedule = create_optimizer(
    init_lr=LEARNING_RATE,
    num_warmup_steps=0, # Standard fine-tuning often uses no warmup or very short
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01, # Typical weight decay
)

# The model output is already logits, loss needs to handle this
# The model itself calculates the loss internally when labels are provided.
# Re-compile the model
logger.info("Compiling Keras model...")
model.compile(optimizer=optimizer)

# --- Prepare TF Datasets MANUALLY ---
logger.info("Converting datasets to tf.data.Dataset format manually...")

def _to_tf_dataset(dataset):
    # Select relevant columns
    columns = ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]
    # Some models might not need token_type_ids
    if "token_type_ids" not in dataset.features:
        columns.remove("token_type_ids")
    
    dataset.set_format(type="tensorflow", columns=columns)
    
    # Create the features dictionary
    features = {col: dataset[col] for col in columns if col not in ["start_positions", "end_positions"]}
    
    # Create the labels tuple
    labels = (dataset["start_positions"], dataset["end_positions"])
    
    return tf.data.Dataset.from_tensor_slices((features, labels))

# Convert train dataset
tf_train_dataset = _to_tf_dataset(train_dataset)
tf_train_dataset = tf_train_dataset.shuffle(len(train_dataset)).batch(BATCH_SIZE)

# For validation, we need features for prediction AND the dataset with IDs/offsets for postprocessing
# We only need inputs for model.predict, not labels
def _to_tf_eval_dataset(dataset):
    columns = ["input_ids", "attention_mask", "token_type_ids"]
    if "token_type_ids" not in dataset.features:
        columns.remove("token_type_ids")
    dataset.set_format(type="tensorflow", columns=columns)
    features = {col: dataset[col] for col in columns}
    return tf.data.Dataset.from_tensor_slices(features)

tf_eval_dataset_for_predict = _to_tf_eval_dataset(val_dataset_processed)
tf_eval_dataset_for_predict = tf_eval_dataset_for_predict.batch(BATCH_SIZE)


# --- Evaluation Setup ---
# Need postprocessing function for SQuAD metrics
# Adapted from: https://huggingface.co/docs/transformers/tasks/question_answering#evaluation
from tqdm.auto import tqdm
import collections

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` best start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not found a non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # We pick our final answer equal to the best answer (unless squad_v2)
        predictions[example["id"]] = best_answer["text"]

    return predictions


squad_metric = load_metric("squad")

# --- Training Arguments ---
logger.info("Setting up Training Arguments...")
# TFTrainingArguments, # Import TFTrainingArguments - REMOVED
# TFTrainer # Import TFTrainer - REMOVED

# --- Compute Metrics Function for Trainer ---
# This function will be called by the Trainer during evaluation
def compute_metrics_for_trainer(eval_preds):
    start_logits, end_logits = eval_preds.predictions
    # The labels here will be the original dataset, need feature mapping
    # It's often easier to do post-processing *after* trainer.predict
    # For now, we'll just return loss (which Trainer calculates anyway)
    # Or we can try the full postprocessing if needed, but it's complex
    # Let's skip metric computation within Trainer for simplicity first
    # and do it manually after training.
    return {} # Return empty dict, rely on loss for now

# --- Initialize Trainer ---
logger.info("Initializing TFTrainer...")
# TFTrainer # Import TFTrainer - REMOVED

# Use DefaultDataCollator for TF
data_collator = DefaultDataCollator(return_tensors="tf")

# Keras callback for evaluation (Re-adding the definition)
class KerasEvalCallback(tf.keras.callbacks.Callback):
    def __init__(self, eval_dataset, raw_val_dataset, features_val_dataset, metric, output_dir):
        self.eval_dataset = eval_dataset # TF dataset format for model.predict
        self.raw_val_dataset = raw_val_dataset # Original HF dataset for context/ID mapping
        self.features_val_dataset = features_val_dataset # Processed HF dataset w/ offset_mapping
        self.metric = metric
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"*** Evaluating on validation set after epoch {epoch+1} ***")
        # raw_predictions is a tuple of (start_logits, end_logits) numpy arrays
        raw_predictions = self.model.predict(self.eval_dataset)

        # Post-process predictions
        final_predictions = postprocess_qa_predictions(
            self.raw_val_dataset, # Needs original examples with context/id
            self.features_val_dataset, # Needs features with offset_mapping/example_id
            raw_predictions,
        )

        # Format for metric
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        formatted_references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.raw_val_dataset]

        # Compute metrics
        metrics = self.metric.compute(predictions=formatted_predictions, references=formatted_references)
        logger.info(f"Epoch {epoch+1} Validation Metrics: {metrics}")

        # Log metrics (e.g., to W&B if integrated)
        if logs is not None:
            logs['val_em'] = metrics['exact_match']
            logs['val_f1'] = metrics['f1']

# --- Training ---
logger.info("Starting training with Keras model.fit()...")

# Instantiate the callback before using it
eval_callback = KerasEvalCallback(
    eval_dataset=tf_eval_dataset_for_predict, # Use the dataset for prediction
    raw_val_dataset=val_dataset_raw,
    features_val_dataset=val_dataset_processed,
    metric=squad_metric,
    output_dir=OUTPUT_DIR
)

# Train using Keras fit method
history = model.fit(
    tf_train_dataset, # Use manually created TF dataset
    # validation_data is not used directly by fit, handled by callback
    epochs=NUM_EPOCHS,
    callbacks=[eval_callback] # Use the evaluation callback
)

# --- Final Evaluation ---
logger.info("Performing final evaluation with Keras model.predict()...")
# Manual Prediction and Evaluation (Simpler for debugging)
logger.info("Manual prediction and evaluation...")
raw_predictions = model.predict(tf_eval_dataset_for_predict) # Use the dataset formatted for prediction

# raw_predictions.predictions contains the output logits (start_logits, end_logits)
# TF model.predict returns a tuple directly if the output is a list/tuple
final_predictions = postprocess_qa_predictions(
    val_dataset_raw, # Original validation examples
    val_dataset_processed, # Validation features
    raw_predictions # Pass the tuple directly
)

formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
formatted_references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_dataset_raw]

final_metrics = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)


logger.info(f"Final Validation Metrics: {final_metrics}")
print("\n" + "=" * 30)
print(f"Baseline Validation Exact Match: {final_metrics['exact_match']:.4f}")
print(f"Baseline Validation F1 Score:    {final_metrics['f1']:.4f}")
print("=" * 30 + "\n")

# --- Save Model ---
# logger.info(f"Saving final model to {OUTPUT_DIR}...")
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
# logger.info("Baseline script finished.") 
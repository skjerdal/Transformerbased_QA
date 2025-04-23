import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForQuestionAnswering,
    create_optimizer,
)
import logging
import collections
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"  # Faster pre-trained model for baseline
MAX_SAMPLES = 500  # Reduced further for initial testing with TF 2.10
VAL_SAMPLES = 100  # Reduced validation samples
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 4  # Significantly reduced batch size for TF 2.10 compatibility
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2  # Reduced epochs for initial testing
OUTPUT_DIR = "baseline_qa_model_tf210"

# Print TensorFlow version and GPU availability for debugging
logger.info(f"TensorFlow version: {tf.__version__}")
gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
logger.info(f"GPU available: {gpu_available}")

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

# --- Preprocessing Functions ---
pad_on_right = tokenizer.padding_side == "right"

def prepare_train_features(examples):
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
model = TFAutoModelForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT)

# --- Convert to TF Datasets (Compatible with TF 2.10) ---
def convert_to_tf_dataset(dataset, batch_size=BATCH_SIZE, shuffle=True):
    """Convert Hugging Face dataset to TF dataset in TF 2.10 compatible way"""
    # Get the feature names excluding labels and offset_mapping
    feature_names = [name for name in dataset.features 
                    if name not in ["start_positions", "end_positions", "offset_mapping", "example_id"]]
    
    # Create numpy arrays for features and labels
    features_dict = {name: np.array(dataset[name]) for name in feature_names}
    
    if "start_positions" in dataset.features and "end_positions" in dataset.features:
        # Training set with labels
        labels = (
            np.array(dataset["start_positions"]),
            np.array(dataset["end_positions"])
        )
        tf_dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
    else:
        # Validation set without labels
        tf_dataset = tf.data.Dataset.from_tensor_slices(features_dict)
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(len(dataset))
    
    return tf_dataset.batch(batch_size)

# Convert datasets to TF format
train_tf_dataset = convert_to_tf_dataset(train_dataset, shuffle=True)
val_tf_dataset = convert_to_tf_dataset(val_dataset_processed, shuffle=False)

# --- PostProcessing Functions for Evaluation ---
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
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
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions

# --- Custom evaluation function ---
def evaluate_qa_model(model, val_dataset, val_features, raw_dataset):
    """Manually evaluate QA model using F1 and EM metrics"""
    from evaluate import load
    squad_metric = load("squad")
    
    # Get predictions
    raw_predictions = model.predict(val_dataset)
    
    # For TF 2.10, the structure might be different, handle both possible outputs
    if isinstance(raw_predictions, tuple):
        # Direct tuple of start_logits, end_logits
        start_logits, end_logits = raw_predictions
    elif hasattr(raw_predictions, 'logits'):
        # HF model returns a 'logits' property in some versions
        start_logits = raw_predictions.logits[0]
        end_logits = raw_predictions.logits[1]
    else:
        try:
            # Try to get start_logits and end_logits as attributes
            start_logits = raw_predictions.start_logits
            end_logits = raw_predictions.end_logits
        except AttributeError:
            # If it's a dict-like object
            try:
                start_logits = raw_predictions['start_logits']
                end_logits = raw_predictions['end_logits']
            except (KeyError, TypeError):
                # Last resort - assuming it's a single tensor output with both logits
                logger.warning("Unable to extract start/end logits in expected format. Using a fallback approach.")
                logits = raw_predictions
                # For QA model output format in TF 2.10
                start_logits = logits[:, 0, :]
                end_logits = logits[:, 1, :]
    
    # Post-process predictions
    final_predictions = postprocess_qa_predictions(
        raw_dataset,
        val_features,
        (start_logits, end_logits),
    )
    
    # Format for metrics
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    formatted_references = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_dataset]
    
    # Compute metrics
    metrics = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
    
    return metrics

# --- Training with custom loop to avoid model.fit() compatibility issues ---
logger.info("Setting up custom training loop...")

# Custom training loop instead of model.fit()
try:
    # Create datasets
    logger.info("Preparing datasets...")
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Track metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    # Training step function
    @tf.function
    def train_step(features, labels):
        with tf.GradientTape() as tape:
            # Forward pass - use the model directly
            outputs = model(features, start_positions=labels[0], end_positions=labels[1], training=True)
            loss = outputs.loss
            
        # Compute gradients and apply
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        train_loss.update_state(loss)
        return loss
    
    # Training loop
    logger.info(f"Starting custom training loop for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss.reset_states()
        
        # Train
        step = 0
        losses = []
        for features, labels in train_tf_dataset:
            loss = train_step(features, labels)
            losses.append(float(loss))
            if step % 10 == 0:
                logger.info(f"  Step {step}, Loss: {loss:.4f}")
            step += 1
        
        # Log epoch results
        avg_loss = sum(losses) / len(losses)
        logger.info(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Evaluate after each epoch
        logger.info(f"Evaluating after epoch {epoch + 1}...")
        metrics = evaluate_qa_model(model, val_tf_dataset, val_dataset_processed, val_dataset_raw)
        logger.info(f"  Validation Metrics - EM: {metrics['exact_match']:.2f}, F1: {metrics['f1']:.2f}")
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    final_metrics = evaluate_qa_model(model, val_tf_dataset, val_dataset_processed, val_dataset_raw)
    
    logger.info("\n" + "=" * 30)
    logger.info(f"Final Validation Exact Match: {final_metrics['exact_match']:.4f}")
    logger.info(f"Final Validation F1 Score:    {final_metrics['f1']:.4f}")
    logger.info("=" * 30 + "\n")
    
    # Save model
    if final_metrics['f1'] > 0:  # Only save if we got meaningful results
        logger.info(f"Saving model to {OUTPUT_DIR}...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Baseline script completed successfully!")
    
except Exception as e:
    logger.error(f"Error during training: {e}")
    import traceback
    traceback.print_exc()
    
    # Provide guidance based on error type
    if "out of memory" in str(e).lower():
        logger.info("OOM error detected. Try the following:")
        logger.info(f"1. Reduce BATCH_SIZE (currently {BATCH_SIZE})")
        logger.info(f"2. Reduce MAX_SAMPLES (currently {MAX_SAMPLES})")
        logger.info(f"3. Reduce MAX_SEQ_LENGTH (currently {MAX_SEQ_LENGTH})")
    elif "dimensions" in str(e).lower() or "shape" in str(e).lower():
        logger.info("Dimension mismatch detected. Check the model inputs and dataset preparation.")
    else:
        logger.info("Try running with fewer samples as a test: change MAX_SAMPLES to 100") 
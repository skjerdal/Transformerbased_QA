import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFDistilBertModel
import logging
import collections
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"  # Base model to use
MAX_SAMPLES = 500  # Reduced for initial testing
VAL_SAMPLES = 100  # Validation samples
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS = 2
OUTPUT_DIR = "tf210_custom_qa_model"

# Print TensorFlow version and GPU availability
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

    # Keep offset mapping for evaluation
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Create offset mapping
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

# --- Convert to TF Datasets ---
def convert_to_tf_dataset(dataset, for_training=True, batch_size=BATCH_SIZE, shuffle=True):
    """Convert Hugging Face dataset to TF dataset"""
    input_ids = np.array(dataset["input_ids"])
    attention_mask = np.array(dataset["attention_mask"])
    
    if for_training:
        # Training set has labels
        start_positions = np.array(dataset["start_positions"])
        end_positions = np.array(dataset["end_positions"])
        
        # Create TF dataset with labels
        tf_dataset = tf.data.Dataset.from_tensor_slices(
            ({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }, 
            {
                "start_positions": start_positions,
                "end_positions": end_positions
            })
        )
    else:
        # Validation set without labels
        tf_dataset = tf.data.Dataset.from_tensor_slices({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(len(dataset))
    
    return tf_dataset.batch(batch_size)

# Create TF datasets
train_tf_dataset = convert_to_tf_dataset(train_dataset, for_training=True, shuffle=True)
val_tf_dataset = convert_to_tf_dataset(val_dataset_processed, for_training=False, shuffle=False)

# --- Create Custom QA Model ---
class CustomQAModel(tf.keras.Model):
    def __init__(self, pretrained_model_name=MODEL_CHECKPOINT):
        super(CustomQAModel, self).__init__()
        self.bert = TFDistilBertModel.from_pretrained(pretrained_model_name)
        self.qa_outputs = tf.keras.layers.Dense(2, name="qa_outputs")  # 2 outputs: start and end positions
    
    def call(self, inputs, training=False):
        # Process input dict or direct tensors
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids = inputs
            attention_mask = None
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        
        sequence_output = outputs[0]  # last hidden states
        
        # Get start/end logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        
        return start_logits, end_logits
    
    def get_config(self):
        config = super().get_config()
        config.update({"pretrained_model_name": MODEL_CHECKPOINT})
        return config

# --- Create Model ---
logger.info("Creating custom QA model...")
model = CustomQAModel(MODEL_CHECKPOINT)

# --- Define loss and metrics ---
def compute_loss(start_positions, end_positions, start_logits, end_logits):
    start_loss = tf.keras.losses.sparse_categorical_crossentropy(
        start_positions, start_logits, from_logits=True
    )
    end_loss = tf.keras.losses.sparse_categorical_crossentropy(
        end_positions, end_logits, from_logits=True
    )
    total_loss = (start_loss + end_loss) / 2
    # Calculate mean over batch
    return tf.reduce_mean(total_loss)

# --- Post-processing for evaluation ---
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

            cls_index = 0  # CLS token is usually at index 0
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

# --- Evaluation Function ---
def evaluate_qa_model(model, val_dataset, val_features, raw_dataset):
    """Manually evaluate QA model using F1 and EM metrics"""
    from evaluate import load
    squad_metric = load("squad")
    
    # Get predictions
    start_logits_list = []
    end_logits_list = []
    
    # Make predictions batch by batch
    for batch in val_dataset:
        start_logits, end_logits = model(batch, training=False)
        start_logits_list.append(start_logits.numpy())
        end_logits_list.append(end_logits.numpy())
    
    # Concatenate batch predictions
    start_logits = np.concatenate(start_logits_list)
    end_logits = np.concatenate(end_logits_list)
    
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

# --- Training Setup ---
logger.info("Setting up training...")
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')

# --- Training Step ---
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        # Forward pass
        start_logits, end_logits = model(features, training=True)
        
        # Compute loss
        loss = compute_loss(
            labels["start_positions"], 
            labels["end_positions"], 
            start_logits, 
            end_logits
        )
    
    # Get gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss.update_state(loss)
    return loss

# --- Training Loop ---
try:
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss.reset_states()
        
        step = 0
        epoch_losses = []
        
        for features, labels in train_tf_dataset:
            loss = train_step(features, labels)
            epoch_losses.append(float(loss.numpy()))
            
            if step % 10 == 0:
                logger.info(f"  Step {step}, Loss: {loss.numpy():.4f}")
            step += 1
        
        # Log epoch results
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        
        # Evaluate
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
    
    # Save model (just the weights since it's a custom model)
    if final_metrics['f1'] > 0:
        logger.info(f"Saving model weights to {OUTPUT_DIR}...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        model.save_weights(os.path.join(OUTPUT_DIR, "qa_model_weights"))
        
        # Save tokenizer for inference
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Save configuration
        import json
        with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
            json.dump({
                "model_name": MODEL_CHECKPOINT,
                "max_seq_length": MAX_SEQ_LENGTH,
                "metrics": {
                    "exact_match": float(final_metrics['exact_match']),
                    "f1": float(final_metrics['f1'])
                }
            }, f)
    
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
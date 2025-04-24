import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFDistilBertModel
import logging
from tqdm.auto import tqdm
import collections
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_CHECKPOINT = "distilbert-base-uncased"
# MAX_SAMPLES = 87000  # Remove - Use full dataset
# VAL_SAMPLES = MAX_SAMPLES // 5  # Remove - Use full dataset
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 16  # Standard batch size
LEARNING_RATE = 5e-5  # Same as your current best configuration
NUM_EPOCHS = 3  # A few epochs for baseline
OUTPUT_DIR = "c:/tf_checkpoints/tf210_custom_qa_model"

# Print TensorFlow version
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

# --- Define the Custom QA Model ---
class CustomQAModel(tf.keras.Model):
    def __init__(self, pretrained_model_name):
        super(CustomQAModel, self).__init__()
        # Load pre-trained DistilBERT model
        self.bert = TFDistilBertModel.from_pretrained(pretrained_model_name)
        # Add QA output layer (single Dense layer for start/end logits)
        self.qa_outputs = tf.keras.layers.Dense(2, name="qa_outputs")
    
    def call(self, inputs, training=False):
        # Process input dict or direct tensors
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask", None)
        else:
            input_ids = inputs[0]
            attention_mask = inputs[1] if len(inputs) > 1 else None
        
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

# --- Load Dataset ---
logger.info("Loading SQuAD dataset...")
squad_dataset = load_dataset("squad")

# --- Select Full Dataset ---
logger.info("Using full train and validation sets...")
train_dataset_raw = squad_dataset["train"] # Use full training set
val_dataset_raw = squad_dataset["validation"] # Use full validation set

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

# --- Create validation dataset WITH labels for loss calculation ---
logger.info("Preprocessing validation data for loss calculation...")
val_dataset_with_labels = val_dataset_raw.map(
    prepare_train_features, # Use the same function as training to get labels
    batched=True,
    remove_columns=val_dataset_raw.column_names,
    desc="Running tokenizer on validation dataset for loss",
)
logger.info(f"Number of features in validation set for loss: {len(val_dataset_with_labels)}")

# --- Convert to TF Datasets ---
def convert_to_tf_dataset(dataset, batch_size=BATCH_SIZE, shuffle=True):
    """Convert Hugging Face dataset to TF dataset in TF 2.10 compatible way"""
    # Extract components
    input_ids = np.array(dataset["input_ids"])
    attention_mask = np.array(dataset["attention_mask"])
    
    # Create features tuple for compatibility
    features = (input_ids, attention_mask)
    
    # For training dataset with labels
    if "start_positions" in dataset.features and "end_positions" in dataset.features:
        start_positions = np.array(dataset["start_positions"])
        end_positions = np.array(dataset["end_positions"])
        labels = (start_positions, end_positions)
        
        # Create as tf.data.Dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        # For validation dataset without labels
        tf_dataset = tf.data.Dataset.from_tensor_slices(features)
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(len(dataset))
    
    return tf_dataset.batch(batch_size)

# Convert datasets to TF format
train_tf_dataset = convert_to_tf_dataset(train_dataset, shuffle=True)
val_tf_dataset = convert_to_tf_dataset(val_dataset_processed, shuffle=False)

# --- Create and compile the model ---
logger.info("Creating and compiling the model...")
model = CustomQAModel(MODEL_CHECKPOINT)

# Define loss function - standard SCE for token classification
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Create optimizer with learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# Define metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_start_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_start_accuracy')
train_end_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_end_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_start_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_start_accuracy')
val_end_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_end_accuracy')

# --- Training step function ---
@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        # Forward pass
        start_logits, end_logits = model(features, training=True)
        
        # Calculate loss (average of start and end positions)
        start_loss = loss_fn(labels[0], start_logits)
        end_loss = loss_fn(labels[1], end_logits)
        loss = (start_loss + end_loss) / 2.0
        
    # Calculate gradients and apply
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss.update_state(loss)
    train_start_accuracy.update_state(labels[0], start_logits)
    train_end_accuracy.update_state(labels[1], end_logits)
    
    return loss

# --- Validation step function ---
@tf.function
def val_step(features, labels):
    # Forward pass (no training)
    start_logits, end_logits = model(features, training=False)
    
    # Ensure both start_logits and labels have same shape except for last dimension
    # The error indicates labels.shape=(384,) and logits.shape=(1, 384)
    # We need to convert labels and ensure consistent dimensions
    batch_size = tf.shape(start_logits)[0]
    
    # Reshape labels if needed to match batch dimension
    start_labels = tf.reshape(labels[0], [batch_size, -1])
    end_labels = tf.reshape(labels[1], [batch_size, -1])
    
    # Calculate loss (average of start and end positions)
    start_loss = loss_fn(tf.squeeze(start_labels), start_logits)
    end_loss = loss_fn(tf.squeeze(end_labels), end_logits)
    loss = (start_loss + end_loss) / 2.0
    
    # Update metrics
    val_loss.update_state(loss)
    val_start_accuracy.update_state(tf.squeeze(start_labels), start_logits)
    val_end_accuracy.update_state(tf.squeeze(end_labels), end_logits)
    
    return start_logits, end_logits

# --- PostProcessing Function for Evaluation ---
def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    """Convert model predictions to readable answers"""
    all_start_logits, all_end_logits = raw_predictions
    
    # Make sure predictions list length matches features
    logger.info(f"Features length: {len(features)}, Predictions length: {len(all_start_logits)}")
    
    # In case lengths don't match, let's truncate to the smaller size
    min_length = min(len(features), len(all_start_logits))
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    
    # Only map features we have predictions for
    for i in range(min_length):
        feature = features[i]
        if i < min_length:  # Safety check
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = collections.OrderedDict()
    logger.info(f"Post-processing {len(examples)} example predictions split into {min_length} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []
        context = example["context"]
        
        for feature_index in feature_indices:
            if feature_index >= min_length:
                continue  # Skip if out of range
                
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
            # In the very rare edge case we have not found a non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions

# --- Evaluation function using SQuAD metrics ---
def compute_metrics(predictions, references):
    """Compute Exact Match and F1 scores for SQuAD"""
    from evaluate import load
    squad_metric = load("squad")
    
    # Format for metrics
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    formatted_references = [{"id": ex["id"], "answers": ex["answers"]} for ex in references]
    
    # Compute metrics
    metrics = squad_metric.compute(predictions=formatted_predictions, references=formatted_references)
    
    return metrics

# --- Full Evaluation ---
def evaluate_model():
    """Run full evaluation with EM/F1 metrics"""
    logger.info("Running full evaluation...")
    val_loss.reset_states() # Reset validation loss metric

    # Lists to collect outputs
    all_start_logits = []
    all_end_logits = []
    
    # Process all validation data in batches
    all_input_ids = []
    all_attention_masks = []
    for i in range(len(val_dataset_processed)):
        all_input_ids.append(val_dataset_processed[i]["input_ids"])
        all_attention_masks.append(val_dataset_processed[i]["attention_mask"])
    
    # Also collect labels for loss calculation
    all_start_positions = np.array(val_dataset_with_labels["start_positions"])
    all_end_positions = np.array(val_dataset_with_labels["end_positions"])

    # Create proper batches for the entire validation set
    total_eval_batches = len(all_input_ids) // BATCH_SIZE + (1 if len(all_input_ids) % BATCH_SIZE > 0 else 0)
    logger.info(f"Processing {len(all_input_ids)} validation features in {total_eval_batches} batches")
    
    for batch_idx in range(total_eval_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_input_ids))
        
        # Get batch data
        batch_input_ids = np.array(all_input_ids[start_idx:end_idx])
        batch_attention_mask = np.array(all_attention_masks[start_idx:end_idx])
        batch_start_labels = all_start_positions[start_idx:end_idx]
        batch_end_labels = all_end_positions[start_idx:end_idx]

        # Forward pass
        features = (batch_input_ids, batch_attention_mask)
        start_logits, end_logits = model(features, training=False)

        # Calculate loss for the batch
        batch_start_loss = loss_fn(batch_start_labels, start_logits)
        batch_end_loss = loss_fn(batch_end_labels, end_logits)
        batch_loss = (batch_start_loss + batch_end_loss) / 2.0
        
        # Update validation loss metric
        val_loss.update_state(batch_loss)

        # Convert to numpy and store logits for postprocessing
        all_start_logits.extend(start_logits.numpy())
        all_end_logits.extend(end_logits.numpy())
    
    # Post-process predictions
    raw_predictions = (all_start_logits, all_end_logits)
    final_predictions = postprocess_qa_predictions(
        val_dataset_raw, val_dataset_processed, raw_predictions
    )
    
    # Compute EM and F1 scores
    metrics = compute_metrics(final_predictions, val_dataset_raw)

    logger.info(f"Validation Loss: {val_loss.result():.4f}") # Log the calculated loss
    logger.info(f"Exact Match: {metrics['exact_match']:.2f}")
    logger.info(f"F1 Score: {metrics['f1']:.2f}")

    return metrics, final_predictions

# --- Best validation results tracking ---
best_val_loss = float('inf')
best_metrics = None
early_stop_patience = 5
no_improvement_count = 0

# --- Training Loop ---
try:
    logger.info(f"Starting training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        # Reset train metrics at start of each epoch
        train_loss.reset_states()
        train_start_accuracy.reset_states()
        train_end_accuracy.reset_states()
        # val_loss.reset_states() # Moved reset inside evaluate_model
        
        # Training loop
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, (features, labels) in enumerate(train_tf_dataset):
            loss = train_step(features, labels)
            
            # Log progress every 50 steps
            if step % 50 == 0:
                logger.info(f"  Step {step}, Loss: {loss:.4f}, "
                           f"Start Acc: {train_start_accuracy.result():.4f}, "
                           f"End Acc: {train_end_accuracy.result():.4f}")
        
        # Log epoch results
        logger.info(f"  Training Loss: {train_loss.result():.4f}")
        logger.info(f"  Training Start Accuracy: {train_start_accuracy.result():.4f}")
        logger.info(f"  Training End Accuracy: {train_end_accuracy.result():.4f}")
        
        # Evaluate after each epoch
        metrics, _ = evaluate_model()
        
        # Check for improvement and early stopping
        current_val_loss = val_loss.result()
        if current_val_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}")
            best_val_loss = current_val_loss
            best_metrics = metrics
            no_improvement_count = 0
            
            # Save model weights
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            model.save_weights(os.path.join(OUTPUT_DIR, "qa_model_weights"))
            
            # Save config
            config = {
                "model_name": MODEL_CHECKPOINT,
                "max_seq_length": MAX_SEQ_LENGTH,
                "doc_stride": DOC_STRIDE,
                "best_epoch": epoch + 1,
                "metrics": {
                    "em": metrics["exact_match"],
                    "f1": metrics["f1"],
                    "val_loss": float(best_val_loss)
                }
            }
            with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
                
            # Save tokenizer
            tokenizer.save_pretrained(OUTPUT_DIR)
        else:
            no_improvement_count += 1
            logger.info(f"No improvement in validation loss for {no_improvement_count} epochs.")
            
            if no_improvement_count >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                break
    
    # Print final results
    logger.info("\n" + "=" * 30)
    logger.info("Training completed!")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Best Exact Match: {best_metrics['exact_match']:.2f}")
    logger.info(f"Best F1 Score: {best_metrics['f1']:.2f}")
    logger.info("=" * 30 + "\n")
    
    logger.info(f"Model saved to {OUTPUT_DIR}")

except Exception as e:
    logger.error(f"Error during training: {e}")
    import traceback
    traceback.print_exc() 
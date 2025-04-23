import numpy as np
import tensorflow as tf
from models.qa_transformer import build_qa_transformer_model
from utils.squad_preprocessing import prepare_squad_data, load_squad_data
from utils.evaluation import time_training, compute_eval_metrics, get_predictions_from_logits, format_predictions_for_evaluation, format_references_for_evaluation
import wandb
# Use the original Wandb Keras callback
from wandb.integration.keras import WandbCallback
# Import new Hugging Face data pipeline
from utils.hf_squad_preprocessing import prepare_squad_data_with_hf
import os
import logging # Make sure logging is imported
# Import AdamW from Tensorflow Addons
import tensorflow_addons as tfa # Comment out or remove

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom Learning Rate Schedule with Warmup --- 
class WarmupPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies linear warmup followed by polynomial decay."""
    def __init__(self, initial_learning_rate, decay_steps, end_learning_rate, power, warmup_steps, name=None):
        super(WarmupPolynomialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.warmup_steps = warmup_steps
        self.name = name

        # Create the underlying polynomial decay schedule
        self.polynomial_decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=power,
            cycle=False
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32) # Ensure step is float
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)

        # Calculate warmup learning rate
        # Linear increase from 0 to initial_learning_rate
        global_step_float = tf.cast(step, tf.float32)
        warmup_percent_done = global_step_float / warmup_steps_float
        warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

        # Calculate polynomial decay learning rate (adjusting step for decay phase)
        is_decay_phase = tf.cast(global_step_float > warmup_steps_float, tf.float32)
        decay_step = global_step_float - warmup_steps_float
        # Make sure decay_step is not negative if global_step_float <= warmup_steps_float
        decay_step = tf.maximum(0.0, decay_step)
        polynomial_decay_learning_rate = self.polynomial_decay_schedule(decay_step)

        # Choose warmup or decay based on step
        # Handle case where warmup_steps is 0
        learning_rate = tf.cond(
            tf.equal(warmup_steps_float, 0.0),
            lambda: self.initial_learning_rate, # No warmup, start decaying immediately
            lambda: tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: polynomial_decay_learning_rate
            )
        )
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "power": self.power,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }

# Set memory growth to avoid OOM errors on smaller GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':

    # --- Hyperparameters Reverted to Training Test 5 Style --- 
    print("\n*** RUNNING WITH CUSTOM MODEL CONFIGURATION (Reverted) ***\n")
    hyperparameters = {
        'TOKENIZER': 'bert-base-uncased', # Back to BERT tokenizer
        'SEQUENCE_LEN': 384,
        # Dimensions from Training Test 1/5
        'D_MODEL': 768,
        'NUM_HEADS': 12,
        'DFF': 2048, # From Training Test 1/5
        'NUM_LAYERS': 6, # Revert to 6 layers like Test 16
        'DROPOUT_RATE': 0.1, # Revert to 0.1 dropout like Test 16
        'BATCH_SIZE': 16,
        'EPOCHS': 40, # Keep high for long run
        'INITIAL_LR': 5e-5, # From Test 16
        'END_LR': 0.0, # For decay schedule
        'WARMUP_STEPS': 0.1, # Keep warmup
        'MAX_SAMPLES': None, # Use full dataset
    }

    # Initialize W&B Run
    wandb.init(project="squad-transformer-qa", config=hyperparameters) # Use main project
    config = wandb.config

    print("Preparing SQuAD dataset using Hugging Face Datasets and BertTokenizerFast...")
    train_dataset, val_dataset, tokenizer, train_examples, val_examples = prepare_squad_data_with_hf(
        tokenizer_name=config.TOKENIZER,
        max_seq_length=config.SEQUENCE_LEN,
        max_samples=config.MAX_SAMPLES,
        batch_size=config.BATCH_SIZE
    )
    actual_vocab_size = tokenizer.vocab_size
    print(f"Using actual vocabulary size: {actual_vocab_size}")
    steps_per_epoch = len(train_dataset)
    validation_steps = len(val_dataset)
    total_steps = steps_per_epoch * config.EPOCHS
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Total training steps: {total_steps}")

    print("Building QA transformer model (custom)...")
    qa_model = build_qa_transformer_model(
        actual_vocab_size,
        config.SEQUENCE_LEN,
        config.D_MODEL,
        config.NUM_HEADS,
        config.DFF,
        config.NUM_LAYERS,
        config.DROPOUT_RATE
        # Removed model_checkpoint and load_pretrained_weights args
    )

    # Loss functions - Revert to standard functional loss without label smoothing
    def start_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    def end_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    # --- Use AdamW with Custom Warmup + PolynomialDecay schedule ---
    print("\n--- USING AdamW Optimizer with WarmupPolynomialDecay --- ")
    # Calculate decay steps (total steps *after* warmup)
    # warmup_steps = int(total_steps * config.WARMUP_STEPS)
    # decay_steps = total_steps - warmup_steps
    # if decay_steps <= 0:
    #     decay_steps = 1 # Avoid division by zero or negative steps
    #     logger.warning(f"Total steps ({total_steps}) less than or equal to warmup steps ({warmup_steps}). Setting decay_steps to 1.")

    # Instantiate the custom learning rate schedule
    # learning_rate_schedule = WarmupPolynomialDecay(
    #     initial_learning_rate=config.INITIAL_LR,
    #     decay_steps=decay_steps,
    #     end_learning_rate=config.END_LR,
    #     power=1.0, # Linear decay after warmup
    #     warmup_steps=warmup_steps
    # )

    # --- Revert to standard Adam (like Test 16) ---
    # print("\n--- USING standard Adam Optimizer with WarmupPolynomialDecay --- ") # Modified print statement
    print("\n--- USING standard Adam Optimizer with Fixed Learning Rate --- ")
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.INITIAL_LR # Pass the fixed initial learning rate directly
        # No weight decay parameter for standard Adam
    )

    qa_model.compile(
        optimizer=optimizer,
        loss=[start_loss, end_loss],
        metrics=['accuracy']
    )
    qa_model.summary()

    print("Fine-tuning the model...")

    # --- Restore Original Callbacks --- 
    print("\n--- RE-ENABLING EARLY STOPPING AND CHECKPOINTING ---")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, # Back to patience 5 from Test 5
        restore_best_weights=True
    )
    checkpoint_dir = 'C:/tf_checkpoints/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'best_qa_model_custom_full'), # Save to directory
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False, # Save entire model (architecture + weights + optimizer)
        verbose=1
    )
    callbacks = [early_stopping, checkpoint, WandbCallback(save_model=False)]

    # --- Add this section before model.fit() ---
    print("\n--- Attempting to load checkpoint if exists ---")
    checkpoint_path = os.path.join(checkpoint_dir, 'best_qa_model_custom_full')
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}. Loading model...")
        # Load the entire model state, including optimizer
        qa_model = tf.keras.models.load_model(checkpoint_path, custom_objects={
            'WarmupPolynomialDecay': WarmupPolynomialDecay,
            'start_loss': start_loss, 
            'end_loss': end_loss
            })
        print("Model loaded successfully.")
        # Optionally find the epoch to start from - simpler just to run fit and let it figure out steps
        # initial_epoch = # Difficult to get accurately without extra saving, but not strictly necessary
    else:
        print("No checkpoint found. Starting training from scratch.")
        # Build and compile model as before (already done earlier in script)
    # --- End of added section ---

    print("\n--- USING ORIGINAL VALIDATION DATA ---")
    history, total_training_time = time_training(
        qa_model,
        train_dataset,
        config.EPOCHS, # Continue training up to the total specified epochs
        validation_data=val_dataset,
        callbacks=callbacks,
        validation_steps=validation_steps
        # initial_epoch=initial_epoch # Optional but helpful for logging
    )
    print(f"Training completed in {total_training_time:.2f} seconds")

    # --- Simplified Evaluation (Just log final metrics) ---
    print("\nEvaluating final model on validation subset...")
    eval_table = wandb.Table(columns=["Example ID", "Question", "Context", "True Answer", "Predicted Answer", "EM", "F1"])
    all_predictions = []
    all_references = []
    num_eval_examples = 0
    max_log_examples = 100
    for batch, labels in val_dataset.take((max_log_examples + config.BATCH_SIZE - 1) // config.BATCH_SIZE):
        input_ids, attention_mask = batch
        start_positions, end_positions = labels
        start_logits, end_logits = qa_model.predict((input_ids, attention_mask), verbose=0)
        predicted_texts = get_predictions_from_logits(input_ids, start_logits, end_logits, tokenizer)
        for i in range(len(input_ids)):
            if num_eval_examples >= max_log_examples: break
            example_index_approx = num_eval_examples
            if example_index_approx < len(val_examples):
                orig_example = val_examples[example_index_approx]
                example_id = orig_example["id"]
                context = orig_example["context"]
                question = orig_example["question"]
                answers = orig_example["answers"]
                prediction = {"id": example_id, "prediction_text": predicted_texts[i]}
                reference = {"id": example_id, "answers": {"text": answers["text"], "answer_start": answers["answer_start"]}}
                metrics = compute_eval_metrics([prediction], [reference])
                eval_table.add_data(example_id, question, context[:100] + "...", answers["text"][0] if answers["text"] else "N/A", predicted_texts[i], metrics["exact_match"], metrics["f1"])
                all_predictions.append(prediction)
                all_references.append(reference)
            num_eval_examples += 1
        if num_eval_examples >= max_log_examples: break

    if all_predictions:
        overall_metrics = compute_eval_metrics(all_predictions, all_references)
    else:
        overall_metrics = {"exact_match": 0.0, "f1": 0.0}

    print("\n" + "=" * 30)
    print(f"Final Validation EM (subset): {overall_metrics['exact_match']:.4f}")
    print(f"Final Validation F1 (subset):    {overall_metrics['f1']:.4f}")
    print("=" * 30 + "\n")

    wandb.log({
        "final_validation_em_subset": overall_metrics["exact_match"],
        "final_validation_f1_subset": overall_metrics["f1"],
        "final_validation_examples": eval_table
    })

    wandb.finish() 
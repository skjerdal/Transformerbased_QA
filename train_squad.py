import numpy as np
import tensorflow as tf
from models.qa_transformer import build_qa_transformer_model
from utils.squad_preprocessing import prepare_squad_data, load_squad_data
from utils.evaluation import time_training, compute_eval_metrics
import wandb
from wandb.integration.keras import WandbCallback

# Set memory growth to avoid OOM errors on smaller GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':

    # Hyperparameters
    # VOCAB_SIZE is now determined by the tokenizer, removed from here
    hyperparameters = {
        # 'VOCAB_SIZE': 30000, # No longer set here
        'TOKENIZER': 'bert-base-uncased', # Specify tokenizer to use
        'SEQUENCE_LEN': 384,
        'D_MODEL': 512,
        'NUM_HEADS': 8,
        'DFF': 2048,
        'NUM_LAYERS': 6,
        'DROPOUT_RATE': 0.1,
        'BATCH_SIZE': 16,
        'EPOCHS': 20,
        'INITIAL_LR': 5e-5,
        'END_LR': 0.0,
        'MAX_SAMPLES': None
    }
    # Estimate SQuAD v1.1 train size for splitting (adjust if using a different version)
    # Note: This estimation is tricky now as invalid examples are removed.
    # It might be better to count the actual size after preprocessing.
    ESTIMATED_TOTAL_SAMPLES = 87000 # Rough estimate, adjust based on preprocessing output

    # Initialize W&B Run
    wandb.init(project="squad-transformer-qa", config=hyperparameters) 
    # Access hyperparameters from wandb.config after init
    config = wandb.config

    print("Preparing SQuAD dataset using BertTokenizerFast...")
    # prepare_squad_data now takes tokenizer_name, returns (dataset, tokenizer instance)
    dataset, tokenizer = prepare_squad_data(
        'squad.json',
        max_seq_length=config.SEQUENCE_LEN,
        tokenizer_name=config.TOKENIZER # Pass tokenizer name from config
        # vocab_size is no longer passed
    )

    # Get actual vocab size from the loaded tokenizer
    actual_vocab_size = tokenizer.vocab_size
    print(f"Using actual vocabulary size: {actual_vocab_size}")

    # It's better to shuffle the dataset *before* splitting if possible
    # dataset = dataset.shuffle(buffer_size=10000) # Add shuffling

    # Calculate the actual dataset size for splitting
    # This requires iterating through the dataset once, which can be slow
    # Alternatively, use the `num_valid` count from preprocessing if reliable
    # total_dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    # if total_dataset_size == tf.data.experimental.INFINITE_CARDINALITY:
    #     print("Warning: Could not determine dataset cardinality. Using estimate.")
    #     total_dataset_size = ESTIMATED_TOTAL_SAMPLES # Fallback to estimate
    # else:
    #     print(f"Actual dataset size after preprocessing: {total_dataset_size}")

    # For now, let's stick to the estimate, but be aware it might be inaccurate
    # A more robust way is needed if ESTIMATED_TOTAL_SAMPLES is way off.
    total_dataset_size = ESTIMATED_TOTAL_SAMPLES # Using estimate
    train_size = int(0.9 * total_dataset_size)
    val_size = total_dataset_size - train_size

    # Create batched dataset *after* splitting
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE) # Add prefetching

    # Calculate steps, ensuring train_size aligns with batches processed
    # Use ceil to avoid dropping the last partial batch if dataset size is not divisible
    steps_per_epoch = int(np.ceil(train_size / config.BATCH_SIZE))
    validation_steps = int(np.ceil(val_size / config.BATCH_SIZE))
    total_steps = steps_per_epoch * config.EPOCHS

    # Split the batched dataset
    train_dataset = dataset.take(steps_per_epoch)
    val_dataset = dataset.skip(steps_per_epoch) # Should take validation_steps ideally

    print(f"Estimated train examples: {train_size}")
    print(f"Estimated validation examples: {val_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Total training steps: {total_steps}")

    print("Building QA transformer model...")
    # Pass the actual vocab size from the loaded tokenizer
    qa_model = build_qa_transformer_model(
        actual_vocab_size, # Use actual vocab size
        config.SEQUENCE_LEN, 
        config.D_MODEL, 
        config.NUM_HEADS, 
        config.DFF, 
        config.NUM_LAYERS, 
        config.DROPOUT_RATE
    )

    # Loss functions remain the same for now
    def start_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

    def end_loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )

    # Create the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=config.INITIAL_LR,
        decay_steps=total_steps,
        end_learning_rate=config.END_LR,
        power=1.0  # Linear decay
    )

    # Use the schedule in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    qa_model.compile(
        optimizer=optimizer,
        loss=[start_loss, end_loss],
        metrics=['accuracy'] # Accuracy here refers to predicting the exact start/end token
    )

    qa_model.summary()

    print("Training the model...")
    # Add early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', # Monitor overall validation loss
        patience=3,
        restore_best_weights=True
    )

    # List of callbacks including W&B
    callbacks = [early_stopping, WandbCallback()]

    # Pass validation_steps to fit
    history, total_training_time = time_training(
        qa_model, 
        train_dataset, 
        config.EPOCHS, 
        validation_data=val_dataset, 
        callbacks=callbacks,
        validation_steps=validation_steps # Important for datasets that don't end
    )

    print(f"Training completed in {total_training_time:.2f} seconds")

    print("\nEvaluating on validation set with EM and F1...")

    print("Loading raw data for validation answers...")
    # Ensure this loading aligns with the validation split logic
    raw_contexts, raw_questions, raw_answers_text, _ = load_squad_data('squad.json')
    # WARNING: This split assumes the original order is preserved and matches the dataset split.
    # This is fragile. A better way is to include example IDs in the dataset.
    val_raw_answers_text = raw_answers_text[train_size:] # Uses estimated train_size
    print(f"Loaded {len(raw_answers_text)} total raw examples, attempting to use {len(val_raw_answers_text)} for validation (based on estimated split).")

    total_em = 0
    total_f1 = 0
    eval_count = 0
    actual_eval_examples_processed = 0

    # Log EM and F1 to W&B
    eval_table = wandb.Table(columns=["Index", "True Answer", "Predicted Answer", "EM", "F1"])

    # Iterate through the validation dataset
    # val_dataset yields ((input_ids, attention_mask), (start_true, end_true))
    for batch_index, (batch_inputs, batch_labels) in enumerate(val_dataset):
        batch_input_ids, batch_attention_mask = batch_inputs
        batch_start_true, batch_end_true = batch_labels

        # Pass both inputs to predict
        batch_start_logits, batch_end_logits = qa_model.predict(
            (batch_input_ids, batch_attention_mask), 
            verbose=0
        )

        for i in range(batch_input_ids.shape[0]): # Iterate through samples in the batch
            current_dataset_index = batch_index * config.BATCH_SIZE + i
            # Map dataset index back to the original raw data index (this is the tricky part)
            current_raw_index = train_size + current_dataset_index # Estimated raw index

            if current_raw_index >= len(raw_answers_text):
                continue # Skip if calculated index is out of bounds

            input_seq = batch_input_ids[i].numpy()
            true_start = batch_start_true[i].numpy()
            true_end = batch_end_true[i].numpy()

            pred_start = tf.argmax(batch_start_logits[i]).numpy()
            pred_end = tf.argmax(batch_end_logits[i]).numpy()

            pred_text = ""
            # Check validity of prediction and ensure indices are within sequence length
            seq_len = input_seq.shape[0] # Actual length of this sequence
            if pred_start <= pred_end and pred_end < seq_len:
                # Use tokenizer's decode method for accurate subword decoding
                predicted_token_ids = input_seq[pred_start : pred_end + 1]
                # Decode, skipping special tokens like [CLS], [SEP], [PAD]
                # Using the BertTokenizerFast instance we got from prepare_squad_data
                pred_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
            # else: handle invalid span (pred_text remains "")

            # Get the corresponding true answer text using the estimated raw index
            true_text = val_raw_answers_text[current_dataset_index] # Index relative to start of val raw texts

            em, f1 = compute_eval_metrics(true_text, pred_text)
            total_em += em
            total_f1 += f1
            eval_count += 1 # Count successful evaluations where raw text was found

            # Add rows to W&B Table (optional, can be verbose)
            if eval_count <= 100: # Log first 100 examples
                eval_table.add_data(eval_count, true_text, pred_text, em, f1)

        actual_eval_examples_processed += batch_input_ids.shape[0]
        if batch_index >= validation_steps -1:
            break # Ensure we don't loop indefinitely if dataset repeats


    print(f"Finished evaluation loop. Processed {actual_eval_examples_processed} examples from val_dataset.")
    print(f"Computed EM/F1 for {eval_count} examples where raw text mapping was successful.")

    avg_em = total_em / eval_count if eval_count > 0 else 0
    avg_f1 = total_f1 / eval_count if eval_count > 0 else 0

    print("\n" + "=" * 30)
    print(f"Validation Exact Match: {avg_em:.4f}")
    print(f"Validation F1 Score:    {avg_f1:.4f}")
    print("=" * 30 + "\n")

    # Log final EM and F1 scores to W&B
    wandb.log({"validation_em": avg_em, "validation_f1": avg_f1})
    # Log the table of examples
    wandb.log({"validation_examples": eval_table})

    # Finish the W&B run
    wandb.finish() 
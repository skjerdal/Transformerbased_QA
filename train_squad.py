import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from models.qa_transformer import build_qa_transformer_model
from utils.squad_preprocessing import prepare_squad_data, load_squad_data
from utils.evaluation import time_training, compute_eval_metrics

# Set memory growth to avoid OOM errors on smaller GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':

    # Updated Hyperparameters for longer training & larger model
    VOCAB_SIZE = 16000  # Adjusted based on previous run's model summary
    SEQUENCE_LEN = 384 # Increased to reduce truncation
    D_MODEL = 512      # Adjusted based on previous run's model summary
    NUM_HEADS = 8      # Adjusted (128 divisible by 4)
    DFF = 2048         # Adjusted (4 * D_MODEL)
    NUM_LAYERS = 6     # Adjusted based on previous run's model summary
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 16   # Kept same to manage memory with increased SEQ_LEN
    EPOCHS = 12
    INITIAL_LR = 5e-5  # Starting learning rate
    END_LR = 0.0       # End learning rate for decay
    MAX_SAMPLES = None # Use full dataset
    # Estimate SQuAD v1.1 train size for splitting (adjust if using a different version)
    ESTIMATED_TOTAL_SAMPLES = 87000

    print("Preparing SQuAD dataset...")
    dataset, tokenizer = prepare_squad_data(
        'squad.json',
        max_seq_length=SEQUENCE_LEN,
        vocab_size=VOCAB_SIZE
    )

    dataset = dataset.batch(BATCH_SIZE)

    train_size = int(0.9 * ESTIMATED_TOTAL_SAMPLES)
    val_size = ESTIMATED_TOTAL_SAMPLES - train_size
    # Calculate steps per epoch and total steps for the learning rate schedule
    steps_per_epoch = train_size // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS

    train_dataset = dataset.take(steps_per_epoch) # Use calculated steps
    val_dataset = dataset.skip(steps_per_epoch)

    print(f"Estimated train dataset size (samples): {train_size}")
    print(f"Estimated validation dataset size (samples): {val_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")

    print("Building QA transformer model...")
    qa_model = build_qa_transformer_model(
        VOCAB_SIZE, SEQUENCE_LEN, D_MODEL, NUM_HEADS, DFF, NUM_LAYERS, DROPOUT_RATE
    )

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
        initial_learning_rate=INITIAL_LR,
        decay_steps=total_steps,
        end_learning_rate=END_LR,
        power=1.0  # Linear decay
    )

    # Use the schedule in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    qa_model.compile(
        optimizer=optimizer,
        loss=[start_loss, end_loss],
        metrics=['accuracy']
    )

    qa_model.summary()

    print("Training the model...")
    # Add early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history, total_training_time = time_training(qa_model, train_dataset, EPOCHS, val_dataset, early_stopping)

    print(f"Training completed in {total_training_time:.2f} seconds")

    print("\nEvaluating on validation set with EM and F1...")

    print("Loading raw data for validation answers...")
    raw_contexts, raw_questions, raw_answers_text, _ = load_squad_data('squad.json')
    val_answers_text = raw_answers_text[train_size:]
    print(f"Loaded {len(raw_answers_text)} total raw examples, attempting to use {len(val_answers_text)} for validation.")

    total_em = 0
    total_f1 = 0
    eval_count = 0

    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

    for batch_index, (batch_inputs, batch_labels) in enumerate(val_dataset):
        batch_start_true, batch_end_true = batch_labels

        batch_start_logits, batch_end_logits = qa_model.predict(batch_inputs, verbose=0)

        for i in range(batch_inputs.shape[0]):
            current_raw_index = batch_index * BATCH_SIZE + i
            if current_raw_index >= len(val_answers_text):
                break

            input_seq = batch_inputs[i].numpy()
            true_start = batch_start_true[i].numpy()
            true_end = batch_end_true[i].numpy()

            pred_start = tf.argmax(batch_start_logits[i]).numpy()
            pred_end = tf.argmax(batch_end_logits[i]).numpy()

            pred_text = ""
            if pred_start <= pred_end:
                predicted_token_ids = input_seq[pred_start : pred_end + 1]
                predicted_token_ids = [tok for tok in predicted_token_ids if tok > 2]
                predicted_words = [reverse_word_index.get(tok_id, '<UNK>') for tok_id in predicted_token_ids]
                pred_text = " ".join(predicted_words)

            true_text = val_answers_text[current_raw_index]

            em, f1 = compute_eval_metrics(true_text, pred_text)
            total_em += em
            total_f1 += f1
            eval_count += 1

            if eval_count <= 5:
                print("-" * 20)
                print(f"Example {eval_count}")
                print(f"  True Answer: '{true_text}'")
                print(f"  Pred Answer: '{pred_text}' (Indices: {pred_start}-{pred_end})")
                print(f"  EM: {em}, F1: {f1}")

    avg_em = total_em / eval_count if eval_count > 0 else 0
    avg_f1 = total_f1 / eval_count if eval_count > 0 else 0

    print("\n" + "=" * 30)
    print(f"Validation Results ({eval_count} examples):")
    print(f"  Average Exact Match (EM): {avg_em:.4f}")
    print(f"  Average F1 Score:         {avg_f1:.4f}")
    print("=" * 30)

    qa_model.save('squad_transformer_model')
    print("\nModel saved to 'squad_transformer_model'") 
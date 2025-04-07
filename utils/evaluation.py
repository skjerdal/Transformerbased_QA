import time
import re
import string
from collections import Counter
import tensorflow as tf # Added for example usage explanation

def time_training(model, dataset, epochs, validation_data=None, callbacks=None, validation_steps=None):
    """Time the training process and return history."""
    start_time = time.time()
    history = model.fit(
        dataset,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks, # Pass callbacks as a list
        validation_steps=validation_steps # Add validation_steps here
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")
    return history, training_time

def evaluate_component(component, input_data):
    """Evaluate a component's execution time and output.
       This can be extended to include memory profiling or other computational metrics."""
    start = time.time()
    output = component(input_data)
    end = time.time()
    compute_time = end - start
    print(f"Component {component.__class__.__name__} compute time: {compute_time:.4f} seconds")
    return output, compute_time

# --- SQuAD Evaluation Metrics ---

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(gold_toks, pred_toks):
    """Computes F1 score based on token overlap."""
    # Ensure tokens are strings (debugging potential issue)
    gold_toks = [str(tok) for tok in gold_toks]
    pred_toks = [str(tok) for tok in pred_toks]

    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, F1 is 1 if both are empty, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_exact(gold_answer, pred_answer):
    """Computes Exact Match score."""
    return int(normalize_text(gold_answer) == normalize_text(pred_answer))

def get_tokens(s):
    """Splits normalized text into tokens."""
    if not s: return []
    return normalize_text(s).split()

def compute_eval_metrics(gold_answer, pred_answer):
    """Computes both F1 and Exact Match scores."""
    normalized_gold = normalize_text(gold_answer)
    normalized_pred = normalize_text(pred_answer)

    gold_tokens = get_tokens(normalized_gold)
    pred_tokens = get_tokens(normalized_pred)

    # Use normalized versions for both EM and F1 for consistency
    exact_score = compute_exact(normalized_gold, normalized_pred)
    f1_score = compute_f1(gold_tokens, pred_tokens)

    return exact_score, f1_score

# --- How to use these (Example Explanation) ---
#
# To get EM/F1 scores after training in train_squad.py:
# 1. Keep aside a validation set *before* creating the tf.data.Dataset,
#    containing the raw texts (contexts, questions, answers_text) and the
#    processed inputs/labels (padded_inputs, start_positions, end_positions).
#    Let's call these val_raw_texts and val_processed_data.
# 2. Get the trained tokenizer object.
# 3. After model training (model.fit), iterate through val_processed_data:
#    for input_seq, (true_start, true_end) in val_processed_data:
#        # Get model predictions (logits)
#        start_logits, end_logits = model.predict(tf.expand_dims(input_seq, axis=0))
#
#        # Find best start/end index from logits
#        pred_start = tf.argmax(start_logits, axis=1).numpy()[0]
#        pred_end = tf.argmax(end_logits, axis=1).numpy()[0]
#
#        # Ensure start <= end (add logic if needed, e.g., if pred_start > pred_end, maybe swap or consider invalid)
#        if pred_start > pred_end:
#            # Handle invalid span, e.g., score as 0 or use only start/end if possible
#            pred_text = "" # Or some other default
#        else:
#            # Convert predicted token indices back to text
#            # Need the original input_seq *before* padding potentially, or handle padding tokens
#            predicted_token_ids = input_seq[pred_start : pred_end + 1]
#            # Filter out special tokens (CLS=1, SEP=2, PAD=0) if they fall in the span
#            predicted_token_ids = [tok for tok in predicted_token_ids if tok > 2]
#            pred_text = tokenizer.sequences_to_texts([predicted_token_ids])[0]
#
#        # Get the ground truth answer text from val_raw_texts using the current index
#        true_text = val_raw_texts['answers'][current_index] # Assuming you stored it
#
#        # Calculate metrics for this example
#        em, f1 = compute_eval_metrics(true_text, pred_text)
#        # Accumulate em_total += em, f1_total += f1
#
# 4. After the loop, calculate average EM and F1:
#    avg_em = em_total / num_validation_examples
#    avg_f1 = f1_total / num_validation_examples
#    print(f"Validation EM: {avg_em:.4f}, Validation F1: {avg_f1:.4f}")
#
# Note: The example explanation assumes you're using a tokenizer to convert token indices back to text.
# If you're using a different method to get predictions, you'll need to adjust the example code accordingly. 
import time
import re
import string
from collections import Counter
import tensorflow as tf # Added for example usage explanation
from evaluate import load as load_metric

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

# --- SQuAD Evaluation Metrics using Hugging Face's evaluate library ---

# Load the SQuAD metric from Hugging Face's evaluate library
squad_metric = load_metric("squad")

def compute_eval_metrics(predictions, references):
    """
    Compute Exact Match and F1 scores using Hugging Face's evaluate library.
    
    Args:
        predictions: List of dictionaries with 'id' and 'prediction_text' fields
        references: List of dictionaries with 'id' and 'answers' fields
                   where 'answers' is a dict with 'text' and 'answer_start' fields
    
    Returns:
        Dictionary with metrics (exact_match, f1)
    """
    # Format predictions and references as required by Hugging Face's SQuAD metric
    formatted_predictions = [
        {"id": p["id"], "prediction_text": p["prediction_text"]} 
        for p in predictions
    ]
    
    formatted_references = [
        {"id": r["id"], "answers": r["answers"]} 
        for r in references
    ]
    
    # Compute metrics using Hugging Face's SQuAD metric
    results = squad_metric.compute(
        predictions=formatted_predictions, 
        references=formatted_references
    )
    
    return results

# --- Helper Functions for Converting Model Outputs to Predictions ---

def get_predictions_from_logits(input_ids, start_logits, end_logits, tokenizer):
    """
    Convert model logits to text predictions using the tokenizer.
    
    Args:
        input_ids: Tensor of token ids [batch_size, seq_len]
        start_logits: Tensor of start position logits [batch_size, seq_len]
        end_logits: Tensor of end position logits [batch_size, seq_len]
        tokenizer: Tokenizer to decode token ids to text
    
    Returns:
        List of prediction texts
    """
    predictions = []
    
    for i in range(len(input_ids)):
        # Get best start/end positions
        start_idx = tf.argmax(start_logits[i]).numpy()
        end_idx = tf.argmax(end_logits[i]).numpy()
        
        # Ensure valid span (start <= end)
        if start_idx > end_idx:
            end_idx = start_idx + 1  # Minimal valid span
        
        # Extract token ids for the predicted span
        predicted_ids_np = input_ids[i][start_idx:end_idx+1].numpy()
        
        # Skip special tokens (Corrected logic)
        # Get all special token IDs from the tokenizer
        special_tokens_ids = set(tokenizer.all_special_ids)
        # Filter out special tokens
        predicted_ids = [id for id in predicted_ids_np if id not in special_tokens_ids]
        
        # Decode to text
        if len(predicted_ids) > 0:
            predicted_text = tokenizer.decode(predicted_ids)
        else:
            predicted_text = ""
        
        predictions.append(predicted_text)
    
    return predictions

def format_predictions_for_evaluation(predicted_texts, example_ids):
    """
    Format predictions for Hugging Face's SQuAD metric.
    
    Args:
        predicted_texts: List of predicted answer texts
        example_ids: List of example IDs corresponding to the predictions
    
    Returns:
        List of dictionaries with 'id' and 'prediction_text' fields
    """
    return [
        {"id": id, "prediction_text": text}
        for id, text in zip(example_ids, predicted_texts)
    ]

def format_references_for_evaluation(references):
    """
    Format references for Hugging Face's SQuAD metric.
    
    Args:
        references: List of dictionaries with answer information
    
    Returns:
        List of dictionaries with 'id' and 'answers' fields
    """
    return [
        {
            "id": ref["id"],
            "answers": {
                "text": ref["answers_text"],
                "answer_start": ref["answers_start"]
            }
        }
        for ref in references
    ]

# --- SQuAD Evaluation Metrics ---

# def normalize_text(s):
#     """Lower text and remove punctuation, articles and extra whitespace."""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
# 
#     def white_space_fix(text):
#         return ' '.join(text.split())
# 
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)
# 
#     def lower(text):
#         return text.lower()
# 
#     return white_space_fix(remove_articles(remove_punc(lower(s))))
# 
# def compute_f1(gold_toks, pred_toks):
#     """Computes F1 score based on token overlap."""
#     # Ensure tokens are strings (debugging potential issue)
#     gold_toks = [str(tok) for tok in gold_toks]
#     pred_toks = [str(tok) for tok in pred_toks]
# 
#     common = Counter(gold_toks) & Counter(pred_toks)
#     num_same = sum(common.values())
# 
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is empty, F1 is 1 if both are empty, 0 otherwise
#         return int(gold_toks == pred_toks)
# 
#     if num_same == 0:
#         return 0
# 
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1
# 
# def compute_exact(gold_answer, pred_answer):
#     """Computes Exact Match score."""
#     return int(normalize_text(gold_answer) == normalize_text(pred_answer))
# 
# def get_tokens(s):
#     """Split on whitespace and return a list of tokens."""
#     if not s:
#         return []
#     return s.split()
# 
# def compute_eval_metrics(gold_answer, pred_answer):
#     """Computes both F1 and Exact Match scores."""
#     normalized_gold = normalize_text(gold_answer)
#     normalized_pred = normalize_text(pred_answer)
# 
#     gold_tokens = get_tokens(normalized_gold)
#     pred_tokens = get_tokens(normalized_pred)
# 
#     # Use normalized versions for both EM and F1 for consistency
#     exact_score = compute_exact(normalized_gold, normalized_pred)
#     f1_score = compute_f1(gold_tokens, pred_tokens)
# 
#     return exact_score, f1_score
# 
# # --- How to use these (Example Explanation) ---
# # ... (explanation comments also removed)
#
# Note: The example explanation assumes you're using a tokenizer to convert token indices back to text.
# If you're using a different method to get predictions, you'll need to adjust the example code accordingly. 
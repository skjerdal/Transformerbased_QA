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

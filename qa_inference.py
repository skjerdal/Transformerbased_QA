import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertModel
import argparse
import json

# Define the custom QA model class (must match the training model)
class CustomQAModel(tf.keras.Model):
    def __init__(self, pretrained_model_name):
        super(CustomQAModel, self).__init__()
        self.bert = TFDistilBertModel.from_pretrained(pretrained_model_name)
        self.qa_outputs = tf.keras.layers.Dense(2, name="qa_outputs")
    
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
        config.update({"pretrained_model_name": None})  # Will be loaded from saved config
        return config

def load_qa_model(model_dir):
    """Load the trained QA model and tokenizer"""
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create model with original pretrained base
    model = CustomQAModel(config["model_name"])
    
    # Load weights
    weights_path = os.path.join(model_dir, "qa_model_weights")
    model.load_weights(weights_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer, config

def get_answer(question, context, model, tokenizer, max_seq_length=384):
    """Get answer for a question-context pair"""
    # Tokenize input
    inputs = tokenizer(
        question, 
        context, 
        max_length=max_seq_length, 
        truncation="only_second",
        padding="max_length",
        return_tensors="tf"
    )
    
    # Get predictions
    start_logits, end_logits = model(inputs, training=False)
    
    # Convert to numpy
    start_logits = start_logits.numpy()[0]
    end_logits = end_logits.numpy()[0]
    
    # Get best answer span
    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)
    
    # If end comes before start, adjust to get shortest valid span
    if end_idx < start_idx:
        # Find next best end position
        end_scores = list(enumerate(end_logits))
        end_scores.sort(key=lambda x: x[1], reverse=True)
        for candidate_end_idx, _ in end_scores:
            if candidate_end_idx >= start_idx:
                end_idx = candidate_end_idx
                break
    
    # Convert token indices to character span in the context
    input_ids = inputs["input_ids"][0].numpy()
    
    # Decode the answer from the input_ids directly
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    answer_tokens = all_tokens[start_idx:end_idx+1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    
    # Clean up answer by removing special tokens or partial tokens
    answer = answer.strip()
    
    # Get scores
    score = start_logits[start_idx] + end_logits[end_idx]
    
    return {
        "answer": answer,
        "score": float(score),
        "start_index": int(start_idx),
        "end_index": int(end_idx)
    }

def main():
    parser = argparse.ArgumentParser(description="QA model inference")
    parser.add_argument("--model_dir", type=str, default="tf210_custom_qa_model", help="Directory with saved model weights and tokenizer")
    parser.add_argument("--question", type=str, required=True, help="The question to answer")
    parser.add_argument("--context", type=str, required=True, help="The context containing the answer")
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_dir}...")
    model, tokenizer, config = load_qa_model(args.model_dir)
    
    print("\nQuestion:", args.question)
    print("Context:", args.context[:100] + "..." if len(args.context) > 100 else args.context)
    
    # Get answer
    result = get_answer(args.question, args.context, model, tokenizer, config.get("max_seq_length", 384))
    
    print("\n" + "="*40)
    print(f"Answer: {result['answer']}")
    print(f"Confidence score: {result['score']:.2f}")
    print("="*40)

if __name__ == "__main__":
    main() 
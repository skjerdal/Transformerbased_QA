import os
import numpy as np
import tensorflow as tf
import gradio as gr
from transformers import AutoTokenizer, TFDistilBertModel
import json

# --- Define the Custom QA Model (same as in training script) ---
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

# --- Load Model Configuration ---
MODEL_DIR = "tf210_custom_qa_model"
config_path = os.path.join(MODEL_DIR, "config.json")

# Load configuration
with open(config_path, "r") as f:
    config = json.load(f)

MODEL_NAME = config["model_name"]
MAX_SEQ_LENGTH = config["max_seq_length"]
DOC_STRIDE = config["doc_stride"]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = CustomQAModel(MODEL_NAME)

# Create dummy inputs to build the model
dummy_ids = tf.ones((1, MAX_SEQ_LENGTH), dtype=tf.int32)
dummy_mask = tf.ones((1, MAX_SEQ_LENGTH), dtype=tf.int32)
_ = model((dummy_ids, dummy_mask))

# Load trained weights
model.load_weights(os.path.join(MODEL_DIR, "qa_model_weights"))

# Configure GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth configuration error: {e}")

# --- Inference Function ---
def predict_answer(context, question):
    """Get answer predictions for a question and context."""
    # Tokenize the inputs
    inputs = tokenizer(
        question,
        context,
        max_length=MAX_SEQ_LENGTH,
        stride=DOC_STRIDE,
        truncation="only_second",
        padding="max_length",
        return_tensors="tf",
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )
    
    # Get offset mapping and remove it from inputs
    offset_mapping = inputs.pop("offset_mapping").numpy()
    
    # Get sample mapping if there are multiple chunks
    sample_mapping = inputs.pop("overflow_to_sample_mapping").numpy() if "overflow_to_sample_mapping" in inputs else None
    
    # Get model inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Get predictions
    start_logits, end_logits = model((input_ids, attention_mask))
    
    # Convert logits to numpy
    start_logits = start_logits.numpy()
    end_logits = end_logits.numpy()
    
    # Get best answer for each feature
    all_answers = []
    for i in range(len(input_ids)):
        # Get non-padded tokens
        unpadded_length = int(tf.reduce_sum(attention_mask[i]).numpy())
        
        # Get the feature scores
        feature_start_logits = start_logits[i][:unpadded_length]
        feature_end_logits = end_logits[i][:unpadded_length]
        
        # Get the top 20 start and end indices
        start_indexes = np.argsort(feature_start_logits)[-20:].tolist()
        end_indexes = np.argsort(feature_end_logits)[-20:].tolist()
        
        valid_answers = []
        for start_idx in start_indexes:
            for end_idx in end_indexes:
                # Skip invalid answer spans
                if end_idx < start_idx or end_idx - start_idx + 1 > 30:
                    continue
                    
                # Get start and end positions in the context
                curr_offset_mapping = offset_mapping[i]
                start_char = int(curr_offset_mapping[start_idx][0])
                end_char = int(curr_offset_mapping[end_idx][1])
                
                # Skip if the answer is not in the context
                if start_char >= len(context) or end_char > len(context):
                    continue
                
                answer_text = context[start_char:end_char]
                score = feature_start_logits[start_idx] + feature_end_logits[end_idx]
                
                valid_answers.append({
                    "text": answer_text,
                    "score": float(score)
                })
        
        if valid_answers:
            # Sort by score
            all_answers.extend(valid_answers)
    
    # If we have answers, return the highest scoring one
    if all_answers:
        best_answer = sorted(all_answers, key=lambda x: x["score"], reverse=True)[0]
        return best_answer["text"]
    else:
        return "No answer found"

# --- Gradio Interface ---
def qa_interface(context, question):
    if not context or not question:
        return "Please provide both a context and a question."
    
    answer = predict_answer(context, question)
    return answer

# Example context and question
example_context = "The SQuAD dataset was developed at Stanford University for question answering research. It contains questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text from the corresponding reading passage."
example_question = "Where was the SQuAD dataset developed?"

# Create Gradio interface
demo = gr.Interface(
    fn=qa_interface,
    inputs=[
        gr.Textbox(lines=10, placeholder="Enter context here...", label="Context"),
        gr.Textbox(lines=2, placeholder="Enter question here...", label="Question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Question Answering System",
    description="Enter a context paragraph and ask a question about it.",
    examples=[[example_context, example_question]],
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    print("Starting Gradio App...")
    demo.launch(share=True)  # Set share=False if you don't want to create a public link 
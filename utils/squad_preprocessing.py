import json
import tensorflow as tf
import numpy as np
from .subword_tokenizer import SubwordTokenizer
from keras.utils import pad_sequences

def load_squad_data(file_path, max_samples=None):
    """Load SQuAD dataset from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    contexts = []
    questions = []
    answers_text = []
    answers_start_char = [] # Store the character start index
    
    # Extract data from SQuAD format
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            for qa in paragraph['qas']:
                question = qa['question']
                
                # Use the first answer and its start position
                if qa['answers']:
                    answer_info = qa['answers'][0]
                    answer = answer_info['text']
                    answer_start = answer_info['answer_start']
                    
                    contexts.append(context)
                    questions.append(question)
                    answers_text.append(answer)
                    answers_start_char.append(answer_start) # Store start char index
                    
                    # Limit samples for faster processing if specified
                    if max_samples and len(contexts) >= max_samples:
                        # Return all collected lists
                        return contexts, questions, answers_text, answers_start_char
    
    # Return all collected lists if max_samples not reached or not specified
    return contexts, questions, answers_text, answers_start_char

def find_subsequence_indices(main_list, sub_list):
    """Find the start and end indices of the first occurrence of sub_list in main_list."""
    # Handle edge case of empty sub_list
    if not sub_list:
        return -1, -1
    len_sub = len(sub_list)
    for i in range(len(main_list) - len_sub + 1):
        if main_list[i:i+len_sub] == sub_list:
            return i, i + len_sub - 1
    return -1, -1 # Not found

def create_qa_inputs(contexts, questions, answers_text, answer_starts_char, tokenizer, max_seq_length):
    """Create input features for question answering, including target start/end token indices."""
    # Tokenize contexts and questions
    print("Tokenizing contexts and questions...")
    combined_inputs = []
    start_positions = []
    end_positions = []
    
    print("Finding answer spans in tokenized sequences...")
    num_not_found = 0
    num_truncated_away = 0
    
    for i, (context, question, answer, ans_start_char) in enumerate(zip(contexts, questions, answers_text, answer_starts_char)):
        # Encode question and context as a pair
        encoded = tokenizer.encode(
            question, 
            context, 
            max_length=max_seq_length,
            padding=True
        )
        input_ids = encoded["input_ids"]
        
        # Find the answer span in the tokenized context
        # First, encode just the answer text
        answer_encoding = tokenizer.encode(answer)
        answer_tokens = answer_encoding["input_ids"][1:-1]  # Remove [CLS] and [SEP]
        
        # Find where the context starts in the combined sequence
        # It starts after [CLS] question [SEP]
        question_encoding = tokenizer.encode(question)
        context_start = len(question_encoding["input_ids"])  # This includes [CLS] and [SEP]
        
        # Get the context part of the combined sequence
        context_tokens = input_ids[context_start:-1]  # Exclude the final [SEP]
        
        # Find the answer span in the context tokens
        start_idx, end_idx = find_subsequence_indices(context_tokens.numpy().tolist(), answer_tokens.numpy().tolist())
        
        if start_idx != -1:
            # Adjust indices to account for [CLS] question [SEP]
            start_idx += context_start
            end_idx += context_start
            
            # Check if the answer span is within max_seq_length
            if start_idx < max_seq_length and end_idx < max_seq_length:
                combined_inputs.append(input_ids)
                start_positions.append(start_idx)
                end_positions.append(end_idx)
            else:
                num_truncated_away += 1
        else:
            num_not_found += 1
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(contexts)} examples...")
    
    print(f"Finished processing. Answers not found in tokenized context: {num_not_found}")
    print(f"Answers truncated away by max_seq_length: {num_truncated_away}")
    
    if not combined_inputs:
        raise ValueError("No valid examples found after preprocessing!")
    
    # Stack all inputs into a single tensor
    padded_inputs = tf.stack(combined_inputs)
    
    # Convert to numpy arrays
    final_start_positions = np.array(start_positions)
    final_end_positions = np.array(end_positions)
    
    return padded_inputs, final_start_positions, final_end_positions

def prepare_squad_data(file_path, max_seq_length=384, vocab_size=30000):
    """Prepare SQuAD data for training."""
    # Load data
    print("Loading SQuAD data...")
    # Call load_squad_data without max_samples to load everything
    contexts, questions, answers_text, answers_start_char = load_squad_data(file_path)
    print(f"Loaded {len(contexts)} examples")
    
    # Create and train tokenizer
    print("Training tokenizer...")
    tokenizer = SubwordTokenizer(vocab_size=vocab_size)
    # Train on all text data
    all_texts = contexts + questions + answers_text
    tokenizer.train(all_texts)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Create inputs
    print("Creating model inputs...")
    inputs, start_positions, end_positions = create_qa_inputs(
        contexts, questions, answers_text, answers_start_char, tokenizer, max_seq_length
    )
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (inputs, (start_positions, end_positions))
    )
    
    return dataset, tokenizer
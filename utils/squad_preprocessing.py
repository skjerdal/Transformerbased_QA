import json
import tensorflow as tf
import numpy as np
# Remove import for custom tokenizer
# from .subword_tokenizer import SubwordTokenizer 
# Import the standard Hugging Face tokenizer
from transformers import BertTokenizerFast

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

def create_qa_inputs(contexts, questions, answers_text, answer_starts_char, tokenizer: BertTokenizerFast, max_seq_length):
    """Create input features for question answering using BertTokenizerFast and character offsets."""
    print("Encoding contexts and questions with BertTokenizerFast, finding answer spans...")
    all_input_ids = []
    all_attention_masks = []
    all_token_type_ids = [] # BERT uses token type IDs
    start_positions = []
    end_positions = []

    num_not_found = 0
    num_truncated_away = 0
    num_valid = 0

    for i, (context, question, answer, ans_start_char) in enumerate(zip(contexts, questions, answers_text, answer_starts_char)):
        ans_end_char = ans_start_char + len(answer)

        # Encode using the BertTokenizerFast __call__ method
        # It handles padding, truncation, attention masks, and token type IDs automatically
        encoded = tokenizer(
            question,
            context,
            max_length=max_seq_length,
            truncation="only_second",  # Truncate context if needed, not question
            padding="max_length",     # Pad to max_seq_length
            return_offsets_mapping=True, # Request character offsets
            return_token_type_ids=True   # Request token type IDs
            # attention_mask is returned by default
        )

        # Access items from the BatchEncoding object (acts like a dict)
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded["token_type_ids"] # 0 for question, 1 for context

        # --- Use char_to_token for more robust mapping --- 
        # Find the start and end token indices corresponding to the answer's character span
        # Need the sequence_ids to distinguish context from question
        sequence_ids = encoded.sequence_ids()
        
        # Find the first and last token indices of the context
        context_start_token_idx = -1
        context_end_token_idx = -1
        for idx, seq_id in enumerate(sequence_ids):
            if seq_id == 1: # Context part
                if context_start_token_idx == -1:
                    context_start_token_idx = idx
                context_end_token_idx = idx # Keep updating to find the last one
                
        token_start_index = -1
        token_end_index = -1
        
        # Use char_to_token to find the start token index
        # It returns None if the char index is out of bounds or maps to padding/CLS etc.
        start_token_candidate = encoded.char_to_token(ans_start_char, sequence_index=1) # sequence_index=1 for context
        
        # Use char_to_token for the *last* character of the answer to find the end token
        end_token_candidate = encoded.char_to_token(ans_end_char - 1, sequence_index=1)

        # Validate: Check if candidates were found and are within the context span
        if (start_token_candidate is not None and 
            end_token_candidate is not None and
            start_token_candidate >= context_start_token_idx and 
            start_token_candidate <= context_end_token_idx and
            end_token_candidate >= context_start_token_idx and
            end_token_candidate <= context_end_token_idx and
            start_token_candidate <= end_token_candidate): # Ensure start <= end
                
            token_start_index = start_token_candidate
            token_end_index = end_token_candidate
        else:
            # Fallback or skip: If char_to_token fails or gives inconsistent results
            # Let's try the offset mapping approach as a fallback (optional, could just skip)
            # Note: Keeping the original offset loop as fallback adds complexity.
            # For simplicity, let's just mark as not found if char_to_token fails.
            num_not_found += 1
            # Log details if needed for debugging
            # print(f"Warn: char_to_token failed for example {i}. Start char: {ans_start_char}, End char: {ans_end_char-1}")
            # print(f"  Start Tok Cand: {start_token_candidate}, End Tok Cand: {end_token_candidate}")
            # print(f"  Context Tok Span: [{context_start_token_idx}, {context_end_token_idx}]")
            continue # Skip this example if mapping failed

        # --- End of char_to_token mapping --- 

        # Find the start and end token indices corresponding to the answer's character span
        # Use token_type_ids to identify context tokens (where token_type_id == 1)
        # context_token_indices = [idx for idx, type_id in enumerate(token_type_ids) if type_id == 1]

        # if not context_token_indices:
        #     # This might happen if the context gets fully truncated
        #     num_not_found += 1
        #     # Log if context was present but got truncated away entirely
        #     # if context: print(f"Warning: Context possibly truncated entirely for example {i}")
        #     continue

        # context_start_token = min(context_token_indices)
        # context_end_token = max(context_token_indices)

        # # Find the token index covering the start character
        # # Iterate only through context tokens
        # token_start_index = -1
        # for idx in range(context_start_token, context_end_token + 1):
        #     start_char, end_char = offsets[idx]
        #     # Check if token is within context (already guaranteed by loop range)
        #     # and its start char offset covers the answer start, and it's a valid span
        #     if start_char <= ans_start_char < end_char:
        #         token_start_index = idx
        #         break

        # # Find the token index covering the end character
        # # Iterate backwards through context tokens
        # token_end_index = -1
        # for idx in range(context_end_token, context_start_token - 1, -1):
        #     start_char, end_char = offsets[idx]
        #     # Check if token is within context (guaranteed by loop range)
        #     # and its end char offset covers the answer end, and it's a valid span
        #     if start_char < ans_end_char <= end_char:
        #         token_end_index = idx
        #         break
                
        # # More robust check: Handle cases where answer starts at the beginning of a token AND ends at the end of the same token.
        # # Ensure the start is found first.
        # # if token_start_index != -1 and token_end_index == -1:
        # #     # Check if the end char is within the start token itself
        # #     start_char, end_char = offsets[token_start_index]
        # #     if start_char < ans_end_char <= end_char:
        # #          token_end_index = token_start_index


        # # Validate the found indices
        # if (token_start_index == -1 or
        #     token_end_index == -1 or
        #     token_start_index > token_end_index): # Ensure start <= end
        #     num_not_found += 1
        #     # Optionally log problematic examples here
        #     # print(f"Warning: Answer span not found or invalid for example {i}. Start={token_start_index}, End={token_end_index}")
        #     # print(f"  Context: {context[:50]}...")
        #     # print(f"  Question: {question}")
        #     # print(f"  Answer: '{answer}' ({ans_start_char}-{ans_end_char})")
        #     # # Log relevant offsets
        #     # context_offsets = [offsets[k] for k in range(context_start_token, context_end_token + 1)]
        #     # print(f"  Context Offsets: {context_offsets}")
        #     continue # Skip if invalid span found by offset mapping

        # Check if the found span is within the max sequence length (redundant with padding='max_length', but safe)
        # Note: This check might be less necessary now as char_to_token handles boundaries
        # elif token_start_index >= max_seq_length or token_end_index >= max_seq_length:
        #      num_truncated_away += 1 # Should ideally be 0 with padding='max_length'
        # else:
        # Valid example found
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        # We don't necessarily need to store token_type_ids unless the model uses them
        # all_token_type_ids.append(token_type_ids) 
        start_positions.append(token_start_index) # Use validated index
        end_positions.append(token_end_index)     # Use validated index
        num_valid += 1

        if (i + 1) % 5000 == 0:
            print(f"Processed {i+1}/{len(contexts)} examples... (Valid: {num_valid}, Not Found: {num_not_found}, Truncated: {num_truncated_away})")

    print(f"Finished processing {len(contexts)} examples.")
    print(f"  Total Valid Examples: {num_valid}")
    print(f"  Answers not found/invalid span in tokenized context: {num_not_found}")
    print(f"  Answers truncated away by max_seq_length: {num_truncated_away}")

    if not all_input_ids:
        raise ValueError("No valid examples found after preprocessing! Check SQuAD data and preprocessing logic.")

    # Stack all inputs into tensors (convert lists of lists/ints to tensors)
    # Use tf.convert_to_tensor for potentially ragged sequences before stacking if needed,
    # but padding="max_length" should ensure uniform lengths.
    final_input_ids = tf.convert_to_tensor(all_input_ids, dtype=tf.int32)
    final_attention_masks = tf.convert_to_tensor(all_attention_masks, dtype=tf.int32)
    final_start_positions = tf.convert_to_tensor(start_positions, dtype=tf.int32)
    final_end_positions = tf.convert_to_tensor(end_positions, dtype=tf.int32)

    # Return input_ids, attention_mask, start_positions, end_positions
    # (Token type IDs are not needed by the current model definition)
    return final_input_ids, final_attention_masks, final_start_positions, final_end_positions

def prepare_squad_data(file_path, max_seq_length=384, vocab_size=None, tokenizer_name='bert-base-uncased', max_samples=None):
    """Prepare SQuAD data for training using a pretrained BertTokenizerFast."""
    # Load data
    print("Loading SQuAD data...")
    contexts, questions, answers_text, answers_start_char = load_squad_data(file_path, max_samples=max_samples)
    print(f"Loaded {len(contexts)} examples")

    # Load pretrained tokenizer
    print(f"Loading pretrained tokenizer: {tokenizer_name}...")
    # The 'Fast' version is important for offset mapping
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name) 
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")
    # vocab_size parameter is now ignored, using the tokenizer's actual size

    # Create inputs using the loaded tokenizer
    print("Creating model inputs (input_ids, attention_mask, start_pos, end_pos)...")
    input_ids, attention_masks, start_positions, end_positions = create_qa_inputs(
        contexts, questions, answers_text, answers_start_char, tokenizer, max_seq_length
    )

    # Create TensorFlow dataset with the new structure: ((input_ids, attention_mask), (start_pos, end_pos))
    print("Creating TensorFlow dataset...")
    dataset = tf.data.Dataset.from_tensor_slices(
        ((input_ids, attention_masks), (start_positions, end_positions))
    )
    print("Dataset created successfully.")

    # Return the dataset and the loaded tokenizer instance
    return dataset, tokenizer
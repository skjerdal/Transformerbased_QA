from datasets import load_dataset
from transformers import BertTokenizerFast
import tensorflow as tf

def prepare_squad_data_with_hf(tokenizer_name='bert-base-uncased', max_seq_length=384, max_samples=None, batch_size=16):
    """
    Prepare SQuAD data using Hugging Face Datasets and Tokenizers
    
    Args:
        tokenizer_name: Name of the tokenizer to load from Hugging Face Hub
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to use (for faster experimentation)
        batch_size: Batch size for training
    
    Returns:
        train_dataset: TF Dataset for training
        val_dataset: TF Dataset for validation
        tokenizer: The loaded tokenizer
        train_examples: Original train examples for reference
        val_examples: Original validation examples for reference
    """
    print(f"Loading SQuAD dataset from Hugging Face Datasets...")
    squad = load_dataset("squad")
    
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    
    # Prepare the train and validation sets
    train_examples = squad["train"]
    val_examples = squad["validation"]
    
    print(f"Original dataset sizes - Train: {len(train_examples)}, Validation: {len(val_examples)}")
    
    # Limit samples if specified
    if max_samples:
        train_examples = train_examples.select(range(min(max_samples, len(train_examples))))
        val_examples = val_examples.select(range(min(max_samples//5, len(val_examples))))
        print(f"Reduced dataset sizes - Train: {len(train_examples)}, Validation: {len(val_examples)}")
    
    # Preprocessing function
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_seq_length,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Create a mapping from original example index to new features
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs.pop("offset_mapping")
        
        # Create lists for start and end positions
        start_positions = []
        end_positions = []
        
        for i, offset in enumerate(offset_mapping):
            # Get the index of the original example this feature came from
            sample_idx = sample_map[i]
            answer = examples["answers"][sample_idx]
            
            # If no answer, set start and end to CLS token position (0)
            if not answer["text"]:
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            # Get the character start and end positions
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            
            # Find the token indices covering these character positions
            sequence_ids = inputs.sequence_ids(i)
            
            # Find the first and last token for the context portion
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
                
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1
            
            # If the answer is not fully in the context, set start/end to CLS
            if (offset[context_start][0] > end_char or 
                offset[context_end][1] < start_char):
                start_positions.append(0)
                end_positions.append(0)
                continue
                
            # Find token positions that contain the answer
            token_start = token_end = 0
            
            # Find the start token position
            for idx in range(context_start, context_end + 1):
                if offset[idx][0] <= start_char < offset[idx][1]:
                    token_start = idx
                    break
            
            # Find the end token position
            for idx in range(context_end, context_start - 1, -1):
                if offset[idx][0] < end_char <= offset[idx][1]:
                    token_end = idx
                    break
            
            # If we couldn't find valid positions, use fallback
            if token_start == 0 and token_end == 0:
                token_start = 0
                token_end = 0
            
            start_positions.append(token_start)
            end_positions.append(token_end)
        
        # Add start and end positions to inputs
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        
        return inputs
    
    print("Processing train dataset...")
    train_dataset = train_examples.map(
        preprocess_function,
        batched=True,
        remove_columns=train_examples.column_names,
        desc="Processing training examples"
    )
    
    print("Processing validation dataset...")
    val_dataset = val_examples.map(
        preprocess_function,
        batched=True,
        remove_columns=val_examples.column_names,
        desc="Processing validation examples"
    )
    
    print(f"Processed dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Function to convert to TensorFlow dataset
    def convert_to_tf_dataset(dataset, shuffle=True):
        """Convert a Hugging Face Dataset to a TensorFlow Dataset"""
        # Convert to TensorFlow dataset
        tf_dataset = dataset.to_tf_dataset(
            columns=["input_ids", "attention_mask", "token_type_ids"],
            label_cols=["start_positions", "end_positions"],
            shuffle=shuffle,
            batch_size=batch_size
        )
        
        # Change to match the expected format: ((input_ids, attention_mask), (start, end))
        def reformat(features, labels):
            return (
                (features["input_ids"], features["attention_mask"]), 
                (labels["start_positions"], labels["end_positions"])
            )
        
        tf_dataset = tf_dataset.map(reformat)
        return tf_dataset
    
    print("Converting to TensorFlow datasets...")
    tf_train_dataset = convert_to_tf_dataset(train_dataset, shuffle=True)
    tf_val_dataset = convert_to_tf_dataset(val_dataset, shuffle=False)
    
    print("Dataset preparation complete!")
    return tf_train_dataset, tf_val_dataset, tokenizer, train_examples, val_examples 
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import List, Dict, Union
import tensorflow as tf

class SubwordTokenizer:
    def __init__(self, vocab_size: int = 30000):
        """Initialize the SubwordTokenizer with HuggingFace's WordPiece tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary.
        """
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Add special tokens and configure how sequences should be handled
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
                ("[UNK]", 3),
                ("[PAD]", 0),
            ],
        )
        
        self.trained = False
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.unk_token_id = 3
    
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer on the given texts.
        
        Args:
            texts: List of texts to train the tokenizer on.
        """
        trainer = WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"],
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)
        self.trained = True
        
        # Cache the vocabulary for quick lookups
        self.vocab = self.tokenizer.get_vocab()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, 
              text: str, 
              text_pair: str = None, 
              max_length: int = None, 
              padding: bool = True) -> Dict[str, tf.Tensor]:
        """Encode text(s) using the trained tokenizer.
        
        Args:
            text: The text to encode.
            text_pair: Optional second text for sequence pair tasks.
            max_length: Maximum length of the encoded sequence.
            padding: Whether to pad sequences to max_length.
            
        Returns:
            Dictionary containing input_ids tensor.
        """
        if not self.trained:
            raise ValueError("Tokenizer must be trained before encoding.")
        
        # Encode the text(s)
        if text_pair:
            encoding = self.tokenizer.encode(text, text_pair)
        else:
            encoding = self.tokenizer.encode(text)
        
        # Convert to tensor
        input_ids = tf.convert_to_tensor(encoding.ids, dtype=tf.int32)
        
        # Handle truncation and padding if needed
        if max_length is not None:
            # First truncate if necessary
            if tf.shape(input_ids)[0] > max_length:
                input_ids = input_ids[:max_length]
            
            # Then pad if needed and requested
            if padding:
                current_length = tf.shape(input_ids)[0]
                pad_length = max_length - current_length
                if pad_length > 0:  # Only pad if we need to
                    input_ids = tf.pad(
                        input_ids,
                        [[0, pad_length]],
                        constant_values=self.pad_token_id
                    )
        
        return {"input_ids": input_ids}
    
    def decode(self, token_ids: Union[List[int], tf.Tensor]) -> str:
        """Decode token IDs back to text.
        
        Args:
            token_ids: List or tensor of token IDs.
            
        Returns:
            Decoded text string.
        """
        if isinstance(token_ids, tf.Tensor):
            token_ids = token_ids.numpy().tolist()
        
        # Filter out padding tokens
        token_ids = [id for id in token_ids if id != self.pad_token_id]
        
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        if not self.trained:
            raise ValueError("Tokenizer must be trained before getting vocab size.")
        return len(self.vocab)
    
    def id_to_token(self, id: int) -> str:
        """Convert a token ID to its string representation."""
        if not self.trained:
            raise ValueError("Tokenizer must be trained before converting IDs.")
        return self.reverse_vocab.get(id, "[UNK]")
    
    def token_to_id(self, token: str) -> int:
        """Convert a token string to its ID."""
        if not self.trained:
            raise ValueError("Tokenizer must be trained before converting tokens.")
        return self.vocab.get(token, self.unk_token_id) 
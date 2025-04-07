import tensorflow as tf
from tensorflow import keras
from keras import layers
from components.positional_encoding import PositionalEncoding
from components.transformer_block import TransformerBlock

def build_qa_transformer_model(vocab_size, sequence_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
    """Build a transformer model for question answering."""
    inputs = layers.Input(shape=(sequence_len,), name='inputs')
    
    # Embedding
    x = layers.Embedding(vocab_size, d_model, name='token_embedding')(inputs)
    
    # Positional encoding
    x = PositionalEncoding(sequence_len, d_model)(x)
    
    # Stack multiple Transformer blocks
    attention_weights_all = {}
    for i in range(num_layers):
        transformer_block = TransformerBlock(d_model, num_heads, dff, dropout_rate)
        x, attn_weights = transformer_block(x, training=True)
        attention_weights_all[f'layer_{i+1}'] = attn_weights
    
    # For QA, we need two outputs: start and end position logits
    # We'll use the sequence output for both
    start_logits = layers.Dense(1, name='start_logits')(x)
    end_logits = layers.Dense(1, name='end_logits')(x)
    
    # Squeeze the last dimension to get shape (batch_size, sequence_len)
    start_logits = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(start_logits)
    end_logits = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(end_logits)
    
    # Create model
    model = keras.Model(
        inputs=inputs, 
        outputs=[start_logits, end_logits]
    )
    
    return model 
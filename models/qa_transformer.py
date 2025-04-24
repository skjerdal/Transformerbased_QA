import tensorflow as tf
from tensorflow import keras
from keras import layers
from components.positional_encoding import PositionalEncoding
from components.transformer_block import TransformerBlock
import logging
import numpy as np 

logger = logging.getLogger(__name__) 

def build_qa_transformer_model(vocab_size, sequence_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
    """Build a transformer model."""
    input_ids = layers.Input(shape=(sequence_len,), dtype='int32', name='input_ids')
    attention_mask = layers.Input(shape=(sequence_len,), dtype='int32', name='attention_mask')

    # Embedding
    embedding_layer = layers.Embedding(vocab_size, d_model, name='token_embedding') 
    x = embedding_layer(input_ids)

    # Positional encoding
    x = PositionalEncoding(sequence_len, d_model)(x)

    # Create the attention mask for the transformer blocks
    # Shape needs to be (batch_size, 1, 1, sequence_len) for broadcasting with attention scores (batch, heads, seq, seq)
    # The mask value should be 0 for tokens to attend to, and -inf (or large negative) for masked tokens.
    # We create it here and pass it down.
    mask_for_attention = tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
    # Invert mask: 0 where attention_mask is 1, 1 where attention_mask is 0
    mask_for_attention = 1.0 - mask_for_attention
    # Multiply by large negative number
    mask_for_attention *= -1e9

    # Stack multiple Transformer blocks, passing the mask
    attention_weights_all = {}
    for i in range(num_layers):
        transformer_block = TransformerBlock(d_model, num_heads, dff, dropout_rate, name=f'transformer_block_{i}')
        # Pass the processed mask_for_attention to the block
        x, attn_weights = transformer_block(x, training=True, mask=mask_for_attention)
        attention_weights_all[f'layer_{i+1}'] = attn_weights

    # Use a single Dense layer to predict both start and end logits together
    qa_outputs = layers.Dense(2, name='qa_outputs')(x) # Shape: (batch_size, sequence_len, 2)

    # Split the output into start_logits and end_logits
    start_logits, end_logits = tf.split(qa_outputs, 2, axis=-1) # Each has shape (batch_size, sequence_len, 1)

    # Squeeze the last dimension to get shape (batch_size, sequence_len)
    start_logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='squeeze_start')(start_logits)
    end_logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='squeeze_end')(end_logits)

    # Create model with two inputs
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[start_logits, end_logits]
    )

    return model 
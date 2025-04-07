import tensorflow as tf
from tensorflow import keras
from keras import layers
from components.positional_encoding import PositionalEncoding
from components.transformer_block import TransformerBlock

def build_qa_transformer_model(vocab_size, sequence_len, d_model, num_heads, dff, num_layers, dropout_rate=0.1):
    """Build a transformer model for question answering, accepting input_ids and attention_mask."""
    # Define two inputs: one for token IDs, one for the attention mask
    input_ids = layers.Input(shape=(sequence_len,), dtype='int32', name='input_ids')
    attention_mask = layers.Input(shape=(sequence_len,), dtype='int32', name='attention_mask')

    # Embedding
    # Masking layer to propagate the mask based on padding (usually 0)
    # embedding_layer = layers.Embedding(vocab_size, d_model, name='token_embedding', mask_zero=True) # Option 1: Use mask_zero
    embedding_layer = layers.Embedding(vocab_size, d_model, name='token_embedding') # Option 2: Pass mask explicitly
    x = embedding_layer(input_ids)
    # x = embedding_layer(input_ids, mask=tf.cast(attention_mask, tf.bool)) # Propagate mask if not using mask_zero

    # Positional encoding
    # Positional encoding doesn't typically use the mask directly, but subsequent layers do
    x = PositionalEncoding(sequence_len, d_model)(x)

    # Create the attention mask for the transformer blocks
    # Shape needs to be (batch_size, 1, 1, sequence_len) for broadcasting with attention scores (batch, heads, seq, seq)
    # The mask value should be 0 for tokens to attend to, and -inf (or large negative) for masked tokens.
    # We create it here and pass it down.
    mask_for_attention = tf.cast(attention_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
    # Invert mask: 0 where attention_mask is 1, 1 where attention_mask is 0
    # mask_for_attention = 1.0 - mask_for_attention
    # Multiply by large negative number
    # mask_for_attention *= -1e9 # This will be added to attention scores

    # Stack multiple Transformer blocks, passing the mask
    attention_weights_all = {}
    for i in range(num_layers):
        transformer_block = TransformerBlock(d_model, num_heads, dff, dropout_rate)
        # Pass the original attention_mask (or derived mask) to the block
        x, attn_weights = transformer_block(x, training=True, mask=attention_mask) # Pass mask here
        attention_weights_all[f'layer_{i+1}'] = attn_weights

    # For QA, we need two outputs: start and end position logits
    # We'll use the sequence output for both
    start_logits = layers.Dense(1, name='start_logits')(x)
    end_logits = layers.Dense(1, name='end_logits')(x)

    # Squeeze the last dimension to get shape (batch_size, sequence_len)
    start_logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='squeeze_start')(start_logits)
    end_logits = layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name='squeeze_end')(end_logits)

    # Create model with two inputs
    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[start_logits, end_logits]
    )

    return model 
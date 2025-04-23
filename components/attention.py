import tensorflow as tf
from tensorflow import keras
from keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Use TensorFlow's built-in MultiHeadAttention layer instead of custom implementation
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            value_dim=d_model // num_heads
        )
        
        # Output projection (equivalent to the dense layer in the original implementation)
        # self.dense = layers.Dense(d_model) # REDUNDANT: Keras MHA includes output projection

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        # Process mask to be compatible with MultiHeadAttention
        # MultiHeadAttention expects mask shape [batch_size, tgt_seq_len, src_seq_len] or broadcastable
        if mask is not None:
            # Convert from [batch, 1, 1, seq_len] to [batch, seq_len, seq_len]
            # mask = tf.squeeze(mask, axis=[1, 2]) # <-- REMOVE THIS LINE
            # The mask in MultiHeadAttention is additive (0 for tokens to attend to, -inf for masked)
            # which is already the format we're using
            # The original shape (batch, 1, 1, seq_len) IS broadcastable.
            pass # No processing needed for the mask shape
        
        # Apply MultiHeadAttention
        attn_output = self.mha(
            query=q,
            key=k, 
            value=v,
            attention_mask=mask,
            return_attention_scores=True
        )
        
        # MultiHeadAttention returns (output, attention_scores) when return_attention_scores=True
        output, attention_weights = attn_output
        
        # Apply final projection (this is handled internally by MultiHeadAttention, but we keep for compatibility)
        # output = self.dense(output) # REMOVED: Redundant
        
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config 
import tensorflow as tf
from tensorflow import keras
from keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Use TensorFlow's built-in MultiHeadAttention layer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            value_dim=d_model // num_heads
        )

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        if mask is not None:
            pass
        
        # Apply MultiHeadAttention
        attn_output = self.mha(
            query=q,
            key=k, 
            value=v,
            attention_mask=mask,
            return_attention_scores=True
        )
        
        output, attention_weights = attn_output
        
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config 
import tensorflow as tf
from tensorflow import keras
from keras import layers
from .attention import MultiHeadSelfAttention

def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        # Multi-head attention, passing the mask
        # The mask passed here should be the original attention mask (batch_size, seq_len)
        attn_output, attn_weights = self.mha(x, x, x, mask) # Pass mask to MHA
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # The FFN output is layer normalized using the sum of the FFN input (out1) and FFN output
        out2 = self.layernorm2(out1 + ffn_output)
        return out2, attn_weights 
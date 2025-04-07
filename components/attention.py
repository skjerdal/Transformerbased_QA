import tensorflow as tf
from tensorflow import keras
from keras import layers

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads

        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def _split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, depth).
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)

        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the mask (if provided) to the attention scores.
        # The mask should prevent attention to padding tokens.
        if mask is not None:
            # Mask shape is (batch_size, seq_len)
            # Needs to be reshaped to broadcast to (batch_size, num_heads, seq_len_q, seq_len_k)
            # Typically, mask is applied to the key sequence length dimension (seq_len_k)
            attention_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            # Add a large negative number where the mask is 0 (padding tokens)
            scaled_attention_logits += (attention_mask * -1e9)

        # Softmax normalization on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth)

        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights 
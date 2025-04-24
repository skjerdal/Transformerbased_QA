import tensorflow as tf
from tensorflow import keras
from keras import layers

class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model):
        super(PositionalEncoding, self).__init__()
        self.sequence_len = sequence_len
        self.d_model = d_model
        self.pos_encoding = self._positional_encoding(sequence_len, d_model)

    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i//2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def _positional_encoding(self, position, d_model):
        angle_rads = self._get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, d_model)
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        """Returns the config of the layer."""
        config = super().get_config()
        config.update({
            "sequence_len": self.sequence_len,
            "d_model": self.d_model,
        })
        return config 
import tensorflow as tf
from tensorflow import keras
from keras import layers

def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        # --- Pre-LN Structure ---
        
        # 1. Layer Norm before Multi-Head Attention
        x_norm1 = self.layernorm1(x)
        
        # 2. Multi-head attention (using normalized input), passing the mask
        attn_output = self.mha(
            query=x_norm1, 
            value=x_norm1, 
            key=x_norm1, 
            attention_mask=mask, 
            training=training
            )
        
        # 4. Residual connection (add original x to attention output)
        out1 = x + attn_output

        # 5. Layer Norm before Feed Forward Network
        out1_norm = self.layernorm2(out1)
        
        # 6. Feed forward network (using normalized input)
        ffn_output = self.ffn(out1_norm)
        
        # 7. Dropout on FFN output
        ffn_output = self.dropout2(ffn_output, training=training)
        
        # 8. Residual connection (add output of first block (out1) to FFN output)
        out2 = out1 + ffn_output
        
        return out2, None

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config 
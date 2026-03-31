"""
bilstm_attention_model.py
=========================
Bidirectional LSTM with Self-Attention for Municipal Complaint
Classification and Named Entity Recognition (NER).

Architecture:
  Embedding → SpatialDropout → BiLSTM → Self-Attention → GlobalAvgPool
           → Dense → BN → Dropout → Softmax

Neural Network used: Bidirectional LSTM + Attention Mechanism
Framework: TensorFlow / Keras
Task: Multi-class classification (department / category / language)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    Dropout, BatchNormalization, SpatialDropout1D,
    GlobalAveragePooling1D, Layer, Multiply, Softmax
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


# ──────────────────────────────────────────────────────────────────
# Custom Attention Layer
# ──────────────────────────────────────────────────────────────────
class BahdanauAttention(Layer):
    """
    Bahdanau (additive) self-attention mechanism.

    Takes BiLSTM hidden states → computes attention weights → weighted sum.
    Allows model to focus on the most complaint-relevant words.

    Example: In "road pothole near school ward 5 very dangerous",
    attention will weight "pothole", "road", "dangerous" more heavily.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, use_bias=False, name="attn_W")
        self.V = Dense(1, use_bias=False, name="attn_V")

    def call(self, hidden_states, training=None):
        """
        Args:
            hidden_states: BiLSTM output, shape (batch, seq_len, hidden_dim)
        Returns:
            context: Weighted sum, shape (batch, hidden_dim)
            weights: Attention weights, shape (batch, seq_len, 1)
        """
        # Score: (batch, seq_len, units) → (batch, seq_len, 1)
        score = self.V(tf.nn.tanh(self.W(hidden_states)))

        # Attention weights
        weights = tf.nn.softmax(score, axis=1)  # (batch, seq_len, 1)

        # Weighted sum (context vector)
        context = weights * hidden_states          # (batch, seq_len, hidden_dim)
        context = tf.reduce_sum(context, axis=1)  # (batch, hidden_dim)

        return context, weights

    def get_config(self):
        config = super().get_config()
        return config


def build_bilstm_attention_model(
    vocab_size: int,
    num_classes: int,
    embedding_dim: int = 64,
    max_len: int = 60,
    lstm_units: int = 128,
    attention_units: int = 64,
    dropout_rate: float = 0.4,
    l2_lambda: float = 0.001,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build the BiLSTM + Attention classification model.

    Args:
        vocab_size:      Vocabulary size (tokenizer vocab + special tokens).
        num_classes:     Number of output classes.
        embedding_dim:   Embedding vector dimensionality.
        max_len:         Maximum input sequence length.
        lstm_units:      Units per LSTM direction (BiLSTM doubles this).
        attention_units: Units in Bahdanau attention dense layer.
        dropout_rate:    Dropout probability.
        l2_lambda:       L2 regularization coefficient.
        learning_rate:   Adam learning rate.

    Returns:
        Compiled Keras model.
    """

    # ── Input ──────────────────────────────────────────────
    inputs = Input(shape=(max_len,), name="token_ids")

    # ── Embedding ──────────────────────────────────────────
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )(inputs)
    x = SpatialDropout1D(0.2, name="spatial_dropout")(x)

    # ── Bidirectional LSTM ──────────────────────────────────
    # return_sequences=True: output hidden state at every time step
    # This is needed for attention to weigh each position
    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,  # shape: (batch, seq_len, lstm_units*2)
            dropout=0.2,
            recurrent_dropout=0.0,  # 0 = faster on GPU; use 0.1 for CPU
            kernel_regularizer=l2(l2_lambda),
        ),
        name="bilstm",
    )(x)

    # ── Attention ───────────────────────────────────────────
    attention = BahdanauAttention(units=attention_units, name="attention")
    context, attn_weights = attention(x)
    # context shape: (batch, lstm_units*2)

    # ── Classification head ─────────────────────────────────
    fc = Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda), name="fc1")(context)
    fc = BatchNormalization(name="bn_fc1")(fc)
    fc = Dropout(dropout_rate, name="dropout1")(fc)

    fc = Dense(64, activation="relu", kernel_regularizer=l2(l2_lambda), name="fc2")(fc)
    fc = Dropout(dropout_rate * 0.5, name="dropout2")(fc)

    outputs = Dense(num_classes, activation="softmax", name="output")(fc)

    # ── Build & compile ─────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Attention_Municipal")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_bilstm_attention_model(vocab_size=10013, num_classes=9)
    m.summary()

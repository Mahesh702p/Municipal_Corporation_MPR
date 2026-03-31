"""
ensemble_model.py
=================
Ensemble of CNN + BiLSTM+Attention for Municipal Complaint Classification.

Strategy: Late-fusion ensemble — both sub-models share the same Embedding
layer but process independently. Their softmax outputs are averaged
(or learned-weighted) to get final prediction.

Architecture:
                     ┌─ TextCNN branch (bigram/trigram/4gram)  ─┐
  Input → Embedding →│                                           │→ Average → Softmax
                     └─ BiLSTM + Attention branch               ─┘

Neural Networks used: CNN + BiLSTM + Bahdanau Attention (Ensemble)
Framework: TensorFlow / Keras
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Bidirectional, LSTM, Dense, Dropout, BatchNormalization,
    SpatialDropout1D, Concatenate, Average, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class BahdanauAttention(Layer):
    """Bahdanau self-attention (duplicated here for standalone import)."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.W = Dense(units, use_bias=False, name="ens_attn_W")
        self.V = Dense(1, use_bias=False, name="ens_attn_V")

    def call(self, hidden_states, training=None):
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights


def build_ensemble_model(
    vocab_size: int,
    num_classes: int,
    embedding_dim: int = 64,
    max_len: int = 60,
    # CNN settings
    filter_sizes: list = None,
    num_filters: int = 128,
    # BiLSTM settings
    lstm_units: int = 128,
    attention_units: int = 64,
    # Common
    dropout_rate: float = 0.4,
    l2_lambda: float = 0.001,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build the CNN + BiLSTM+Attention ensemble.

    Both branches share a single Embedding layer (tied weights).
    Their pre-softmax logits are concatenated and fed to a fusion head.

    Args:
        vocab_size:     Vocabulary size from MunicipalTokenizer.get_vocab_size().
        num_classes:    Number of output classes (departments).
        embedding_dim:  Embedding vector size.
        max_len:        Maximum padded sequence length.
        filter_sizes:   n-gram window sizes for CNN branch.
        num_filters:    Conv1D feature maps per window.
        lstm_units:     Units per LSTM direction.
        attention_units: Bahdanau attention hidden size.
        dropout_rate:   Shared dropout rate.
        l2_lambda:      L2 regularization.
        learning_rate:  Adam learning rate.

    Returns:
        Compiled Keras Model.
    """
    if filter_sizes is None:
        filter_sizes = [2, 3, 4]

    # ── Shared Input & Embedding ────────────────────────────
    inputs = Input(shape=(max_len,), name="token_ids")

    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="shared_embedding",
    )(inputs)
    embedding = SpatialDropout1D(0.2, name="spatial_dropout")(embedding)

    # ────────────────────────────────────────────────────────
    # Branch 1: TextCNN
    # ────────────────────────────────────────────────────────
    cnn_branches = []
    for fs in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=fs,
            activation="relu",
            padding="valid",
            kernel_regularizer=l2(l2_lambda),
            name=f"cnn_conv_{fs}gram",
        )(embedding)
        conv = BatchNormalization(name=f"cnn_bn_{fs}gram")(conv)
        pool = GlobalMaxPooling1D(name=f"cnn_pool_{fs}gram")(conv)
        cnn_branches.append(pool)

    cnn_concat = Concatenate(name="cnn_concat")(cnn_branches)
    cnn_fc = Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda),
                   name="cnn_fc")(cnn_concat)
    cnn_fc = BatchNormalization(name="cnn_fc_bn")(cnn_fc)
    cnn_fc = Dropout(dropout_rate, name="cnn_dropout")(cnn_fc)
    # CNN logits (pre-softmax)
    cnn_logits = Dense(num_classes, name="cnn_logits")(cnn_fc)

    # ────────────────────────────────────────────────────────
    # Branch 2: BiLSTM + Attention
    # ────────────────────────────────────────────────────────
    lstm_out = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True,
            dropout=0.2,
            kernel_regularizer=l2(l2_lambda),
        ),
        name="bilstm",
    )(embedding)

    attention_layer = BahdanauAttention(units=attention_units, name="attention")
    context, _ = attention_layer(lstm_out)

    lstm_fc = Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda),
                    name="lstm_fc")(context)
    lstm_fc = BatchNormalization(name="lstm_fc_bn")(lstm_fc)
    lstm_fc = Dropout(dropout_rate, name="lstm_dropout")(lstm_fc)
    # BiLSTM logits (pre-softmax)
    lstm_logits = Dense(num_classes, name="lstm_logits")(lstm_fc)

    # ────────────────────────────────────────────────────────
    # Fusion Head: Learn to weight CNN vs BiLSTM
    # ────────────────────────────────────────────────────────
    # Concatenate logits → learned fusion (better than simple average)
    fused = Concatenate(name="fusion_concat")([cnn_logits, lstm_logits])
    fused = Dense(64, activation="relu", name="fusion_fc")(fused)
    fused = Dropout(0.2, name="fusion_dropout")(fused)
    outputs = Dense(num_classes, activation="softmax", name="output")(fused)

    # ── Build & compile ─────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="Ensemble_CNN_BiLSTM_Municipal")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_ensemble_model(vocab_size=10013, num_classes=9)
    m.summary()

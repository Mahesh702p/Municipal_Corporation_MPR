"""
cnn_model.py
============
Multi-Filter 1D Convolutional Neural Network for Municipal Complaint
Classification. Detects local n-gram patterns across complaint text.

Architecture:
  Embedding → SpatialDropout → [Conv1D(2g) | Conv1D(3g) | Conv1D(4g)]
           → GlobalMaxPool × 3 → Concat → Dense → BN → Dropout → Softmax

Neural Network used: 1D-CNN (TextCNN)
Framework: TensorFlow / Keras
Task: Multi-class classification (department + category)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, BatchNormalization,
    SpatialDropout1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def build_cnn_model(
    vocab_size: int,
    num_classes: int,
    embedding_dim: int = 64,
    max_len: int = 60,
    filter_sizes: list = None,
    num_filters: int = 128,
    dropout_rate: float = 0.5,
    l2_lambda: float = 0.001,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """
    Build the Multi-Filter TextCNN model.

    Args:
        vocab_size:     Size of vocabulary (from tokenizer + special tokens).
        num_classes:    Number of output classes.
        embedding_dim:  Dimensionality of embedding vectors.
        max_len:        Maximum sequence length.
        filter_sizes:   List of Conv1D kernel sizes (n-gram windows).
        num_filters:    Number of filters (feature maps) per kernel size.
        dropout_rate:   Dropout probability.
        l2_lambda:      L2 regularization coefficient.
        learning_rate:  Adam optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    if filter_sizes is None:
        filter_sizes = [2, 3, 4]

    # ── Input ──────────────────────────────────────────────
    inputs = Input(shape=(max_len,), name="token_ids")

    # ── Embedding ──────────────────────────────────────────
    # mask_zero=True: ignore PAD tokens (id=0) in attention/pooling
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=False,  # GlobalMaxPool doesn't support masking directly
        name="embedding",
    )(inputs)

    # Spatial dropout on entire feature maps (better than regular dropout for embeddings)
    x = SpatialDropout1D(0.2, name="spatial_dropout")(x)

    # ── Parallel Conv branches (different n-gram windows) ──
    conv_branches = []
    for fs in filter_sizes:
        conv = Conv1D(
            filters=num_filters,
            kernel_size=fs,
            activation="relu",
            padding="valid",
            kernel_regularizer=l2(l2_lambda),
            name=f"conv_{fs}gram",
        )(x)
        conv = BatchNormalization(name=f"bn_conv_{fs}gram")(conv)
        pool = GlobalMaxPooling1D(name=f"maxpool_{fs}gram")(conv)
        conv_branches.append(pool)

    # ── Concatenate all branches ────────────────────────────
    concat = Concatenate(name="concat")(conv_branches)  # shape: (batch, num_filters * len(filter_sizes))

    # ── Dense classification head ───────────────────────────
    fc = Dense(256, activation="relu", kernel_regularizer=l2(l2_lambda), name="fc1")(concat)
    fc = BatchNormalization(name="bn_fc1")(fc)
    fc = Dropout(dropout_rate, name="dropout1")(fc)

    fc = Dense(128, activation="relu", kernel_regularizer=l2(l2_lambda), name="fc2")(fc)
    fc = Dropout(dropout_rate * 0.5, name="dropout2")(fc)

    outputs = Dense(num_classes, activation="softmax", name="output")(fc)

    # ── Build & compile ────────────────────────────────────
    model = Model(inputs=inputs, outputs=outputs, name="TextCNN_Municipal")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # Quick test
    m = build_cnn_model(vocab_size=10013, num_classes=9)
    m.summary()

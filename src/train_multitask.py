"""
Multi-Task Municipal Complaint Classifier - Training Script (450k Dataset)
---------------------------------------------------------------------------
Trains a shared backbone (TextCNN + BiLSTM + Attention) with two output heads:
  Head 1: Department classification (9 classes)
  Head 2: Intent classification (5 classes)

Datasets used:
  - data/complaints_labeled.csv    (100,000 records)
  - data/complaints_robust.csv     (350,000 records)
  Total: 450,000 training samples
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, BatchNormalization,
    Bidirectional, LSTM, Attention, Flatten,
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---- Reproducibility ----
np.random.seed(42)
tf.random.set_seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ================================
# Configuration
# ================================
CONFIG = {
    "labeled_data_path":  os.path.join(PROJECT_DIR, "data/complaints_labeled.csv"),
    "robust_data_path":   os.path.join(PROJECT_DIR, "data/complaints_robust.csv"),
    "model_dir":          os.path.join(PROJECT_DIR, "models_multitask"),
    "max_sequence_length": 80,
    "max_vocab_size":      20000,
    "embedding_dim":       128,
    "filter_sizes":        [2, 3, 4],
    "num_filters":         128,
    "lstm_units":          128,
    "dropout_rate":        0.5,
    "spatial_dropout":     0.3,
    "l2_reg":              0.001,
    "batch_size":          256,
    "epochs":              20,
    "learning_rate":       0.001,
    "test_size":           0.2,
}


# ================================
# Attention Layer (custom)
# ================================
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                  initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                  initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)


# ================================
# Load & Combine Data
# ================================
def load_data():
    print("\n📥 Loading datasets...")

    df_labeled = pd.read_csv(CONFIG["labeled_data_path"])
    df_robust  = pd.read_csv(CONFIG["robust_data_path"])

    print(f"  ✅ complaints_labeled.csv : {len(df_labeled):,} rows")
    print(f"  ✅ complaints_robust.csv  : {len(df_robust):,} rows")

    df = pd.concat([df_labeled, df_robust], ignore_index=True)
    df = df.dropna(subset=["text", "department", "intent"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"\n  🔢 Total combined dataset : {len(df):,} rows")
    print("\n  Department distribution:")
    print(df["department"].value_counts().to_string())
    print("\n  Intent distribution:")
    print(df["intent"].value_counts().to_string())

    return df


# ================================
# Tokenize Text
# ================================
def tokenize(df):
    print("\n🔠 Tokenizing text...")

    tokenizer = Tokenizer(num_words=CONFIG["max_vocab_size"], oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"].astype(str))
    sequences = tokenizer.texts_to_sequences(df["text"].astype(str))
    X = pad_sequences(sequences, maxlen=CONFIG["max_sequence_length"], padding="post", truncating="post")

    vocab_size = min(len(tokenizer.word_index) + 1, CONFIG["max_vocab_size"])
    print(f"  Vocabulary size  : {vocab_size:,}")
    print(f"  Sequence shape   : {X.shape}")

    return X, tokenizer, vocab_size


# ================================
# Encode Labels
# ================================
def encode_labels(df):
    dept_encoder   = LabelEncoder()
    intent_encoder = LabelEncoder()

    y_dept   = dept_encoder.fit_transform(df["department"])
    y_intent = intent_encoder.fit_transform(df["intent"])

    print(f"\n  Department classes ({len(dept_encoder.classes_)}): {dept_encoder.classes_.tolist()}")
    print(f"  Intent classes    ({len(intent_encoder.classes_)}): {intent_encoder.classes_.tolist()}")

    return y_dept, y_intent, dept_encoder, intent_encoder


# ================================
# Build Multi-Task Model
# ================================
def build_model(vocab_size, num_dept_classes, num_intent_classes):
    print("\n🔧 Building Multi-Task model...")

    inp = Input(shape=(CONFIG["max_sequence_length"],), name="input")

    # --- Embedding ---
    x = Embedding(vocab_size, CONFIG["embedding_dim"], name="embedding")(inp)
    x = tf.keras.layers.SpatialDropout1D(CONFIG["spatial_dropout"])(x)

    # --- Parallel CNN (bi-gram, tri-gram, 4-gram) ---
    conv_outputs = []
    for k in CONFIG["filter_sizes"]:
        c = Conv1D(CONFIG["num_filters"], k, activation="relu",
                   kernel_regularizer=l2(CONFIG["l2_reg"]), name=f"conv_{k}")(x)
        c = BatchNormalization()(c)
        c = GlobalMaxPooling1D()(c)
        conv_outputs.append(c)

    cnn_out = Concatenate(name="cnn_concat")(conv_outputs)   # (batch, 128*3=384)

    # --- BiLSTM + Attention ---
    lstm_in  = tf.keras.layers.Reshape((CONFIG["max_sequence_length"], CONFIG["embedding_dim"]))(
        tf.keras.layers.Embedding(vocab_size, CONFIG["embedding_dim"], name="embedding_lstm")(inp)
    )
    bilstm   = Bidirectional(LSTM(CONFIG["lstm_units"], return_sequences=True), name="bilstm")(lstm_in)
    att_out  = AttentionLayer(name="attention")(bilstm)      # (batch, 256)

    # --- Merge CNN + BiLSTM+Attention ---
    merged = Concatenate(name="merged")([cnn_out, att_out])  # (batch, 640)
    shared = Dense(256, activation="relu", kernel_regularizer=l2(CONFIG["l2_reg"]), name="shared_dense")(merged)
    shared = BatchNormalization()(shared)
    shared = Dropout(CONFIG["dropout_rate"])(shared)
    shared = Dense(128, activation="relu", name="shared_dense2")(shared)
    shared = Dropout(CONFIG["dropout_rate"] * 0.5)(shared)

    # --- Head 1: Department ---
    dept_out = Dense(num_dept_classes, activation="softmax", name="department")(shared)

    # --- Head 2: Intent ---
    intent_out = Dense(num_intent_classes, activation="softmax", name="intent")(shared)

    model = Model(inputs=inp, outputs=[dept_out, intent_out])
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss={
            "department": "sparse_categorical_crossentropy",
            "intent":     "sparse_categorical_crossentropy",
        },
        loss_weights={"department": 1.0, "intent": 1.0},
        metrics={"department": "accuracy", "intent": "accuracy"}
    )

    model.summary()
    return model


# ================================
# Main Training Pipeline
# ================================
def main():
    print("=" * 65)
    print("🏛️  Municipal Multi-Task Classifier — 450k Training Pipeline")
    print("=" * 65)

    # 1. Load data
    df = load_data()

    # 2. Tokenize
    X, tokenizer, vocab_size = tokenize(df)

    # 3. Encode labels
    y_dept, y_intent, dept_encoder, intent_encoder = encode_labels(df)

    # 4. Train/Test split
    (X_train, X_test,
     yd_train, yd_test,
     yi_train, yi_test) = train_test_split(
        X, y_dept, y_intent,
        test_size=CONFIG["test_size"],
        random_state=42
    )
    print(f"\n  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")

    # 5. Build model
    model = build_model(vocab_size,
                        len(dept_encoder.classes_),
                        len(intent_encoder.classes_))

    # 6. Callbacks
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(CONFIG["model_dir"], "best_model.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        )
    ]

    # 7. Train
    print("\n🚀 Training started...")
    history = model.fit(
        X_train,
        {"department": yd_train, "intent": yi_train},
        validation_data=(X_test, {"department": yd_test, "intent": yi_test}),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        callbacks=callbacks
    )

    # 8. Evaluate
    print("\n📊 Evaluation on Test Set:")
    dept_pred_prob, intent_pred_prob = model.predict(X_test, verbose=1, batch_size=512)
    dept_pred   = np.argmax(dept_pred_prob, axis=1)
    intent_pred = np.argmax(intent_pred_prob, axis=1)

    print("\n--- DEPARTMENT CLASSIFICATION ---")
    print(classification_report(yd_test, dept_pred, target_names=dept_encoder.classes_))
    dept_acc = accuracy_score(yd_test, dept_pred)
    print(f"🎯 Department Test Accuracy: {dept_acc:.2%}")

    print("\n--- INTENT CLASSIFICATION ---")
    print(classification_report(yi_test, intent_pred, target_names=intent_encoder.classes_))
    intent_acc = accuracy_score(yi_test, intent_pred)
    print(f"🎯 Intent Test Accuracy: {intent_acc:.2%}")

    # 9. Save everything
    model.save(os.path.join(CONFIG["model_dir"], "multitask_model.keras"))
    joblib.dump(tokenizer,      os.path.join(CONFIG["model_dir"], "multitask_tokenizer.pkl"))
    joblib.dump(dept_encoder,   os.path.join(CONFIG["model_dir"], "dept_encoder.pkl"))
    joblib.dump(intent_encoder, os.path.join(CONFIG["model_dir"], "intent_encoder.pkl"))
    joblib.dump(CONFIG,         os.path.join(CONFIG["model_dir"], "multitask_config.pkl"))

    print(f"\n✅ All model files saved to: {CONFIG['model_dir']}/")
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Department Accuracy : {dept_acc:.2%}")
    print(f"   Intent Accuracy     : {intent_acc:.2%}")


if __name__ == "__main__":
    main()

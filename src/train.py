"""
CNN Municipal Complaint Classifier - Training Script
-----------------------------------------------------
Trains a 1D CNN model to classify municipal complaints into categories.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, Concatenate, BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ================================
# Configuration
# ================================
CONFIG = {
    "data_path": os.path.join(PROJECT_DIR, "data/municipal_complaints_clean.csv"),
    "model_dir": os.path.join(PROJECT_DIR, "models"),
    "max_sequence_length": 60,
    "max_vocab_size": 3000,
    "embedding_dim": 64,
    "filter_sizes": [2, 3, 4],
    "num_filters": 64,
    "dropout_rate": 0.6,
    "spatial_dropout": 0.4,
    "l2_reg": 0.01,
    "batch_size": 16,
    "epochs": 30,
    "learning_rate": 0.0005,
    "test_size": 0.2,
}


def main():
    print("="*60)
    print("🏛️  Municipal Complaint CNN Classifier - Training")
    print("="*60)
    
    # Load data
    print("\n📥 Loading dataset...")
    df = pd.read_csv(CONFIG["data_path"])
    print(f"Total samples: {len(df)}")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    num_classes = len(label_encoder.classes_)
    
    # Tokenize
    tokenizer = Tokenizer(num_words=CONFIG["max_vocab_size"], oov_token="<OOV>")
    tokenizer.fit_on_texts(df["clean_text"])
    sequences = tokenizer.texts_to_sequences(df["clean_text"])
    X = pad_sequences(sequences, maxlen=CONFIG["max_sequence_length"], padding="post")
    
    vocab_size = min(len(tokenizer.word_index) + 1, CONFIG["max_vocab_size"])
    print(f"Vocabulary size: {vocab_size}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=42, stratify=y
    )
    print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Build model
    print("\n🔧 Building CNN model...")
    input_layer = Input(shape=(CONFIG["max_sequence_length"],))
    embedding = Embedding(vocab_size, CONFIG["embedding_dim"])(input_layer)
    embedding = tf.keras.layers.SpatialDropout1D(CONFIG["spatial_dropout"])(embedding)
    
    conv_outputs = []
    for filter_size in CONFIG["filter_sizes"]:
        conv = Conv1D(CONFIG["num_filters"], filter_size, activation="relu",
                      kernel_regularizer=l2(CONFIG["l2_reg"]))(embedding)
        conv = BatchNormalization()(conv)
        pool = GlobalMaxPooling1D()(conv)
        conv_outputs.append(pool)
    
    concatenated = Concatenate()(conv_outputs)
    dense = Dense(128, activation="relu", kernel_regularizer=l2(CONFIG["l2_reg"]))(concatenated)
    dense = BatchNormalization()(dense)
    dense = Dropout(CONFIG["dropout_rate"])(dense)
    dense = Dense(64, activation="relu", kernel_regularizer=l2(CONFIG["l2_reg"]))(dense)
    dense = Dropout(CONFIG["dropout_rate"] * 0.5)(dense)
    output = Dense(num_classes, activation="softmax")(dense)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    # Train
    print("\n🚀 Training...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=CONFIG["epochs"], batch_size=CONFIG["batch_size"],
              callbacks=callbacks, class_weight=class_weight_dict)
    
    # Evaluate
    print("\n📊 Evaluation:")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"\n🎯 Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    
    # Save
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    model.save(f"{CONFIG['model_dir']}/cnn_category_model.keras")
    joblib.dump(tokenizer, f"{CONFIG['model_dir']}/cnn_tokenizer.pkl")
    joblib.dump(label_encoder, f"{CONFIG['model_dir']}/cnn_label_encoder.pkl")
    joblib.dump(CONFIG, f"{CONFIG['model_dir']}/cnn_config.pkl")
    
    print("\n✅ Model saved to models/")


if __name__ == "__main__":
    main()

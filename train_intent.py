"""
train_intent.py
===============
Trains the Level-1 Intent Classifier for the Municipal Corporation MLM.

Classes:
  - complaint       (citizen reporting a problem)
  - query           (asking how to do something / what documents)
  - status_check    (checking complaint/application status)
  - emergency       (urgent danger situations)
  - service_request (applying for a new service)

Uses the same Ensemble CNN+BiLSTM architecture.
Loads data from complaints_labeled.csv (which has 'intent' column after generate_intents.py).

Usage:
  python train_intent.py

Output:
  artifacts/intent_model/   ← saved Keras model
  artifacts/label_encoders.json updated with 'intent' key
"""

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "abctokz_repo" / "src"))
sys.path.insert(0, str(ROOT))

DATA_PATH = str(ROOT / "data" / "processed" / "complaints_robust.csv")
TOK_PATH = str(ROOT / "artifacts" / "municipal_bpe_tok")
SAVE_PATH = str(ROOT / "artifacts" / "intent_model")
LABEL_ENC_PATH = str(ROOT / "artifacts" / "label_encoders.json")

MAX_LEN = 60
VOCAB_SIZE = 10000
EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5


def main():
    import tensorflow as tf

    # ── Load data ───────────────────────────────────────────
    print(f"\nLoading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total rows: {len(df):,}")

    if "intent" not in df.columns:
        print("ERROR: 'intent' column not found. Run generate_intents.py first.")
        return

    df = df.dropna(subset=["text", "intent"])
    texts = df["text"].astype(str).tolist()
    labels_raw = df["intent"].astype(str).tolist()

    label_classes = sorted(set(labels_raw))
    label2idx = {lbl: i for i, lbl in enumerate(label_classes)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    labels = np.array([label2idx[l] for l in labels_raw])
    num_classes = len(label_classes)

    print(f"Intent classes ({num_classes}): {label_classes}")
    dist = {lbl: int((labels == i).sum()) for lbl, i in label2idx.items()}
    print(f"Distribution: {dist}")

    # ── Tokenizer ───────────────────────────────────────────
    from tokenizer.municipal_tokenizer import MunicipalTokenizer
    if os.path.exists(os.path.join(TOK_PATH, "manifest.json")):
        tok = MunicipalTokenizer.load(TOK_PATH)
    else:
        print("Tokenizer not found — run train.py first to build tokenizer.")
        return

    vocab_size = tok.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size:,}")

    # ── Encode ──────────────────────────────────────────────
    print(f"\nEncoding {len(texts):,} texts...")
    X = tok.encode_batch(texts, max_len=MAX_LEN)
    y = labels
    print(f"X shape: {X.shape}")

    # ── Split ───────────────────────────────────────────────
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )
    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Model ───────────────────────────────────────────────
    from models.ensemble_model import build_ensemble_model
    model = build_ensemble_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
    )
    model.summary(line_length=72)

    os.makedirs(SAVE_PATH, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_PATH, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    # ── Train ───────────────────────────────────────────────
    # Compute dynamic class weights to balance dataset
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    print(f"Computed Class Weights: {class_weight_dict}")

    print("\nTraining Intent Classifier (Ensemble)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ────────────────────────────────────────────
    print("\n--- Test Evaluation ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    label_names = [idx2label[i] for i in sorted(idx2label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    # ── Save ─────────────────────────────────────────────────
    model.save(os.path.join(SAVE_PATH, "model.keras"))

    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(SAVE_PATH, "history.json"), "w") as f:
        json.dump(hist_data, f, indent=2)

    # Update label encoders
    enc_data = {}
    if os.path.exists(LABEL_ENC_PATH):
        with open(LABEL_ENC_PATH) as f:
            enc_data = json.load(f)
    enc_data["intent"] = {"label2idx": label2idx, "idx2label": idx2label}
    with open(LABEL_ENC_PATH, "w") as f:
        json.dump(enc_data, f, indent=2)

    print(f"\n✓ Intent model saved → {SAVE_PATH}/")
    print(f"✓ Label encoders updated → {LABEL_ENC_PATH}")


if __name__ == "__main__":
    main()

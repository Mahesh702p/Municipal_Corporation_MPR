"""
train.py
========
End-to-End Training Script for Municipal Corporation MLM.

Pipeline:
  1. Load labeled data from complaints_labeled.csv
  2. Encode text with MunicipalTokenizer (train if needed)
  3. Encode department/category labels
  4. Train/Val/Test split (80/10/10)
  5. Train all three models (CNN, BiLSTM+Attention, Ensemble)
  6. Evaluate and save best model
  7. Print classification report

Usage:
  python train.py                    # train all models
  python train.py --model cnn        # train only CNN
  python train.py --model bilstm     # train only BiLSTM+Attention
  python train.py --model ensemble   # train only Ensemble
  python train.py --target category  # train on category labels (20 classes)

Outputs:
  artifacts/municipal_bpe_tok/     ← trained tokenizer
  artifacts/<model_name>/          ← saved Keras model
  artifacts/label_encoders.json    ← label → int mapping
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from pathlib import Path

# ── Project paths ──────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "abctokz_repo" / "src"))
sys.path.insert(0, str(ROOT))

# ── Data & artifacts paths ──────────────────────────────────
DATA_PATH = str(ROOT / "data" / "processed" / "complaints_robust.csv")
CORPUS_PATH = str(ROOT / "data" / "processed" / "pretrain_corpus.txt")
TOK_PATH = str(ROOT / "artifacts" / "municipal_bpe_tok")
ARTIFACTS_DIR = str(ROOT / "artifacts")
LABEL_ENC_PATH = str(ROOT / "artifacts" / "label_encoders.json")

# ── Hyperparameters ─────────────────────────────────────────
MAX_LEN = 60
VOCAB_SIZE = 10000
EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5              # early stopping patience


def load_data(target: str = "department"):
    """Load CSV, encode labels, return texts + label array."""
    print(f"\n{'='*55}")
    print(f" Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f" Total rows: {len(df):,}")
    print(f" Columns: {list(df.columns)}")

    df = df.dropna(subset=["text", target])
    df = df[df[target] != "Unknown"]
    texts = df["text"].astype(str).tolist()

    # Encode labels
    labels_raw = df[target].astype(str).tolist()
    label_classes = sorted(set(labels_raw))
    label2idx = {lbl: i for i, lbl in enumerate(label_classes)}
    idx2label = {i: lbl for lbl, i in label2idx.items()}
    labels = np.array([label2idx[l] for l in labels_raw])

    print(f" Classes ({target}): {len(label_classes)}")
    print(f"   {label_classes}")

    dist = {lbl: int((labels == i).sum()) for lbl, i in label2idx.items()}
    print(f" Class distribution: {dist}")

    # Save label encoders
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    enc_data = {}
    try:
        if os.path.exists(LABEL_ENC_PATH):
            with open(LABEL_ENC_PATH) as f:
                enc_data = json.load(f)
    except Exception:
        pass
    enc_data[target] = {"label2idx": label2idx, "idx2label": idx2label}
    with open(LABEL_ENC_PATH, "w") as f:
        json.dump(enc_data, f, indent=2)

    return texts, labels, label2idx, idx2label


def get_or_train_tokenizer():
    """Load existing tokenizer or train a new one."""
    from tokenizer.municipal_tokenizer import MunicipalTokenizer

    if os.path.exists(os.path.join(TOK_PATH, "manifest.json")):
        print(f"\n Loading tokenizer from: {TOK_PATH}")
        tok = MunicipalTokenizer.load(TOK_PATH)
    else:
        print(f"\n Training tokenizer (vocab_size={VOCAB_SIZE})...")
        tok = MunicipalTokenizer(vocab_size=VOCAB_SIZE, model_type="bpe", max_len=MAX_LEN)
        tok.train([CORPUS_PATH])
        tok.save(TOK_PATH)

    print(f" Tokenizer vocab size: {tok.get_vocab_size():,}")
    return tok


def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Stratified 80/10/10 split."""
    from sklearn.model_selection import train_test_split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_ratio + test_ratio, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed, stratify=y_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(model_type: str, vocab_size: int, num_classes: int):
    """Build and return the selected model."""
    if model_type == "cnn":
        from models.cnn_model import build_cnn_model
        return build_cnn_model(vocab_size=vocab_size, num_classes=num_classes,
                               embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN)
    elif model_type == "bilstm":
        from models.bilstm_attention_model import build_bilstm_attention_model
        return build_bilstm_attention_model(vocab_size=vocab_size, num_classes=num_classes,
                                            embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN)
    elif model_type == "ensemble":
        from models.ensemble_model import build_ensemble_model
        return build_ensemble_model(vocab_size=vocab_size, num_classes=num_classes,
                                    embedding_dim=EMBEDDING_DIM, max_len=MAX_LEN)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_model(model_type: str, X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    vocab_size, num_classes, idx2label):
    """Train, evaluate, and save one model."""
    import tensorflow as tf

    print(f"\n{'='*55}")
    print(f" Training: {model_type.upper()}")
    print(f"{'='*55}")

    model = build_model(model_type, vocab_size, num_classes)
    model.summary(line_length=72)

    save_path = os.path.join(ARTIFACTS_DIR, f"{model_type}_model")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_path, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    os.makedirs(save_path, exist_ok=True)

    # Compute dynamic class weights to balance dataset
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(classes, weights)}
    print(f"Computed Class Weights: {class_weight_dict}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print(f"\n--- Test Evaluation ({model_type.upper()}) ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Classification report
    from sklearn.metrics import classification_report
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    label_names = [idx2label[i] for i in sorted(idx2label.keys())]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=3))

    # Save full model
    model.save(os.path.join(save_path, "model.keras"))

    # Save training history
    hist_path = os.path.join(save_path, "history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)

    print(f"\n✓ Model saved → {save_path}/")

    return test_acc, history


def main():
    global EPOCHS, BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "bilstm", "ensemble", "all"],
                        default="all", help="Model to train")
    parser.add_argument("--target", choices=["department", "category"],
                        default="department", help="Label target")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size


    # ── Load data ───────────────────────────────────────────
    texts, labels, label2idx, idx2label = load_data(target=args.target)
    num_classes = len(label2idx)

    # ── Tokenizer ───────────────────────────────────────────
    tok = get_or_train_tokenizer()
    vocab_size = tok.get_vocab_size()

    # ── Encode all texts ────────────────────────────────────
    print(f"\n Encoding {len(texts):,} texts → (batch, {MAX_LEN})...")
    X = tok.encode_batch(texts, max_len=MAX_LEN)
    y = labels

    print(f" X shape: {X.shape}, dtype: {X.dtype}")
    print(f" y shape: {y.shape}, classes: {num_classes}")

    # ── Split ───────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"\n Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Train selected models ────────────────────────────────
    models_to_train = ["cnn", "bilstm", "ensemble"] if args.model == "all" else [args.model]
    results = {}

    for mtype in models_to_train:
        acc, hist = train_one_model(
            mtype, X_train, X_val, X_test,
            y_train, y_val, y_test,
            vocab_size, num_classes, idx2label
        )
        results[mtype] = acc

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(" FINAL RESULTS SUMMARY")
    print(f"{'='*55}")
    for mtype, acc in results.items():
        print(f"  {mtype.upper():<12}  Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*55}\n")

    if results:
        best = max(results, key=results.get)
        print(f" Best model: {best.upper()} ({results[best]*100:.2f}%)")
        print(f" Artifacts: {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()

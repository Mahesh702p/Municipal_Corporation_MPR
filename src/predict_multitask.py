"""
predict_multitask.py
--------------------
CLI Script to run predictions using the trained Multi-Task Model (450k dataset).
Loads model artifacts from Municipal_Multitask_Results/ directory.

Usage:
    python src/predict_multitask.py
    python src/predict_multitask.py "urgent fire near transformer in ward 5"
"""

import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Base path to multi-task model artifacts
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "Municipal_Multitask_Results")

MODEL_PATH     = os.path.join(RESULTS_DIR, "multitask_model.keras")
TOKENIZER_PATH = os.path.join(RESULTS_DIR, "multitask_tokenizer.pkl")
DEPT_ENC_PATH  = os.path.join(RESULTS_DIR, "dept_encoder.pkl")
INTENT_ENC_PATH= os.path.join(RESULTS_DIR, "intent_encoder.pkl")
CONFIG_PATH    = os.path.join(RESULTS_DIR, "multitask_config.pkl")

# Custom Attention Layer required for model loading
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_W', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1),  initializer='zeros',          trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)


def load_multitask_artifacts():
    print("⏳ Loading Multi-Task Model Artifacts...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load model with custom layer mapping
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )
    tokenizer  = joblib.load(TOKENIZER_PATH)
    dept_enc   = joblib.load(DEPT_ENC_PATH)
    intent_enc = joblib.load(INTENT_ENC_PATH)
    config     = joblib.load(CONFIG_PATH)

    print("✅ Multi-Task Model & Encoders Loaded Successfully!\n")
    return model, tokenizer, dept_enc, intent_enc, config


def predict_text(text: str, model, tokenizer, dept_enc, intent_enc, max_seq_len=60):
    # Preprocess text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_seq_len, padding="post", truncating="post")

    # Model inference
    dept_probs, intent_probs = model.predict(padded, verbose=0)

    # Top predictions
    dept_idx   = np.argmax(dept_probs[0])
    intent_idx = np.argmax(intent_probs[0])

    dept_pred   = dept_enc.classes_[dept_idx]
    dept_conf   = dept_probs[0][dept_idx]

    intent_pred = intent_enc.classes_[intent_idx]
    intent_conf = intent_probs[0][intent_idx]

    return {
        "text": text,
        "department": dept_pred,
        "department_confidence": float(dept_conf),
        "intent": intent_pred,
        "intent_confidence": float(intent_conf)
    }


def main():
    model, tokenizer, dept_enc, intent_enc, config = load_multitask_artifacts()
    max_seq_len = config.get("max_seq_len", 60)

    # Single argument evaluation
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        res = predict_text(text, model, tokenizer, dept_enc, intent_enc, max_seq_len)
        print("=" * 60)
        print(f"📝 Input Text: \"{res['text']}\"")
        print(f"🏛️ Department : {res['department']} ({res['department_confidence']:.2%})")
        print(f"🎯 Intent     : {res['intent']} ({res['intent_confidence']:.2%})")
        print("=" * 60)
        return

    # Interactive Loop Mode
    print("=" * 60)
    print("🏛️ Municipal Multi-Task Grievance Routing Engine (Interactive Prompt)")
    print("Type any complaint or query in English or Hinglish.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Exiting prediction CLI.")
                break

            res = predict_text(user_input, model, tokenizer, dept_enc, intent_enc, max_seq_len)
            print("-" * 50)
            print(f"🏛️  Department Prediction : {res['department'].upper()} ({res['department_confidence']:.2%})")
            print(f"🎯  Intent Prediction     : {res['intent'].upper()} ({res['intent_confidence']:.2%})")
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

"""
Municipal Complaint Classifier - Prediction Script
---------------------------------------------------
Classifies municipal complaints and returns department, category, and priority.
"""

import os
import re
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Load model and artifacts
print("📥 Loading model...")
model = load_model(os.path.join(MODELS_DIR, "cnn_category_model.keras"))
tokenizer = joblib.load(os.path.join(MODELS_DIR, "cnn_tokenizer.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
config = joblib.load(os.path.join(MODELS_DIR, "cnn_config.pkl"))
print("✅ Model loaded!")

# Category to Department/Priority mapping
CATEGORY_INFO = {
    "Leakage": {"department": "Water Supply", "priority": "High"},
    "Shortage": {"department": "Water Supply", "priority": "High"},
    "Power Cut": {"department": "Electricity", "priority": "High"},
    "Pothole": {"department": "Roads", "priority": "Medium"},
    "Road Damage": {"department": "Roads", "priority": "Medium"},
    "Garbage": {"department": "Sanitation", "priority": "Low"}
}


def clean_text(text):
    """Clean and normalize input text."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!? ]", "", text)
    return text.strip()


def predict(complaint_text):
    """
    Classify a complaint and return prediction details.
    
    Args:
        complaint_text: The complaint text to classify
        
    Returns:
        dict with department, category, priority, and confidence
    """
    # Preprocess
    cleaned = clean_text(complaint_text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=config["max_sequence_length"], 
                          padding="post", truncating="post")
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    
    # Get category and info
    category = label_encoder.inverse_transform([predicted_class])[0]
    info = CATEGORY_INFO.get(category, {"department": "Unknown", "priority": "Unknown"})
    
    return {
        "category": category,
        "department": info["department"],
        "priority": info["priority"],
        "confidence": round(confidence * 100, 2)
    }


def main():
    """Interactive CLI for testing predictions."""
    print("\n" + "="*55)
    print("🏛️  Municipal Complaint Classifier")
    print("="*55)
    print("Categories: Water Leakage, Water Shortage, Power Cut,")
    print("            Pothole, Road Damage, Garbage")
    print("="*55)
    
    while True:
        complaint = input("\nEnter complaint (or 'quit' to exit): ").strip()
        
        if complaint.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not complaint:
            print("⚠️  Please enter a valid complaint.")
            continue
        
        result = predict(complaint)
        
        print("\n📋 Classification Result")
        print("-" * 35)
        print(f"  Category:   {result['category']}")
        print(f"  Department: {result['department']}")
        print(f"  Priority:   {result['priority']}")
        print(f"  Confidence: {result['confidence']}%")


if __name__ == "__main__":
    main()

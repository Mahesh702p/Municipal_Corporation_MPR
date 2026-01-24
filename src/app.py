import streamlit as st
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
from spellchecker import SpellChecker
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(
    page_title="Municipal Complaint Classifier",
    layout="centered"
)

# Constants & Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

CATEGORY_INFO = {
    "Leakage": {"department": "Water Supply", "priority": "High"},
    "Shortage": {"department": "Water Supply", "priority": "High"},
    "Power Cut": {"department": "Electricity", "priority": "High"},
    "Pothole": {"department": "Roads", "priority": "Medium"},
    "Road Damage": {"department": "Roads", "priority": "Medium"},
    "Garbage": {"department": "Sanitation", "priority": "Low"}
}

@st.cache_resource
def load_prediction_model():
    """Load the model and tokenizer once."""
    try:
        model = load_model(os.path.join(MODELS_DIR, "cnn_category_model.keras"))
        tokenizer = joblib.load(os.path.join(MODELS_DIR, "cnn_tokenizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
        config = joblib.load(os.path.join(MODELS_DIR, "cnn_config.pkl"))
        return model, tokenizer, label_encoder, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def clean_text(text):
    """Clean and normalize input text."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!? ]", "", text)
    return text.strip()

def correct_spelling(text):
    """Correct spelling using pyspellchecker with repeated char reduction."""
    spell = SpellChecker()
    # Reduce repeated chars pattern (e.g. "gaaarbbbage" -> "gaarbbbage")
    # Actually, we want to reduce ANY run of 3+ same chars to 2 chars.
    # regex: (.)\1+ matches a char followed by itself 1 or more times.
    # But we want to keep 2 if there are 3+.
    # Simpler approach from testing: re.sub(r'(.)\1+', r'\1\1', text) 
    # This turns "aaa" -> "aa", "bb" -> "bb".
    text = re.sub(r'(.)\1+', r'\1\1', text)
    
    words = text.split()
    corrected_words = []
    
    for word in words:
        # Get the one `most likely` answer. 
        # spell.correction returns None if word is unknown and no correction found, 
        # or the word itself if it's known.
        # Actually spell.correction(word) checks if it needs correction.
        # If word is in dictionary, it returns the word.
        corrected = spell.correction(word)
        if corrected:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)

def main():
    st.title("Municipal Complaint Classifier")

    # Load resources
    model, tokenizer, label_encoder, config = load_prediction_model()

    if not model:
        st.warning("Could not load model. Please ensure the model files are in the 'models/' directory.")
        return

    # User Input
    with st.container():
        complaint_text = st.text_area("Describe the issue:", height=150, placeholder="e.g., There is a huge pothole on Main Street causing traffic.")

        if st.button("Classify Complaint", type="primary"):
            if not complaint_text.strip():
                st.warning("Please enter a complaint description.")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocess
                    # Spell Check
                    corrected_text = correct_spelling(complaint_text)
                    if corrected_text.lower() != complaint_text.lower():
                        st.info(f"**Spelling Corrected:** {corrected_text}")
                    
                    cleaned = clean_text(corrected_text)
                    sequence = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(sequence, maxlen=config["max_sequence_length"], 
                                         padding="post", truncating="post")
                    
                    # Predict
                    prediction = model.predict(padded, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = float(np.max(prediction)) * 100
                    
                    category = label_encoder.inverse_transform([predicted_class])[0]
                    info = CATEGORY_INFO.get(category, {"department": "Unknown", "priority": "Unknown"})

                    # Display Results
                    st.divider()
                    
                    # Custom CSS for cards
                    st.markdown("""
                    <style>
                    .metric-card {
                        background-color: #0e1117;
                        border: 1px solid #262730;
                        padding: 20px;
                        border-radius: 10px;
                        text-align: center;
                    }
                    .metric-label {
                        font-size: 14px;
                        color: #fafafa;
                        margin-bottom: 5px;
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    
                    priority_color = "#28a745"  # Green
                    if info['priority'] == "Medium":
                        priority_color = "#ffc107" # Amber
                    elif info['priority'] == "High":
                        priority_color = "#dc3545" # Red

                    with c1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Category</div>
                            <div class="metric-value">{category}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Department</div>
                            <div class="metric-value">{info['department']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c3:
                        st.markdown(f"""
                        <div class="metric-card" style="border-color: {priority_color};">
                            <div class="metric-label">Priority</div>
                            <div class="metric-value" style="color: {priority_color};">{info['priority']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.divider()
                    st.progress(confidence / 100, text=f"Confidence: {confidence:.2f}%")
                    
                    # Contextual Message
                    if info['priority'] == "High":
                        st.error("This is a high-priority issue. A ticket has been raised immediately.")
                    elif info['priority'] == "Medium":
                        st.warning("This issue has been logged for review.")
                    else:
                        st.info("Thank you for your feedback. We will attend to it shortly.")

    # Footer
    st.markdown("---")
    st.caption("Powered by CNN & TensorFlow | Municipal Corporation Portal")

if __name__ == "__main__":
    main()

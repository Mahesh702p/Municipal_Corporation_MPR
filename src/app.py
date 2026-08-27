import streamlit as st
import os
import sys
import warnings

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import re
import random
import datetime
import json
import numpy as np
import pandas as pd
import joblib
from spellchecker import SpellChecker
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page config
st.set_page_config(
    page_title="Municipal Complaint Classifier & Routing System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
ARCH_IMAGE_PATH = os.path.join(PROJECT_DIR, "architecture_diagram.png")

# Comprehensive Category / Department Mapping for both Clean & Labeled Datasets
CATEGORY_INFO = {
    "water_supply": {"department": "Water Supply Department", "priority": "High", "icon": "🚰", "code": "WTR"},
    "electricity": {"department": "Electricity Board", "priority": "High", "icon": "⚡", "code": "ELE"},
    "disaster_management": {"department": "Disaster & Emergency Services", "priority": "High", "icon": "🚨", "code": "EMG"},
    "sewerage": {"department": "Sewerage & Drainage", "priority": "High", "icon": "🌊", "code": "SEW"},
    "roads": {"department": "Roads & Infrastructure", "priority": "Medium", "icon": "🛣️", "code": "RD"},
    "solid_waste": {"department": "Sanitation & Solid Waste", "priority": "Medium", "icon": "🗑️", "code": "SAN"},
    "health": {"department": "Public Health & Sanitation", "priority": "Medium", "icon": "🏥", "code": "HLT"},
    "revenue": {"department": "Revenue & Property Tax", "priority": "Low", "icon": "📑", "code": "REV"},
    "parks": {"department": "Parks & Recreation", "priority": "Low", "icon": "🌳", "code": "PRK"},

    # Clean Dataset Categories
    "Leakage": {"department": "Water Supply Department", "priority": "High", "icon": "💧", "code": "WTR-LEAK"},
    "Shortage": {"department": "Water Supply Department", "priority": "High", "icon": "🚰", "code": "WTR-SHORT"},
    "Power Cut": {"department": "Electricity Board", "priority": "High", "icon": "⚡", "code": "ELE-CUT"},
    "Pothole": {"department": "Roads & Infrastructure", "priority": "Medium", "icon": "🛣️", "code": "RD-POT"},
    "Road Damage": {"department": "Roads & Infrastructure", "priority": "Medium", "icon": "🧱", "code": "RD-DMG"},
    "Garbage": {"department": "Sanitation & Solid Waste", "priority": "Low", "icon": "🗑️", "code": "SAN-GARB"}
}

SAMPLE_COMPLAINTS = {
    "💧 Water Shortage (Hinglish)": "paani nai aaraha teen dino se ghar me severe crisis hai",
    "⚡ Power Outage (Hinglish)": "bijli chali gayi hai poore sector 15 me bohot garmi hai",
    "🛣️ Pothole Issue (Hinglish)": "main sadak par bada gadda hai traffic jam ho raha hai",
    "🗑️ Garbage Problem (Hinglish)": "gali me kachra pada hai 4 dino se koi nahi aaya",
    "🚨 Emergency Disaster": "massive fire outbreak near transformer behind society",
    "🚰 Water Supply (English)": "No drinking water supply in Ward 7 for past 3 days"
}

# Common Hinglish / Transliterated Indian keywords to preserve
HINGLISH_KEYWORDS = [
    "paani", "pani", "ghar", "bijli", "sadak", "kachra", "naali", "gadda", 
    "safai", "dino", "aaraha", "araha", "aa", "raha", "me", "se", "nai", 
    "nhi", "kaafi", "bhi", "bohot", "bahut", "gali", "mora", "pani", "light", "teen"
]

# Custom Premium Styling
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Card */
    .header-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #60a5fa;
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #9ca3af;
        font-size: 15px;
    }

    /* Result Card Styling */
    .res-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .res-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .res-val {
        font-size: 22px;
        font-weight: 700;
        color: #f0f6fc;
    }

    /* Ticket Box Styling */
    .ticket-box {
        background-color: #0d1117;
        border: 2px dashed #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    .ticket-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #30363d;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    .ticket-id {
        font-family: monospace;
        font-weight: bold;
        color: #38bdf8;
        font-size: 18px;
    }

    /* Badge Pills */
    .badge-high {
        background-color: rgba(220, 38, 38, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-medium {
        background-color: rgba(217, 119, 6, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load trained Keras 1D CNN model & tokenizers."""
    try:
        model = load_model(os.path.join(MODELS_DIR, "cnn_category_model.keras"))
        tokenizer = joblib.load(os.path.join(MODELS_DIR, "cnn_tokenizer.pkl"))
        label_encoder = joblib.load(os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
        config = joblib.load(os.path.join(MODELS_DIR, "cnn_config.pkl"))
        return model, tokenizer, label_encoder, config
    except Exception as e:
        st.error(f"Error loading AI model artifacts: {e}")
        return None, None, None, None


@st.cache_data
def load_dataset():
    """Load dataset for analytics tab."""
    for fn in ["complaints_labeled.csv", "municipal_complaints_clean.csv"]:
        csv_path = os.path.join(DATA_DIR, fn)
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
    return None


def clean_text(text):
    """Normalize input complaint text."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9.,!? ]", "", text)
    return text.strip()


def edit_distance(s1, s2):
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def correct_spelling(text):
    """Spell correction with Hinglish protection & distance constraints."""
    spell = SpellChecker()
    spell.word_frequency.load_words(HINGLISH_KEYWORDS)
    
    # Reduce 3 or more repeated characters
    text_reduced = re.sub(r'(.)\1{2,}', r'\1', text)
    
    words = text_reduced.split()
    corrected_words = []
    
    for word in words:
        clean_w = re.sub(r'[^a-z0-9]', '', word.lower())
        if len(clean_w) > 3:
            if clean_w in spell.known([clean_w]):
                corrected_words.append(word)
            else:
                candidate = spell.correction(clean_w)
                if candidate and candidate != clean_w:
                    if edit_distance(clean_w, candidate) <= 1:
                        corrected_words.append(word.replace(clean_w, candidate))
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words)


def main():
    # Header Banner
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            🏛️ Municipal Complaint Classifier & Smart Dispatch Engine
        </div>
        <div class="header-subtitle">
            Automated Deep Learning (1D CNN + NLP Pipeline) for Citizen Grievance Categorization, Department Assignment & Urgent Priority Routing
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model
    model, tokenizer, label_encoder, config = load_model_artifacts()

    if model is None:
        st.error("Failed to load model. Please run `python src/train.py` first.")
        return

    # Tabs
    tab_classifier, tab_analytics, tab_arch = st.tabs([
        "🚀 Live Grievance Classifier", 
        "📊 Municipal Workload Analytics", 
        "🛠️ DL Model & NLP Architecture"
    ])

    # ==========================================
    # TAB 1: LIVE CLASSIFIER
    # ==========================================
    with tab_classifier:
        col_left, col_right = st.columns([1.1, 0.9], gap="large")

        with col_left:
            st.subheader("📝 Submit Citizen Grievance")
            
            # Quick Sample Selection
            st.markdown("**Try a sample complaint scenario:**")
            selected_sample = st.selectbox(
                "Choose sample preset:",
                ["-- Select Sample Preset --"] + list(SAMPLE_COMPLAINTS.keys()),
                label_visibility="collapsed"
            )
            
            default_text = ""
            if selected_sample != "-- Select Sample Preset --":
                default_text = SAMPLE_COMPLAINTS[selected_sample]

            complaint_input = st.text_area(
                "Enter Complaint Description:",
                value=default_text,
                height=130,
                placeholder="Describe the issue in detail (e.g. paani nai aaraha teen dino se, severe leak near MG road...)"
            )

            st.caption("💡 **Tip**: *Supports both Hinglish (e.g. 'paani nai aaraha', 'bijli chali gayi') and English complaints.*")

            c_chk, c_empty = st.columns([1.2, 0.8])
            with c_chk:
                enable_spellcheck = st.checkbox("✨ Enable Smart Typo Correction", value=True)

            c_btn1, c_btn2 = st.columns([1, 1])
            with c_btn1:
                classify_clicked = st.button("⚡ Classify & Dispatch Ticket", type="primary", use_container_width=True)
            with c_btn2:
                if st.button("🧹 Clear Input", use_container_width=True):
                    st.rerun()

        with col_right:
            st.subheader("🎯 Real-Time Analysis & Routing")

            if classify_clicked and complaint_input.strip():
                with st.spinner("Processing NLP pipeline & 1D CNN Inference..."):
                    # 1. Spell Check & Cleaning
                    if enable_spellcheck:
                        corrected_text = correct_spelling(complaint_input)
                    else:
                        corrected_text = complaint_input

                    has_spelling_diff = corrected_text.strip().lower() != complaint_input.strip().lower()
                    cleaned = clean_text(corrected_text)
                    
                    # 2. Tokenize & Pad
                    sequence = tokenizer.texts_to_sequences([cleaned])
                    padded = pad_sequences(
                        sequence, 
                        maxlen=config["max_sequence_length"], 
                        padding="post", 
                        truncating="post"
                    )

                    # 3. Model Inference
                    prediction = model.predict(padded, verbose=0)[0]
                    predicted_class_idx = int(np.argmax(prediction))
                    confidence_pct = float(prediction[predicted_class_idx]) * 100
                    raw_category = label_encoder.inverse_transform([predicted_class_idx])[0]
                    
                    info = CATEGORY_INFO.get(
                        raw_category, 
                        {
                            "department": raw_category.replace("_", " ").title() + " Department", 
                            "priority": "Medium", 
                            "icon": "🏛️", 
                            "code": raw_category[:3].upper()
                        }
                    )

                    category_display = raw_category.replace("_", " ").title()

                    # Display Spell Correction Notice if any
                    if has_spelling_diff:
                        st.info(f"✨ **Smart Typo Correction Applied:**\n*{corrected_text}*")

                    # Display Metric Cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f"""
                        <div class="res-card">
                            <div class="res-label">Category / Dept</div>
                            <div class="res-val">{info['icon']} {category_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m2:
                        st.markdown(f"""
                        <div class="res-card">
                            <div class="res-label">Assigned Unit</div>
                            <div class="res-val" style="font-size: 16px; color: #38bdf8;">{info['department']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m3:
                        badge_cls = f"badge-{info['priority'].lower()}"
                        st.markdown(f"""
                        <div class="res-card">
                            <div class="res-label">Priority Level</div>
                            <div style="margin-top: 6px;"><span class="{badge_cls}">{info['priority'].upper()}</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.progress(confidence_pct / 100.0, text=f"AI Model Confidence: {confidence_pct:.2f}%")

                    # Category Probabilities Chart
                    st.markdown("**Softmax Class Probability Distribution:**")
                    prob_df = pd.DataFrame({
                        "Category": [c.replace("_", " ").title() for c in label_encoder.classes_],
                        "Probability (%)": [round(float(p) * 100, 2) for p in prediction]
                    }).sort_values(by="Probability (%)", ascending=True)
                    
                    st.bar_chart(prob_df, x="Category", y="Probability (%)", color="#38bdf8", height=200)

                    # Generate Ticket Receipt Card
                    ticket_number = f"MPR-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
                    
                    ticket_json = {
                        "ticket_id": ticket_number,
                        "timestamp": timestamp_str,
                        "complaint_text": complaint_input,
                        "processed_text": corrected_text,
                        "category": category_display,
                        "department": info["department"],
                        "priority": info["priority"],
                        "confidence": f"{confidence_pct:.2f}%",
                        "status": "DISPATCHED_TO_FIELD_UNIT"
                    }

                    st.markdown(f"""
                    <div class="ticket-box">
                        <div class="ticket-header">
                            <div class="ticket-id">🎫 Ticket ID: {ticket_number}</div>
                            <div style="color: #10b981; font-size: 13px; font-weight: bold;">STATUS: DISPATCHED</div>
                        </div>
                        <div style="font-size: 13px; color: #9ca3af; margin-bottom: 8px;">
                            <b>Assigned Unit:</b> {info['department']} | <b>Code:</b> {info['code']}
                        </div>
                        <div style="font-size: 13px; color: #9ca3af;">
                            <b>Created At:</b> {timestamp_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Official Dispatch Ticket (JSON)",
                        data=json.dumps(ticket_json, indent=2),
                        file_name=f"{ticket_number}.json",
                        mime="application/json",
                        use_container_width=True
                    )

            elif classify_clicked and not complaint_input.strip():
                st.warning("⚠️ Please enter a complaint description before classifying.")
            else:
                st.info("👈 Enter a complaint on the left or select a preset to analyze classification & ticket routing.")

    # ==========================================
    # TAB 2: WORKLOAD ANALYTICS
    # ==========================================
    with tab_analytics:
        st.subheader("📊 Dataset & Department Workload Analytics")
        
        df = load_dataset()
        if df is not None:
            col_target = "department" if "department" in df.columns else "category"
            col_text = "text" if "text" in df.columns else "complaint_text"
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Dataset Records", f"{len(df):,}")
            with col2:
                st.metric("Departments / Categories", df[col_target].nunique())
            with col3:
                st.metric("Languages Supported", "Hinglish + English")
            with col4:
                st.metric("Model Architecture", "1D CNN Multi-Scale")

            st.divider()

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Complaints per Department")
                cat_counts = df[col_target].value_counts().reset_index()
                cat_counts.columns = ["Department", "Count"]
                st.bar_chart(cat_counts, x="Department", y="Count", color="#60a5fa")

            with c2:
                st.markdown("#### Intent Distribution" if "intent" in df.columns else "#### Department Workload")
                if "intent" in df.columns:
                    intent_counts = df["intent"].value_counts().reset_index()
                    intent_counts.columns = ["Intent", "Count"]
                    st.bar_chart(intent_counts, x="Intent", y="Count", color="#34d399")
                else:
                    dept_counts = df[col_target].value_counts().reset_index()
                    dept_counts.columns = ["Department", "Count"]
                    st.bar_chart(dept_counts, x="Department", y="Count", color="#34d399")

            st.divider()
            st.markdown("#### Sample Dataset Explorer")
            st.dataframe(df[[col_text, col_target]].head(25), use_container_width=True)
        else:
            st.error("Dataset not found in `data/`.")

    # ==========================================
    # TAB 3: ARCHITECTURE & SPECS
    # ==========================================
    with tab_arch:
        st.subheader("🛠️ Deep Learning Architecture & NLP Pipeline")
        
        if os.path.exists(ARCH_IMAGE_PATH):
            st.image(ARCH_IMAGE_PATH, caption="Municipal Complaint Classifier System Architecture", use_container_width=True)

        st.markdown("""
        ### Technical Model Specifications
        - **Model Architecture**: 1D Convolutional Neural Network (CNN)
        - **Embedding Layer**: 64-dimensional learned word embedding
        - **Regularization**: `SpatialDropout1D` (rate=0.4), L2 Regularization (0.01), Batch Normalization, Dense Dropout (0.6)
        - **Parallel Conv1D Filters**:
          - Filter sizes: `[2, 3, 4]` (capturing bi-grams, tri-grams, and 4-grams)
          - Number of filters per size: `64`
        - **Pooling**: `GlobalMaxPooling1D` across parallel filter channels
        - **Dataset Scale**: **100,000 Multi-Lingual Complaints** (`complaints_labeled.csv`)
        
        ### Preprocessing Pipeline
        1. **Character Run Reduction**: Reduces 3+ consecutive duplicate characters (e.g. `paaaani` → `paani`).
        2. **Hinglish Vocabulary Protection**: Preserves transliterated keywords (`paani`, `ghar`, `bijli`, `sadak`, `naali`) and enforces max edit distance of 1.
        3. **Tokenization & Padding**: Text sequences padded/truncated to `max_sequence_length = 60`.
        """)


if __name__ == "__main__":
    main()

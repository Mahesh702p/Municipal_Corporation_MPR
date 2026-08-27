# Municipal Complaint Classifier & Smart Dispatch Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red.svg)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-92.54%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent, Deep Learning-powered municipal grievance classification and dispatch engine built using a **1D Convolutional Neural Network (CNN)** and **Natural Language Processing (NLP)** pipeline. The application automatically categorizes unstructured citizen complaints (e.g., Water Leakages, Power Outages, Potholes, Road Damage, Garbage Overflow) into relevant municipal departments and assigns priority levels with downloadable JSON dispatch tickets.

---

## 🏛️ System Architecture

![Architecture Diagram](architecture_diagram.png)

### High-Level End-to-End Workflow

1. **Input Interface & Spell Correction**:
   - **Repeated Character Reduction**: Converts runs of 3+ repeated characters (e.g., `gaaarbbbage` → `garbage`, `heeeelp` → `help`).
   - **Contextual Typo Correction**: Utilizes `pyspellchecker` to map misspelled words to standard municipal lexicon.
2. **NLP Tokenization & Padding**:
   - Cleans non-alphanumeric noise, normalizes text, tokenizes into word sequences, and applies sequence padding to a fixed vector length ($N = 60$).
3. **1D CNN Model Architecture**:
   - **Embedding Layer**: 64-dimensional learned dense word representation.
   - **Regularization**: `SpatialDropout1D` (0.4) and L2 Regularization (0.01) to prevent over-fitting.
   - **Parallel Conv1D Extractors**: Three parallel 1D convolutional layers with kernel sizes of `2`, `3`, and `4` (capturing bi-grams, tri-grams, and 4-grams) with 64 filters each, Batch Normalization, and `GlobalMaxPooling1D`.
   - **Dense Classification Head**: Concatenated feature representations routed through dense layers (128 → 64 units) with Dropout (0.6) to a 6-class Softmax layer.
4. **Automated Department & Priority Dispatch**:
   - Categorizes grievance into one of 6 classes, assigns department routing, flags emergency priority, and generates an official downloadable JSON ticket.

---

## 📊 Classification Categories & Department Routing

| Category | Assigned Department | Priority Level | Emergency Response |
| :--- | :--- | :--- | :--- |
| **Water Leakage** | Water Supply Department | 🔴 High | Immediate Field Crew Alert |
| **Water Shortage** | Water Supply Department | 🔴 High | Tanker Dispatch Pipeline |
| **Power Cut** | Electricity Board | 🔴 High | Grid Inspection Unit |
| **Pothole** | Roads & Infrastructure | 🟡 Medium | Road Patch Work Ticket |
| **Road Damage** | Roads & Infrastructure | 🟡 Medium | Maintenance Inspection |
| **Garbage Overflow** | Sanitation & Solid Waste | 🟢 Low | Routine Sanitation Truck |

---

## 🎯 Model Performance & Metrics

Retrained and evaluated on **1,943 annotated citizen grievance records**:

- **Overall Test Accuracy**: **92.54%**
- **Classification Performance Breakdown**:

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Garbage** | 0.90 | 0.98 | 0.94 | 65 |
| **Leakage** | 0.95 | 0.90 | 0.93 | 83 |
| **Pothole** | 0.90 | 0.97 | 0.93 | 58 |
| **Power Cut** | 0.97 | 0.94 | 0.95 | 67 |
| **Road Damage** | 0.94 | 0.80 | 0.86 | 59 |
| **Shortage** | 0.89 | 0.96 | 0.92 | 57 |
| **Macro Average** | **0.93** | **0.93** | **0.92** | **389** |

---

## 💻 Quickstart & Local Setup

### 1. Clone Repository
```bash
git clone https://github.com/Mahesh702p/Municipal_Corporation_MPR.git
cd Municipal_Corporation_MPR
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv

# Linux / MacOS
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Application
```bash
streamlit run src/app.py
```
The Web UI will be accessible locally at `http://localhost:8501`.

---

## 🚀 450,000 Sample Multi-Task Deep Learning Pipeline (TextCNN + BiLSTM + Attention)

To scale to city-wide municipal deployments, the engine features an upgraded **Multi-Task Architecture** trained on **450,000 complaint records** (100,000 labeled + 350,000 robust noisy Hinglish/ALL-CAPS records) on GPU across 20 epochs.

### Multi-Task Model Features:
- **Shared Representation**: Embedding Layer + Parallel 1D CNNs (bi/tri/4-grams) + Bidirectional LSTM + Custom Attention Mechanism.
- **Dual Output Heads**:
  1. **Department Classifier**: 9 Municipal Departments (*Health, Disaster, Revenue, Solid Waste, Water, Electricity, Sewerage, Parks, Roads*).
  2. **Intent Detector**: 5 Intent Types (*Emergency, Complaint, Query, Service Request, Status Check*).

### 📊 Multi-Task Model Performance (90,000 Unseen Test Set):
- **Intent Detection Accuracy**: **100.00%** ($F1 = 1.00$ across all 5 intent classes)
- **Department Routing Accuracy**: **82.87%** ($F1$ Scores: Health 0.93, Disaster 0.92, Revenue 0.84, Solid Waste 0.86, Water 0.85, Electricity 0.86)

### 💻 Running Multi-Task Predictions
```bash
# Predict on single sentence (English / Hinglish)
python3 src/predict_multitask.py "paani nahi aa raha hai subah se pipe phut gaya hai"

# Output:
# 🏛️ Department : WATER_SUPPLY (100.00%)
# 🎯 Intent     : COMPLAINT (100.00%)

# Interactive mode prompt:
python3 src/predict_multitask.py
```

---

## 📂 Project Structure

```
Municipal_Corporation_MPR/
├── data/
│   ├── municipal_complaints_clean.csv   # Prototype Dataset (1,943 records)
│   ├── complaints_labeled.csv           # Clean Dataset (100,000 records)
│   └── complaints_robust.csv            # Robust Noisy Dataset (350,000 records)
├── models/
│   ├── cnn_category_model.keras        # Single-task Keras 1D CNN model
│   └── cnn_tokenizer.pkl               # Single-task Tokenizer
├── Municipal_Multitask_Results/         # 450k Colab GPU Model Artifacts & Metrics
│   ├── multitask_model.keras           # Trained Dual-Head Multi-Task Model
│   ├── multitask_tokenizer.pkl         # Subword Tokenizer
│   ├── dept_encoder.pkl / intent_encoder.pkl
│   ├── summary.json                    # Full evaluation metrics
│   ├── training_history.png            # Training curves
│   └── confusion_matrices.png          # Heatmap plots
├── src/
│   ├── app.py                          # Streamlit Web Application & Dashboard
│   ├── predict.py                      # Single-Task Prediction CLI
│   ├── predict_multitask.py            # Multi-Task 450k Model Prediction Engine
│   └── train_multitask.py              # Multi-Task Training Script
├── requirements.txt                    # Project Python Dependencies
└── README.md                           # Documentation
```

---

## 🤝 Contributing & License

Contributions are welcome! Please feel free to open an Issue or submit a Pull Request.

This project is open-source under the [MIT License](LICENSE).


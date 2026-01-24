# Municipal Complaint Classifier

A Streamlit-based web application that uses Deep Learning (CNN) to automatically categorize municipal complaints (e.g., "Pothole", "No Water", "Garbage") and assign them to the correct department with a priority level.

![App Screenshot](https://raw.githubusercontent.com/YOUR_USERNAME/REPO_NAME/main/screenshots/app_preview.png)
*(Note: Replace with actual screenshot link after upload)*

## 🚀 Features

*   **Automatic Classification**: Uses a trained CNN (Convolutional Neural Network) to classify text into 6 categories (Leakage, Shortage, Power Cut, Pothole, Road Damage, Garbage).
*   **Smart Spell Check**: Includes a custom NLP pipeline to handle severe typos (e.g., `gaaarbbbage` -> `garbage`).
*   **Priority Assignment**: Automatically flags high-priority issues (e.g., Water Leakage, Power Cuts).
*   **User-Friendly Interface**: Simple web UI built with Streamlit.

## 🛠️ Architecture

### High-Level Overview
The application follows a standard Model-View-Controller (MVC) pattern adapted for Streamlit:

1.  **View (Frontend)**: Streamlit UI handles user input and visualizes results (cards, progress bars).
2.  **Controller (Logic)**: `app.py` manages the data flow, loads models, and calls the preprocessing pipeline.
3.  **Model (Backend)**: TensorFlow/Keras model performs the actual inference.

### NLP Pipeline
When you enter a complaint, the following happens:

1.  **Spell Correction**:
    *   **Character Reduction**: Reduces repeated characters (e.g., `heeeelp` -> `heelp`).
    *   **Correction**: Uses `pyspellchecker` to map the result to the nearest valid word.
2.  **Normalization**: Converts to lowercase, removes special characters & extra spaces.
3.  **Tokenization**: Converts text into a sequence of integers using the pickle-saved tokenizer.
4.  **Padding**: Ensures the sequence is exactly `max_len` long.
5.  **Inference**: The CNN model predicts the probability for each category.

### Model Details
*   **Type**: 1D Convolutional Neural Network (CNN)
*   **Embedding**: Learned embedding layer trained on municipal complaint data.
*   **Input**: Text sequences (padded).
*   **Output**: Softmax probability over 6 classes.

## 💻 How to Run Locally

### Prerequisites
*   Python 3.10+
*   `pip` (Python package manager)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/Municipal_Corporation_CNN.git
    cd Municipal_Corporation_CNN
    ```

2.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
```bash
streamlit run src/app.py
```
The app will open in your browser at `http://localhost:8501`.

## 🤝 Collaboration

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to set up your dev environment and submit Pull Requests.

## 📂 Project Structure
```
Municipal_Corporation_CNN/
├── models/                 # Trained model files (.keras, .pkl)
├── src/
│   └── app.py              # Main application script
├── data/                   # (Optional) Raw data csvs
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 📜 License
[MIT](LICENSE)

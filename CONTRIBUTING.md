# Contributing to Municipal Complaint Classifier

Hey friends! 👋 Thanks for wanting to check out or improve this project.

## Setting Up for Development

1.  **Get the Code**: `git clone` this repo.
2.  **Environment**: Make sure you use a virtual environment (`venv`) so you don't mess up your global Python install.
    ```bash
    python -m venv venv
    source venv/bin/activate  # or .\venv\Scripts\activate on Windows
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Test
Before pushing any changes, please run the app locally and verify the following:
*   **Normal Input**: Try a clear sentence like "There is no water supply".
*   **Typo Input**: Try a messy sentence like "waaater not commming".
*   **Priority Check**: Ensure "Power Cut" shows up as High Priority (Red).

## Updating the Model
If you retrain the model (using `train.py` or a notebook):
1.  Save the new model as `cnn_category_model.keras` in the `models/` folder.
2.  **Crucial**: You MUST also save the updated `cnn_tokenizer.pkl` and `cnn_label_encoder.pkl`. If you don't, the new model won't understand the old tokenizer's numbers!

## Pull Requests
1.  Create a branch for your feature: `git checkout -b new-feature`
2.  Commit your changes: `git commit -m "Added cool feature"`
3.  Push to the branch: `git push origin new-feature`
4.  Open a Pull Request!

# Mental Health Sentiment Prediction (Deep Learning)

A Streamlit web app that classifies personal statements into mental‑health related sentiment categories using multiple deep learning architectures (LSTM, BiLSTM, CNN‑BiLSTM, and a Multi‑View model). The app downloads the trained models and tokenizer from Hugging Face at runtime and provides class probabilities for each prediction.

## Highlights

- **4 model choices**: LSTM, BiLSTM, CNN‑BiLSTM, Multi‑View
- **7 classes**: Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality disorder
- **Interactive UI** with probability chart
- **On‑demand model download** from Hugging Face

## Project Structure

- app.py — Streamlit application
- requirements.txt — Python dependencies
- *.keras — local model files (auto‑downloaded if missing)

## How It Works

1. The app downloads a **tokenizer** and selected **model weights** from Hugging Face (only if they don’t already exist locally).
2. User input is minimally preprocessed (lower‑cased; placeholders for additional preprocessing/stopword removal exist).
3. The text is tokenized and padded to a fixed sequence length.
4. The selected model predicts class probabilities and the app displays the top class plus a bar chart of all probabilities.

## Models & Artifacts

The app fetches models and the tokenizer from the following Hugging Face repository:

- Base URL: https://huggingface.co/shubhamprabhukhanolkar/mental-health-sentiment-models/resolve/main/

Models referenced in the app:

- lstm_sentiment_model.keras
- bilstm_sentiment_model.keras
- cnn_bilstm_mental_health_sentiment_model.keras
- new.keras (Multi‑View)
- tokenizer.pickle

## Setup

### 1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

The app will open in your browser. Paste a personal statement, select a model, and click **Predict**.

## Notes

- The app loads models with `compile=False` to reduce load time and avoid training‑time dependencies.
- `preprocess_text()` and `remove_stopwords_simple()` are intentionally minimal and can be extended to match your training pipeline.
- First launch may take longer due to model downloads.

## Example

Input:

> “I feel empty and hopeless lately.”

Output (example):

- Predicted sentiment: **Depression**
- Confidence: **0.84** (varies by model)

## Dependencies

- streamlit
- tensorflow
- numpy
- scikit‑learn
- pandas
- requests

## License

Add your license here.

## Acknowledgements

- Hugging Face for model hosting
- Streamlit for the interactive UI

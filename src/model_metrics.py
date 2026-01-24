# src/model_metrics.py
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = "data/news.csv"

def get_model_metrics():
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )

    results = {}

    # ===== TF-IDF Logistic Regression =====
    tfidf_model = joblib.load("models/tfidf_model.pkl")
    y_pred = tfidf_model.predict(X_test)
    results["TF-IDF + LR"] = accuracy_score(y_test, y_pred)

    # ===== Random Forest =====
    rf_model = joblib.load("models/rf_model.pkl")
    rf_vectorizer = joblib.load("models/rf_vectorizer.pkl")
    X_test_vec = rf_vectorizer.transform(X_test)
    y_pred = rf_model.predict(X_test_vec)
    results["Random Forest"] = accuracy_score(y_test, y_pred)

    # ===== XGBoost =====
    xgb_model = joblib.load("models/xgb_model.pkl")
    xgb_vectorizer = joblib.load("models/xgb_vectorizer.pkl")
    X_test_vec = xgb_vectorizer.transform(X_test)
    y_pred = xgb_model.predict(X_test_vec)
    results["XGBoost"] = accuracy_score(y_test, y_pred)

    # ===== LSTM =====
    lstm_model = load_model("models/lstm_model.h5")
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(X_train)

    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=200)
    y_pred = (lstm_model.predict(X_test_pad) > 0.5).astype(int).reshape(-1)
    results["LSTM"] = accuracy_score(y_test, y_pred)

    # ===== BERT =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer_bert = BertTokenizerFast.from_pretrained("models/bert_model")
    bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
    bert_model.to(device)
    bert_model.eval()

    inputs = tokenizer_bert(list(X_test), truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    results["BERT"] = accuracy_score(y_test, preds)

    return results
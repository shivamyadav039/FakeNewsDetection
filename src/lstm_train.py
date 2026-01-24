# src/lstm_train.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "data/news.csv"
MODEL_PATH = "models/lstm_model.h5"

MAX_WORDS = 20000
MAX_LEN = 200

def train_lstm_model():
    # =========================
    # 1. Load Dataset
    # =========================
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["label"] = df["label"].astype(int)

    X = df["text"].values
    y = df["label"].values

    # =========================
    # 2. Train-Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # =========================
    # 3. Tokenization
    # =========================
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post")

    # =========================
    # 4. Build LSTM Model
    # =========================
    model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # =========================
    # 5. Training
    # =========================
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=5,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # =========================
    # 6. Save Model
    # =========================
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)

    print(f"✅ LSTM model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_lstm_model()
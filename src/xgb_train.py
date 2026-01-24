import os
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "data/news.csv"
MODEL_PATH = "models/xgb_model.pkl"

def train_xgboost():
    print("🚀 Training XGBoost (Bias Fixed)...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["label"] = df["label"].astype(int)

    X = df["text"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Calculate class imbalance ratio 🔥
    counter = Counter(y_train)
    ratio = counter[0] / counter[1]
    print("⚖️ Class imbalance ratio:", ratio)

    # TF-IDF Vectorizer (Improved)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=6000,
        ngram_range=(1, 2),
        min_df=3
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # XGBoost with class balancing 🔥
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,  # ✅ FIX BIAS
        eval_metric="logloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ XGBoost Accuracy: {acc:.4f}")
    print(f"✅ XGBoost F1-score: {f1:.4f}")

    # Save model & vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, "models/xgb_vectorizer.pkl")

    print("💾 XGBoost model saved successfully!")

if __name__ == "__main__":
    train_xgboost()
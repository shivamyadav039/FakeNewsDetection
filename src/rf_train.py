import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

DATA_PATH = "data/news.csv"
MODEL_PATH = "models/rf_model.pkl"

def train_random_forest():
    print("🚀 Training Random Forest (Bias Fixed)...")

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

    # TF-IDF Vectorizer (Improved)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=6000,   # 🔥 more features
        ngram_range=(1, 2),  # 🔥 bigrams
        min_df=3
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Random Forest with balanced classes 🔥
    base_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"  # ✅ FIX BIAS
    )

    # Probability Calibration 🔥
    model = CalibratedClassifierCV(base_model, method="sigmoid")
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ Random Forest Accuracy: {acc:.4f}")
    print(f"✅ Random Forest F1-score: {f1:.4f}")

    # Save model & vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, "models/rf_vectorizer.pkl")

    print("💾 Random Forest model saved successfully!")

if __name__ == "__main__":
    train_random_forest()
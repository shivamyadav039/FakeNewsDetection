import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("🚀 Starting TF-IDF model training...")

DATA_PATH = "data/news.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded. Rows:", len(df))

# Combine title + text
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["label"] = df["label"].astype(int)

X = df["text"]
y = df["label"]

# Train-test split
print("🔀 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorizer
print("🧠 Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model
print("📊 Evaluating model...")
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print("✅ TF-IDF Model Accuracy:", round(acc, 4))
print(classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, os.path.join(MODEL_DIR, "tfidf_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("💾 TF-IDF model saved in models/")
print("🎉 Training completed successfully!")
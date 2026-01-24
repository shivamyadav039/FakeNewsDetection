# src/dataset_builder.py
import pandas as pd
import os
import re

FAKE_PATH = "data/Fake.csv"
TRUE_PATH = "data/True.csv"
OUTPUT_PATH = "data/news.csv"

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_dataset():
    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    fake_df = fake_df[["title", "text"]]
    true_df = true_df[["title", "text"]]

    fake_df["label"] = 0  # Fake
    true_df["label"] = 1  # Real

    for col in ["title", "text"]:
        fake_df[col] = fake_df[col].apply(clean_text)
        true_df[col] = true_df[col].apply(clean_text)

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("✅ Dataset created:", OUTPUT_PATH)
    print(df["label"].value_counts())

if __name__ == "__main__":
    build_dataset()
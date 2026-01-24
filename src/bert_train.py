import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

DATA_PATH = "data/news.csv"
MODEL_DIR = "models/bert_model"

# Load dataset
df = pd.read_csv(DATA_PATH)
df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["label"] = df["label"].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.1,   # smaller validation for speed
    stratify=df["label"],
    random_state=42
)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(texts):
    return tokenizer(texts, truncation=True, padding=True, max_length=128)

train_enc = tokenize(train_texts)
val_enc = tokenize(val_texts)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_enc, train_labels)
val_dataset = NewsDataset(val_enc, val_labels)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,  # keep 1 epoch for Mac
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=50,
    report_to="none",
    fp16=False  # Mac CPU/MPS safe
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

os.makedirs(MODEL_DIR, exist_ok=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("✅ BERT model trained and saved successfully!")
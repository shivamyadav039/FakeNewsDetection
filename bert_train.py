# bert_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch, os

# Load and prepare data
df = pd.read_csv("news.csv")
df.dropna(subset=['title', 'text', 'label'], inplace=True)
df['combined'] = df['title'] + " " + df['text']
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['combined'], df['label'], test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_enc = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_enc = tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_enc, list(train_labels))
val_dataset = NewsDataset(val_enc, list(val_labels))

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./bert_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save model & tokenizer
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")
print("✅ BERT model and tokenizer saved in ./bert_model")

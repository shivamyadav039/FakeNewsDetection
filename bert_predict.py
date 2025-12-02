# bert_predict.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "./bert_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_news(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
        label = "REAL" if predicted_class == 1 else "FAKE"
        return label, round(confidence, 2)

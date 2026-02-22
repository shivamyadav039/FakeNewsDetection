# src/bert_predict.py
import os
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_PATH = "models/bert_model"

# =========================
# 1. Device Setup (CPU / GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. Load Model & Tokenizer (Only Once, with error handling)
# =========================
tokenizer = None
model = None

if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH):
    try:
        tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print(f"✅ BERT model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load BERT model: {e}")
        tokenizer = None
        model = None
else:
    print(f"⚠️ BERT model not found at {MODEL_PATH}. Run src/bert_train.py to train it.")

# =========================
# 3. Prediction Function
# =========================
def predict_news(text):
    # Check if model is loaded
    if tokenizer is None or model is None:
        raise RuntimeError(
            "BERT model not available. "
            "Please train the model first by running: python src/bert_train.py"
        )
    
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    results = []
    for p in probs:
        label_id = int(p.argmax())
        label = "REAL" if label_id == 1 else "FAKE"
        confidence = float(p[label_id])
        results.append({
            "label": label,
            "confidence": round(confidence, 4),
            "fake_prob": round(float(p[0]), 4),
            "real_prob": round(float(p[1]), 4)
        })

    return results[0] if len(results) == 1 else results
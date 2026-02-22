import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from src.bert_predict import predict_news
from src.model_loader import load_all_models

# =========================
# Load Models Once (Using Centralized Loader)
# =========================
_models = load_all_models()

tfidf_model = _models.get("TF-IDF")
tfidf_vectorizer = _models.get("TFIDF_Vectorizer")

rf_model = _models.get("Random Forest")
rf_vectorizer = _models.get("RF_Vectorizer")

xgb_model = _models.get("XGBoost")
xgb_vectorizer = _models.get("XGB_Vectorizer")

# =========================
# Ensemble Prediction Function
# =========================
def ensemble_predict(text):
    probs = {}
    weights = {}

    # =========================
    # 1️⃣ BERT (Always Available)
    # =========================
    try:
        bert_result = predict_news(text)
        probs["bert"] = bert_result["real_prob"]
        weights["bert"] = 0.4  # highest weight
    except Exception as e:
        print("⚠️ BERT failed:", e)

    # =========================
    # 2️⃣ TF-IDF Model
    # =========================
    if tfidf_model is not None and tfidf_vectorizer is not None:
        try:
            vec = tfidf_vectorizer.transform([text])
            prob = tfidf_model.predict_proba(vec)[0][1]
            probs["tfidf"] = prob
            weights["tfidf"] = 0.2
        except Exception as e:
            print("⚠️ TF-IDF failed:", e)

    # =========================
    # 3️⃣ Random Forest Model
    # =========================
    if rf_model is not None and rf_vectorizer is not None:
        try:
            vec = rf_vectorizer.transform([text])
            prob = rf_model.predict_proba(vec)[0][1]
            probs["rf"] = prob
            weights["rf"] = 0.2
        except Exception as e:
            print("⚠️ Random Forest failed:", e)

    # =========================
    # 4️⃣ XGBoost Model (Safe)
    # =========================
    if xgb_model is not None and xgb_vectorizer is not None:
        try:
            vec = xgb_vectorizer.transform([text])
            prob = xgb_model.predict_proba(vec)[0][1]
            probs["xgb"] = prob
            weights["xgb"] = 0.2
        except Exception as e:
            print("⚠️ XGBoost failed:", e)

    # =========================
    # If No Models Loaded (Fallback)
    # =========================
    if len(probs) == 0:
        return {
            "label": "UNKNOWN",
            "confidence": 0.5,
            "real_prob": 0.5,
            "fake_prob": 0.5,
            "individual_probs": {}
        }

    # =========================
    # Normalize Weights Dynamically
    # =========================
    total_weight = sum(weights.values())
    for k in weights:
        weights[k] = weights[k] / total_weight

    # =========================
    # Weighted Ensemble Probability
    # =========================
    final_real_prob = sum(probs[m] * weights[m] for m in probs)
    final_fake_prob = 1 - final_real_prob

    label = "REAL" if final_real_prob >= 0.5 else "FAKE"
    confidence = max(final_real_prob, final_fake_prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "real_prob": round(final_real_prob, 4),
        "fake_prob": round(final_fake_prob, 4),
        "individual_probs": {k: round(v, 4) for k, v in probs.items()},
        "active_models": list(probs.keys())
    }
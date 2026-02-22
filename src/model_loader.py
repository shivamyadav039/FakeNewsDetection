# src/model_loader.py
import joblib
import os

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

MODEL_DIR = "models"

def safe_load(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def load_all_models():
    """Load models and vectorizers used by the app.
    Returns a dict with keys:
      - TF-IDF, TFIDF_Vectorizer
      - Random Forest, RF_Vectorizer
      - XGBoost, XGB_Vectorizer
      - LSTM
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {}

    # TF-IDF
    models["TF-IDF"] = safe_load(os.path.join(MODEL_DIR, "tfidf_model.pkl"))
    models["TFIDF_Vectorizer"] = safe_load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    # Random Forest
    models["Random Forest"] = safe_load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    models["RF_Vectorizer"] = safe_load(os.path.join(MODEL_DIR, "rf_vectorizer.pkl"))

    # XGBoost
    models["XGBoost"] = safe_load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    models["XGB_Vectorizer"] = safe_load(os.path.join(MODEL_DIR, "xgb_vectorizer.pkl"))

    # LSTM (TensorFlow)
    if TENSORFLOW_AVAILABLE:
        try:
            models["LSTM"] = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
        except Exception:
            models["LSTM"] = None
    else:
        models["LSTM"] = None

    return models

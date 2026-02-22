import streamlit as st
import joblib
import pandas as pd

# =========================
# SAFE IMPORTS (REPLACED)
# =========================
try:
    from src.bert_predict import predict_news
    BERT_AVAILABLE = True
except Exception as e:
    BERT_AVAILABLE = False
    print(f"⚠️ BERT not available: {e}")

from src.model_loader import load_all_models

try:
    from src.ensemble_predict import ensemble_predict
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# =========================
# CUSTOM CSS (UI DESIGN 🔥)
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
    color: #2E86C1;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #555;
    text-align: center;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background-color: #f8f9fa;
    border: 1px solid #ddd;
    margin-top: 10px;
}
.result-real {
    color: green;
    font-size: 22px;
    font-weight: bold;
}
.result-fake {
    color: red;
    font-size: 22px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("<div class='main-title'>📰 Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-powered system using Machine Learning & BERT</div>", unsafe_allow_html=True)

st.write("")

# =========================
# SIDEBAR DASHBOARD
# =========================
with st.sidebar:
    st.header("📊 Project Dashboard")
    st.write("**Models Used:**")
    st.write("✔ BERT (Transformer)")
    st.write("✔ TF-IDF + Logistic Regression")
    st.write("✔ Random Forest")
    st.write("✔ XGBoost")
    st.write("✔ LSTM (optional)")
    st.markdown("---")
    st.write("👨‍💻 Developer: **Shivam Yadav**")
    st.write("🎓 Project: Fake News Detection")

# =========================
# LOAD MODELS SAFELY
# =========================
@st.cache_resource
def load_models():
    # delegate to centralized loader (loads models + vectorizers + LSTM if available)
    return load_all_models()

models = load_models()

# =========================
# MODEL SELECTOR (REPLACED)
# =========================
st.markdown("### 🤖 Select AI Model")

model_options = []

# Add models based on availability
if BERT_AVAILABLE and models.get("BERT") is not None:
    model_options.append("BERT")

if ENSEMBLE_AVAILABLE:
    model_options.append("Ensemble")
    
if models.get("TF-IDF") is not None:
    model_options.append("TF-IDF")
    
if models.get("Random Forest") is not None:
    model_options.append("Random Forest")
    
if models.get("XGBoost") is not None:
    model_options.append("XGBoost")
    
if TENSORFLOW_AVAILABLE and models.get("LSTM") is not None:
    model_options.append("LSTM")

# Default to available models or show error
if not model_options:
    st.error("❌ No models available. Please train models first.")
    st.stop()

model_choice = st.selectbox(
    "",
    model_options
)

# =========================
# INPUT AREA
# =========================
st.markdown("### ✍️ Enter News Article or Headline")

news_input = st.text_area(
    "",
    height=160,
    placeholder="Paste your news text here..."
)

# =========================
# PREDICTION BUTTON
# =========================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Analyze News", use_container_width=True)

# =========================
# PREDICTION LOGIC (REPLACED)
# =========================
if predict_btn:
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("🤖 AI is analyzing the news..."):

            label = "UNKNOWN"
            confidence = 0.0
            fake_prob = 0.5
            real_prob = 0.5
            individual_probs = {}
            active_models = []

            # ===== BERT =====
            if model_choice == "BERT":
                if not BERT_AVAILABLE:
                    st.error("⚠️ BERT model not available. Please train it first: `python src/bert_train.py`")
                    st.stop()
                try:
                    result = predict_news(news_input)
                    if isinstance(result, list):
                        result = result[0]
                    label = result.get("label", "UNKNOWN")
                    confidence = float(result.get("confidence", 0.0))
                    fake_prob = float(result.get("fake_prob", 1 - confidence))
                    real_prob = float(result.get("real_prob", confidence))
                except Exception as e:
                    st.error(f"❌ BERT prediction failed: {e}")
                    st.info("💡 Train BERT model: `python src/bert_train.py`")

            # ===== Ensemble =====
            elif model_choice == "Ensemble":
                if not ENSEMBLE_AVAILABLE:
                    st.error("❌ Ensemble module not available.")
                    st.stop()
                try:
                    ens = ensemble_predict(news_input)
                    label = ens.get("label", "UNKNOWN")
                    confidence = float(ens.get("confidence", 0.0))
                    fake_prob = float(ens.get("fake_prob", 1 - confidence))
                    real_prob = float(ens.get("real_prob", confidence))
                    individual_probs = ens.get("individual_probs", {})
                    active_models = ens.get("active_models", [])
                except Exception as e:
                    st.error(f"❌ Ensemble prediction failed: {e}")

            # ===== TF-IDF =====
            elif model_choice == "TF-IDF":
                if models.get("TF-IDF") is None or models.get("TFIDF_Vectorizer") is None:
                    st.error("❌ TF-IDF model or vectorizer not found.")
                    st.stop()
                try:
                    vec = models["TFIDF_Vectorizer"].transform([news_input])
                    try:
                        prob = float(models["TF-IDF"].predict_proba(vec)[0][1])
                    except Exception:
                        pred = int(models["TF-IDF"].predict([news_input])[0])
                        prob = 1.0 if pred == 1 else 0.0
                    pred = 1 if prob >= 0.5 else 0
                    label = "REAL" if pred == 1 else "FAKE"
                    confidence = float(prob if label == "REAL" else 1 - prob)
                    fake_prob = 1 - confidence if label == "REAL" else confidence
                    real_prob = confidence if label == "REAL" else 1 - confidence
                except Exception as e:
                    st.error(f"❌ TF-IDF prediction failed: {e}")

            # ===== Random Forest =====
            elif model_choice == "Random Forest":
                if models.get("Random Forest") is None or models.get("RF_Vectorizer") is None:
                    st.error("❌ Random Forest model or vectorizer not found.")
                    st.stop()
                try:
                    vec = models["RF_Vectorizer"].transform([news_input])
                    try:
                        prob = float(models["Random Forest"].predict_proba(vec)[0][1])
                    except Exception:
                        pred = int(models["Random Forest"].predict(vec)[0])
                        prob = 1.0 if pred == 1 else 0.0
                    pred = 1 if prob >= 0.5 else 0
                    label = "REAL" if pred == 1 else "FAKE"
                    confidence = float(prob if label == "REAL" else 1 - prob)
                    fake_prob = 1 - confidence if label == "REAL" else confidence
                    real_prob = confidence if label == "REAL" else 1 - confidence
                except Exception as e:
                    st.error(f"❌ Random Forest prediction failed: {e}")

            # ===== XGBoost =====
            elif model_choice == "XGBoost":
                if models.get("XGBoost") is None or models.get("XGB_Vectorizer") is None:
                    st.error("❌ XGBoost model or vectorizer not found.")
                    st.stop()
                try:
                    vec = models["XGB_Vectorizer"].transform([news_input])
                    try:
                        prob = float(models["XGBoost"].predict_proba(vec)[0][1])
                    except Exception:
                        pred = int(models["XGBoost"].predict(vec)[0])
                        prob = 1.0 if pred == 1 else 0.0
                    pred = 1 if prob >= 0.5 else 0
                    label = "REAL" if pred == 1 else "FAKE"
                    confidence = float(prob if label == "REAL" else 1 - prob)
                    fake_prob = 1 - confidence if label == "REAL" else confidence
                    real_prob = confidence if label == "REAL" else 1 - confidence
                except Exception as e:
                    st.error(f"❌ XGBoost prediction failed: {e}")

            # ===== LSTM =====
            elif model_choice == "LSTM":
                if not TENSORFLOW_AVAILABLE or models.get("LSTM") is None:
                    st.error("⚠️ LSTM model not available (TensorFlow not installed).")
                    st.stop()
                try:
                    tokenizer = Tokenizer(num_words=20000)
                    tokenizer.fit_on_texts([news_input])
                    seq = tokenizer.texts_to_sequences([news_input])
                    pad = pad_sequences(seq, maxlen=200)
                    prob = float(models["LSTM"].predict(pad)[0][0])
                    pred = 1 if prob > 0.5 else 0
                    label = "REAL" if pred == 1 else "FAKE"
                    confidence = float(prob if pred == 1 else 1 - prob)
                    fake_prob = 1 - confidence if label == "REAL" else confidence
                    real_prob = confidence if label == "REAL" else 1 - confidence
                except Exception as e:
                    st.error(f"❌ LSTM prediction failed: {e}")

        # =========================
        # RESULT UI (CARD STYLE 🔥)
        # =========================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(f"### 📊 Prediction Result ({model_choice})")

        if label == "REAL":
            st.markdown("<div class='result-real'>✅ REAL NEWS</div>", unsafe_allow_html=True)
        elif label == "FAKE":
            st.markdown("<div class='result-fake'>🚨 FAKE NEWS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-fake'>❓ UNKNOWN</div>", unsafe_allow_html=True)

        st.write(f"**Confidence Score:** {confidence:.4f}")

        # Progress Bar
        try:
            st.progress(min(max(confidence, 0.0), 1.0))
        except Exception:
            pass

        # Probability Chart
        chart_data = pd.DataFrame({
            "Probability": [round(fake_prob, 4), round(real_prob, 4)]
        }, index=["Fake", "Real"])

        st.bar_chart(chart_data)

        # If ensemble, show individual model probabilities
        if model_choice == "Ensemble" and individual_probs:
            with st.expander("🔎 Individual Model Probabilities"):
                st.json(individual_probs)
                if active_models:
                    st.write(f"**Active models used:** {', '.join(active_models)}")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>🚀 Built with Machine Learning & BERT | Fake News Detection Project</div>", unsafe_allow_html=True)

# Add model status to sidebar (shows which models loaded)
with st.sidebar:
    st.markdown("---")
    st.subheader("Model Status")
    try:
        loaded = [k for k, v in models.items() if v is not None and not k.endswith('Vectorizer')]
        missing = [k for k, v in models.items() if v is None and not k.endswith('Vectorizer')]
        st.write("**Loaded:**", ", ".join(loaded) if loaded else "None")
        st.write("**Missing:**", ", ".join(missing) if missing else "None")
    except Exception:
        pass
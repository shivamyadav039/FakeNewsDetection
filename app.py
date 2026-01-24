
import streamlit as st
import joblib
import pandas as pd

# =========================
# SAFE IMPORTS
# =========================
from src.bert_predict import predict_news

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
    models = {}

    try:
        models["TF-IDF"] = joblib.load("models/tfidf_model.pkl")
    except:
        models["TF-IDF"] = None

    try:
        models["Random Forest"] = joblib.load("models/rf_model.pkl")
        models["RF_Vectorizer"] = joblib.load("models/rf_vectorizer.pkl")
    except:
        models["Random Forest"] = None

    try:
        models["XGBoost"] = joblib.load("models/xgb_model.pkl")
        models["XGB_Vectorizer"] = joblib.load("models/xgb_vectorizer.pkl")
    except:
        models["XGBoost"] = None

    if TENSORFLOW_AVAILABLE:
        try:
            models["LSTM"] = load_model("models/lstm_model.h5")
        except:
            models["LSTM"] = None
    else:
        models["LSTM"] = None

    return models

models = load_models()

# =========================
# MODEL SELECTOR
# =========================
st.markdown("### 🤖 Select AI Model")

model_choice = st.selectbox(
    "",
    ["BERT", "TF-IDF", "Random Forest", "XGBoost", "LSTM"]
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
# PREDICTION LOGIC
# =========================
if predict_btn:
    if not news_input.strip():
        st.warning("⚠️ Please enter some news text.")
    else:
        with st.spinner("🤖 AI is analyzing the news..."):

            # ===== BERT =====
            if model_choice == "BERT":
                result = predict_news(news_input)
                label = result["label"]
                confidence = result["confidence"]
                fake_prob = result.get("fake_prob", 1-confidence)
                real_prob = result.get("real_prob", confidence)

            # ===== TF-IDF =====
            elif model_choice == "TF-IDF":
                if models["TF-IDF"] is None:
                    st.error("❌ TF-IDF model not found.")
                    st.stop()
                pred = models["TF-IDF"].predict([news_input])[0]
                label = "REAL" if pred == 1 else "FAKE"
                confidence = 1.0
                fake_prob = 1-confidence if label=="REAL" else confidence
                real_prob = confidence if label=="REAL" else 1-confidence

            # ===== Random Forest =====
            elif model_choice == "Random Forest":
                if models["Random Forest"] is None:
                    st.error("❌ Random Forest model not found.")
                    st.stop()
                vec = models["RF_Vectorizer"].transform([news_input])
                pred = models["Random Forest"].predict(vec)[0]
                label = "REAL" if pred == 1 else "FAKE"
                confidence = 1.0
                fake_prob = 1-confidence if label=="REAL" else confidence
                real_prob = confidence if label=="REAL" else 1-confidence

            # ===== XGBoost =====
            elif model_choice == "XGBoost":
                if models["XGBoost"] is None:
                    st.error("❌ XGBoost model not found.")
                    st.stop()
                vec = models["XGB_Vectorizer"].transform([news_input])
                pred = models["XGBoost"].predict(vec)[0]
                label = "REAL" if pred == 1 else "FAKE"
                confidence = 1.0
                fake_prob = 1-confidence if label=="REAL" else confidence
                real_prob = confidence if label=="REAL" else 1-confidence

            # ===== LSTM =====
            elif model_choice == "LSTM":
                if not TENSORFLOW_AVAILABLE or models["LSTM"] is None:
                    st.error("⚠️ LSTM model not available (TensorFlow not installed).")
                    st.stop()
                tokenizer = Tokenizer(num_words=20000)
                tokenizer.fit_on_texts([news_input])
                seq = tokenizer.texts_to_sequences([news_input])
                pad = pad_sequences(seq, maxlen=200)
                prob = models["LSTM"].predict(pad)[0][0]
                pred = 1 if prob > 0.5 else 0
                label = "REAL" if pred == 1 else "FAKE"
                confidence = float(prob if pred == 1 else 1 - prob)
                fake_prob = 1-confidence if label=="REAL" else confidence
                real_prob = confidence if label=="REAL" else 1-confidence

        # =========================
        # RESULT UI (CARD STYLE 🔥)
        # =========================
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(f"### 📊 Prediction Result ({model_choice})")

        if label == "REAL":
            st.markdown("<div class='result-real'>✅ REAL NEWS</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result-fake'>🚨 FAKE NEWS</div>", unsafe_allow_html=True)

        st.write(f"**Confidence Score:** {confidence:.4f}")

        # Progress Bar
        st.progress(confidence)

        # Probability Chart
        chart_data = pd.DataFrame({
            "Probability": [fake_prob, real_prob]
        }, index=["Fake", "Real"])

        st.bar_chart(chart_data)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<div class='footer'>🚀 Built with Machine Learning & BERT | Fake News Detection Project</div>", unsafe_allow_html=True)
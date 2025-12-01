import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.title("📰 Fake News Detection")

article = st.text_area("Paste a news article here:")

if st.button("Check if it's Fake or Real"):
    if not article.strip():
        st.warning("Please enter a news article.")
    else:
        vector = vectorizer.transform([article])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("✅ This article is Real.")
        else:
            st.error("⚠️ This article is Fake.")


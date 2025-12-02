# app.py
import streamlit as st
from bert_predict import predict_news

st.title("📰 Fake News Detection System (BERT)")
st.write("Paste a news article or headline below 👇")

news_input = st.text_area("Enter News Text Here", height=200)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        label, confidence = predict_news(news_input)
        st.success(f"✅ This news is predicted to be: **{label}**")
        st.info(f"🧠 Prediction confidence: {confidence}")

Fake News Detection System
A machine learning–powered system to detect whether news content is real or fake. Built using Python, NLP techniques, and a logistic regression classifier.

📌 Overview
This project demonstrates how Natural Language Processing and classical machine learning models can be applied to classify news articles as either “Real” or “Fake”. It features a trained logistic regression model, real-time prediction using Streamlit, and support for saving/loading models.

✨ Features
✅ Preprocessing: Clean text using stopword removal, punctuation stripping, and lowercasing.

🧠 Model: Logistic Regression with TF-IDF vectorizer.

📦 Data: CSV-based dataset containing real and fake news samples.

💾 Persistence: Save and load model and vectorizer using joblib.

🌐 Web Interface: Streamlit-based UI for entering article text and viewing prediction.

📊 Evaluation: Accuracy, confusion matrix, and cross-validation during training.

🔁 Flowchart
plaintext
Copy
Edit
[Input News Article]
        ↓
[Preprocessing (Lowercase, Punctuation Removal, etc.)]
        ↓
[TF-IDF Vectorization]
        ↓
[Trained Logistic Regression Model]
        ↓
[Output: Fake or Real]
⚙️ Installation
🧰 Prerequisites
Python 3.x

pip (Python package manager)

📦 Required Python Libraries
pandas

numpy

scikit-learn

joblib

streamlit

Install all using:

bash
Copy
Edit
pip install -r requirements.txt
🛠️ Build Instructions (Manual Setup)
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/FakeNewsDetection.git
cd FakeNewsDetection
Create and activate a virtual environment (optional but recommended):

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# OR
venv\Scripts\activate      # On Windows
Run the app:

bash
Copy
Edit
streamlit run app.py
💡 Usage
When prompted, enter the text of a news article into the input box.

The model will output whether it believes the news is Fake or Real.

Behind the scenes:

The text is transformed using the saved TF-IDF vectorizer.

The logistic regression model makes a binary prediction.

🧠 Prediction Labels:

Label	Meaning
0	Fake News
1	Real News

📂 Code Structure
File	Description
app.py	Streamlit web interface
fake_news_detection.py	Training script (loads data, trains model, saves model/vectorizer)
data/fake_or_real_news.csv	Dataset used for training
models/model.pkl	Trained classifier
models/vectorizer.pkl	TF-IDF vectorizer
requirements.txt	Python dependencies
README.md	Project documentation

📊 Dataset
Dataset used from Kaggle:

📎 https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

Contains labeled real and fake news headlines and body text.

Dataset was preprocessed before training.

🔮 Future Enhancements
Add additional classifiers (e.g., SVM, BERT).

Improve preprocessing pipeline (lemmatization, named entity filtering).

Add text explainability using LIME/SHAP.

Deploy on cloud (Streamlit Sharing, HuggingFace Spaces, etc).

Build API endpoint using Flask/FastAPI.

👨‍💻 Author
Shivam Yadav

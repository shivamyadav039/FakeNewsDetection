# 📰 Fake News Detection System using Ensemble AI & BERT

🚀 An advanced AI-powered Fake News Detection System that combines multiple machine learning and deep learning models using ensemble learning to accurately classify news as REAL or FAKE.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/NLP-Deep%20Learning-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/BERT-Transformers-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge">
</p>

---

## 🌟 Features

✅ **Multiple AI Models**
- 🤖 BERT (Bidirectional Encoder Representations from Transformers)
- 📊 TF-IDF + Logistic Regression
- 🌲 Random Forest (with Calibrated Probabilities)
- 🚀 XGBoost (with Class Imbalance Handling)
- 🧠 LSTM (Long Short-Term Memory)
- 🎯 Ensemble Stacking (Meta-Learner)

✅ **Production-Ready Features**
- Beautiful Streamlit Web Interface
- Centralized Model Loader with Caching
- Robust Error Handling & Fallback Mechanisms
- Real-time Confidence Scores & Probability Charts
- Hyperparameter Tuning Pipeline
- Docker Support for Easy Deployment

---

## � Project Structure

```
FakeNewsDetection/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Dockerfile                  # Docker configuration
├── .gitignore                  # Git ignore rules
│
├── data/                       # Dataset directory
│   ├── README.md              # Dataset setup instructions
│   ├── Fake.csv               # Fake news dataset (not in git)
│   ├── True.csv               # Real news dataset (not in git)
│   └── news.csv               # Combined dataset (generated)
│
├── models/                     # Trained models (not in git)
│   ├── bert_model/            # BERT fine-tuned model
│   ├── tfidf_model.pkl        # TF-IDF model
│   ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│   ├── rf_model.pkl           # Random Forest model
│   ├── rf_vectorizer.pkl      # RF vectorizer
│   ├── xgb_model.pkl          # XGBoost model
│   ├── xgb_vectorizer.pkl     # XGB vectorizer
│   ├── lstm_model.h5          # LSTM model
│   └── stacking_ensemble.pkl  # Stacking meta-learner
│
└── src/                        # Source code
    ├── dataset_builder.py      # Dataset preparation script
    ├── bert_train.py           # BERT training script
    ├── bert_predict.py         # BERT prediction wrapper
    ├── tfidf_train.py          # TF-IDF model training
    ├── rf_train.py             # Random Forest training
    ├── xgb_train.py            # XGBoost training
    ├── lstm_train.py           # LSTM training
    ├── model_loader.py         # Centralized model loader
    ├── ensemble_predict.py     # Ensemble prediction logic
    ├── tune_and_ensemble.py    # Hyperparameter tuning + stacking
    ├── model_metrics.py        # Model evaluation utilities
    └── model_compare.py        # Model comparison script
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager
- 8GB+ RAM (for BERT training)
- GPU recommended (optional, for faster training)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/shivamyadav039/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2️⃣ Set Up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note**: This will install TensorFlow, PyTorch, and Transformers. Installation may take 5-10 minutes.

### 4️⃣ Download Dataset

1. Download the dataset from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
2. Extract and place `Fake.csv` and `True.csv` in the `data/` folder
3. See `data/README.md` for detailed instructions

### 5️⃣ Build Dataset

```bash
python src/dataset_builder.py
```

This creates `data/news.csv` from the raw CSV files.

### 6️⃣ Train Models (Optional)

You can train individual models or use the automated tuning pipeline:

**Option A: Train Individual Models**
```bash
python src/tfidf_train.py      # TF-IDF + Logistic Regression
python src/rf_train.py          # Random Forest
python src/xgb_train.py         # XGBoost
python src/lstm_train.py        # LSTM (requires TensorFlow)
python src/bert_train.py        # BERT (slow, requires GPU for best results)
```

**Option B: Automated Tuning + Ensemble (Recommended)**
```bash
python src/tune_and_ensemble.py
```

This script:
- Performs hyperparameter tuning for TF-IDF, Random Forest, and XGBoost
- Trains a stacking ensemble meta-learner
- Saves all optimized models
- Targets 92%+ accuracy

### 7️⃣ Run Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t fake-news-detector .
```

### Run Container

```bash
docker run -p 8501:8501 fake-news-detector
```

Access the app at `http://localhost:8501`

---

## 📊 Model Performance

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| BERT | ~95% | ~0.95 | ~2-4 hours |
| TF-IDF + LR | ~92% | ~0.92 | ~2 minutes |
| Random Forest | ~91% | ~0.91 | ~5 minutes |
| XGBoost | ~93% | ~0.93 | ~3 minutes |
| LSTM | ~89% | ~0.89 | ~30 minutes |
| **Ensemble** | **~96%** | **~0.96** | N/A |

*Performance measured on test set (20% holdout)*

---

## 🎯 Usage Examples

### Web Interface (Streamlit)

1. Select a model from the dropdown (BERT, Ensemble, TF-IDF, etc.)
2. Paste news article text in the input box
3. Click "🔍 Analyze News"
4. View prediction, confidence score, and probability chart

### Python API

```python
from src.bert_predict import predict_news
from src.ensemble_predict import ensemble_predict

# Single model prediction
result = predict_news("Breaking news article text here...")
print(result)  # {'label': 'FAKE', 'confidence': 0.9234, ...}

# Ensemble prediction
result = ensemble_predict("News article text here...")
print(result)  # Includes individual model probabilities
```

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'pandas'`

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### Issue: `FileNotFoundError: data/news.csv not found`

**Solution**: Build the dataset first
```bash
python src/dataset_builder.py
```

### Issue: Models not loading in Streamlit

**Solution**: Train models first or check `models/` directory exists

### Issue: BERT training too slow / out of memory

**Solutions**:
- Use GPU (CUDA) for faster training
- Reduce batch size in `src/bert_train.py`
- Use CPU-only for inference (prediction is fast)

### Issue: Import warnings in VS Code

**Solution**: These are IDE linter warnings, not runtime errors. Install packages to resolve:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🛠️ Advanced Configuration

### Hyperparameter Tuning

Edit `src/tune_and_ensemble.py` to customize:
- `n_iter`: Number of random search iterations
- `cv`: Cross-validation folds
- Parameter grids for each model

### Ensemble Weights

Edit `src/ensemble_predict.py` to adjust model weights:
```python
weights["bert"] = 0.4    # BERT weight
weights["tfidf"] = 0.2   # TF-IDF weight
weights["rf"] = 0.2      # Random Forest weight
weights["xgb"] = 0.2     # XGBoost weight
```

---

## 📚 Technologies Used

- **Python 3.11**
- **Machine Learning**: scikit-learn, XGBoost
- **Deep Learning**: TensorFlow, PyTorch, Hugging Face Transformers
- **NLP**: BERT, TF-IDF, Word Embeddings
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

---

## 👨‍💻 Developer

**Shivam Yadav**
- GitHub: [@shivamyadav039](https://github.com/shivamyadav039)

---

## 📝 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- BERT Model: [Hugging Face Transformers](https://huggingface.co/bert-base-uncased)
- Inspiration: Combating misinformation with AI

---

## 🚀 Future Enhancements

- [ ] Add multi-language support
- [ ] Implement explainability (LIME/SHAP)
- [ ] Add API endpoint (FastAPI/Flask)
- [ ] Real-time news scraping & classification
- [ ] Mobile app version
- [ ] Add more ensemble techniques (voting, boosting)

---

**⭐ If you find this project helpful, please give it a star!**



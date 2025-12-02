# Fake News Detection

A simple NLP + Machine Learning project that classifies news articles as Fake or Real using TF-IDF features and Logistic Regression. This repository includes a Streamlit demo for live inference and a training script to reproduce the model artifacts.

## Status
This branch prepares the project to run locally: it adds a training script, refactors utilities under `src/`, adds a small sample dataset and developer tooling (requirements, CI, Dockerfile). After running `python train.py` the trained model and vectorizer will be saved to `model/` and the Streamlit app will be runnable.

## Quick start
1. Create and activate a Python virtual environment (recommended):
   - python3 -m venv .venv
   - source .venv/bin/activate  # macOS / Linux
   - .\venv\Scripts\activate  # Windows
2. Install dependencies:
   pip install -r requirements.txt
3. Train the model (creates `model/model.joblib` and `model/vectorizer.joblib`):
   python train.py --data data/sample.csv --out model/
4. Run the Streamlit app:
   streamlit run app.py

Open the URL printed by Streamlit (usually http://localhost:8501).

## Files added / modified
- README.md — this file
- requirements.txt — pinned dependencies
- .gitignore
- LICENSE (MIT)
- data/sample.csv — small example dataset with `text` and `label` columns
- train.py — training script (TF-IDF + LogisticRegression)
- app.py — Streamlit app for inference
- src/preprocess.py — text cleaning & preprocessing utilities
- src/model_utils.py — train/save/load helpers
- tests/test_preprocess.py — basic unit tests
- .github/workflows/ci.yml — CI for tests & linting
- Dockerfile — container for the Streamlit app

## Dataset format
The training script expects a CSV with columns:
- text — the article text or headline to use as input
- label — integer label (0 = Fake, 1 = Real)

## Reproducible training
To train on your own dataset, place a CSV with the required columns and run:

python train.py --data path/to/your.csv --out model/

The script will save `model.joblib` and `vectorizer.joblib` into the specified output directory.

## Development
- Run tests: `pytest -q`
- Lint: `flake8`

## Contributing
Fork the repository, create a feature branch, add tests and open a PR.

## License
MIT — see the LICENSE file for details.
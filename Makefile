.PHONY: help setup install clean run train tune test docker-build docker-run

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📰 Fake News Detection - Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🔧 Setup & Installation:"
	@echo "  make setup        - Create venv and install dependencies"
	@echo "  make install      - Install dependencies only"
	@echo "  make clean        - Remove cache and temp files"
	@echo ""
	@echo "📊 Data & Training:"
	@echo "  make build-data   - Build dataset from CSV files"
	@echo "  make train        - Train all models individually"
	@echo "  make tune         - Run hyperparameter tuning + ensemble"
	@echo ""
	@echo "🚀 Running:"
	@echo "  make run          - Run Streamlit app"
	@echo "  make test         - Run tests (if available)"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

setup:
	@echo "🚀 Setting up environment..."
	bash setup_env.sh

install:
	@echo "📥 Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Cleanup complete!"

build-data:
	@echo "📊 Building dataset..."
	python3 src/dataset_builder.py

train:
	@echo "🤖 Training all models..."
	@echo "1/5 Training TF-IDF model..."
	python3 src/tfidf_train.py
	@echo "2/5 Training Random Forest..."
	python3 src/rf_train.py
	@echo "3/5 Training XGBoost..."
	python3 src/xgb_train.py
	@echo "4/5 Training LSTM..."
	python3 src/lstm_train.py
	@echo "5/5 Training BERT (this may take a while)..."
	python3 src/bert_train.py
	@echo "✅ All models trained!"

tune:
	@echo "⚡ Running hyperparameter tuning + ensemble training..."
	python3 src/tune_and_ensemble.py

run:
	@echo "🚀 Starting Streamlit app..."
	streamlit run app.py

test:
	@echo "🧪 Running tests..."
	pytest -v

docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t fake-news-detector .

docker-run:
	@echo "🐳 Running Docker container..."
	docker run -p 8501:8501 fake-news-detector

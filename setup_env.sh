#!/bin/bash

echo "🚀 Setting up Fake News Detection environment..."
echo ""

# Check if Python 3.11+ is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "✅ Virtual environment created"
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "📥 Installing dependencies from requirements.txt..."
echo "   (This may take 5-10 minutes for TensorFlow, PyTorch, etc.)"
pip install -r requirements.txt

# Check installation
echo ""
echo "✅ Dependencies installed successfully!"
echo ""

# Create necessary directories
mkdir -p models data

echo "📁 Created models/ and data/ directories"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Environment setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📌 Next steps:"
echo ""
echo "1. Activate environment (if not already active):"
echo "   source .venv/bin/activate"
echo ""
echo "2. Download dataset:"
echo "   See data/README.md for instructions"
echo ""
echo "3. Build dataset:"
echo "   python src/dataset_builder.py"
echo ""
echo "4. Train models (optional):"
echo "   python src/tune_and_ensemble.py"
echo ""
echo "5. Run Streamlit app:"
echo "   streamlit run app.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

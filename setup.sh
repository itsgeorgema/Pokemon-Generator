#!/bin/sh
set -e

echo "🔧 Setting up Pokemon Generator for Python 3.13..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.13."
    exit 1
fi

# Create and activate virtual environment
echo "🔄 Creating virtual environment..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing core build dependencies..."
pip install --upgrade pip setuptools wheel

echo "📦 Installing project dependencies..."
pip install -r requirements.txt

echo "🗄️ Setting up directories..."
mkdir -p static/generated
mkdir -p static/generated_samples

echo "✅ Setup complete! You can now run the application with:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "🐳 Or using Docker:"
echo "   docker-compose up --build"
echo ""
echo "🌐 The application will be available at:"
echo "   http://localhost:8080" 
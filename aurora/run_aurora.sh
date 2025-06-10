#!/bin/bash
echo "🧠 Aurora - Autonomous Creative Artist"
echo "====================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install:"
    echo "   sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 First run - Setting up Aurora's environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
if ! python -c "import numpy" 2>/dev/null; then
    echo "📦 Installing Aurora's core requirements..."
    pip install --upgrade pip
    pip install numpy colorama pillow
    
    echo "📦 Installing optional features..."
    pip install pygame librosa opencv-python || echo "Some optional features may not have installed"
    
    # Heavier dependencies (install separately to handle failures)
    echo "📦 Installing AI components..."
    pip install chromadb sentence-transformers || echo "Advanced memory features unavailable"
    pip install llama-cpp-python || echo "LLM features unavailable"
fi

# Create necessary directories
mkdir -p aurora_memory dream_logs conversation_logs models

echo ""
echo "✨ Launching Aurora..."
echo ""

# Run Aurora
cd aurora
python3 -m main

# Deactivate when done
deactivate

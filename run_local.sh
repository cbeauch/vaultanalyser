#!/bin/bash

# HyperLiquid Vault Analyzer - Local Run Script
# This script sets up and runs the vault analyzer locally

set -e  # Exit on error

echo "🚀 Starting HyperLiquid Vault Analyzer..."

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the vaultanalyser directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create cache directory if it doesn't exist
if [ ! -d "cache" ]; then
    echo "📁 Creating cache directory..."
    mkdir -p cache
fi

# Run the Streamlit app
echo "🌐 Starting Streamlit server..."
echo "📡 The app will be available at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run main.py 
#!/bin/bash

echo "Setting up Hollywoodbets AI Horse Racing Predictor..."
echo "====================================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install ChromeDriver for Selenium
echo "Installing ChromeDriver..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo apt-get update
    sudo apt-get install -y chromium-browser chromium-chromedriver
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    brew install --cask google-chrome
    brew install chromedriver
elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows (Git Bash)
    echo "Please install Chrome and ChromeDriver manually on Windows"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw/race_cards
mkdir -p data/raw/results
mkdir -p data/raw/tipsters
mkdir -p data/raw/form_guides
mkdir -p data/processed/training_data
mkdir -p data/processed/feature_sets
mkdir -p data/processed/test_sets
mkdir -p data/processed/predictions
mkdir -p data/models/experiments
mkdir -p data/models/deployment_packages
mkdir -p logs
mkdir -p output

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please update .env with your configuration"
fi

# Initialize database
echo "Initializing database..."
python -c "
from src.utils.database import db
print('Database initialized successfully')
"

echo ""
echo "Setup complete!"
echo "==============="
echo ""
echo "To activate virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the system:"
echo "  python main.py --mode once"
echo ""
echo "For continuous operation:"
echo "  python main.py --mode continuous"
echo ""
echo "To run tests:"
echo "  python -m pytest tests/"

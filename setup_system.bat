@echo off
echo ============================================
echo AI Horse Racing Predictor - Complete Setup
echo ============================================
echo.

REM Create directory structure
echo Creating project structure...
mkdir "C:\AI_Horse_Racing" 2>nul
cd "C:\AI_Horse_Racing"

mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\raw\race_cards 2>nul
mkdir data\raw\results 2>nul
mkdir data\raw\tipsters 2>nul
mkdir data\raw\form_guides 2>nul

mkdir data\processed 2>nul
mkdir data\processed\training_data 2>nul
mkdir data\processed\feature_sets 2>nul
mkdir data\processed\test_sets 2>nul
mkdir data\processed\predictions 2>nul

mkdir models 2>nul
mkdir models\experiments 2>nul
mkdir models\deployment_packages 2>nul

mkdir logs 2>nul

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.9...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe' -OutFile 'python_installer.exe'"
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1
    del python_installer.exe
)

echo.
echo Installing Python packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Creating virtual environment...
python -m venv venv

echo.
echo Generating sample data...
cd "C:\AI_Horse_Racing\data\raw\race_cards"
python sample_race_cards.json

echo.
echo Setup complete!
echo.
echo To start the system:
echo 1. Run: start_server.bat
echo 2. Open: http://localhost:5000
echo.
pause

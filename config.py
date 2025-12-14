import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Scraping configuration
class ScrapingConfig:
    HOLLYWOODBETS_URL = "https://www.hollywoodbets.net"
    RACE_CARDS_URL = f"{HOLLYWOODBETS_URL}/horse-racing/race-cards"
    RESULTS_URL = f"{HOLLYWOODBETS_URL}/horse-racing/results"
    TIPSTERS_URL = f"{HOLLYWOODBETS_URL}/horse-racing/tips"
    
    # Headers to mimic browser
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    # Request delays (seconds)
    REQUEST_DELAY = 2
    MAX_RETRIES = 3
    
    # Proxy settings (if needed)
    USE_PROXY = False
    PROXY_LIST = []  # Add your proxy list here

# Database configuration
class DBConfig:
    DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # sqlite, postgresql, mysql
    SQLITE_PATH = DATA_DIR / "horse_racing.db"
    
    # PostgreSQL/MySQL settings
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "horse_racing")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    
    @property
    def connection_string(self):
        if self.DB_TYPE == "sqlite":
            return f"sqlite:///{self.SQLITE_PATH}"
        elif self.DB_TYPE == "postgresql":
            return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        elif self.DB_TYPE == "mysql":
            return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        else:
            return f"sqlite:///{self.SQLITE_PATH}"

# Model configuration
class ModelConfig:
    # Feature engineering
    FEATURE_WINDOW_DAYS = 365  # How many days of history to use
    MIN_RACES_PER_HORSE = 3
    MIN_RACES_PER_JOCKEY = 10
    MIN_RACES_PER_TRAINER = 10
    
    # Model parameters
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Training
    N_ESTIMATORS = 1000
    EARLY_STOPPING_ROUNDS = 50
    CV_FOLDS = 5
    
    # Model paths
    XGBOOST_PATH = MODELS_DIR / "xgboost_model.joblib"
    LGBM_PATH = MODELS_DIR / "lightgbm_model.joblib"
    ENSEMBLE_PATH = MODELS_DIR / "ensemble_model.joblib"
    SCALER_PATH = MODELS_DIR / "scaler.joblib"
    ENCODER_PATH = MODELS_DIR / "encoder.joblib"

# Betting configuration
class BettingConfig:
    MIN_CONFIDENCE = 0.65  # Minimum confidence to place a bet
    MAX_BETS_PER_DAY = 10
    BANKROLL = 10000  # Starting bankroll in Rands
    STAKE_PERCENTAGE = 0.02  # 2% of bankroll per bet
    
    # Pick Six specific
    PICK_SIX_MIN_CONFIDENCE = 0.55
    PICK_SIX_COMBINATIONS = 10  # Number of combinations to generate

# Initialize configs
SCRAPING_CONFIG = ScrapingConfig()
DB_CONFIG = DBConfig()
MODEL_CONFIG = ModelConfig()
BETTING_CONFIG = BettingConfig()

import logging
import sys
from datetime import datetime
from pathlib import Path
import colorama
from colorama import Fore, Style

colorama.init()

class Logger:
    def __init__(self, name="HorseRacingPredictor"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f"horse_racing_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Color formatter for console
        class ColorFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: Fore.CYAN + formatter._fmt + Style.RESET_ALL,
                logging.INFO: Fore.GREEN + formatter._fmt + Style.RESET_ALL,
                logging.WARNING: Fore.YELLOW + formatter._fmt + Style.RESET_ALL,
                logging.ERROR: Fore.RED + formatter._fmt + Style.RESET_ALL,
                logging.CRITICAL: Fore.RED + Style.BRIGHT + formatter._fmt + Style.RESET_ALL,
            }
            
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
                return formatter.format(record)
        
        color_formatter = ColorFormatter()
        console_handler.setFormatter(color_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        return self.logger
    
    def log_scraping_start(self, url):
        self.logger.info(f"🔄 Starting scraping: {url}")
    
    def log_scraping_success(self, url, items_found):
        self.logger.info(f"✅ Successfully scraped {url} - Found {items_found} items")
    
    def log_scraping_error(self, url, error):
        self.logger.error(f"❌ Error scraping {url}: {error}")
    
    def log_model_training_start(self, model_name):
        self.logger.info(f"🤖 Starting {model_name} training")
    
    def log_model_training_success(self, model_name, accuracy):
        self.logger.info(f"🎯 {model_name} trained successfully - Accuracy: {accuracy:.4f}")
    
    def log_prediction(self, race_id, predictions):
        self.logger.info(f"📊 Predictions for race {race_id}: {predictions}")

# Singleton instance
logger = Logger().get_logger()

"""
Horse Racing Prediction System - Main Package
"""

__version__ = "1.0.0"
__author__ = "Horse Racing AI Team"
__description__ = "AI-powered horse racing prediction system for South African racing"

from src.scraper.hollywoodbets_scraper import HollywoodBetsScraper
from src.models.predictor import HorseRacingPredictor
from src.models.model_trainer import ModelTrainer
from src.betting.punters_challenge import PuntersChallenge
from src.betting.pick_six import PickSix
from src.utils.logger import logger
from src.utils.database import db

__all__ = [
    'HollywoodBetsScraper',
    'HorseRacingPredictor',
    'ModelTrainer',
    'PuntersChallenge',
    'PickSix',
    'logger',
    'db'
]

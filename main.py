#!/usr/bin/env python3
"""
Main entry point for Hollywood Bets AI Horse Racing Predictor
"""

import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse
import json
from pathlib import Path

from src.scraper.hollywoodbets_scraper import HollywoodBetsScraper
from src.models.predictor import HorseRacingPredictor
from src.models.model_trainer import ModelTrainer
from src.betting.punters_challenge import PuntersChallenge
from src.betting.pick_six import PickSix
from src.utils.logger import logger
from src.utils.database import db
from config import MODEL_CONFIG, BETTING_CONFIG

class HorseRacingPredictionSystem:
    def __init__(self):
        self.scraper = HollywoodBetsScraper(use_selenium=True)
        self.predictor = HorseRacingPredictor()
        self.punters_challenge = PuntersChallenge(self.predictor)
        self.pick_six = PickSix(self.predictor)
        self.is_running = False
        
    def run_daily_pipeline(self):
        """Run the complete daily prediction pipeline."""
        logger.info("=" * 60)
        logger.info("Starting daily prediction pipeline")
        logger.info("=" * 60)
        
        try:
            today = datetime.now()
            
            # Step 1: Scrape today's race cards
            logger.info("Step 1: Scraping today's race cards...")
            races = self.scraper.scrape_race_cards(today)
            
            if not races:
                logger.warning("No races found for today")
                return
            
            # Save races to database
            for race in races:
                self.scraper.save_to_database(race)
            
            logger.info(f"Found {len(races)} races for today")
            
            # Step 2: Train/update model if needed
            logger.info("Step 2: Checking model...")
            self._update_model_if_needed()
            
            # Step 3: Make predictions
            logger.info("Step 3: Making predictions...")
            predictions = self.predictor.get_daily_predictions(today)
            
            if not predictions:
                logger.warning("No predictions generated")
                return
            
            # Save predictions to database
            self._save_predictions_to_db(predictions)
            
            # Step 4: Generate betting recommendations
            logger.info("Step 4: Generating betting recommendations...")
            betting_recommendations = self.predictor.generate_betting_recommendations(predictions)
            
            # Step 5: Generate Punters Challenge strategy
            logger.info("Step 5: Generating Punters Challenge strategy...")
            challenge_strategy = self.punters_challenge.generate_daily_challenge(today)
            
            # Step 6: Generate Pick Six combinations
            logger.info("Step 6: Generating Pick Six combinations...")
            pick_six_combinations = self.pick_six.generate_pick_six_combinations(races)
            
            # Step 7: Output results
            logger.info("Step 7: Outputting results...")
            self._output_results(predictions, betting_recommendations, 
                               challenge_strategy, pick_six_combinations)
            
            # Step 8: Scrape yesterday's results for learning
            logger.info("Step 8: Scraping yesterday's results...")
            yesterday = today - timedelta(days=1)
            results = self.scraper.scrape_race_results(yesterday)
            
            for result in results:
                self.scraper.update_results(result)
            
            # Step 9: Update model with new results
            if results:
                logger.info("Step 9: Updating model with new results...")
                self._update_model_with_results(predictions, results)
            
            logger.info("=" * 60)
            logger.info("Daily pipeline completed successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            self.scraper.close()
    
    def _update_model_if_needed(self):
        """Update model if it's outdated or doesn't exist."""
        model_path = MODEL_CONFIG.ENSEMBLE_PATH
        
        if not model_path.exists():
            logger.info("No model found, training new model...")
            self._train_new_model()
        else:
            # Check if model is older than 7 days
            model_age = datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)
            if model_age.days >= 7:
                logger.info(f"Model is {model_age.days} days old, retraining...")
                self._train_new_model()
            else:
                logger.info(f"Model is {model_age.days} days old, using existing model")
    
    def _train_new_model(self):
        """Train a new model with all available data."""
        try:
            trainer = ModelTrainer()
            
            # Prepare training data
            X, y, feature_cols = trainer.prepare_training_data()
            
            # Train ensemble model
            ensemble = trainer.train_ensemble(X, y)
            
            # Save model
            trainer.save_models()
            
            # Reload predictor with new model
            self.predictor.load_model()
            
            logger.info("New model trained and loaded successfully")
            
        except Exception as e:
            logger.error(f"Error training new model: {e}")
    
    def _save_predictions_to_db(self, predictions: List[Dict[str, Any]]):
        """Save predictions to database."""
        for pred in predictions:
            try:
                prediction_data = {
                    'race_id': pred['race_id'],
                    'model_name': 'ensemble',
                    'prediction_data': json.dumps({
                        'predictions': pred['predictions'],
                        'top_pick': pred['top_pick'],
                        'value_pick': pred.get('value_pick')
                    }),
                    'top_pick': pred['top_pick']['horse_name'] if pred['top_pick'] else None,
                    'top_pick_confidence': pred['top_pick']['combined_probability'] if pred['top_pick'] else None,
                    'accuracy': pred.get('confidence', 0.0)
                }
                
                db.save_prediction(prediction_data)
                
            except Exception as e:
                logger.error(f"Error saving prediction to database: {e}")
    
    def _output_results(self, predictions: List[Dict[str, Any]],
                       betting_recommendations: Dict[str, Any],
                       challenge_strategy: Dict[str, Any],
                       pick_six_combinations: Dict[str, Any]):
        """Output results to files and console."""
        
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        today_str = datetime.now().strftime("%Y%m%d")
        
        # 1. Save predictions to JSON
        predictions_file = output_dir / f"predictions_{today_str}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        # 2. Save betting recommendations
        if betting_recommendations:
            betting_file = output_dir / f"betting_recommendations_{today_str}.json"
            with open(betting_file, 'w') as f:
                json.dump(betting_recommendations, f, indent=2, default=str)
        
        # 3. Save Punters Challenge strategy
        if challenge_strategy:
            challenge_file = output_dir / f"punters_challenge_{today_str}.json"
            with open(challenge_file, 'w') as f:
                json.dump(challenge_strategy, f, indent=2, default=str)
        
        # 4. Save Pick Six combinations
        if pick_six_combinations:
            pick_six_file = output_dir / f"pick_six_{today_str}.json"
            with open(pick_six_file, 'w') as f:
                json.dump(pick_six_combinations, f, indent=2, default=str)
        
        # 5. Print summary to console
        self._print_summary(predictions, betting_recommendations)
    
    def _print_summary(self, predictions: List[Dict[str, Any]],
                      betting_recommendations: Dict[str, Any]):
        """Print summary of predictions to console."""
        print("\n" + "=" * 60)
        print("HORSE RACING PREDICTIONS SUMMARY")
        print("=" * 60)
        
        print(f"\nTotal Races: {len(predictions)}")
        print(f"Predictions Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        print("\n" + "-" * 60)
        print("TOP PICKS FOR TODAY:")
        print("-" * 60)
        
        for i, race in enumerate(predictions[:10]):  # Show first 10 races
            if race['top_pick']:
                print(f"\nRace {i+1}: {race['race_name']}")
                print(f"  Time: {race['race_time'].strftime('%H:%M') if isinstance(race['race_time'], datetime) else race['race_time']}")
                print(f"  Top Pick: {race['top_pick']['horse_name']}")
                print(f"  Probability: {race['top_pick']['combined_probability']:.3f}")
                print(f"  Confidence: {race['confidence']:.3f}")
                
                if race.get('value_pick'):
                    print(f"  Value Pick: {race['value_pick']['horse_name']} "
                          f"(Value: {race['value_pick'].get('value_rating', 'N/A')})")
        
        if betting_recommendations and betting_recommendations.get('recommended_bets'):
            print("\n" + "-" * 60)
            print("BETTING RECOMMENDATIONS:")
            print("-" * 60)
            
            print(f"\nTotal Recommended Bets: {betting_recommendations['total_bets']}")
            print(f"Total Stake: R{betting_recommendations['total_stake']:.2f}")
            print(f"Expected Value: R{betting_recommendations['expected_value']:.2f}")
            
            for bet in betting_recommendations['recommended_bets'][:5]:  # Show first 5 bets
                print(f"\n  Race: {bet['race_name']}")
                print(f"  Horse: {bet['horse']}")
                print(f"  Probability: {bet['probability']:.3f}")
                print(f"  Stake: R{bet['stake']:.2f}")
                print(f"  Value Rating: {bet['value_rating']}")
        
        print("\n" + "=" * 60)
        print("Predictions saved to output directory")
        print("=" * 60 + "\n")
    
    def _update_model_with_results(self, predictions: List[Dict[str, Any]],
                                 results: List[Dict[str, Any]]):
        """Update model with new race results."""
        try:
            self.predictor.update_model_feedback(predictions, results)
            logger.info("Model updated with new results")
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    def run_continuous(self):
        """Run the system continuously with scheduled tasks."""
        logger.info("Starting continuous prediction system...")
        self.is_running = True
        
        # Schedule daily tasks
        schedule.every().day.at("08:00").do(self.run_daily_pipeline)  # Morning predictions
        schedule.every().day.at("18:00").do(self._scrape_results)     # Evening results
        
        # Run immediate first execution
        self.run_daily_pipeline()
        
        # Keep running
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scrape_results(self):
        """Scrape results from completed races."""
        logger.info("Scraping today's race results...")
        
        try:
            today = datetime.now()
            results = self.scraper.scrape_race_results(today)
            
            for result in results:
                self.scraper.update_results(result)
            
            logger.info(f"Scraped {len(results)} race results")
            
        except Exception as e:
            logger.error(f"Error scraping results: {e}")
    
    def stop(self):
        """Stop the continuous system."""
        self.is_running = False
        logger.info("Prediction system stopped")
    
    def run_once(self):
        """Run the pipeline once."""
        self.run_daily_pipeline()

def main():
    parser = argparse.ArgumentParser(description='Hollywood Bets AI Horse Racing Predictor')
    parser.add_argument('--mode', choices=['once', 'continuous', 'train', 'scrape'],
                       default='once', help='Run mode')
    parser.add_argument('--date', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    system = HorseRacingPredictionSystem()
    
    try:
        if args.mode == 'once':
            system.run_once()
        
        elif args.mode == 'continuous':
            system.run_continuous()
        
        elif args.mode == 'train':
            logger.info("Training new model...")
            system._train_new_model()
        
        elif args.mode == 'scrape':
            logger.info("Running scraper only...")
            if args.date:
                date = datetime.strptime(args.date, '%Y-%m-%d')
            else:
                date = datetime.now()
            
            races = system.scraper.scrape_race_cards(date)
            logger.info(f"Scraped {len(races)} races")
            
            for race in races:
                system.scraper.save_to_database(race)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        system.stop()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import joblib
from scipy.special import softmax

from config import MODEL_CONFIG, BETTING_CONFIG
from src.utils.logger import logger
from src.utils.database import db
from src.scraper.data_cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer
from src.features.tipster_analyzer import TipsterAnalyzer

class HorseRacingPredictor:
    def __init__(self):
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.data_cleaner = DataCleaner()
        self.tipster_analyzer = TipsterAnalyzer()
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = joblib.load(MODEL_CONFIG.ENSEMBLE_PATH)
            self.feature_engineer.load_transformations(MODEL_CONFIG.ENCODER_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def predict_race(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions for a single race."""
        if not self.model:
            logger.error("Model not loaded")
            return {}
        
        try:
            # Prepare race data
            df = pd.DataFrame(race_data['runners'])
            df['race_id'] = race_data['race_id']
            df['race_date'] = race_data['race_date']
            df['distance'] = race_data['distance']
            df['going'] = race_data.get('going', 'Unknown')
            df['race_class'] = race_data.get('race_class', 'Unknown')
            df['meeting_name'] = race_data.get('meeting_name', 'Unknown')
            
            # Clean and engineer features
            df_clean = self.data_cleaner.clean_race_data(df)
            df_features = self.feature_engineer.create_comprehensive_features(df_clean)
            
            # Prepare for prediction
            X_pred = self.data_cleaner.prepare_prediction_data(df_features.to_dict('records'))
            
            # Separate horse info from features
            horse_info = X_pred[['horse_id', 'horse_name']]
            X_features = X_pred.drop(['horse_id', 'horse_name'], axis=1, errors='ignore')
            
            # Handle categorical features
            categorical_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                X_features = self.feature_engineer.encode_categorical_features(X_features, categorical_cols)
            
            # Scale features
            X_scaled, _ = self.feature_engineer.scale_features(X_features)
            
            # Make predictions
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Combine with horse info
            predictions = []
            for idx, (horse_id, horse_name) in enumerate(zip(horse_info['horse_id'], horse_info['horse_name'])):
                predictions.append({
                    'horse_id': horse_id,
                    'horse_name': horse_name,
                    'probability': float(probabilities[idx]),
                    'confidence': self._calculate_confidence(probabilities[idx])
                })
            
            # Sort by probability
            predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            # Add tipster analysis
            tipster_predictions = self.tipster_analyzer.combine_tipster_predictions(race_data['race_id'])
            
            # Combine with model predictions
            for pred in predictions:
                horse_name = pred['horse_name']
                if horse_name in tipster_predictions:
                    tipster_conf = tipster_predictions[horse_name]
                    # Weighted combination: 70% model, 30% tipsters
                    pred['combined_probability'] = 0.7 * pred['probability'] + 0.3 * tipster_conf
                else:
                    pred['combined_probability'] = pred['probability']
            
            # Re-sort by combined probability
            predictions.sort(key=lambda x: x['combined_probability'], reverse=True)
            
            # Calculate value bets
            for pred in predictions:
                if 'odds' in df.loc[df['horse_name'] == pred['horse_name']].iloc[0]:
                    odds = df.loc[df['horse_name'] == pred['horse_name']].iloc[0]['odds']
                    if odds > 0:
                        pred['odds'] = odds
                        pred['implied_probability'] = 1 / odds
                        pred['value'] = pred['combined_probability'] - pred['implied_probability']
                        pred['value_rating'] = self._calculate_value_rating(pred['value'])
            
            result = {
                'race_id': race_data['race_id'],
                'race_name': race_data['race_name'],
                'race_time': race_data['race_time'],
                'predictions': predictions,
                'top_pick': predictions[0] if predictions else None,
                'value_pick': self._find_value_pick(predictions),
                'confidence': self._calculate_overall_confidence(predictions)
            }
            
            logger.info(f"Predictions made for race {race_data['race_id']}: {result['top_pick']['horse_name']} "
                       f"({result['top_pick']['combined_probability']:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error predicting race: {e}")
            return {}
    
    def _calculate_confidence(self, probability: float) -> str:
        """Convert probability to confidence level."""
        if probability >= 0.8:
            return 'Very High'
        elif probability >= 0.7:
            return 'High'
        elif probability >= 0.6:
            return 'Medium'
        elif probability >= 0.5:
            return 'Low'
        else:
            return 'Very Low'
    
    def _calculate_value_rating(self, value: float) -> str:
        """Calculate value rating based on value."""
        if value >= 0.2:
            return 'Excellent Value'
        elif value >= 0.1:
            return 'Good Value'
        elif value >= 0.05:
            return 'Some Value'
        elif value >= 0:
            return 'Fair Value'
        else:
            return 'Poor Value'
    
    def _find_value_pick(self, predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best value pick (good probability at good odds)."""
        value_picks = [p for p in predictions if p.get('value', -1) > 0.05]
        
        if not value_picks:
            return None
        
        # Sort by value
        value_picks.sort(key=lambda x: x['value'], reverse=True)
        
        # Return the best value pick that also has decent probability
        for pick in value_picks:
            if pick['combined_probability'] >= 0.3:  # Minimum probability threshold
                return pick
        
        return None
    
    def _calculate_overall_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in predictions."""
        if not predictions:
            return 0.0
        
        # Confidence based on top pick margin
        if len(predictions) >= 2:
            margin = predictions[0]['combined_probability'] - predictions[1]['combined_probability']
            confidence = min(margin * 10, 1.0)  # Scale margin to 0-1
        else:
            confidence = predictions[0]['combined_probability']
        
        return confidence
    
    def predict_multiple_races(self, races_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make predictions for multiple races."""
        results = []
        
        for race_data in races_data:
            prediction = self.predict_race(race_data)
            if prediction:
                results.append(prediction)
        
        return results
    
    def get_daily_predictions(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get predictions for all races on a specific date."""
        if date is None:
            date = datetime.now()
        
        # Get races from database
        races = db.get_races_for_date(date)
        
        if not races:
            logger.warning(f"No races found for {date.date()}")
            return []
        
        # Convert to prediction format
        races_data = []
        
        with db.get_session() as session:
            for race in races:
                # Get runners for this race
                runners = session.query(db.Runner).filter_by(race_id=race.id).all()
                
                race_data = {
                    'race_id': race.race_id,
                    'race_name': race.race_name,
                    'race_date': race.race_date,
                    'race_time': race.race_time,
                    'distance': race.distance,
                    'going': race.going,
                    'race_class': race.race_class,
                    'meeting_name': 'Unknown',  # Would need to be populated
                    'runners': []
                }
                
                for runner in runners:
                    runner_data = {
                        'horse_id': runner.horse_id,
                        'horse_name': runner.horse_name,
                        'jockey_name': runner.jockey_name,
                        'trainer_name': runner.trainer_name,
                        'weight': runner.weight,
                        'draw': runner.draw,
                        'age': runner.age,
                        'form_rating': runner.form_rating,
                        'official_rating': runner.official_rating,
                        'days_since_last_run': runner.days_since_last_run,
                        'career_starts': runner.career_starts,
                        'career_wins': runner.career_wins,
                        'odds': runner.odds
                    }
                    race_data['runners'].append(runner_data)
                
                races_data.append(race_data)
        
        # Make predictions
        predictions = self.predict_multiple_races(races_data)
        
        logger.info(f"Made predictions for {len(predictions)} races on {date.date()}")
        
        return predictions
    
    def generate_betting_recommendations(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate betting recommendations based on predictions."""
        recommendations = {
            'date': datetime.now().date(),
            'recommended_bets': [],
            'total_bets': 0,
            'total_stake': 0.0,
            'expected_value': 0.0
        }
        
        bankroll = BETTING_CONFIG.BANKROLL
        stake_per_bet = bankroll * BETTING_CONFIG.STAKE_PERCENTAGE
        
        for race_pred in predictions:
            top_pick = race_pred['top_pick']
            
            # Check confidence threshold
            if (top_pick['combined_probability'] >= BETTING_CONFIG.MIN_CONFIDENCE and 
                race_pred['confidence'] >= 0.6):
                
                # Check for value
                value_rating = top_pick.get('value_rating', 'Poor Value')
                if value_rating not in ['Poor Value', 'Fair Value']:
                    
                    bet = {
                        'race_id': race_pred['race_id'],
                        'race_name': race_pred['race_name'],
                        'race_time': race_pred['race_time'],
                        'horse': top_pick['horse_name'],
                        'probability': top_pick['combined_probability'],
                        'confidence': race_pred['confidence'],
                        'stake': stake_per_bet,
                        'bet_type': 'win',
                        'value_rating': value_rating,
                        'odds': top_pick.get('odds', 0.0)
                    }
                    
                    recommendations['recommended_bets'].append(bet)
        
        recommendations['total_bets'] = len(recommendations['recommended_bets'])
        recommendations['total_stake'] = sum(b['stake'] for b in recommendations['recommended_bets'])
        
        # Calculate expected value
        expected_value = 0.0
        for bet in recommendations['recommended_bets']:
            if bet['odds'] > 0:
                ev = (bet['probability'] * (bet['odds'] - 1) * bet['stake'] - 
                      (1 - bet['probability']) * bet['stake'])
                expected_value += ev
        
        recommendations['expected_value'] = expected_value
        
        logger.info(f"Generated {recommendations['total_bets']} betting recommendations "
                   f"with total stake R{recommendations['total_stake']:.2f}")
        
        return recommendations
    
    def evaluate_predictions(self, predictions: List[Dict[str, Any]], 
                           results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate prediction accuracy against actual results."""
        evaluation = {
            'total_races': len(predictions),
            'correct_predictions': 0,
            'accuracy': 0.0,
            'profitable_races': 0,
            'total_profit': 0.0,
            'roi': 0.0
        }
        
        for pred in predictions:
            # Find matching result
            result = next((r for r in results if r['race_id'] == pred['race_id']), None)
            
            if result:
                top_pick = pred['top_pick']
                winning_horse = result.get('winning_horse_name')
                
                if top_pick and winning_horse and top_pick['horse_name'] == winning_horse:
                    evaluation['correct_predictions'] += 1
                    
                    # Calculate profit (simplified)
                    if 'odds' in top_pick and top_pick['odds'] > 0:
                        profit = top_pick['odds'] - 1  # For 1 unit stake
                        evaluation['total_profit'] += profit
                        evaluation['profitable_races'] += 1
        
        if evaluation['total_races'] > 0:
            evaluation['accuracy'] = evaluation['correct_predictions'] / evaluation['total_races']
        
        if evaluation['profitable_races'] > 0:
            evaluation['roi'] = (evaluation['total_profit'] / evaluation['profitable_races']) * 100
        
        logger.info(f"Evaluation: {evaluation['accuracy']:.3f} accuracy, "
                   f"{evaluation['roi']:.1f}% ROI")
        
        return evaluation
    
    def update_model_feedback(self, predictions: List[Dict[str, Any]], 
                            results: List[Dict[str, Any]]):
        """Update model based on prediction results (continuous learning)."""
        # Prepare data for retraining
        new_data = []
        
        for pred in predictions:
            result = next((r for r in results if r['race_id'] == pred['race_id']), None)
            
            if result and pred.get('race_data'):
                # Add actual outcome to race data
                race_data = pred['race_data'].copy()
                winning_horse = result.get('winning_horse_name')
                
                # Mark winners in runner data
                for runner in race_data.get('runners', []):
                    runner['is_winner'] = 1 if runner['horse_name'] == winning_horse else 0
                
                new_data.append(race_data)
        
        if new_data:
            # Convert to DataFrame
            df_list = []
            for race in new_data:
                for runner in race.get('runners', []):
                    runner_df = pd.DataFrame([runner])
                    runner_df['race_id'] = race['race_id']
                    runner_df['race_date'] = race['race_date']
                    runner_df['distance'] = race['distance']
                    runner_df['going'] = race.get('going', 'Unknown')
                    df_list.append(runner_df)
            
            if df_list:
                new_df = pd.concat(df_list, ignore_index=True)
                
                # Update model
                from src.models.model_trainer import ModelTrainer
                trainer = ModelTrainer()
                trainer.continuous_learning(new_df)
                
                # Reload updated model
                self.load_model()
                
                logger.info(f"Model updated with {len(new_data)} new race results")
    
    def predict_with_explanations(self, race_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions with feature importance explanations."""
        prediction = self.predict_race(race_data)
        
        if not prediction or not self.model:
            return prediction
        
        try:
            # Prepare features for SHAP (if available)
            import shap
            
            # Prepare data
            df = pd.DataFrame(race_data['runners'])
            df['race_id'] = race_data['race_id']
            df['race_date'] = race_data['race_date']
            df['distance'] = race_data['distance']
            df['going'] = race_data.get('going', 'Unknown')
            
            df_clean = self.data_cleaner.clean_race_data(df)
            df_features = self.feature_engineer.create_comprehensive_features(df_clean)
            
            X_pred = self.data_cleaner.prepare_prediction_data(df_features.to_dict('records'))
            X_features = X_pred.drop(['horse_id', 'horse_name'], axis=1, errors='ignore')
            
            categorical_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                X_features = self.feature_engineer.encode_categorical_features(X_features, categorical_cols)
            
            X_scaled, _ = self.feature_engineer.scale_features(X_features)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_scaled)
            
            # Add explanations to predictions
            for idx, pred in enumerate(prediction['predictions']):
                if idx < len(shap_values):
                    # Get top contributing features
                    feature_names = X_scaled.columns
                    shap_vals = shap_values[idx]
                    
                    # Sort by absolute contribution
                    contributions = []
                    for j, (name, val) in enumerate(zip(feature_names, shap_vals)):
                        contributions.append({
                            'feature': name,
                            'contribution': float(val),
                            'absolute_contribution': abs(float(val))
                        })
                    
                    contributions.sort(key=lambda x: x['absolute_contribution'], reverse=True)
                    pred['top_contributors'] = contributions[:5]  # Top 5 features
            
            prediction['explanations_available'] = True
            
        except Exception as e:
            logger.warning(f"Could not generate explanations: {e}")
            prediction['explanations_available'] = False
        
        return prediction
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        # Load historical predictions and results
        with db.get_session() as session:
            # Get recent predictions
            recent_preds = session.query(db.Prediction).order_by(
                db.Prediction.created_at.desc()
            ).limit(100).all()
            
            metrics = {
                'total_predictions': len(recent_preds),
                'accuracy_by_model': {},
                'recent_trend': 0.0
            }
            
            # Group by model
            models = {}
            for pred in recent_preds:
                model_name = pred.model_name
                if model_name not in models:
                    models[model_name] = {'total': 0, 'correct': 0}
                
                models[model_name]['total'] += 1
                if pred.accuracy and pred.accuracy > 0.5:
                    models[model_name]['correct'] += 1
            
            # Calculate accuracy
            for model_name, stats in models.items():
                if stats['total'] > 0:
                    metrics['accuracy_by_model'][model_name] = stats['correct'] / stats['total']
            
            # Calculate recent trend
            if len(recent_preds) >= 20:
                recent = recent_preds[:10]
                older = recent_preds[10:20]
                
                recent_acc = sum(1 for p in recent if p.accuracy and p.accuracy > 0.5) / len(recent)
                older_acc = sum(1 for p in older if p.accuracy and p.accuracy > 0.5) / len(older)
                
                metrics['recent_trend'] = recent_acc - older_acc
        
        return metrics

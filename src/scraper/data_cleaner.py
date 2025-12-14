import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
from scipy import stats

from src.utils.logger import logger

class DataCleaner:
    def __init__(self):
        self.going_mapping = {
            'good': 1.0,
            'good to firm': 0.9,
            'firm': 0.8,
            'yielding': 0.7,
            'soft': 0.6,
            'heavy': 0.5,
            'slow': 0.4,
            'fast': 0.9,
            'wet': 0.6,
            'dry': 0.8,
            'unknown': 0.7
        }
    
    def clean_race_data(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess race data."""
        df = race_df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Convert data types
        df = self._convert_data_types(df)
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Normalize numerical features
        df = self._normalize_features(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        
        # Numerical columns
        num_cols = ['weight', 'draw', 'age', 'form_rating', 'official_rating',
                   'days_since_last_run', 'career_starts', 'career_wins',
                   'course_wins', 'distance_wins']
        
        for col in num_cols:
            if col in df.columns:
                # Fill with median for numerical columns
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns
        cat_cols = ['jockey_name', 'trainer_name', 'going', 'race_class']
        
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Odds handling
        if 'odds' in df.columns:
            df['odds'] = df['odds'].replace(0, np.nan)
            df['odds'] = df['odds'].fillna(df['odds'].median())
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        
        # Convert going to numerical score
        if 'going' in df.columns:
            df['going_score'] = df['going'].apply(
                lambda x: self.going_mapping.get(str(x).lower(), 0.7)
            )
        
        # Convert race class to numerical
        if 'race_class' in df.columns:
            df['class_score'] = df['race_class'].apply(self._extract_class_score)
        
        # Convert time features
        if 'race_time' in df.columns:
            df['hour'] = pd.to_datetime(df['race_time']).dt.hour
            df['minute'] = pd.to_datetime(df['race_time']).dt.minute
            df['time_of_day'] = df['hour'] + df['minute'] / 60
        
        # Convert date to features
        if 'race_date' in df.columns:
            df['race_date'] = pd.to_datetime(df['race_date'])
            df['day_of_week'] = df['race_date'].dt.dayofweek
            df['month'] = df['race_date'].dt.month
            df['year'] = df['race_date'].dt.year
        
        return df
    
    def _extract_class_score(self, class_str: str) -> float:
        """Extract numerical score from race class."""
        if not isinstance(class_str, str):
            return 5.0
        
        class_str = class_str.lower()
        
        # Maiden races
        if 'maiden' in class_str:
            return 1.0
        
        # Class races
        class_match = re.search(r'class\s*(\d+)', class_str)
        if class_match:
            class_num = int(class_match.group(1))
            return max(10.0 - class_num, 1.0)  # Higher class = lower number
        
        # Graded races
        if 'grade 1' in class_str or 'group 1' in class_str:
            return 10.0
        elif 'grade 2' in class_str or 'group 2' in class_str:
            return 9.0
        elif 'grade 3' in class_str or 'group 3' in class_str:
            return 8.0
        
        # Listed races
        if 'listed' in class_str:
            return 7.0
        
        # Handicap races
        if 'handicap' in class_str:
            return 6.0
        
        return 5.0  # Default
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing data."""
        
        # Win percentages
        if 'career_starts' in df.columns and 'career_wins' in df.columns:
            df['career_win_pct'] = np.where(
                df['career_starts'] > 0,
                df['career_wins'] / df['career_starts'],
                0
            )
        
        # Recent form (last 3 races)
        if 'recent_form' in df.columns:
            df['recent_form_score'] = df['recent_form'].apply(self._calculate_form_score)
        
        # Jockey and trainer stats
        if 'jockey_name' in df.columns:
            df = self._calculate_jockey_stats(df)
        
        if 'trainer_name' in df.columns:
            df = self._calculate_trainer_stats(df)
        
        # Horse consistency
        if 'form_rating' in df.columns:
            df['rating_consistency'] = df.groupby('horse_id')['form_rating'].transform('std').fillna(0)
        
        # Distance suitability
        if 'distance' in df.columns and 'distance_wins' in df.columns:
            df['distance_suitability'] = df.apply(
                lambda row: self._calculate_distance_suitability(row), axis=1
            )
        
        # Weight carried
        if 'weight' in df.columns:
            df['weight_vs_avg'] = df['weight'] - df.groupby('race_id')['weight'].transform('mean')
        
        # Draw advantage
        if 'draw' in df.columns:
            df['draw_advantage'] = df.apply(self._calculate_draw_advantage, axis=1)
        
        # Days since last run
        if 'days_since_last_run' in df.columns:
            df['fitness_index'] = np.exp(-df['days_since_last_run'] / 30)  # 30-day half-life
        
        # Age factor
        if 'age' in df.columns:
            df['age_factor'] = np.where(
                df['age'] <= 4, 1.0,
                np.where(df['age'] <= 6, 0.9,
                np.where(df['age'] <= 8, 0.8, 0.7))
            )
        
        return df
    
    def _calculate_form_score(self, form_str: str) -> float:
        """Calculate form score from recent form string."""
        if not isinstance(form_str, str):
            return 0.5
        
        # Form string like "1-2-3" or "412"
        form_chars = form_str.replace('-', '')
        
        if not form_chars:
            return 0.5
        
        # Convert positions to scores
        scores = []
        for char in form_chars[:3]:  # Last 3 races
            if char.isdigit():
                pos = int(char)
                if pos <= 3:
                    score = 1.0 - (pos - 1) * 0.25
                else:
                    score = 0.25
                scores.append(score)
        
        if not scores:
            return 0.5
        
        return np.mean(scores)
    
    def _calculate_jockey_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate jockey statistics."""
        jockey_stats = df.groupby('jockey_name').agg({
            'is_winner': ['mean', 'count'],
            'odds': 'mean'
        }).reset_index()
        
        jockey_stats.columns = ['jockey_name', 'jockey_win_rate', 'jockey_rides', 'jockey_avg_odds']
        
        # Merge back
        df = df.merge(jockey_stats, on='jockey_name', how='left')
        
        # Calculate jockey form (last 10 rides)
        df['jockey_recent_form'] = df.groupby('jockey_name')['is_winner'].transform(
            lambda x: x.rolling(10, min_periods=1).mean()
        ).fillna(0)
        
        return df
    
    def _calculate_trainer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trainer statistics."""
        trainer_stats = df.groupby('trainer_name').agg({
            'is_winner': ['mean', 'count'],
            'odds': 'mean'
        }).reset_index()
        
        trainer_stats.columns = ['trainer_name', 'trainer_win_rate', 'trainer_runners', 'trainer_avg_odds']
        
        # Merge back
        df = df.merge(trainer_stats, on='trainer_name', how='left')
        
        # Calculate trainer form (last 20 runners)
        df['trainer_recent_form'] = df.groupby('trainer_name')['is_winner'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        ).fillna(0)
        
        return df
    
    def _calculate_distance_suitability(self, row) -> float:
        """Calculate distance suitability score."""
        try:
            distance = float(row['distance'])
            distance_wins = float(row.get('distance_wins', 0))
            career_starts = float(row.get('career_starts', 1))
            
            if distance_wins > 0 and career_starts > 0:
                win_rate = distance_wins / career_starts
                
                # Adjust for distance (optimal distance range)
                if 1000 <= distance <= 1400:
                    optimal_range = [1000, 1400]
                elif 1400 <= distance <= 2000:
                    optimal_range = [1400, 2000]
                else:
                    optimal_range = [2000, 3000]
                
                # Calculate distance suitability
                if optimal_range[0] <= distance <= optimal_range[1]:
                    suitability = win_rate * 1.2
                else:
                    suitability = win_rate * 0.8
                
                return min(suitability, 1.0)
            
        except:
            pass
        
        return 0.5
    
    def _calculate_draw_advantage(self, row) -> float:
        """Calculate draw advantage based on track and distance."""
        try:
            draw = int(row['draw'])
            distance = float(row.get('distance', 1200))
            
            # Different tracks have different draw biases
            # For South African tracks
            track = str(row.get('meeting_name', '')).lower()
            
            if 'greyville' in track:
                # Greyville has draw bias for sprints
                if distance <= 1400:
                    # Lower draws better for sprints
                    advantage = 1.0 - (draw / 20)
                else:
                    advantage = 0.5
            elif 'kenilworth' in track or 'durham' in track:
                # Kenilworth/Durban bias
                advantage = 0.5 + (draw / 40)
            else:
                # Default: middle draws are best
                advantage = 1.0 - abs(draw - 10) / 20
            
            return max(0.0, min(1.0, advantage))
            
        except:
            return 0.5
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        numerical_cols = [
            'weight', 'draw', 'age', 'form_rating', 'official_rating',
            'days_since_last_run', 'career_starts', 'career_wins',
            'career_win_pct', 'going_score', 'class_score',
            'jockey_win_rate', 'trainer_win_rate', 'odds'
        ]
        
        # Only normalize columns that exist
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        for col in numerical_cols:
            if col == 'odds':
                # Inverse normalization for odds (lower odds = better)
                df[col] = 1 / (df[col] + 1)
            else:
                # Min-max normalization
                if df[col].max() > df[col].min():
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers."""
        numerical_cols = [
            'weight', 'form_rating', 'official_rating',
            'days_since_last_run', 'career_starts', 'odds'
        ]
        
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        for col in numerical_cols:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
                df = df[z_scores < 3]  # Remove beyond 3 standard deviations
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for model training."""
        # Select features
        feature_cols = [
            'weight', 'draw', 'age', 'form_rating', 'official_rating',
            'days_since_last_run', 'career_win_pct', 'going_score',
            'class_score', 'jockey_win_rate', 'trainer_win_rate',
            'rating_consistency', 'distance_suitability', 'weight_vs_avg',
            'draw_advantage', 'fitness_index', 'age_factor',
            'jockey_recent_form', 'trainer_recent_form', 'odds'
        ]
        
        # Only use columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols]
        y = df['is_winner'] if 'is_winner' in df.columns else None
        
        return X, y, feature_cols
    
    def prepare_prediction_data(self, race_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare race data for prediction."""
        df = pd.DataFrame(race_data)
        
        # Clean the data
        df = self.clean_race_data(df)
        
        # Select features
        feature_cols = [
            'weight', 'draw', 'age', 'form_rating', 'official_rating',
            'days_since_last_run', 'career_win_pct', 'going_score',
            'class_score', 'jockey_win_rate', 'trainer_win_rate',
            'rating_consistency', 'distance_suitability', 'weight_vs_avg',
            'draw_advantage', 'fitness_index', 'age_factor',
            'jockey_recent_form', 'trainer_recent_form', 'odds'
        ]
        
        # Only use columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df[['horse_id', 'horse_name'] + feature_cols]

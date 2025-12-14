import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TrainingDataGenerator:
    def __init__(self, raw_data_dir="../raw", processed_dir="."):
        self.raw_data_dir = raw_data_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
        
    def generate_training_dataset(self, num_samples=10000):
        """Generate comprehensive training dataset"""
        np.random.seed(42)
        
        data = []
        
        for i in range(num_samples):
            sample = self._generate_training_sample(i)
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Save to CSV
        filename = f"{self.processed_dir}/training_dataset_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"Generated training dataset with {len(df)} samples")
        print(f"Features: {list(df.columns)}")
        
        return df
    
    def _generate_training_sample(self, sample_id):
        """Generate individual training sample"""
        # Base horse characteristics
        sample = {
            'sample_id': sample_id,
            'horse_age': np.random.randint(2, 8),
            'horse_sex_encoded': np.random.choice([0, 1, 2, 3]),  # Colt, Filly, Gelding, Mare
            'days_since_last_race': np.random.randint(14, 120),
            'lifetime_starts': np.random.randint(1, 30),
            'lifetime_wins': np.random.randint(0, 8),
            'win_percentage': round(np.random.uniform(0, 0.4), 3),
            'in_the_money_percentage': round(np.random.uniform(0.1, 0.6), 3),
            'avg_finish_position': round(np.random.uniform(3.0, 8.0), 2),
            
            # Recent form
            'last_race_finish': np.random.randint(1, 12),
            'last_race_beaten_lengths': round(np.random.uniform(0, 15), 1),
            'last_race_speed_figure': np.random.randint(65, 95),
            'last_race_class_level': np.random.choice([1, 2, 3, 4]),  # Mdn, Clm, Alw, Stk
            
            # Trainer stats
            'trainer_win_percentage': round(np.random.uniform(0.1, 0.3), 3),
            'trainer_roi': round(np.random.uniform(0.8, 1.2), 2),
            'trainer_jockey_combo_win_pct': round(np.random.uniform(0.15, 0.35), 3),
            
            # Jockey stats
            'jockey_win_percentage': round(np.random.uniform(0.12, 0.25), 3),
            'jockey_mount_earnings': np.random.randint(100000, 10000000),
            
            # Race conditions
            'post_position': np.random.randint(1, 12),
            'morning_line_odds': round(np.random.uniform(1.5, 20.0), 2),
            'weight_carried': np.random.randint(1140, 1220),
            'claimed_last_out': np.random.choice([0, 1], p=[0.8, 0.2]),
            'first_time_blinkers': np.random.choice([0, 1], p=[0.9, 0.1]),
            'first_time_lasix': np.random.choice([0, 1], p=[0.85, 0.15]),
            
            # Track/distance factors
            'distance': round(np.random.uniform(5.0, 10.0), 1),
            'surface_dirt': np.random.choice([0, 1], p=[0.3, 0.7]),
            'track_familiarity': round(np.random.uniform(0.5, 1.5), 2),
            'distance_preference': round(np.random.uniform(0.8, 1.2), 2),
            
            # Workout data
            'last_workout_days_ago': np.random.randint(3, 21),
            'workout_rank_percentage': round(np.random.uniform(0.3, 0.9), 2),
            'bullet_workout': np.random.choice([0, 1], p=[0.8, 0.2]),
            
            # Pedigree factors
            'sire_stakes_win_percentage': round(np.random.uniform(0.05, 0.15), 3),
            'dam_producer_rating': np.random.randint(1, 10),
            'mud_pedigree': round(np.random.uniform(0.5, 1.0), 2),
            
            # Pace/style
            'running_style_early': np.random.choice([0, 1], p=[0.7, 0.3]),
            'early_speed_points': np.random.randint(70, 95),
            'late_speed_points': np.random.randint(70, 95),
            
            # Competition factors
            'field_size': np.random.randint(6, 12),
            'average_opponent_win_pct': round(np.random.uniform(0.1, 0.3), 3),
            'favorite_present': np.random.choice([0, 1], p=[0.3, 0.7]),
            
            # Target variables
            'finished_win': np.random.choice([0, 1], p=[0.85, 0.15]),
            'finished_in_the_money': np.random.choice([0, 1], p=[0.6, 0.4]),
            'finish_position': np.random.randint(1, 12),
            'beaten_lengths': round(np.random.uniform(0, 15), 1),
            'final_odds': round(np.random.uniform(1.5, 25.0), 2),
            'value_indicator': round(np.random.uniform(0.8, 1.3), 2)
        }
        
        return sample
    
    def _add_derived_features(self, df):
        """Add calculated/derived features"""
        
        # Form indicators
        df['form_rating'] = (df['last_race_speed_figure'] * 0.4 + 
                            df['win_percentage'] * 100 * 0.3 + 
                            (1 - df['days_since_last_race'] / 120) * 100 * 0.3)
        
        # Class drop/rise
        df['class_change'] = np.random.uniform(0.8, 1.2, len(df))
        
        # Speed figure trend
        df['speed_figure_trend'] = df['last_race_speed_figure'] / df.groupby(
            'sample_id')['last_race_speed_figure'].transform('mean').fillna(80)
        
        # Consistency metric
        df['consistency_score'] = 1 / (df['avg_finish_position'] * df['lifetime_starts'])
        
        # Power rating
        df['power_rating'] = (
            df['form_rating'] * 0.25 +
            df['trainer_win_percentage'] * 100 * 0.2 +
            df['jockey_win_percentage'] * 100 * 0.15 +
            df['early_speed_points'] * 0.2 +
            df['sire_stakes_win_percentage'] * 100 * 0.1 +
            df['workout_rank_percentage'] * 100 * 0.1
        )
        
        # Value metrics
        df['implied_probability'] = 1 / df['morning_line_odds']
        df['value_score'] = df['power_rating'] / 100 / df['implied_probability']
        df['is_value_bet'] = (df['value_score'] > 1.2).astype(int)
        
        # Race dynamics
        df['post_position_bias'] = np.where(
            df['post_position'] <= 3, 1.1,
            np.where(df['post_position'] >= 10, 0.9, 1.0)
        )
        
        # Experience factor
        df['experience_factor'] = np.log(df['lifetime_starts'] + 1)
        
        return df
    
    def load_and_preprocess_real_data(self):
        """Load real data from raw directories and preprocess"""
        # Load race cards
        race_cards_path = f"{self.raw_data_dir}/race_cards/race_cards_*.json"
        
        # Load results
        results_path = f"{self.raw_data_dir}/results/results_*.json"
        
        # Load form guides
        form_guides_path = f"{self.raw_data_dir}/form_guides/form_guides_*.json"
        
        # This would merge and process all data sources
        # For now, return generated data
        return self.generate_training_dataset(5000)
    
    def create_train_test_split(self, df, test_size=0.2):
        """Create training and test sets"""
        from sklearn.model_selection import train_test_split
        
        # Features and targets
        feature_columns = [col for col in df.columns if not col.startswith(('finished_', 'finish_', 'beaten_', 'final_', 'value_'))]
        target_columns = ['finished_win', 'finished_in_the_money', 'finish_position']
        
        X = df[feature_columns].copy()
        y = df[target_columns].copy()
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y['finished_win']
        )
        
        # Save splits
        X_train.to_csv(f"{self.processed_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{self.processed_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{self.processed_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{self.processed_dir}/y_test.csv", index=False)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    generator = TrainingDataGenerator()
    df = generator.generate_training_dataset(10000)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = generator.create_train_test_split(df)
    
    print("Training data generation complete!")

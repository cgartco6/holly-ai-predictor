import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
import json
import os
from datetime import datetime, timedelta

class TestDataSplitter:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def create_time_based_splits(self, df, date_column='race_date', test_size=0.2):
        """Create time-based train/test splits"""
        
        # Ensure date column exists
        if date_column not in df.columns:
            df[date_column] = pd.date_range(
                start='2023-01-01',
                periods=len(df),
                freq='D'
            )
        
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Determine split index
        split_idx = int(len(df) * (1 - test_size))
        
        # Create splits
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Save splits
        train_df.to_csv(f"{self.data_dir}/time_based_train.csv", index=False)
        test_df.to_csv(f"{self.data_dir}/time_based_test.csv", index=False)
        
        print(f"Time-based split:")
        print(f"  Training: {len(train_df)} samples ({train_df[date_column].min()} to {train_df[date_column].max()})")
        print(f"  Test: {len(test_df)} samples ({test_df[date_column].min()} to {test_df[date_column].max()})")
        
        return train_df, test_df
    
    def create_stratified_splits(self, df, target_column='finished_win', test_size=0.2):
        """Create stratified train/test splits"""
        
        if target_column not in df.columns:
            df[target_column] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        
        # Create stratified split
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df[target_column],
            random_state=42
        )
        
        # Save splits
        train_df.to_csv(f"{self.data_dir}/stratified_train.csv", index=False)
        test_df.to_csv(f"{self.data_dir}/stratified_test.csv", index=False)
        
        print(f"Stratified split:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        print(f"  Target distribution - Train: {train_df[target_column].mean():.3f}, Test: {test_df[target_column].mean():.3f}")
        
        return train_df, test_df
    
    def create_cross_validation_folds(self, df, n_splits=5, target_column='finished_win'):
        """Create cross-validation folds"""
        
        if target_column not in df.columns:
            df[target_column] = np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        folds = []
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(df, df[target_column])):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            # Save each fold
            train_fold.to_csv(f"{self.data_dir}/fold_{fold_num}_train.csv", index=False)
            val_fold.to_csv(f"{self.data_dir}/fold_{fold_num}_val.csv", index=False)
            
            fold_info = {
                'fold': fold_num,
                'train_size': len(train_fold),
                'val_size': len(val_fold),
                'train_positive_ratio': train_fold[target_column].mean(),
                'val_positive_ratio': val_fold[target_column].mean()
            }
            folds.append(fold_info)
            
            print(f"Fold {fold_num}: Train={len(train_fold)}, Val={len(val_fold)}, "
                  f"Target ratio: Train={fold_info['train_positive_ratio']:.3f}, "
                  f"Val={fold_info['val_positive_ratio']:.3f}")
        
        # Save fold information
        with open(f"{self.data_dir}/cv_folds_info.json", 'w') as f:
            json.dump(folds, f, indent=2)
        
        return folds
    
    def create_time_series_cv_folds(self, df, date_column='race_date', n_splits=5):
        """Create time-series cross-validation folds"""
        
        if date_column not in df.columns:
            df[date_column] = pd.date_range(
                start='2023-01-01',
                periods=len(df),
                freq='D'
            )
        
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        folds = []
        for fold_num, (train_idx, val_idx) in enumerate(tscv.split(df)):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            # Save each fold
            train_fold.to_csv(f"{self.data_dir}/ts_fold_{fold_num}_train.csv", index=False)
            val_fold.to_csv(f"{self.data_dir}/ts_fold_{fold_num}_val.csv", index=False)
            
            fold_info = {
                'fold': fold_num,
                'train_dates': f"{train_fold[date_column].min()} to {train_fold[date_column].max()}",
                'val_dates': f"{val_fold[date_column].min()} to {val_fold[date_column].max()}",
                'train_size': len(train_fold),
                'val_size': len(val_fold)
            }
            folds.append(fold_info)
            
            print(f"Time Series Fold {fold_num}:")
            print(f"  Train: {fold_info['train_dates']} ({len(train_fold)} samples)")
            print(f"  Val: {fold_info['val_dates']} ({len(val_fold)} samples)")
        
        # Save fold information
        with open(f"{self.data_dir}/ts_cv_folds_info.json", 'w') as f:
            json.dump(folds, f, indent=2)
        
        return folds
    
    def create_validation_sets(self, df, validation_size=0.15, test_size=0.15):
        """Create train/validation/test sets"""
        
        # First split: train + temp, test
        train_temp_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=df['finished_win'] if 'finished_win' in df.columns else None
        )
        
        # Second split: train, validation
        train_df, val_df = train_test_split(
            train_temp_df,
            test_size=validation_size/(1-test_size),
            random_state=42,
            stratify=train_temp_df['finished_win'] if 'finished_win' in train_temp_df.columns else None
        )
        
        # Save all sets
        train_df.to_csv(f"{self.data_dir}/final_train.csv", index=False)
        val_df.to_csv(f"{self.data_dir}/final_validation.csv", index=False)
        test_df.to_csv(f"{self.data_dir}/final_test.csv", index=False)
        
        # Create sample submission/test format
        submission_df = test_df[['sample_id']].copy()
        if 'finished_win' in test_df.columns:
            submission_df['prediction'] = np.random.random(len(submission_df))
        submission_df.to_csv(f"{self.data_dir}/sample_submission.csv", index=False)
        
        # Create metadata
        metadata = {
            'split_date': datetime.now().isoformat(),
            'total_samples': len(df),
            'train_samples': len(train_df),
            'validation_samples': len(val_df),
            'test_samples': len(test_df),
            'train_percentage': len(train_df)/len(df),
            'validation_percentage': len(val_df)/len(df),
            'test_percentage': len(test_df)/len(df)
        }
        
        if 'finished_win' in df.columns:
            metadata.update({
                'train_win_rate': train_df['finished_win'].mean(),
                'val_win_rate': val_df['finished_win'].mean(),
                'test_win_rate': test_df['finished_win'].mean()
            })
        
        with open(f"{self.data_dir}/split_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Final splits created:")
        print(f"  Train: {len(train_df)} samples ({metadata.get('train_win_rate', 0):.3f} win rate)")
        print(f"  Validation: {len(val_df)} samples ({metadata.get('val_win_rate', 0):.3f} win rate)")
        print(f"  Test: {len(test_df)} samples ({metadata.get('test_win_rate', 0):.3f} win rate)")
        
        return train_df, val_df, test_df

if __name__ == "__main__":
    from training_data_generator import TrainingDataGenerator
    
    # Generate sample data
    generator = TrainingDataGenerator()
    df = generator.generate_training_dataset(5000)
    
    # Create splits
    splitter = TestDataSplitter()
    
    print("Creating time-based splits...")
    train_time, test_time = splitter.create_time_based_splits(df)
    
    print("\nCreating stratified splits...")
    train_strat, test_strat = splitter.create_stratified_splits(df)
    
    print("\nCreating cross-validation folds...")
    folds = splitter.create_cross_validation_folds(df, n_splits=5)
    
    print("\nCreating time-series CV folds...")
    ts_folds = splitter.create_time_series_cv_folds(df, n_splits=5)
    
    print("\nCreating final train/validation/test splits...")
    train_final, val_final, test_final = splitter.create_validation_sets(df)
    
    print("\nAll test sets created successfully!")

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG
from src.utils.logger import logger

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_selector = None
        self.label_encoders = {}
        
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for horse racing prediction."""
        
        # Basic features already created by DataCleaner
        # Add advanced features
        
        # 1. Temporal features
        df = self._create_temporal_features(df)
        
        # 2. Interaction features
        df = self._create_interaction_features(df)
        
        # 3. Statistical features
        df = self._create_statistical_features(df)
        
        # 4. Trend features
        df = self._create_trend_features(df)
        
        # 5. Competition features
        df = self._create_competition_features(df)
        
        # 6. Momentum features
        df = self._create_momentum_features(df)
        
        # 7. Consistency features
        df = self._create_consistency_features(df)
        
        # 8. Specialization features
        df = self._create_specialization_features(df)
        
        # 9. Market features
        df = self._create_market_features(df)
        
        # 10. Environmental features
        df = self._create_environmental_features(df)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal-based features."""
        
        if 'race_date' in df.columns:
            df['race_date'] = pd.to_datetime(df['race_date'])
            
            # Seasonality features
            df['day_of_year'] = df['race_date'].dt.dayofyear
            df['week_of_year'] = df['race_date'].dt.isocalendar().week
            df['quarter'] = df['race_date'].dt.quarter
            
            # Cyclical encoding for temporal features
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
            
            df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
            df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
            
            # Time since last win
            if 'horse_id' in df.columns and 'is_winner' in df.columns:
                df['days_since_last_win'] = self._calculate_days_since_last_win(df)
        
        return df
    
    def _calculate_days_since_last_win(self, df: pd.DataFrame) -> pd.Series:
        """Calculate days since last win for each horse."""
        result = pd.Series(0, index=df.index)
        
        for horse_id in df['horse_id'].unique():
            horse_mask = df['horse_id'] == horse_id
            horse_df = df[horse_mask].sort_values('race_date')
            
            days_since = []
            last_win_date = None
            
            for idx, row in horse_df.iterrows():
                if last_win_date is None:
                    days_since.append(365)  # Default if no previous win
                else:
                    days_diff = (row['race_date'] - last_win_date).days
                    days_since.append(min(days_diff, 365))
                
                if row['is_winner'] == 1:
                    last_win_date = row['race_date']
            
            result.loc[horse_mask] = days_since
        
        return result
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables."""
        
        # Jockey-Trainer combination
        if 'jockey_name' in df.columns and 'trainer_name' in df.columns:
            df['jockey_trainer_combo'] = df['jockey_name'] + '_' + df['trainer_name']
            
            # Calculate combo success rate
            combo_stats = df.groupby('jockey_trainer_combo').agg({
                'is_winner': ['mean', 'count']
            }).reset_index()
            
            combo_stats.columns = ['jockey_trainer_combo', 'combo_win_rate', 'combo_attempts']
            df = df.merge(combo_stats, on='jockey_trainer_combo', how='left')
        
        # Weight-Distance interaction
        if 'weight' in df.columns and 'distance' in df.columns:
            df['weight_per_distance'] = df['weight'] / df['distance']
        
        # Age-Experience interaction
        if 'age' in df.columns and 'career_starts' in df.columns:
            df['experience_per_age'] = df['career_starts'] / df['age']
        
        # Form-Going interaction
        if 'recent_form_score' in df.columns and 'going_score' in df.columns:
            df['form_going_interaction'] = df['recent_form_score'] * df['going_score']
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        
        # Z-scores within race
        if 'form_rating' in df.columns:
            df['form_rating_zscore'] = df.groupby('race_id')['form_rating'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        
        if 'official_rating' in df.columns:
            df['official_rating_zscore'] = df.groupby('race_id')['official_rating'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        
        # Percentile ranks
        if 'career_win_pct' in df.columns:
            df['win_pct_percentile'] = df.groupby('race_id')['career_win_pct'].rank(pct=True)
        
        # Moving averages
        if 'horse_id' in df.columns and 'form_rating' in df.columns:
            df['form_rating_ma'] = df.groupby('horse_id')['form_rating'].transform(
                lambda x: x.rolling(5, min_periods=1).mean()
            )
        
        return df
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend features."""
        
        if 'horse_id' in df.columns and 'form_rating' in df.columns:
            # Linear trend of last 3 ratings
            df['rating_trend'] = df.groupby('horse_id').apply(
                lambda x: self._calculate_linear_trend(x['form_rating'].tail(3))
            ).reset_index(level=0, drop=True)
        
        # Improvement/decline indicators
        if 'recent_form_score' in df.columns:
            df['form_momentum'] = df.groupby('horse_id')['recent_form_score'].transform(
                lambda x: x.diff().rolling(3, min_periods=1).mean()
            )
        
        return df
    
    def _calculate_linear_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of a series."""
        if len(series) < 2:
            return 0
        
        x = np.arange(len(series))
        y = series.values
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def _create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features about competition strength."""
        
        if 'race_id' in df.columns and 'form_rating' in df.columns:
            # Competition strength
            df['competition_strength'] = df.groupby('race_id')['form_rating'].transform('std')
            
            # Relative strength
            df['relative_strength'] = df.groupby('race_id').apply(
                lambda x: x['form_rating'] / x['form_rating'].max() if x['form_rating'].max() > 0 else 0
            ).reset_index(level=0, drop=True)
        
        # Field size
        df['field_size'] = df.groupby('race_id')['horse_id'].transform('count')
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        
        # Winning streak
        if 'horse_id' in df.columns and 'is_winner' in df.columns:
            df['winning_streak'] = df.groupby('horse_id')['is_winner'].transform(
                lambda x: x.rolling(window=3, min_periods=1).sum()
            )
        
        # Place streak (top 3 finishes)
        if 'position' in df.columns:
            df['placed'] = df['position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
            df['place_streak'] = df.groupby('horse_id')['placed'].transform(
                lambda x: x.rolling(window=5, min_periods=1).sum()
            )
        
        # Recent improvement
        if 'form_rating' in df.columns:
            df['rating_improvement'] = df.groupby('horse_id')['form_rating'].transform(
                lambda x: x.pct_change().rolling(3, min_periods=1).mean()
            )
        
        return df
    
    def _create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create consistency features."""
        
        # Position consistency
        if 'position' in df.columns and 'horse_id' in df.columns:
            df['position_consistency'] = df.groupby('horse_id')['position'].transform('std').fillna(0)
        
        # Rating consistency over time
        if 'form_rating' in df.columns and 'horse_id' in df.columns:
            df['rating_std_5race'] = df.groupby('horse_id')['form_rating'].transform(
                lambda x: x.rolling(5, min_periods=1).std()
            ).fillna(0)
        
        # Win interval consistency
        if 'horse_id' in df.columns and 'is_winner' in df.columns:
            df['win_interval_cv'] = self._calculate_win_interval_cv(df)
        
        return df
    
    def _calculate_win_interval_cv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate coefficient of variation of win intervals."""
        result = pd.Series(0.0, index=df.index)
        
        for horse_id in df['horse_id'].unique():
            horse_mask = df['horse_id'] == horse_id
            horse_df = df[horse_mask].sort_values('race_date')
            
            # Find win dates
            win_dates = horse_df[horse_df['is_winner'] == 1]['race_date']
            
            if len(win_dates) >= 2:
                intervals = np.diff(sorted(win_dates)).astype('timedelta64[D]').astype(int)
                cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
            else:
                cv = 0
            
            result.loc[horse_mask] = cv
        
        return result
    
    def _create_specialization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create specialization features."""
        
        # Distance specialization
        if 'distance' in df.columns and 'horse_id' in df.columns:
            df['preferred_distance'] = df.groupby('horse_id')['distance'].transform(lambda x: x.mode()[0] if not x.mode().empty else 0)
            df['distance_specialization'] = np.abs(df['distance'] - df['preferred_distance'])
        
        # Going specialization
        if 'going_score' in df.columns and 'horse_id' in df.columns:
            df['preferred_going'] = df.groupby('horse_id')['going_score'].transform('mean')
            df['going_specialization'] = np.abs(df['going_score'] - df['preferred_going'])
        
        # Track specialization
        if 'meeting_name' in df.columns and 'horse_id' in df.columns:
            track_success = df.groupby(['horse_id', 'meeting_name'])['is_winner'].mean().reset_index()
            track_success.columns = ['horse_id', 'meeting_name', 'track_success_rate']
            df = df.merge(track_success, on=['horse_id', 'meeting_name'], how='left')
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-based features."""
        
        if 'odds' in df.columns:
            # Implied probability
            df['implied_probability'] = 1 / df['odds']
            
            # Market efficiency
            df['market_efficiency'] = df.groupby('race_id')['implied_probability'].transform('sum')
            
            # Value indicator
            df['value_indicator'] = df['predicted_probability'] - df['implied_probability'] if 'predicted_probability' in df.columns else 0
            
            # Odds movement (would need historical odds data)
        
        # Favorite status
        if 'odds' in df.columns and 'race_id' in df.columns:
            df['is_favorite'] = df.groupby('race_id')['odds'].transform(
                lambda x: x == x.min()
            ).astype(int)
        
        return df
    
    def _create_environmental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create environmental features."""
        
        # Weather impact (simulated - would need actual weather data)
        if 'going_score' in df.columns:
            df['weather_impact'] = np.where(
                df['going_score'] < 0.6, 0.8,  # Heavy going
                np.where(df['going_score'] < 0.7, 0.9,  # Soft going
                np.where(df['going_score'] > 0.9, 1.1,  # Firm going
                1.0))  # Good going
            )
        
        # Time of day impact
        if 'time_of_day' in df.columns:
            df['time_impact'] = np.where(
                df['time_of_day'] < 12, 0.95,  # Morning
                np.where(df['time_of_day'] < 16, 1.0,  # Afternoon
                1.05)  # Evening
            )
        
        # Season impact
        if 'month' in df.columns:
            df['season_impact'] = np.where(
                df['month'].isin([12, 1, 2]), 1.05,  # Summer
                np.where(df['month'].isin([3, 4, 5]), 1.0,  # Autumn
                np.where(df['month'].isin([6, 7, 8]), 0.95,  # Winter
                1.0))  # Spring
            )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Scale features using StandardScaler."""
        # Fit on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return X_train_scaled, X_test_scaled
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> pd.DataFrame:
        """Select top k features using mutual information."""
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def apply_pca(self, X: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction."""
        self.pca = PCA(n_components=n_components, random_state=MODEL_CONFIG.RANDOM_STATE)
        X_pca = self.pca.fit_transform(X)
        
        # Create column names
        pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        return pd.DataFrame(X_pca, columns=pca_cols, index=X.index)
    
    def save_transformations(self, path: str):
        """Save transformation objects."""
        transformations = {
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_selector': self.feature_selector,
            'label_encoders': self.label_encoders
        }
        joblib.dump(transformations, path)
        logger.info(f"Transformations saved to {path}")
    
    def load_transformations(self, path: str):
        """Load transformation objects."""
        transformations = joblib.load(path)
        self.scaler = transformations['scaler']
        self.pca = transformations['pca']
        self.feature_selector = transformations['feature_selector']
        self.label_encoders = transformations['label_encoders']
        logger.info(f"Transformations loaded from {path}")
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Get feature importance scores."""
        # Use mutual information
        mi_scores = mutual_info_classif(X, y, random_state=MODEL_CONFIG.RANDOM_STATE)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_lagged_features(self, df: pd.DataFrame, horse_id_col: str, 
                              value_col: str, lags: List[int]) -> pd.DataFrame:
        """Create lagged features for time series data."""
        df_lagged = df.copy()
        
        for lag in lags:
            df_lagged[f'{value_col}_lag{lag}'] = df.groupby(horse_id_col)[value_col].shift(lag)
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame, horse_id_col: str,
                               value_col: str, windows: List[int]) -> pd.DataFrame:
        """Create rolling window features."""
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'{value_col}_rolling_mean_{window}'] = df.groupby(horse_id_col)[value_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df_rolling[f'{value_col}_rolling_std_{window}'] = df.groupby(horse_id_col)[value_col].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
        
        return df_rolling

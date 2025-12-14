import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import json
import os

class FeatureEngineering:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_sets = {}
        
    def create_feature_sets(self, df):
        """Create multiple feature sets for different models"""
        
        # Basic features set
        basic_features = self._create_basic_features(df)
        self.feature_sets['basic'] = basic_features
        
        # Advanced features set
        advanced_features = self._create_advanced_features(df)
        self.feature_sets['advanced'] = advanced_features
        
        # Expert features set
        expert_features = self._create_expert_features(df)
        self.feature_sets['expert'] = expert_features
        
        # Ensemble features set (combination)
        ensemble_features = self._create_ensemble_features(basic_features, advanced_features, expert_features)
        self.feature_sets['ensemble'] = ensemble_features
        
        # Save feature sets
        self._save_feature_sets()
        
        return self.feature_sets
    
    def _create_basic_features(self, df):
        """Create basic feature set"""
        features = [
            'horse_age',
            'days_since_last_race',
            'lifetime_starts',
            'win_percentage',
            'in_the_money_percentage',
            'last_race_finish',
            'last_race_speed_figure',
            'trainer_win_percentage',
            'jockey_win_percentage',
            'post_position',
            'morning_line_odds',
            'weight_carried',
            'distance',
            'field_size',
            'form_rating'
        ]
        
        # Select only existing columns
        existing_features = [f for f in features if f in df.columns]
        basic_df = df[existing_features].copy()
        
        # Fill missing values
        basic_df = basic_df.fillna(basic_df.median())
        
        # Scale features
        basic_scaled = pd.DataFrame(
            self.scaler.fit_transform(basic_df),
            columns=basic_df.columns,
            index=basic_df.index
        )
        
        return basic_scaled
    
    def _create_advanced_features(self, df):
        """Create advanced feature set"""
        advanced_features = []
        
        # Form and consistency features
        if 'form_rating' in df.columns:
            advanced_features.append('form_rating')
        
        if 'consistency_score' in df.columns:
            advanced_features.append('consistency_score')
        
        if 'speed_figure_trend' in df.columns:
            advanced_features.append('speed_figure_trend')
        
        # Pace and style features
        pace_features = [col for col in df.columns if 'speed' in col.lower() or 'pace' in col.lower()]
        advanced_features.extend(pace_features)
        
        # Trainer/Jockey combo features
        combo_features = [col for col in df.columns if 'combo' in col.lower()]
        advanced_features.extend(combo_features)
        
        # Workout features
        workout_features = [col for col in df.columns if 'workout' in col.lower()]
        advanced_features.extend(workout_features)
        
        # Pedigree features
        pedigree_features = [col for col in df.columns if 'pedigree' in col.lower() or 'sire' in col.lower()]
        advanced_features.extend(pedigree_features)
        
        # Select existing features
        existing_features = [f for f in advanced_features if f in df.columns]
        advanced_df = df[existing_features].copy()
        
        # Fill and scale
        advanced_df = advanced_df.fillna(advanced_df.median())
        advanced_scaled = pd.DataFrame(
            self.scaler.fit_transform(advanced_df),
            columns=advanced_df.columns,
            index=advanced_df.index
        )
        
        return advanced_scaled
    
    def _create_expert_features(self, df):
        """Create expert feature set with engineered features"""
        expert_df = pd.DataFrame(index=df.index)
        
        # Interaction features
        if all(col in df.columns for col in ['form_rating', 'trainer_win_percentage']):
            expert_df['form_trainer_interaction'] = df['form_rating'] * df['trainer_win_percentage'] * 100
        
        if all(col in df.columns for col in ['jockey_win_percentage', 'post_position']):
            expert_df['jockey_post_interaction'] = df['jockey_win_percentage'] * (1 / df['post_position'])
        
        # Ratio features
        if all(col in df.columns for col in ['last_race_speed_figure', 'days_since_last_race']):
            expert_df['speed_per_day'] = df['last_race_speed_figure'] / (df['days_since_last_race'] + 1)
        
        if all(col in df.columns for col in ['weight_carried', 'horse_age']):
            expert_df['weight_age_ratio'] = df['weight_carried'] / (df['horse_age'] * 100)
        
        # Trend features
        if 'power_rating' in df.columns and 'morning_line_odds' in df.columns:
            expert_df['value_power_ratio'] = df['power_rating'] / df['morning_line_odds']
        
        # Polynomial features
        if 'form_rating' in df.columns:
            expert_df['form_rating_squared'] = df['form_rating'] ** 2
            expert_df['form_rating_cubed'] = df['form_rating'] ** 3
        
        # Distance suitability
        if all(col in df.columns for col in ['distance', 'distance_preference']):
            expert_df['distance_suitability'] = 1 / (1 + abs(df['distance'] - df['distance_preference'] * 6))
        
        # Class consistency
        if all(col in df.columns for col in ['last_race_class_level', 'win_percentage']):
            expert_df['class_consistency'] = df['win_percentage'] * (5 - df['last_race_class_level'])
        
        # Fill and scale
        expert_df = expert_df.fillna(expert_df.median())
        expert_scaled = pd.DataFrame(
            self.scaler.fit_transform(expert_df),
            columns=expert_df.columns,
            index=expert_df.index
        )
        
        return expert_scaled
    
    def _create_ensemble_features(self, basic, advanced, expert):
        """Combine all feature sets"""
        ensemble_df = pd.concat([basic, advanced, expert], axis=1)
        
        # Remove duplicate columns
        ensemble_df = ensemble_df.loc[:, ~ensemble_df.columns.duplicated()]
        
        return ensemble_df
    
    def select_best_features(self, X, y, k=50):
        """Select best features using various methods"""
        
        # ANOVA F-value
        selector_anova = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_anova = selector_anova.fit_transform(X, y)
        anova_features = X.columns[selector_anova.get_support()].tolist()
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns)
        rf_features = importances.nlargest(k).index.tolist()
        
        # Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50), n_features_to_select=k)
        rfe.fit(X, y)
        rfe_features = X.columns[rfe.support_].tolist()
        
        # Combine feature selection results
        all_selected = list(set(anova_features + rf_features + rfe_features))
        
        feature_selection_results = {
            'anova_features': anova_features,
            'rf_features': rf_features,
            'rfe_features': rfe_features,
            'all_selected_features': all_selected,
            'feature_importances': importances.to_dict()
        }
        
        # Save results
        with open(f"{self.data_dir}/feature_selection_results.json", 'w') as f:
            json.dump(feature_selection_results, f, indent=2, default=str)
        
        return all_selected
    
    def _save_feature_sets(self):
        """Save feature sets to files"""
        for set_name, features in self.feature_sets.items():
            # Save as CSV
            features.to_csv(f"{self.data_dir}/{set_name}_features.csv")
            
            # Save metadata
            metadata = {
                'feature_set': set_name,
                'num_features': len(features.columns),
                'features': features.columns.tolist(),
                'creation_date': pd.Timestamp.now().isoformat()
            }
            
            with open(f"{self.data_dir}/{set_name}_features_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(self.feature_sets)} feature sets")

if __name__ == "__main__":
    # Example usage
    from training_data_generator import TrainingDataGenerator
    
    generator = TrainingDataGenerator()
    df = generator.generate_training_dataset(1000)
    
    feature_engineer = FeatureEngineering()
    feature_sets = feature_engineer.create_feature_sets(df)
    
    print("Feature engineering complete!")
    for set_name, features in feature_sets.items():
        print(f"{set_name}: {features.shape[1]} features")

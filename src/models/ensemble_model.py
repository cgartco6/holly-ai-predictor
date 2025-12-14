import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import joblib
from collections import defaultdict

# Ensemble methods
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV

# Bayesian optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from config import MODEL_CONFIG
from src.utils.logger import logger

class AdvancedEnsemble:
    def __init__(self):
        self.ensemble = None
        self.meta_model = None
        self.base_models = {}
        self.feature_importances = {}
        
    def create_advanced_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create advanced ensemble using multiple techniques."""
        logger.info("Creating advanced ensemble model...")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            stratify=y
        )
        
        # Step 1: Create diverse base models
        base_models = self._create_diverse_base_models()
        
        # Step 2: Train base models
        self._train_base_models(base_models, X_train, y_train)
        
        # Step 3: Create meta-features
        meta_features = self._create_meta_features(base_models, X_val)
        
        # Step 4: Train meta-model
        meta_model = self._train_meta_model(meta_features, y_val)
        
        # Step 5: Create stacking ensemble
        ensemble = self._create_stacking_ensemble(base_models, meta_model)
        
        self.ensemble = ensemble
        self.meta_model = meta_model
        self.base_models = base_models
        
        # Evaluate ensemble
        ensemble_score = ensemble.score(X_val, y_val)
        logger.info(f"Advanced ensemble created with validation score: {ensemble_score:.4f}")
        
        return ensemble
    
    def _create_diverse_base_models(self) -> Dict[str, Any]:
        """Create a diverse set of base models."""
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        
        models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                use_label_encoder=False
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                random_seed=MODEL_CONFIG.RANDOM_STATE,
                verbose=0
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                n_jobs=-1
            ),
            'logistic': LogisticRegression(
                C=1.0,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                max_iter=1000
            ),
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=MODEL_CONFIG.RANDOM_STATE
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                random_state=MODEL_CONFIG.RANDOM_STATE,
                max_iter=1000
            )
        }
        
        return models
    
    def _train_base_models(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series):
        """Train all base models."""
        logger.info("Training base models...")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y)
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[name] = model.feature_importances_
    
    def _create_meta_features(self, models: Dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
        """Create meta-features from base model predictions."""
        meta_features = []
        
        for name, model in models.items():
            try:
                # Get predicted probabilities
                y_pred_proba = model.predict_proba(X)[:, 1]
                meta_features.append(y_pred_proba)
            except Exception as e:
                logger.warning(f"Could not get predictions from {name}: {e}")
                # Use zeros as fallback
                meta_features.append(np.zeros(len(X)))
        
        # Create DataFrame
        meta_df = pd.DataFrame(
            np.column_stack(meta_features),
            columns=[f'meta_{name}' for name in models.keys()],
            index=X.index
        )
        
        # Add original features (optional)
        # meta_df = pd.concat([meta_df, X], axis=1)
        
        return meta_df
    
    def _train_meta_model(self, X_meta: pd.DataFrame, y: pd.Series) -> Any:
        """Train meta-model on meta-features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier
        
        # Use simple model as meta-learner
        meta_model = LogisticRegression(
            C=0.1,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            max_iter=1000
        )
        
        meta_model.fit(X_meta, y)
        
        # Evaluate meta-model
        score = meta_model.score(X_meta, y)
        logger.info(f"Meta-model trained with accuracy: {score:.4f}")
        
        return meta_model
    
    def _create_stacking_ensemble(self, base_models: Dict[str, Any], meta_model: Any) -> Any:
        """Create stacking ensemble."""
        # Convert dict to list of tuples for StackingClassifier
        estimators = [(name, model) for name, model in base_models.items()]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_bayesian_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create ensemble using Bayesian optimization."""
        logger.info("Creating Bayesian optimized ensemble...")
        
        import xgboost as xgb
        from sklearn.model_selection import StratifiedKFold
        
        # Define search spaces for different models
        search_spaces = [
            {
                'model': xgb.XGBClassifier(use_label_encoder=False),
                'params': {
                    'n_estimators': Integer(100, 500),
                    'max_depth': Integer(3, 10),
                    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                    'subsample': Real(0.5, 1.0),
                    'colsample_bytree': Real(0.5, 1.0)
                }
            }
        ]
        
        # Perform Bayesian optimization
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=MODEL_CONFIG.RANDOM_STATE)
        
        optimized_models = []
        
        for space in search_spaces:
            opt = BayesSearchCV(
                estimator=space['model'],
                search_spaces=space['params'],
                n_iter=50,
                cv=cv,
                scoring='roc_auc',
                random_state=MODEL_CONFIG.RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            
            opt.fit(X, y)
            optimized_models.append(opt.best_estimator_)
            
            logger.info(f"Best score for {space['model'].__class__.__name__}: {opt.best_score_:.4f}")
        
        # Create voting ensemble with optimized models
        from sklearn.ensemble import VotingClassifier
        
        voting_ensemble = VotingClassifier(
            estimators=[(f'opt_{i}', model) for i, model in enumerate(optimized_models)],
            voting='soft',
            weights=[1] * len(optimized_models)
        )
        
        voting_ensemble.fit(X, y)
        
        self.ensemble = voting_ensemble
        
        return voting_ensemble
    
    def create_dynamic_weighted_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Create ensemble with dynamically weighted models."""
        logger.info("Creating dynamically weighted ensemble...")
        
        from sklearn.model_selection import cross_val_score
        
        # Get base models
        base_models = self._create_diverse_base_models()
        
        # Calculate weights based on cross-validation performance
        weights = {}
        
        for name, model in base_models.items():
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                weights[name] = scores.mean()
                logger.info(f"{name} CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
                weights[name] = 0.5
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1/len(weights) for k in weights.keys()}
        
        logger.info(f"Model weights: {weights}")
        
        # Create weighted voting ensemble
        from sklearn.ensemble import VotingClassifier
        
        # Filter out models with very low weight
        filtered_models = {
            name: model for name, model in base_models.items()
            if weights[name] > 0.05  # Minimum weight threshold
        }
        
        filtered_weights = [weights[name] for name in filtered_models.keys()]
        
        weighted_ensemble = VotingClassifier(
            estimators=list(filtered_models.items()),
            voting='soft',
            weights=filtered_weights
        )
        
        weighted_ensemble.fit(X, y)
        
        self.ensemble = weighted_ensemble
        
        return weighted_ensemble
    
    def create_time_series_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                   time_column: str = 'race_date') -> Any:
        """Create ensemble that accounts for time series nature of data."""
        logger.info("Creating time-series aware ensemble...")
        
        # Sort by time
        if time_column in X.columns:
            time_series = X[time_column]
            X_sorted = X.drop(columns=[time_column])
            
            # Create time-based folds
            from sklearn.model_selection import TimeSeriesSplit
            
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Train models with time series cross-validation
            models = []
            
            for train_idx, val_idx in tscv.split(X_sorted):
                X_train, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train a model on this fold
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=MODEL_CONFIG.RANDOM_STATE,
                    use_label_encoder=False
                )
                model.fit(X_train, y_train)
                models.append(model)
            
            # Create ensemble of time-based models
            from sklearn.ensemble import VotingClassifier
            
            time_ensemble = VotingClassifier(
                estimators=[(f'ts_model_{i}', model) for i, model in enumerate(models)],
                voting='soft'
            )
            
            # Fit on entire dataset
            time_ensemble.fit(X_sorted, y)
            
            self.ensemble = time_ensemble
            
            return time_ensemble
        
        else:
            logger.warning("Time column not found, using regular ensemble")
            return self.create_dynamic_weighted_ensemble(X, y)
    
    def get_model_contributions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get contribution of each base model to final predictions."""
        if not self.base_models or not self.meta_model:
            logger.error("Ensemble not trained or base models not available")
            return {}
        
        contributions = {}
        
        # Get predictions from each base model
        for name, model in self.base_models.items():
            try:
                contributions[name] = model.predict_proba(X)[:, 1]
            except:
                contributions[name] = np.zeros(len(X))
        
        # Get meta-model coefficients
        if hasattr(self.meta_model, 'coef_'):
            coefficients = self.meta_model.coef_[0]
            
            # Weight contributions by coefficients
            for i, (name, contribution) in enumerate(contributions.items()):
                if i < len(coefficients):
                    contributions[name] = contributions[name] * coefficients[i]
        
        return contributions
    
    def save_ensemble(self, path: str):
        """Save ensemble model and components."""
        ensemble_data = {
            'ensemble': self.ensemble,
            'meta_model': self.meta_model,
            'base_models': self.base_models,
            'feature_importances': self.feature_importances
        }
        
        joblib.dump(ensemble_data, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load_ensemble(self, path: str):
        """Load ensemble model and components."""
        ensemble_data = joblib.load(path)
        
        self.ensemble = ensemble_data['ensemble']
        self.meta_model = ensemble_data['meta_model']
        self.base_models = ensemble_data['base_models']
        self.feature_importances = ensemble_data.get('feature_importances', {})
        
        logger.info(f"Ensemble loaded from {path}")
        
        return self.ensemble
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if self.ensemble is None:
            raise ValueError("Ensemble not trained")
        
        # Get predictions from all base models
        all_predictions = []
        
        for name, model in self.base_models.items():
            try:
                preds = model.predict_proba(X)[:, 1]
                all_predictions.append(preds)
            except:
                logger.warning(f"Could not get predictions from {name}")
        
        if not all_predictions:
            raise ValueError("No base model predictions available")
        
        # Stack predictions
        predictions_stack = np.column_stack(all_predictions)
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(predictions_stack, axis=1)
        std_predictions = np.std(predictions_stack, axis=1)
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble.predict_proba(X)[:, 1]
        
        return ensemble_pred, std_predictions
    
    def adaptive_weighting(self, X: pd.DataFrame, recent_performance: Dict[str, float]) -> Any:
        """Adapt model weights based on recent performance."""
        logger.info("Adapting ensemble weights based on recent performance...")
        
        # Calculate new weights
        total_performance = sum(recent_performance.values())
        
        if total_performance > 0:
            weights = {k: v/total_performance for k, v in recent_performance.items()}
        else:
            weights = {k: 1/len(recent_performance) for k in recent_performance.keys()}
        
        # Create new ensemble with updated weights
        from sklearn.ensemble import VotingClassifier
        
        # Get models with non-zero weight
        active_models = {
            name: model for name, model in self.base_models.items()
            if name in weights and weights[name] > 0.01
        }
        
        active_weights = [weights[name] for name in active_models.keys()]
        
        adaptive_ensemble = VotingClassifier(
            estimators=list(active_models.items()),
            voting='soft',
            weights=active_weights
        )
        
        adaptive_ensemble.fit(X, y)  # Note: Would need y here
        
        self.ensemble = adaptive_ensemble
        
        logger.info(f"Updated ensemble weights: {weights}")
        
        return adaptive_ensemble

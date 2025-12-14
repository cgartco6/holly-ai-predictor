import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Advanced models
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from config import MODEL_CONFIG
from src.utils.logger import logger
from src.features.feature_engineering import FeatureEngineer
from src.utils.database import db

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.feature_engineer = FeatureEngineer()
        
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare training data from database."""
        logger.info("Preparing training data...")
        
        # Get data from database
        df = db.get_training_data(days_back=MODEL_CONFIG.FEATURE_WINDOW_DAYS)
        
        if df.empty:
            raise ValueError("No training data available")
        
        logger.info(f"Loaded {len(df)} records for training")
        
        # Clean and engineer features
        from src.scraper.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        df_clean = cleaner.clean_race_data(df)
        
        # Add comprehensive features
        df_features = self.feature_engineer.create_comprehensive_features(df_clean)
        
        # Prepare for training
        X, y, feature_cols = cleaner.prepare_training_data(df_features)
        
        # Handle class imbalance
        X, y = self._handle_class_imbalance(X, y)
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = self.feature_engineer.encode_categorical_features(X, categorical_cols)
        
        # Scale features
        X_scaled, _ = self.feature_engineer.scale_features(X)
        
        # Feature selection
        if len(feature_cols) > 50:
            X_selected = self.feature_engineer.select_features(X_scaled, y, k=50)
        else:
            X_selected = X_scaled
        
        logger.info(f"Training data prepared: {X_selected.shape[0]} samples, {X_selected.shape[1]} features")
        
        return X_selected, y, X_selected.columns.tolist()
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE."""
        from imblearn.over_sampling import SMOTE
        
        # Check imbalance
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        
        if imbalance_ratio < 0.3:  # Significant imbalance
            logger.info(f"Handling class imbalance (ratio: {imbalance_ratio:.3f})")
            
            smote = SMOTE(random_state=MODEL_CONFIG.RANDOM_STATE)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            logger.info(f"After SMOTE: {len(y_resampled)} samples ({y_resampled.mean():.3f} positive)")
            
            return X_resampled, y_resampled
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train multiple models and select the best one."""
        logger.info("Training multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG.TEST_SIZE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            stratify=y
        )
        
        # Define models to train
        models_to_train = {
            'xgboost': self._train_xgboost,
            'lightgbm': self._train_lightgbm,
            'catboost': self._train_catboost,
            'random_forest': self._train_random_forest,
            'gradient_boosting': self._train_gradient_boosting
        }
        
        results = {}
        
        for model_name, train_func in models_to_train.items():
            logger.info(f"Training {model_name}...")
            
            try:
                model, metrics = train_func(X_train, X_test, y_train, y_test)
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'feature_importance': self._get_feature_importance(model, X.columns)
                }
                
                logger.info(f"{model_name} trained - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        # Select best model based on AUC
        best_model_name = max(
            results.items(), 
            key=lambda x: x[1]['metrics']['auc']
        )[0]
        
        self.best_model = results[best_model_name]['model']
        logger.info(f"Best model: {best_model_name} (AUC: {results[best_model_name]['metrics']['auc']:.4f})")
        
        # Store all models
        self.models = results
        
        return results
    
    def _train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train XGBoost model."""
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Base model
        xgb_model = xgb.XGBClassifier(
            random_state=MODEL_CONFIG.RANDOM_STATE,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1
        )
        
        # Randomized search
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=MODEL_CONFIG.CV_FOLDS,
            scoring='roc_auc',
            random_state=MODEL_CONFIG.RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        best_model = random_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = random_search.best_params_
        
        return best_model, metrics
    
    def _train_lightgbm(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train LightGBM model."""
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': MODEL_CONFIG.RANDOM_STATE
        }
        
        # Train with early stopping
        model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Convert to sklearn interface for consistency
        sklearn_model = lgb.LGBMClassifier(**params)
        sklearn_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_pred_proba = model.predict(X_test)
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return sklearn_model, metrics
    
    def _train_catboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train CatBoost model."""
        # Identify categorical features
        cat_features = X_train.select_dtypes(include=['int64', 'uint8']).columns.tolist()
        
        model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=MODEL_CONFIG.RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=50
        )
        
        # Train
        model.fit(
            X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test)
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        return model, metrics
    
    def _train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train Random Forest model."""
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=MODEL_CONFIG.RANDOM_STATE),
            param_grid,
            cv=MODEL_CONFIG.CV_FOLDS,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['best_params'] = grid_search.best_params_
        
        return best_model, metrics
    
    def _train_gradient_boosting(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train Gradient Boosting model."""
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=MODEL_CONFIG.RANDOM_STATE
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=MODEL_CONFIG.CV_FOLDS,
            scoring='roc_auc'
        )
        
        # Train final model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        return model, metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            # Create dummy importances
            importances = np.ones(len(feature_names))
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train ensemble model combining multiple algorithms."""
        logger.info("Training ensemble model...")
        
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG.TEST_SIZE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            stratify=y
        )
        
        # Define base models
        estimators = [
            ('xgb', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=MODEL_CONFIG.RANDOM_STATE,
                use_label_encoder=False
            )),
            ('lgb', lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=MODEL_CONFIG.RANDOM_STATE
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                random_state=MODEL_CONFIG.RANDOM_STATE
            ))
        ]
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[2, 2, 1]  # Give more weight to boosting models
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        logger.info(f"Ensemble model trained - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
        
        # Store ensemble as best model
        self.best_model = ensemble
        self.models['ensemble'] = {
            'model': ensemble,
            'metrics': metrics
        }
        
        return ensemble
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'xgboost'):
        """Perform advanced hyperparameter tuning."""
        logger.info(f"Performing hyperparameter tuning for {model_type}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=MODEL_CONFIG.TEST_SIZE,
            random_state=MODEL_CONFIG.RANDOM_STATE,
            stratify=y
        )
        
        if model_type == 'xgboost':
            # Extensive parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0]
            }
            
            model = xgb.XGBClassifier(
                random_state=MODEL_CONFIG.RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        elif model_type == 'lightgbm':
            param_grid = {
                'num_leaves': [20, 31, 40, 50],
                'max_depth': [-1, 5, 10, 15],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300, 500],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0.5, 1.0, 1.5, 2.0]
            }
            
            model = lgb.LGBMClassifier(
                random_state=MODEL_CONFIG.RANDOM_STATE,
                verbose=-1
            )
        
        else:
            logger.error(f"Unsupported model type for tuning: {model_type}")
            return None
        
        # Bayesian optimization or random search
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=50,
            cv=MODEL_CONFIG.CV_FOLDS,
            scoring='roc_auc',
            random_state=MODEL_CONFIG.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        random_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Evaluate on test set
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        logger.info(f"Test set performance: AUC={metrics['auc']:.4f}")
        
        return best_model
    
    def save_models(self):
        """Save trained models to disk."""
        for model_name, model_data in self.models.items():
            if 'model' in model_data:
                model_path = MODEL_CONFIG.MODELS_DIR / f"{model_name}_model.joblib"
                joblib.dump(model_data['model'], model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save feature transformations
        self.feature_engineer.save_transformations(MODEL_CONFIG.ENCODER_PATH)
        
        # Save best model separately
        if self.best_model:
            joblib.dump(self.best_model, MODEL_CONFIG.ENSEMBLE_PATH)
            logger.info(f"Saved best model to {MODEL_CONFIG.ENSEMBLE_PATH}")
    
    def load_models(self):
        """Load trained models from disk."""
        try:
            # Load ensemble model
            self.best_model = joblib.load(MODEL_CONFIG.ENSEMBLE_PATH)
            logger.info(f"Loaded best model from {MODEL_CONFIG.ENSEMBLE_PATH}")
            
            # Load feature transformations
            self.feature_engineer.load_transformations(MODEL_CONFIG.ENCODER_PATH)
            
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def continuous_learning(self, new_data: pd.DataFrame):
        """Update models with new data (continuous learning)."""
        logger.info("Performing continuous learning with new data...")
        
        if new_data.empty:
            logger.warning("No new data for continuous learning")
            return
        
        # Prepare new data
        from src.scraper.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        df_clean = cleaner.clean_race_data(new_data)
        df_features = self.feature_engineer.create_comprehensive_features(df_clean)
        
        X_new, y_new, _ = cleaner.prepare_training_data(df_features)
        
        # Encode categorical features
        categorical_cols = X_new.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X_new = self.feature_engineer.encode_categorical_features(X_new, categorical_cols)
        
        # Scale features
        X_new_scaled, _ = self.feature_engineer.scale_features(X_new)
        
        # Update models (online learning where possible)
        if hasattr(self.best_model, 'partial_fit'):
            try:
                self.best_model.partial_fit(X_new_scaled, y_new)
                logger.info("Updated best model with new data")
            except:
                logger.warning("Model doesn't support partial_fit, retraining from scratch")
                self.retrain_with_all_data()
        else:
            # For models that don't support online learning, retrain periodically
            self.retrain_with_all_data()
    
    def retrain_with_all_data(self):
        """Retrain models with all available data."""
        logger.info("Retraining models with all available data...")
        
        X, y, feature_cols = self.prepare_training_data()
        
        # Retrain ensemble
        ensemble = self.train_ensemble(X, y)
        
        # Update best model
        self.best_model = ensemble
        
        # Save updated models
        self.save_models()
        
        logger.info("Models retrained and saved")
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate model performance on recent data."""
        logger.info("Evaluating model performance...")
        
        # Get recent data (last 30 days)
        recent_data = db.get_training_data(days_back=30)
        
        if recent_data.empty:
            return {"error": "No recent data available"}
        
        # Prepare data
        from src.scraper.data_cleaner import DataCleaner
        cleaner = DataCleaner()
        df_clean = cleaner.clean_race_data(recent_data)
        df_features = self.feature_engineer.create_comprehensive_features(df_clean)
        
        X, y, _ = cleaner.prepare_training_data(df_features)
        
        # Encode and scale
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = self.feature_engineer.encode_categorical_features(X, categorical_cols)
        
        X_scaled, _ = self.feature_engineer.scale_features(X)
        
        # Make predictions
        if self.best_model is None:
            self.load_models()
        
        y_pred = self.best_model.predict(X_scaled)
        y_pred_proba = self.best_model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred, y_pred_proba)
        
        # Additional analysis
        metrics['total_predictions'] = len(y)
        metrics['positive_predictions'] = sum(y_pred)
        metrics['actual_positives'] = sum(y)
        
        # Calculate profitability (simplified)
        profitable = self._calculate_profitability(X_scaled, y, y_pred_proba)
        metrics.update(profitable)
        
        logger.info(f"Model performance: Accuracy={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}")
        
        return metrics
    
    def _calculate_profitability(self, X: pd.DataFrame, y_true: pd.Series, 
                                y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate potential profitability of model predictions."""
        # This is a simplified profitability calculation
        # In reality, you'd need actual odds and betting amounts
        
        # Assume betting on predictions with probability > threshold
        thresholds = [0.5, 0.6, 0.7, 0.8]
        profitability = {}
        
        for threshold in thresholds:
            bets_mask = y_pred_proba > threshold
            n_bets = sum(bets_mask)
            
            if n_bets > 0:
                # Calculate win rate for these bets
                correct_bets = sum((y_true == 1) & bets_mask)
                win_rate = correct_bets / n_bets
                
                # Simplified ROI calculation
                # Assume average odds of 3.0 for simplicity
                average_odds = 3.0
                roi = (win_rate * average_odds - 1) * 100
                
                profitability[f'roi_threshold_{threshold}'] = roi
                profitability[f'win_rate_threshold_{threshold}'] = win_rate
                profitability[f'n_bets_threshold_{threshold}'] = n_bets
        
        return profitability

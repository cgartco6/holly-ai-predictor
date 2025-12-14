import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class ModelExperiment2:
    """Advanced Tuning and Ensemble Experiment"""
    
    def __init__(self, experiment_dir="."):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        self.tuned_models = {}
        self.results = {}
        
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for multiple models"""
        
        print("=" * 50)
        print("HYPERPARAMETER TUNING")
        print("=" * 50)
        
        # 1. Random Forest Tuning
        print("\n1. Tuning Random Forest...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_grid = RandomizedSearchCV(
            rf, rf_param_grid, n_iter=20, 
            cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
        )
        rf_grid.fit(X_train, y_train)
        
        self.tuned_models['RandomForest'] = rf_grid.best_estimator_
        print(f"  Best params: {rf_grid.best_params_}")
        print(f"  Best CV score: {rf_grid.best_score_:.4f}")
        
        # 2. XGBoost Tuning
        print("\n2. Tuning XGBoost...")
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_grid = RandomizedSearchCV(
            xgb_model, xgb_param_grid, n_iter=20,
            cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
        )
        xgb_grid.fit(X_train, y_train)
        
        self.tuned_models['XGBoost'] = xgb_grid.best_estimator_
        print(f"  Best params: {xgb_grid.best_params_}")
        print(f"  Best CV score: {xgb_grid.best_score_:.4f}")
        
        # 3. LightGBM Tuning
        print("\n3. Tuning LightGBM...")
        lgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        lgb_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
        lgb_grid = RandomizedSearchCV(
            lgb_model, lgb_param_grid, n_iter=20,
            cv=3, scoring='roc_auc', n_jobs=-1, random_state=42
        )
        lgb_grid.fit(X_train, y_train)
        
        self.tuned_models['LightGBM'] = lgb_grid.best_estimator_
        print(f"  Best params: {lgb_grid.best_params_}")
        print(f"  Best CV score: {lgb_grid.best_score_:.4f}")
        
        # Save tuned parameters
        tuned_params = {
            model_name: str(model.get_params())
            for model_name, model in self.tuned_models.items()
        }
        
        with open(f"{self.experiment_dir}/tuned_parameters.json", 'w') as f:
            json.dump(tuned_params, f, indent=2)
        
        return self.tuned_models
    
    def create_stacked_ensemble(self, X_train, y_train, X_val, y_val):
        """Create stacked ensemble model"""
        print("\n" + "=" * 50)
        print("STACKED ENSEMBLE")
        print("=" * 50)
        
        from sklearn.ensemble import StackingClassifier
        
        # Base models
        estimators = [
            ('rf', self.tuned_models['RandomForest']),
            ('xgb', self.tuned_models['XGBoost']),
            ('lgb', self.tuned_models['LightGBM'])
        ]
        
        # Meta-model
        meta_model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=3,
            n_jobs=-1,
            passthrough=True
        )
        
        # Train stacking model
        print("Training Stacking Ensemble...")
        stacking.fit(X_train, y_train)
        
        self.tuned_models['StackingEnsemble'] = stacking
        
        # Evaluate on validation set
        stacking_score = stacking.score(X_val, y_val)
        print(f"Stacking Ensemble Validation Accuracy: {stacking_score:.4f}")
        
        return stacking
    
    def create_weighted_voting(self, weights=None):
        """Create weighted voting ensemble"""
        print("\n" + "=" * 50)
        print("WEIGHTED VOTING ENSEMBLE")
        print("=" * 50)
        
        if weights is None:
            # Default weights based on typical performance
            weights = {
                'RandomForest': 0.3,
                'XGBoost': 0.4,
                'LightGBM': 0.3
            }
        
        # Create voting classifier
        voting = VotingClassifier(
            estimators=[
                ('rf', self.tuned_models['RandomForest']),
                ('xgb', self.tuned_models['XGBoost']),
                ('lgb', self.tuned_models['LightGBM'])
            ],
            voting='soft',
            weights=[weights['RandomForest'], weights['XGBoost'], weights['LightGBM']]
        )
        
        self.tuned_models['WeightedVoting'] = voting
        
        print(f"Weighted Voting Ensemble created with weights: {weights}")
        return voting
    
    def calibrate_probabilities(self, X_train, y_train):
        """Calibrate model probabilities"""
        print("\n" + "=" * 50)
        print("PROBABILITY CALIBRATION")
        print("=" * 50)
        
        calibrated_models = {}
        
        for name, model in self.tuned_models.items():
            if name not in ['StackingEnsemble', 'WeightedVoting']:
                print(f"Calibrating {name}...")
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated.fit(X_train, y_train)
                calibrated_models[f"Calibrated_{name}"] = calibrated
                
                print(f"  {name} calibration complete")
        
        self.tuned_models.update(calibrated_models)
        return calibrated_models
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all tuned models"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        for name, model in self.tuned_models.items():
            print(f"\nEvaluating {name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1': f1_score(y_test, y_pred, zero_division=0),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                self.results[name] = metrics
                
                # Print results
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                
                # Save detailed predictions
                predictions_df = pd.DataFrame({
                    'actual': y_test,
                    'predicted': y_pred,
                    'probability': y_pred_proba
                })
                predictions_df.to_csv(f"{self.experiment_dir}/{name}_predictions.csv", index=False)
                
            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
        
        # Save all results
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(f"{self.experiment_dir}/all_model_results.csv")
        
        # Find best model
        best_model_name = results_df['roc_auc'].idxmax()
        best_score = results_df.loc[best_model_name, 'roc_auc']
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"ROC-AUC Score: {best_score:.4f}")
        print(f"{'='*50}")
        
        # Save best model
        joblib.dump(self.tuned_models[best_model_name], 
                   f"{self.experiment_dir}/best_tuned_model.pkl")
        
        # Save experiment summary
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'best_model': best_model_name,
            'best_score': float(best_score),
            'all_results': self.results,
            'models_tuned': list(self.tuned_models.keys())
        }
        
        with open(f"{self.experiment_dir}/experiment_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return self.results
    
    def create_diagnostics(self, X_test, y_test, model_name):
        """Create diagnostic plots for a specific model"""
        print(f"\nCreating diagnostics for {model_name}...")
        
        model = self.tuned_models[model_name]
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(15, 10))
        
        # ROC Curve plot
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, 'b-', label=f'ROC curve (AUC = {self.results[model_name]["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        # Probability distribution
        plt.subplot(2, 2, 2)
        plt.hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='Class 0', color='red')
        plt.hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Class 1', color='blue')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title(f'Probability Distribution - {model_name}')
        plt.legend()
        plt.grid(True)
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.subplot(2, 2, 3)
            importances = pd.Series(model.feature_importances_, index=X_test.columns)
            top_10 = importances.nlargest(10)
            top_10.plot(kind='barh')
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importances - {model_name}')
            plt.grid(True)
        
        # Confusion Matrix
        plt.subplot(2, 2, 4)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        plt.savefig(f"{self.experiment_dir}/{model_name}_diagnostics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Diagnostic plots saved for {model_name}")

if __name__ == "__main__":
    # Run Experiment 2
    experiment = ModelExperiment2(experiment_dir="model_experiment_2")
    
    # Load data
    from training_data_generator import TrainingDataGenerator
    from sklearn.model_selection import train_test_split
    
    generator = TrainingDataGenerator()
    df = generator.generate_training_dataset(5000)
    
    # Prepare data
    X = df.drop(['finished_win', 'finished_in_the_money', 'finish_position', 
                'beaten_lengths', 'final_odds', 'value_indicator'], axis=1, errors='ignore')
    y = df['finished_win'] if 'finished_win' in df.columns else np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    
    # Split into train, validation, test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    print(f"Data splits:")
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Run experiment
    experiment.hyperparameter_tuning(X_train, y_train)
    experiment.create_stacked_ensemble(X_train, y_train, X_val, y_val)
    experiment.create_weighted_voting()
    experiment.calibrate_probabilities(X_train, y_train)
    
    # Train voting ensemble
    experiment.tuned_models['WeightedVoting'].fit(X_train, y_train)
    
    # Evaluate all models
    experiment.evaluate_all_models(X_test, y_test)
    
    # Create diagnostics for best model
    best_model_name = max(experiment.results, key=lambda x: experiment.results[x]['roc_auc'])
    experiment.create_diagnostics(X_test, y_test, best_model_name)
    
    print("\nExperiment 2 completed successfully!")

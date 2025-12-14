import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib
import json
import os
from datetime import datetime

class ModelExperiment1:
    """Base Models Experiment: Compare different algorithms"""
    
    def __init__(self, experiment_dir="."):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def load_data(self, train_path, test_path):
        """Load training and test data"""
        self.X_train = pd.read_csv(train_path)
        self.X_test = pd.read_csv(test_path)
        
        # Assume target is in separate files or columns
        if 'finished_win' in self.X_train.columns:
            self.y_train = self.X_train['finished_win']
            self.X_train = self.X_train.drop('finished_win', axis=1)
            
            self.y_test = self.X_test['finished_win']
            self.X_test = self.X_test.drop('finished_win', axis=1)
        else:
            # Create synthetic target for demo
            self.y_train = np.random.choice([0, 1], size=len(self.X_train), p=[0.85, 0.15])
            self.y_test = np.random.choice([0, 1], size=len(self.X_test), p=[0.85, 0.15])
        
        print(f"Training data: {self.X_train.shape}")
        print(f"Test data: {self.X_test.shape}")
        
    def train_models(self):
        """Train multiple models"""
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(self.X_train, self.y_train)
        self.models['RandomForest'] = rf
        
        # 2. Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            subsample=0.8
        )
        gb.fit(self.X_train, self.y_train)
        self.models['GradientBoosting'] = gb
        
        # 3. XGBoost
        print("Training XGBoost...")
        xg = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        xg.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = xg
        
        # 4. LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        lgb_model.fit(self.X_train, self.y_train)
        self.models['LightGBM'] = lgb_model
        
        # 5. CatBoost
        print("Training CatBoost...")
        cat = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
        cat.fit(self.X_train, self.y_train)
        self.models['CatBoost'] = cat
        
        # 6. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
        lr.fit(self.X_train, self.y_train)
        self.models['LogisticRegression'] = lr
        
        # 7. Neural Network
        print("Training Neural Network...")
        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        nn.fit(self.X_train, self.y_train)
        self.models['NeuralNetwork'] = nn
        
        print("All models trained successfully!")
        
    def evaluate_models(self):
        """Evaluate all models"""
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'f1': f1_score(self.y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'positive_rate': self.y_test.mean()
            }
            
            self.results[name] = metrics
            
            # Print results
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, 
                                       index=self.X_train.columns)
                top_features = importances.nlargest(10)
                
                feature_importance = {
                    'top_features': top_features.index.tolist(),
                    'importance_values': top_features.values.tolist()
                }
                
                # Save feature importance
                with open(f"{self.experiment_dir}/{name}_feature_importance.json", 'w') as f:
                    json.dump(feature_importance, f, indent=2)
                
                print(f"  Top feature: {top_features.index[0]} ({top_features.values[0]:.4f})")
        
        # Determine best model
        self._select_best_model()
        
        # Save results
        self._save_results()
        
    def _select_best_model(self):
        """Select the best model based on ROC-AUC"""
        best_score = -1
        best_model_name = None
        
        for name, metrics in self.results.items():
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model_name = name
        
        self.best_model = {
            'name': best_model_name,
            'model': self.models[best_model_name],
            'score': best_score,
            'all_metrics': self.results[best_model_name]
        }
        
        print(f"\n{'='*50}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"ROC-AUC Score: {best_score:.4f}")
        print(f"{'='*50}")
        
        # Save best model
        joblib.dump(self.models[best_model_name], 
                   f"{self.experiment_dir}/best_model_{best_model_name}.pkl")
        
    def _save_results(self):
        """Save experiment results"""
        # Save metrics
        metrics_df = pd.DataFrame(self.results).T
        metrics_df.to_csv(f"{self.experiment_dir}/model_metrics.csv")
        
        # Save detailed results
        results_data = {
            'experiment_date': datetime.now().isoformat(),
            'best_model': self.best_model['name'],
            'best_score': self.best_model['score'],
            'models_trained': list(self.models.keys()),
            'metrics': self.results,
            'data_info': {
                'train_shape': self.X_train.shape,
                'test_shape': self.X_test.shape,
                'train_positive_rate': self.y_train.mean(),
                'test_positive_rate': self.y_test.mean()
            }
        }
        
        with open(f"{self.experiment_dir}/experiment_results.json", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\nResults saved to {self.experiment_dir}")
        
    def create_ensemble(self):
        """Create ensemble model from top performers"""
        print("\nCreating Ensemble Model...")
        
        # Select top 3 models by ROC-AUC
        sorted_models = sorted(self.results.items(), 
                              key=lambda x: x[1]['roc_auc'], 
                              reverse=True)[:3]
        
        top_model_names = [name for name, _ in sorted_models]
        top_models = {name: self.models[name] for name in top_model_names}
        
        # Simple averaging ensemble
        class EnsembleModel:
            def __init__(self, models):
                self.models = models
                
            def predict_proba(self, X):
                probas = [model.predict_proba(X)[:, 1] for model in self.models.values()]
                avg_proba = np.mean(probas, axis=0)
                return np.column_stack([1 - avg_proba, avg_proba])
            
            def predict(self, X, threshold=0.5):
                probas = self.predict_proba(X)[:, 1]
                return (probas >= threshold).astype(int)
        
        ensemble = EnsembleModel(top_models)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(self.X_test)
        y_pred_proba_ensemble = ensemble.predict_proba(self.X_test)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred_ensemble),
            'precision': precision_score(self.y_test, y_pred_ensemble, zero_division=0),
            'recall': recall_score(self.y_test, y_pred_ensemble, zero_division=0),
            'f1': f1_score(self.y_test, y_pred_ensemble, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_ensemble)
        }
        
        print(f"Ensemble Performance:")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        # Save ensemble
        ensemble_info = {
            'ensemble_models': top_model_names,
            'ensemble_metrics': ensemble_metrics,
            'creation_date': datetime.now().isoformat()
        }
        
        with open(f"{self.experiment_dir}/ensemble_info.json", 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        joblib.dump(ensemble, f"{self.experiment_dir}/ensemble_model.pkl")
        
        return ensemble

if __name__ == "__main__":
    # Run experiment
    experiment = ModelExperiment1(experiment_dir="model_experiment_1")
    
    # Load data (using generated data)
    from training_data_generator import TrainingDataGenerator
    
    generator = TrainingDataGenerator()
    df = generator.generate_training_dataset(5000)
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    X = df.drop(['finished_win', 'finished_in_the_money', 'finish_position', 
                'beaten_lengths', 'final_odds', 'value_indicator'], axis=1, errors='ignore')
    y = df['finished_win'] if 'finished_win' in df.columns else np.random.choice([0, 1], size=len(df), p=[0.85, 0.15])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save to CSV for loading
    X_train.to_csv("X_train_temp.csv", index=False)
    X_test.to_csv("X_test_temp.csv", index=False)
    
    # Add target back for loading
    X_train['finished_win'] = y_train
    X_test['finished_win'] = y_test
    
    X_train.to_csv("train_data_temp.csv", index=False)
    X_test.to_csv("test_data_temp.csv", index=False)
    
    # Run experiment
    experiment.load_data("train_data_temp.csv", "test_data_temp.csv")
    experiment.train_models()
    experiment.evaluate_models()
    experiment.create_ensemble()
    
    print("\nExperiment 1 completed successfully!")

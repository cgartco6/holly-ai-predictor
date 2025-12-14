import joblib
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class ModelDeployer:
    """Deploy trained models for production use"""
    
    def __init__(self, deployment_dir="."):
        self.deployment_dir = deployment_dir
        os.makedirs(deployment_dir, exist_ok=True)
        self.models = {}
        self.metadata = {}
        
    def load_model(self, model_path: str, model_name: str):
        """Load a trained model"""
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            print(f"Loaded model: {model_name}")
            return model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def create_deployment_package(self, model_name: str, model, 
                                 feature_columns: List[str],
                                 model_info: Dict[str, Any]):
        """Create a deployment package for a model"""
        
        package_dir = f"{self.deployment_dir}/{model_name}_v{model_info.get('version', '1.0')}"
        os.makedirs(package_dir, exist_ok=True)
        
        # 1. Save the model
        model_path = f"{package_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        
        # 2. Save feature information
        feature_info = {
            'feature_columns': feature_columns,
            'feature_count': len(feature_columns),
            'required_features': feature_columns,
            'feature_dtypes': {}  # Would be populated with actual dtypes
        }
        
        with open(f"{package_dir}/feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # 3. Save model metadata
        metadata = {
            'model_name': model_name,
            'version': model_info.get('version', '1.0'),
            'model_type': str(type(model).__name__),
            'deployment_date': datetime.now().isoformat(),
            'performance_metrics': model_info.get('metrics', {}),
            'training_info': model_info.get('training_info', {}),
            'dependencies': {
                'python': '3.9+',
                'packages': [
                    'scikit-learn>=1.0',
                    'numpy>=1.21',
                    'pandas>=1.3',
                    'joblib>=1.1'
                ]
            }
        }
        
        with open(f"{package_dir}/model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 4. Create prediction script
        self._create_prediction_script(package_dir, model_name, feature_columns)
        
        # 5. Create validation script
        self._create_validation_script(package_dir)
        
        # 6. Create API wrapper
        self._create_api_wrapper(package_dir, model_name)
        
        print(f"Deployment package created at: {package_dir}")
        return package_dir
    
    def _create_prediction_script(self, package_dir: str, model_name: str, feature_columns: List[str]):
        """Create a standalone prediction script"""
        
        script_content = f'''"""
Prediction Script for {model_name}
Usage: python predict.py --input input_data.csv --output predictions.csv
"""

import joblib
import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path

class HorseRacingPredictor:
    def __init__(self, model_path=".", model_name="{model_name}"):
        self.model_path = Path(model_path)
        self.model_name = model_name
        
        # Load model
        self.model = joblib.load(self.model_path / f"{model_name}.pkl")
        
        # Load feature info
        with open(self.model_path / "feature_info.json", 'r') as f:
            self.feature_info = json.load(f)
        
        # Load metadata
        with open(self.model_path / "model_metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def preprocess_input(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training format"""
        
        # Ensure all required features are present
        required_features = self.feature_info['required_features']
        
        # Add missing features with default values
        for feature in required_features:
            if feature not in input_data.columns:
                print(f"Warning: Missing feature {{feature}}, filling with 0")
                input_data[feature] = 0
        
        # Select only required features in correct order
        processed_data = input_data[required_features].copy()
        
        # Handle missing values
        processed_data = processed_data.fillna(0)
        
        return processed_data
    
    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions"""
        
        # Preprocess
        processed_data = self.preprocess_input(input_data)
        
        # Make predictions
        probabilities = self.model.predict_proba(processed_data)
        predictions = self.model.predict(processed_data)
        
        # Create results DataFrame
        results = input_data.copy()
        results['predicted_class'] = predictions
        results['predicted_probability'] = probabilities[:, 1]
        results['prediction_confidence'] = np.max(probabilities, axis=1)
        
        # Add additional insights
        results['recommendation'] = results.apply(
            lambda row: 'BET' if row['predicted_probability'] > 0.3 else 'AVOID', axis=1
        )
        
        return results
    
    def predict_single(self, features: dict) -> dict:
        """Predict for a single sample"""
        input_df = pd.DataFrame([features])
        result = self.predict(input_df)
        
        return {{
            'predicted_class': int(result.iloc[0]['predicted_class']),
            'predicted_probability': float(result.iloc[0]['predicted_probability']),
            'confidence': float(result.iloc[0]['prediction_confidence']),
            'recommendation': result.iloc[0]['recommendation']
        }}

def main():
    parser = argparse.ArgumentParser(description='Horse Racing Predictor')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--model_path', default='.', help='Path to model directory')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HorseRacingPredictor(args.model_path)
    
    # Load input data
    try:
        input_data = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error loading input file: {{e}}")
        sys.exit(1)
    
    # Make predictions
    print(f"Making predictions for {{len(input_data)}} samples...")
    predictions = predictor.predict(input_data)
    
    # Save predictions
    predictions.to_csv(args.output, index=False)
    print(f"Predictions saved to {{args.output}}")
    
    # Print summary
    win_predictions = predictions['predicted_class'].sum()
    win_rate = predictions['predicted_class'].mean()
    print(f"\\nPrediction Summary:")
    print(f"  Total predictions: {{len(predictions)}}")
    print(f"  Predicted winners: {{win_predictions}} ({{win_rate:.1%}})")
    print(f"  Average confidence: {{predictions['prediction_confidence'].mean():.1%}}")

if __name__ == "__main__":
    main()
'''
        
        with open(f"{package_dir}/predict.py", 'w') as f:
            f.write(script_content)
    
    def _create_validation_script(self, package_dir: str):
        """Create model validation script"""
        
        script_content = '''"""
Model Validation Script
Validates that the deployed model works correctly
"""

import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_model(model_path="."):
    """Validate the deployed model"""
    
    print("=" * 50)
    print("MODEL VALIDATION")
    print("=" * 50)
    
    # Check required files
    required_files = ['model_metadata.json', 'feature_info.json', '*.pkl']
    model_files = list(Path(model_path).glob('*'))
    
    print("\\n1. File Structure Check:")
    for file in model_files:
        print(f"   ✓ {file.name}")
    
    # Load metadata
    try:
        with open(Path(model_path) / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        print(f"\\n2. Model Metadata:")
        print(f"   Model: {metadata['model_name']}")
        print(f"   Version: {metadata['version']}")
        print(f"   Type: {metadata['model_type']}")
    except Exception as e:
        print(f"   ✗ Error loading metadata: {e}")
        return False
    
    # Load model
    try:
        model_files = list(Path(model_path).glob('*.pkl'))
        if not model_files:
            print("   ✗ No model file found (*.pkl)")
            return False
        
        model = joblib.load(model_files[0])
        print(f"\\n3. Model Load Test:")
        print(f"   ✓ Model loaded successfully")
        print(f"   Model class: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False
    
    # Test prediction with dummy data
    try:
        with open(Path(model_path) / 'feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        # Create dummy data
        dummy_data = {}
        for feature in feature_info['required_features']:
            dummy_data[feature] = [0]  # Default value
        
        dummy_df = pd.DataFrame(dummy_data)
        
        # Make prediction
        predictions = model.predict(dummy_df)
        probabilities = model.predict_proba(dummy_df)
        
        print(f"\\n4. Prediction Test:")
        print(f"   ✓ Predictions made successfully")
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Probability shape: {probabilities.shape}")
        
    except Exception as e:
        print(f"   ✗ Error making predictions: {e}")
        return False
    
    # Performance check from metadata
    if 'performance_metrics' in metadata:
        print(f"\\n5. Performance Metrics:")
        metrics = metadata['performance_metrics']
        for metric, value in metrics.items():
            print(f"   {metric}: {value}")
    
    print("\\n" + "=" * 50)
    print("VALIDATION COMPLETE - All checks passed!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate deployed model')
    parser.add_argument('--model_path', default='.', help='Path to model directory')
    
    args = parser.parse_args()
    
    success = validate_model(args.model_path)
    
    if not success:
        print("\\n✗ Model validation failed!")
        exit(1)
'''
        
        with open(f"{package_dir}/validate.py", 'w') as f:
            f.write(script_content)
    
    def _create_api_wrapper(self, package_dir: str, model_name: str):
        """Create Flask API wrapper for the model"""
        
        api_content = f'''"""
Flask API for {model_name}
Run with: python api.py
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import traceback

app = Flask(__name__)

class PredictionAPI:
    def __init__(self):
        self.model = None
        self.feature_info = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = Path(__file__).parent
            model_files = list(model_path.glob('*.pkl'))
            
            if model_files:
                self.model = joblib.load(model_files[0])
            
            # Load feature info
            with open(model_path / 'feature_info.json', 'r') as f:
                self.feature_info = json.load(f)
            
            # Load metadata
            with open(model_path / 'model_metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            print(f"Model loaded: {{self.metadata['model_name']}} v{{self.metadata['version']}}")
            
        except Exception as e:
            print(f"Error loading model: {{e}}")
            self.model = None
    
    def preprocess(self, data):
        """Preprocess input data"""
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in self.feature_info['required_features']:
            if feature not in df.columns:
                df[feature] = 0
        
        return df[self.feature_info['required_features']].fillna(0)
    
    def predict(self, data):
        """Make prediction"""
        if self.model is None:
            return {{"error": "Model not loaded"}}
        
        try:
            # Preprocess
            processed_data = self.preprocess(data)
            
            # Predict
            probability = self.model.predict_proba(processed_data)[0, 1]
            prediction = self.model.predict(processed_data)[0]
            
            # Calculate confidence
            confidence = probability if prediction == 1 else 1 - probability
            
            return {{
                "prediction": int(prediction),
                "probability": float(probability),
                "confidence": float(confidence),
                "recommendation": "BET" if probability > 0.3 else "AVOID",
                "model_info": {{
                    "name": self.metadata['model_name'],
                    "version": self.metadata['version']
                }}
            }}
        
        except Exception as e:
            return {{"error": str(e)}}

# Initialize API
api = PredictionAPI()

@app.route('/')
def home():
    return jsonify({{
        "service": "Horse Racing Prediction API",
        "model": api.metadata['model_name'] if api.metadata else "Unknown",
        "version": api.metadata['version'] if api.metadata else "Unknown",
        "status": "ready" if api.model else "error"
    }})

@app.route('/health')
def health():
    return jsonify({{
        "status": "healthy" if api.model else "unhealthy",
        "timestamp": pd.Timestamp.now().isoformat()
    }})

@app.route('/predict', methods=['POST'])
def predict():
    """Make a single prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({{"error": "No data provided"}}), 400
        
        result = api.predict(data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({{"error": str(e), "traceback": traceback.format_exc()}}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make batch predictions"""
    try:
        data = request.json
        
        if not data or not isinstance(data, list):
            return jsonify({{"error": "Data must be a list of records"}}), 400
        
        results = []
        for record in data:
            result = api.predict(record)
            results.append(result)
        
        return jsonify({{
            "predictions": results,
            "count": len(results),
            "summary": {{
                "predicted_winners": sum(1 for r in results if r.get('prediction') == 1),
                "average_confidence": np.mean([r.get('confidence', 0) for r in results])
            }}
        }})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    if api.metadata:
        return jsonify(api.metadata)
    return jsonify({{"error": "Model not loaded"}}), 500

if __name__ == '__main__':
    print("Starting Horse Racing Prediction API...")
    print("Endpoint: http://localhost:5000")
    print("Available routes:")
    print("  GET  /          - Service info")
    print("  GET  /health    - Health check")
    print("  GET  /model_info - Model information")
    print("  POST /predict   - Single prediction")
    print("  POST /batch_predict - Batch predictions")
    app.run(host='0.0.0.0', port=5000, debug=True)
'''
        
        with open(f"{package_dir}/api.py", 'w') as f:
            f.write(api_content)
    
    def create_dockerfile(self, package_dir: str):
        """Create Dockerfile for containerization"""
        
        dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the application
CMD ["python", "api.py"]
'''
        
        with open(f"{package_dir}/Dockerfile", 'w') as f:
            f.write(dockerfile_content)
    
    def create_requirements(self, package_dir: str):
        """Create requirements.txt file"""
        
        requirements = '''flask>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
joblib>=1.1.0
gunicorn>=20.0.0
'''
        
        with open(f"{package_dir}/requirements.txt", 'w') as f:
            f.write(requirements)
    
    def package_for_production(self, model_name: str, model_path: str, 
                              feature_columns: List[str], model_info: Dict[str, Any]):
        """Create complete production package"""
        
        print(f"\nPackaging {model_name} for production...")
        
        # Create deployment package
        package_dir = self.create_deployment_package(
            model_name, 
            joblib.load(model_path), 
            feature_columns, 
            model_info
        )
        
        # Create Dockerfile
        self.create_dockerfile(package_dir)
        
        # Create requirements.txt
        self.create_requirements(package_dir)
        
        # Create README
        readme_content = f'''# {model_name} - Horse Racing Prediction Model

## Overview
This package contains a trained machine learning model for predicting horse racing outcomes.

## Model Details
- **Model Name**: {model_name}
- **Version**: {model_info.get('version', '1.0')}
- **Type**: {str(type(joblib.load(model_path)).__name__)}
- **Deployment Date**: {datetime.now().strftime('%Y-%m-%d')}

## Features
- {len(feature_columns)} input features
- Predicts win probability
- Provides confidence scores
- Includes betting recommendations

## Usage

### 1. Command Line Prediction
```bash
python predict.py --input data.csv --output predictions.csv

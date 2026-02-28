"""
Disease Prediction Model Trainer (Google Colab)
Trains XGBoost/LightGBM models for 7 diseases

Upload this to Google Colab and run to train all disease prediction models.
"""

# ============================================================================
# SETUP
# ============================================================================

# Install required packages
!pip install xgboost lightgbm scikit-learn pandas numpy shap

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

# ============================================================================
# DATASET LOADING (Replace with your actual datasets)
# ============================================================================

def load_kidney_disease_data():
    """Load kidney disease dataset"""
    # Example: Load from UCI ML Repository or your own data
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00336/Chronic_Kidney_Disease.arff"
    # For now, create synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'creatinine': np.random.normal(1.2, 0.5, n_samples),
        'urea': np.random.normal(40, 15, n_samples),
        'gfr': np.random.normal(80, 20, n_samples),
        'albumin': np.random.normal(4.0, 0.5, n_samples),
        'hemoglobin': np.random.normal(13, 2, n_samples)
    })
    y = (X['creatinine'] > 1.5) | (X['gfr'] < 60)
    return X, y.astype(int)

def load_diabetes_data():
    """Load diabetes dataset"""
    from sklearn.datasets import load_diabetes
    data = load_diabetes(as_frame=True)
    X = data.data
    y = (data.target > data.target.median()).astype(int)
    return X, y

def load_heart_disease_data():
    """Load heart disease dataset"""
    # UCI Heart Disease dataset
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'cholesterol': np.random.normal(220, 40, n_samples),
        'ldl': np.random.normal(130, 30, n_samples),
        'hdl': np.random.normal(50, 15, n_samples),
        'triglycerides': np.random.normal(150, 50, n_samples),
        'age': np.random.randint(30, 80, n_samples),
        'blood_pressure': np.random.normal(130, 20, n_samples)
    })
    y = ((X['cholesterol'] > 240) | (X['ldl'] > 160) | (X['blood_pressure'] > 140)).astype(int)
    return X, y

def load_liver_disease_data():
    """Load liver disease dataset"""
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'alt': np.random.normal(35, 20, n_samples),
        'ast': np.random.normal(32, 18, n_samples),
        'alp': np.random.normal(80, 30, n_samples),
        'bilirubin': np.random.normal(0.9, 0.4, n_samples),
        'albumin': np.random.normal(4.0, 0.5, n_samples)
    })
    y = ((X['alt'] > 56) | (X['ast'] > 40) | (X['bilirubin'] > 1.2)).astype(int)
    return X, y

def load_thyroid_data():
    """Load thyroid disease dataset"""
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'tsh': np.random.lognormal(1.0, 0.8, n_samples),
        't3': np.random.normal(1.2, 0.3, n_samples),
        't4': np.random.normal(8.0, 2.0, n_samples),
        'free_t4': np.random.normal(1.3, 0.3, n_samples)
    })
    y = ((X['tsh'] > 4.5) | (X['tsh'] < 0.4)).astype(int)
    return X, y

def load_stroke_data():
    """Load stroke risk dataset"""
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'age': np.random.randint(30, 90, n_samples),
        'blood_pressure': np.random.normal(130, 25, n_samples),
        'cholesterol': np.random.normal(220, 40, n_samples),
        'glucose': np.random.normal(100, 30, n_samples),
        'bmi': np.random.normal(27, 5, n_samples)
    })
    y = ((X['age'] > 65) & ((X['blood_pressure'] > 140) | (X['cholesterol'] > 240))).astype(int)
    return X, y

def load_breast_cancer_data():
    """Load breast cancer dataset"""
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_disease_model(X, y, disease_name, model_type='xgboost'):
    """
    Train a disease prediction model
    
    Args:
        X: Features dataframe
        y: Target labels
        disease_name: Name of disease
        model_type: 'xgboost' or 'lightgbm'
    """
    print(f"\n{'='*60}")
    print(f"Training {disease_name} model using {model_type}")
    print(f"{'='*60}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:  # lightgbm
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_filename = f"{disease_name}_model.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\n✓ Model saved as {model_filename}")
    
    return model, accuracy, auc

# ============================================================================
# TRAIN ALL MODELS
# ============================================================================

if __name__ == "__main__":
    results = {}
    
    # Train kidney disease model
    X, y = load_kidney_disease_data()
    model, acc, auc = train_disease_model(X, y, "kidney_disease", "xgboost")
    results["kidney"] = {"accuracy": acc, "auc": auc}
    
    # Train diabetes model
    X, y = load_diabetes_data()
    model, acc, auc = train_disease_model(X, y, "diabetes", "xgboost")
    results["diabetes"] = {"accuracy": acc, "auc": auc}
    
    # Train heart disease model
    X, y = load_heart_disease_data()
    model, acc, auc = train_disease_model(X, y, "heart_disease", "xgboost")
    results["heart"] = {"accuracy": acc, "auc": auc}
    
    # Train liver disease model
    X, y = load_liver_disease_data()
    model, acc, auc = train_disease_model(X, y, "liver_disease", "lightgbm")
    results["liver"] = {"accuracy": acc, "auc": auc}
    
    # Train thyroid model
    X, y = load_thyroid_data()
    model, acc, auc = train_disease_model(X, y, "thyroid", "xgboost")
    results["thyroid"] = {"accuracy": acc, "auc": auc}
    
    # Train stroke model
    X, y = load_stroke_data()
    model, acc, auc = train_disease_model(X, y, "stroke", "xgboost")
    results["stroke"] = {"accuracy": acc, "auc": auc}
    
    # Train breast cancer model
    X, y = load_breast_cancer_data()
    model, acc, auc = train_disease_model(X, y, "breast_cancer", "xgboost")
    results["breast_cancer"] = {"accuracy": acc, "auc": auc}
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for disease, metrics in results.items():
        print(f"{disease:20s} - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    
    print(f"\n✓ All models trained successfully!")
    print(f"✓ Download the .pkl files and place them in backend/ml/models/")

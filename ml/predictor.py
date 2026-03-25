"""
Clinical ML Predictor
Loads and runs disease prediction models (XGBoost, LightGBM, etc.) and SHAP explainers.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path

class ClinicalPredictor:
    """Manages all disease prediction models"""
    
    # Updated path for Docker/Project Root resolution
    MODEL_DIR = Path("/app/data/trained model data")
    if not MODEL_DIR.exists():
        # Fallback for local development
        MODEL_DIR = Path(__file__).parent.parent / "backend" / "data" / "trained model data"
    
    # Model file names based on actual files found
    MODELS = {
        "kidney": {
            "model": "kidney_xgb.pkl",
            "explainer": "kidney_explainer.pkl"
        },
        "thyroid": {
            "model": "thyroid_xgboost.pkl", 
            # "explainer": "thyroid_explainer.pkl", # Not found, maybe check later
            "encoder": "thyroid_le.pkl" # Label encoder found
        },
        "cancer": {
            "model": "cancer_rf.pkl",
            "explainer": "cancer_explainer.pkl"
        },
        "stroke": {
            "model": "stroke_rf.pkl",
            "explainer": "stroke_explainer.pkl"
        },
        "liver": {
            "model": "liver_lgbm.pkl",
            "explainer": "liver_explainer.pkl"
        }
    }
    
    # Aliases to help map extracted text to model features
    ALIASES = {
        "sc": ["creatinine", "serum_creatinine"],
        "bu": ["urea", "blood_urea"],
        "hemo": ["hemoglobin"],
        "pot": ["potassium"],
        "sod": ["sodium"],
        "wc": ["white_blood_cell_count", "wbc"],
        "rc": ["red_blood_cell_count", "rbc"],
        "Total_Bilirubin": ["bilirubin", "total_bilirubin"],
        "Alamine_Aminotransferase": ["alt", "sgpt"],
        "Aspartate_Aminotransferase": ["ast", "sgot"],
        "Alkaline_Phosphotase": ["alp"],
        "avg_glucose_level": ["glucose", "sugar", "blood_sugar"]
    }
    
    def __init__(self):
        """Initialize predictor (lazy load)"""
        self.loaded_models = {}
        self.loaded_explainers = {}
        self.encoders = {}
        self.loaded = False
        # self._load_all_models() is now called on demand
    
    def _load_all_models(self):
        """Load all available models from disk"""
        print(f"Loading models from: {self.MODEL_DIR}")
        
        for disease, files in self.MODELS.items():
            # Load Model
            model_path = self.MODEL_DIR / files["model"]
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.loaded_models[disease] = pickle.load(f)
                    print(f"✓ Loaded {disease} model")
                except Exception as e:
                    print(f"✗ Failed to load {disease} model: {e}")
            else:
                print(f"⚠ Model file not found: {model_path}")
                
            # Load Explainer (if exists)
            if "explainer" in files:
                explainer_path = self.MODEL_DIR / files["explainer"]
                if explainer_path.exists():
                    try:
                        with open(explainer_path, 'rb') as f:
                            self.loaded_explainers[disease] = pickle.load(f)
                        print(f"✓ Loaded {disease} explainer")
                    except Exception as e:
                         print(f"✗ Failed to load {disease} explainer: {e}")

            # Load Encoder (if exists)
            if "encoder" in files:
                encoder_path = self.MODEL_DIR / files["encoder"]
                if encoder_path.exists():
                    try:
                        with open(encoder_path, 'rb') as f:
                            self.encoders[disease] = pickle.load(f)
                        print(f"✓ Loaded {disease} encoder")
                    except Exception as e:
                        print(f"✗ Failed to load {disease} encoder: {e}")
        
        self.loaded = True

    
    def predict_all_diseases(self, lab_markers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict disease risks based on lab markers
        
        Args:
            lab_markers: List of extracted lab values with name, value, unit
            
        Returns:
            Dictionary with disease predictions and risk scores
        """
        if not self.loaded:
            self._load_all_models()

        # Extract marker values into a feature dictionary
        features = self._extract_features(lab_markers)
        
        predictions = {}
        
        # If no models loaded, return mock predictions for development
        if not self.loaded_models:
            return self._mock_predictions(features)
        
        # Core features that MUST be present to run a prediction
        # If none of these are found, we skip the prediction to avoid false alarms
        required_vitals = {
            "kidney": ["sc", "bu", "al", "su", "sg", "bgr"], # Creatinine, Urea, Albumin, Sugar, Specific Gravity, Blood Glucose
            "liver": ["Total_Bilirubin", "Direct_Bilirubin", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Alkaline_Phosphotase"],
            "thyroid": ["TSH", "T3", "TT4", "FTI"],
            "stroke": ["avg_glucose_level", "bmi", "hypertension", "heart_disease"],
            "cancer": ["radius_mean", "texture_mean", "perimeter_mean"] # Breast Cancer features usually come as a set
        }

        # Run predictions for each loaded model
        for disease, model in self.loaded_models.items():
            try:
                # 1. Feature Coverage Check
                # Does the input have ANY meaningful data for this disease?
                disease_reqs = required_vitals.get(disease, [])
                
                # Check using both raw names and mapped aliases
                has_data = False
                for req in disease_reqs:
                    # Check direct capability
                    if features.get(req.lower()) is not None:
                        has_data = True
                        break
                    # Check aliases
                    if req in self.ALIASES:
                        for alias in self.ALIASES[req]:
                            if features.get(alias) is not None:
                                has_data = True
                                break
                    if has_data: break
                
                if not has_data:
                    # Skip prediction if no relevant biomarkers are found
                    continue

                # Convert features to model input format
                X, feature_names = self._prepare_features(features, disease)
                
                if X is None:
                    continue

                # Get prediction and probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0] 
                    # Assuming binary classification where 1 = Disease
                    risk_score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    pred = model.predict(X)[0]
                    risk_score = 1.0 if pred == 1 else 0.0
                
                risk_level = "High" if risk_score > 0.5 else "Low"
                if risk_score > 0.8: risk_level = "Critical"
                
                # Get SHAP explanation if available
                shap_values = []
                if disease in self.loaded_explainers:
                    try:
                        explainer = self.loaded_explainers[disease]
                        # TreeExplainer expects dataframe or array
                        shap_out = explainer.shap_values(X)
                        
                        # Handle different SHAP output formats (list of arrays for classification vs single array)
                        if isinstance(shap_out, list):
                            # Binary classification usually returns [class0_shap, class1_shap]
                            vals = shap_out[1][0] 
                        else:
                            vals = shap_out[0]
                            
                        # Map top 3 features
                        # feature_names is needed here. If X is numpy, we might need manual mapping.
                        # For now, we'll try to zip if we have names
                        if len(vals) == len(feature_names):
                            combined = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
                            shap_values = [{"feature": k, "impact": float(v)} for k, v in combined[:3]]
                            
                    except Exception as e:
                        print(f"SHAP Error for {disease}: {e}")

                predictions[disease] = {
                    "risk": risk_level,
                    "probability": round(risk_score, 4),
                    "confidence": round(max(risk_score, 1-risk_score), 2),
                    "top_contributors": shap_values
                }
            except Exception as e:
                print(f"Error predicting {disease}: {e}")
                # Fallback to mock if real model fails (e.g. input shape mismatch)
                # predictions[disease] = self._mock_single(disease, features)
        
        return predictions
    
    def _extract_features(self, lab_markers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract numerical features from lab markers"""
        features = {}
        
        for marker in lab_markers:
            name = marker.get("name", "").lower()
            val = marker.get("value")
            
            # Add some cleaning logic here if needed
            if val is not None:
                try:
                    features[name] = float(val)
                    # Also add cleaned keys (e.g. "Hemoglobin (Hb)" -> "hemoglobin")
                    clean_name = name.split("(")[0].strip().replace(" ", "_")
                    features[clean_name] = float(val)
                except (ValueError, TypeError):
                    pass
        
        return features
    
    def _prepare_features(self, features: Dict[str, float], disease: str) -> tuple:
        """
        Prepare features for specific disease model
        Returns (Reshaped Array/DataFrame, Feature Names List)
        """
        # Feature mappings based on likely training data (Common datasets)
        # We need to fill missing values with means or 0
        feature_maps = {
            "kidney": ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"],
             # Mapping common names to these cryptic codes
             # age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells...
            
            "liver": ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"],
            
            "thyroid": ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych", "TSH", "T3", "TT4", "T4U", "FTI"],
            
            "stroke": ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"],
            
            "cancer": ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
        }
        
        # Aliases are now defined at the class level as self.ALIASES

        required_cols = feature_maps.get(disease)
        if not required_cols:
            return None, None

        # Build row
        row_data = {}
        for col in required_cols:
            # 1. Direct match
            val = features.get(col.lower())
            
            # 2. Alias match
            if val is None and col in self.ALIASES:
                for alias in self.ALIASES[col]:
                    if alias in features:
                        val = features[alias]
                        break
            
            # 3. Default (Mean imputation substitute 0 for now)
            if val is None:
                val = 0.0
                
            row_data[col] = [val]

        # Convert to DataFrame (XGBoost/LightGBM often prefer DF with names)
        df = pd.DataFrame(row_data)
        
        # Handle categorical encoding if needed (simplified)
        # Real implementation would need the original encoders
        
        return df, required_cols
    
    def _mock_predictions(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Return mock predictions when models aren't loaded
        Useful for development/testing
        """
        predictions = {}
        creatinine = features.get("creatinine", 1.0)
        risk_score = 0.8 if creatinine > 1.5 else 0.1
        predictions["kidney"] = {
            "risk": "High" if risk_score > 0.5 else "Low", 
            "probability": risk_score,
            "confidence": 0.95,
            "top_contributors": [{"feature": "creatinine", "impact": 0.85}]
        }
        return predictions


# Global predictor instance
clinical_predictor = ClinicalPredictor()

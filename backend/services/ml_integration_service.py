import sys
import os
from typing import Dict, Any, List

# Add root ml directory to sys.path to allow importing the predictor
# This works both locally and in Docker
# In Docker, we mapped ./ml to /ml
ML_ROOT = "/ml"
if not os.path.exists(ML_ROOT):
    # Fallback for local development or different Docker mapping
    # Assuming backend/services/ml_integration_service.py -> ../../ml
    ML_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ml")

if os.path.exists(ML_ROOT) and ML_ROOT not in sys.path:
    sys.path.append(ML_ROOT)
    print(f"ML Integration: Added {ML_ROOT} to sys.path")

try:
    from predictor import clinical_predictor
    print("ML Integration: Successfully imported clinical_predictor")
except ImportError as e:
    clinical_predictor = None
    print(f"WARNING: Could not import ClinicalPredictor from {ML_ROOT}: {e}")

class MLIntegrationService:
    @staticmethod
    def get_predictions(lab_markers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Bridge to the ML Predictor for disease risk scores.
        
        Args:
            lab_markers: List of structured markers extracted from the report.
            
        Returns:
            Dictionary with disease-specific risk scores and explanations.
        """
        if not clinical_predictor:
            print("ML Integration: Predictor not available, returning empty results.")
            return {}
            
        try:
            # ClinicalPredictor expects: List[Dict[str, Any]] (matches our structured markers)
            return clinical_predictor.predict_all_diseases(lab_markers)
        except Exception as e:
            print(f"ML Prediction Error: {e}")
            return {}

ml_bridge = MLIntegrationService()


from typing import Dict, Any, List
from backend.ml.predictor import clinical_predictor

class CDSSService:
    def predict_disease(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict disease based on input data using ClinicalPredictor.
        Expected input format:
        {
            "lab_results": [
                {"name": "hemoglobin", "value": 12.5, "unit": "g/dL"},
                ...
            ]
        }
        """
        lab_markers = data.get("lab_results", [])
        if not lab_markers:
            return {"status": "error", "message": "No lab results provided"}
            
        predictions = clinical_predictor.predict_all_diseases(lab_markers)
        
        return {
            "status": "success",
            "predictions": predictions
        }

cdss_service = CDSSService()

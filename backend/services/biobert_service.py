"""
BioBERT Service for Medical Named Entity Recognition
Extracts clinical entities from medical text
"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

class BioBERTService:
    """Medical NER using BioBERT (with fallback to rule-based extraction)"""
    
    MODEL_PATH = Path(__file__).parent.parent / "data" / "trained model data" / "mediexplain_biobert_final"
    
    def __init__(self):
        """Initialize BioBERT model"""
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load BioBERT model from disk"""
        try:
            if self.MODEL_PATH.exists():
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.MODEL_PATH))
                self.model = AutoModelForTokenClassification.from_pretrained(str(self.MODEL_PATH))
                print("BioBERT model loaded successfully")
            else:
                print(f"BioBERT model not found at {self.MODEL_PATH}")
                print("Using rule-based extraction fallback")
        except Exception as e:
            print(f"Failed to load BioBERT: {e}")
            print("Using rule-based extraction fallback")
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract medical entities from text
        
        Args:
            text: Medical report text
            
        Returns:
            List of entities with type, value, and position
        """
        if self.model and self.tokenizer:
            return self._extract_with_biobert(text)
        else:
            return self._extract_with_rules(text)
    
    def _extract_with_biobert(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using BioBERT model"""
        import torch
        
        entities = []
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Decode predictions
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = predictions[0].tolist()
            
            # Group tokens into entities
            current_entity = None
            current_tokens = []
            
            for token, label_id in zip(tokens, labels):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                
                # Label mapping (simplified)
                label_map = {
                    0: "O",  # Outside
                    1: "B-TEST",  # Beginning of test
                    2: "I-TEST",  # Inside test
                    3: "B-VALUE",  # Beginning of value
                    4: "I-VALUE",  # Inside value
                    5: "B-DISEASE",  # Beginning of disease
                    6: "I-DISEASE"  # Inside disease
                }
                
                label = label_map.get(label_id, "O")
                
                if label.startswith("B-"):
                    # Save previous entity
                    if current_entity:
                        entities.append({
                            "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                            "type": current_entity,
                            "confidence": 0.85
                        })
                    # Start new entity
                    current_entity = label[2:]
                    current_tokens = [token]
                elif label.startswith("I-") and current_entity:
                    current_tokens.append(token)
                else:
                    # Save and reset
                    if current_entity:
                        entities.append({
                            "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                            "type": current_entity,
                            "confidence": 0.85
                        })
                    current_entity = None
                    current_tokens = []
            
            # Save last entity
            if current_entity:
                entities.append({
                    "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                    "type": current_entity,
                    "confidence": 0.85
                })
        
        except Exception as e:
            print(f"BioBERT extraction error: {e}")
            return self._extract_with_rules(text)
        
        return entities
    
    def _extract_with_rules(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback rule-based entity extraction
        Uses regex patterns to identify common lab tests and values
        """
        entities = []
        
        # Common lab test patterns
        test_patterns = [
            (r'\b(hemoglobin|hb|hgb)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(glucose|blood sugar)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(creatinine)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(cholesterol|total cholesterol)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(tsh|thyroid stimulating hormone)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(alt|sgpt)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(ast|sgot)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(hba1c|a1c)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(wbc|white blood cell)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
            (r'\b(rbc|red blood cell)\s*:?\s*(\d+\.?\d*)', 'TEST', 'VALUE'),
        ]
        
        for pattern, test_type, value_type in test_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(1),
                    "type": test_type,
                    "confidence": 0.75,
                    "value": match.group(2) if len(match.groups()) > 1 else None
                })
        
        # Disease mentions
        disease_keywords = [
            "diabetes", "hypertension", "anemia", "infection", "inflammation",
            "kidney disease", "liver disease", "heart disease", "thyroid disorder"
        ]
        
        for disease in disease_keywords:
            if disease.lower() in text.lower():
                entities.append({
                    "text": disease,
                    "type": "DISEASE",
                    "confidence": 0.70
                })
        
        return entities


# Global BioBERT service instance
biobert_service = BioBERTService()

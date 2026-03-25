from datetime import datetime
from db.mongo import audit_logs_collection

class AuditService:
    @staticmethod
    def log_event(user_email: str, action: str, details: dict = None, ip_address: str = None):
        """
        Logs a governance event to the persistent audit trail.
        
        Args:
            user_email: The user performing the action.
            action: A standardized string key (e.g., "LAB_ANALYSIS_GENERATED").
            details: A dictionary of relevant metadata (e.g., findings, filenames).
            ip_address: Optional IP of the request.
        """
        if details is None:
            details = {}

        event = {
            "timestamp": datetime.utcnow(),
            "user_email": user_email,
            "action": action,
            "details": details,
            "ip_address": ip_address
        }
        
        try:
            audit_logs_collection.insert_one(event)
            print(f"Audit Logged: {action} by {user_email}")
        except Exception as e:
            print(f"Audit Log Error: {e}")

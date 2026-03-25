from core.config import client, db, users_collection, reports_collection, conversations_collection, MONGO_URI
from pymongo import ASCENDING, DESCENDING

# Backwards compatibility / export expected symbols
mongo_client = client
database = db
MONGO_URL = MONGO_URI

# Additional collections
audit_logs_collection = db["audit_logs"]
role_history_collection = db["role_history"]
lab_results_collection = db["lab_results"]

# =============================
# DEPENDENCY
# =============================
def get_db():
    return db

# =============================
# INDEXES (PERFORMANCE + SAFETY)
# =============================
def ensure_indexes():
    # Users
    users_collection.create_index(
        [("email", ASCENDING)], unique=True
    )

    # Conversations
    conversations_collection.create_index(
        [("user_email", ASCENDING), ("timestamp", DESCENDING)]
    )

    # Medical reports
    reports_collection.create_index(
        [("report_id", ASCENDING)], unique=True
    )
    reports_collection.create_index(
        [("user_email", ASCENDING), ("uploaded_at", DESCENDING)]
    )

    # Lab results (time-series ready)
    lab_results_collection.create_index(
        [("patient_id", ASCENDING), ("test_name", ASCENDING)]
    )
    lab_results_collection.create_index(
        [("test_name", ASCENDING), ("recorded_at", ASCENDING)]
    )

    # Audit logs
    audit_logs_collection.create_index(
        [("event_type", ASCENDING), ("timestamp", DESCENDING)]
    )

    # Role history
    role_history_collection.create_index(
        [("target_user", ASCENDING), ("timestamp", DESCENDING)]
    )

# =============================
# INIT (CALL ON STARTUP)
# =============================
def init_mongo():
    """
    Initialize MongoDB collections and indexes.
    Call once at application startup.
    """
    ensure_indexes()
    print("MongoDB initialized with indexes")

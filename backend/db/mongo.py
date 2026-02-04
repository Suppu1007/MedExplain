# app/db/mongo.py

import os
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure

load_dotenv()

# =============================
# CONFIG
# =============================
MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "medexplain_db")

if not MONGO_URL:
    raise RuntimeError("❌ MONGO_URL missing in environment")

# =============================
# CLIENT
# =============================
try:
    client = MongoClient(
        MONGO_URL,
        tls=True,
        tlsAllowInvalidCertificates=True,
        serverSelectionTimeoutMS=5000,
    )
    # Test connection
    client.admin.command("ping")
except ConnectionFailure as e:
    raise RuntimeError(f"❌ MongoDB connection failed: {e}")

db = client[DB_NAME]

# =============================
# COLLECTIONS
# =============================
users_collection = db["users"]
conversations_collection = db["conversations"]
reports_collection = db["medical_reports"]
audit_logs_collection = db["audit_logs"]
role_history_collection = db["role_history"]
lab_results_collection = db["lab_results"]   # ⬅ NEW (important)

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
# DEPENDENCY
# =============================
def get_db():
    """
    FastAPI dependency.
    Returns main MongoDB database.
    """
    return db


# =============================
# INIT (CALL ON STARTUP)
# =============================
def init_mongo():
    """
    Initialize MongoDB collections and indexes.
    Call once at application startup.
    """
    ensure_indexes()
    print("✅ MongoDB initialized with indexes")

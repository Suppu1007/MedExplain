# app/db/mongo.py

import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME", "medexplain_db")

if not MONGO_URL:
    raise RuntimeError("MONGO_URL missing in environment!")

client = MongoClient(
    MONGO_URL,
    tls=True,
    tlsAllowInvalidCertificates=True,
)

db = client[DB_NAME]

# =============================
# COLLECTIONS
# =============================
users_collection = db["users"]
conversations_collection = db["conversations"]
reports_collection = db["medical_reports"]
audit_logs_collection = db["audit_logs"]
role_history_collection = db["role_history"]


def get_collections():
    return {
        "users": users_collection,
        "conversations": conversations_collection,
        "reports": reports_collection,
        "audit_logs": audit_logs_collection,
        "role_history": role_history_collection,
    }


def get_db():
    """Return main MedExplain database reference."""
    return db

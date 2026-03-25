import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "medexplain")

# Security Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.getenv("SECRET_KEY", "supersecretkey123"))
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Email Configuration
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:1007/auth/google/callback")

# Database Connection
client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

# Collections
users_collection = db["users"]
reports_collection = db["reports"]
conversations_collection = db["conversations"]
knowledge_collection = db["knowledge"]
role_history_collection = db["role_history"]

def ensure_default_admin():
    """
    Ensures that a default admin account exists in the database.
    Uses environment variables ADMIN_EMAIL and ADMIN_PASSWORD if available,
    otherwise defaults to medexplain.ai / admin@123.
    """
    # Import hash_password here to avoid circular import
    from core.security import hash_password

    admin_email = os.getenv("ADMIN_EMAIL", "admin@medexplain.ai")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin@123")
    
    existing_admin = users_collection.find_one({"email": admin_email})

    if not existing_admin:
        print(f"Creating default admin account: {admin_email}")
        users_collection.insert_one({
            "name": "System Admin",
            "email": admin_email,
            "password": hash_password(admin_password),
            "role": "Admin",
            "status": "Active",
            "created_at": datetime.utcnow(),
        })
    else:
        print(f"Default admin account already exists: {admin_email}")

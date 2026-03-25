from fastapi.templating import Jinja2Templates
import os

# Centralized template configuration
# This ensures all modules use the correct relative path whether running locally or in Docker
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Docker-safe absolute path fallback
TEMPLATE_DIR = "/frontend/templates"
if not os.path.exists(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.join(BASE_DIR, "frontend", "templates")

print(f"DEBUG: Using TEMPLATE_DIR: {TEMPLATE_DIR}")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

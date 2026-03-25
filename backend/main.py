# app/main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.openapi.utils import get_openapi
from fastapi.templating import Jinja2Templates
import os

from core.security import decode_token
from core.config import ensure_default_admin
from core.templates import templates
from db.mongo import init_mongo


# =====================================================
# FASTAPI APP INITIALIZATION
# =====================================================
app = FastAPI(
    title="MedExplain API",
    version="1.0.0",
    description=(
        "MedExplain – Explainable Medical AI System\n\n"
        "• ML + DL + LLM (RAG)\n"
        "• Clinical Decision Support (Not Diagnosis)\n"
        "• Audit-safe & Ethics-first"
    ),
)


# =====================================================
# STATIC FILES & TEMPLATES
# =====================================================
# templates = Jinja2Templates(directory="frontend/templates") # Moved to core/templates.py

# We resolve static paths relative to the project root for local reliability
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "static")
if not os.path.exists(STATIC_DIR):
    # Fallback to standard Docker path
    STATIC_DIR = "/app/frontend/static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static",
)

from modules.diagnosis import router as diagnosis_router

app.include_router(diagnosis_router.router)
# =====================================================
# CUSTOM OPENAPI (JWT SUPPORT)
# =====================================================
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("components", {}).setdefault("securitySchemes", {})
    schema["components"]["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }

    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

# =====================================================
# ROUTERS — UI
# =====================================================
from modules.auth.router import ui_router as auth_ui_router
from modules.home.router import ui_router as home_ui_router
from modules.profile.router import ui_router as profile_ui_router
from modules.users.router import api_router as users_api_router

from modules.lab_analysis.router import router as lab_router
from modules.imaging.router import router as imaging_router
from modules.history.router import router as history_router

app.include_router(auth_ui_router)
app.include_router(home_ui_router)
app.include_router(profile_ui_router)

# Admin Routers (UI + API)
from modules.admin.router import ui_router as admin_ui_router
from modules.admin.router import api_router as admin_api_router
app.include_router(admin_ui_router)
app.include_router(admin_api_router)

# Knowledge Router
from modules.knowledge.router import ui_router as knowledge_ui_router
app.include_router(knowledge_ui_router)

app.include_router(users_api_router)
app.include_router(lab_router)
app.include_router(imaging_router)
app.include_router(history_router)


# =====================================================
# ROUTERS — API (CORE MODULES TEMPORARILY DISABLED)
# =====================================================
from modules.auth.router import api_router as auth_api_router

app.include_router(auth_api_router)

# CORE MEDICAL MODULES RE-ENABLED
from modules.assistant.router import api_router as assistant_api_router
app.include_router(assistant_api_router)

# Note: Lab and Imaging have combined routers already included above

# =====================================================
# PUBLIC & PROTECTED UI ROUTE GUARD
# =====================================================
PUBLIC_PATHS = {
    "/", "/login", "/signup", "/try",
    "/forgot-password", "/reset-password",
    "/auth/google", "/auth/google/callback",
}

PUBLIC_PREFIXES = [
    "/static", "/favicon", "/docs", "/openapi.json",
]


@app.middleware("http")
async def authentication_guard(request: Request, call_next):
    path = request.url.path.lower()

    token = request.cookies.get("access_token")
    email = decode_token(token) if token else None
    logged_in = email is not None

    # Allow all API routes
    if path.startswith("/api"):
        return await call_next(request)

    # Public UI routes
    if path in PUBLIC_PATHS or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        # Premium Flow: Redirect logged-in users away from auth pages
        if logged_in and path in ["/", "/login", "/signup"]:
            return RedirectResponse("/dashboard", status_code=303)
        return await call_next(request)

    # Protected UI routes
    if not logged_in:
        return RedirectResponse("/login", status_code=303)

    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# =====================================================
# STARTUP INITIALIZATION
# =====================================================
@app.on_event("startup")
def initialize():
    """
    Governance initialization:
    • Ensures default admin exists
    • System ready for secure operation
    """
    ensure_default_admin()
    init_mongo()

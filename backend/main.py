# app/main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.openapi.utils import get_openapi
from fastapi.templating import Jinja2Templates

from app.core.security import decode_token
from app.core.config import ensure_default_admin


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
templates = Jinja2Templates(directory="app/frontend/templates")

app.mount(
    "/static",
    StaticFiles(directory="app/frontend/static"),
    name="static",
)


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
from app.modules.auth.router import ui_router as auth_ui_router
from app.modules.home.router import ui_router as home_ui_router
from app.modules.profile.router import ui_router as profile_ui_router
from app.modules.users.router import api_router as users_api_router

app.include_router(auth_ui_router)
app.include_router(home_ui_router)
app.include_router(profile_ui_router)
app.include_router(users_api_router)


# =====================================================
# ROUTERS — API (CORE MODULES TEMPORARILY DISABLED)
# =====================================================
from app.modules.auth.router import api_router as auth_api_router

app.include_router(auth_api_router)

# 🔒 COMMENTED CORE MEDICAL MODULES
from app.modules.assistant.router import api_router as assistant_api_router
app.include_router(assistant_api_router)
# from app.modules.ingestion.router import api_router as ingestion_api_router
# from app.modules.lab_analysis.router import api_router as lab_api_router
# from app.modules.imaging.router import api_router as imaging_api_router
# from app.modules.cdss.router import api_router as cdss_api_router

# app.include_router(assistant_api_router, prefix="/api/assistant", tags=["Medical Assistant"])
# app.include_router(ingestion_api_router, prefix="/api/ingestion", tags=["Data Ingestion"])
# app.include_router(lab_api_router, prefix="/api/labs", tags=["Lab Analysis"])
# app.include_router(imaging_api_router, prefix="/api/imaging", tags=["Imaging Analysis"])
# app.include_router(cdss_api_router, prefix="/api/cdss", tags=["Clinical Decision Support"])

# =====================================================
# PUBLIC & PROTECTED UI ROUTE GUARD
# =====================================================
PUBLIC_PATHS = {
    "/", "/login", "/signup",
    "/forgot-password", "/reset-password",
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
        if logged_in and path == "/":
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

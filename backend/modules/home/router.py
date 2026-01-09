# app/routes/home.py

from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from datetime import datetime

from app.core.config import (
    users_collection,
    reports_collection,
    conversations_collection,
)
from app.core.dependencies import get_current_user, is_admin
from app.main import templates


# =====================================================
# ROUTERS
# =====================================================
ui_router = APIRouter(tags=["Dashboard"])
api_router = APIRouter(
    prefix="/api/dashboard",
    tags=["Dashboard API"],
    dependencies=[Depends(get_current_user)],
)

# =====================================================
# HOME (POST-LOGIN LANDING)
# =====================================================
@ui_router.get("/home", response_class=HTMLResponse, include_in_schema=False)
async def home(
    request: Request,
    user_email: str = Depends(get_current_user),
):
    """
    Home page after login.
    Lightweight overview + navigation.
    """
    flash = request.cookies.get("flash")
    user = users_collection.find_one({"email": user_email})
    is_admin_user = is_admin(user_email)

    context = {
        "request": request,
        "user": user.get("name") if user else user_email,
        "active_page": "home",
        "is_admin": is_admin_user,
        "flash": flash,
    }

    resp = templates.TemplateResponse("home.html", context)
    if flash:
        resp.delete_cookie("flash")
    return resp


# =====================================================
# DASHBOARD (CORE VIEW)
# =====================================================
@ui_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard(
    request: Request,
    user_email: str = Depends(get_current_user),
):
    """
    Main dashboard.
    Admin → system-wide metrics
    User  → personal medical reports
    """

    flash = request.cookies.get("flash")
    user = users_collection.find_one({"email": user_email})
    is_admin_user = is_admin(user_email)

    # ---------------- ADMIN METRICS ----------------
    if is_admin_user:
        total_reports = reports_collection.count_documents({})
        analyzed_reports = reports_collection.count_documents(
            {"status": "Analyzed"}
        )
        active_users = users_collection.count_documents(
            {"status": "Active"}
        )
        assistant_queries = conversations_collection.count_documents({})

        recent_activity = list(
            reports_collection
            .find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(5)
        )

        context = {
            "request": request,
            "active_page": "dashboard",
            "is_admin": True,
            "user": user.get("name") if user else user_email,
            "total_reports": total_reports,
            "analyzed_reports": analyzed_reports,
            "active_users": active_users,
            "assistant_queries": assistant_queries,
            "recent_activity": recent_activity,
            "flash": flash,
        }

    # ---------------- USER METRICS ----------------
    else:
        my_reports = list(
            reports_collection
            .find({"user_email": user_email}, {"_id": 0})
            .sort("created_at", -1)
        )

        my_reports_count = len(my_reports)
        my_analyzed_reports = len(
            [r for r in my_reports if r.get("status") == "Analyzed"]
        )
        my_pending_reports = len(
            [r for r in my_reports if r.get("status") != "Analyzed"]
        )

        context = {
            "request": request,
            "active_page": "dashboard",
            "is_admin": False,
            "user": user.get("name") if user else user_email,
            "my_reports": my_reports,
            "my_reports_count": my_reports_count,
            "my_analyzed_reports": my_analyzed_reports,
            "my_pending_reports": my_pending_reports,
            "flash": flash,
        }

    resp = templates.TemplateResponse("dashboard.html", context)
    if flash:
        resp.delete_cookie("flash")
    return resp


# =====================================================
# DASHBOARD SUMMARY API (JSON)
# =====================================================
@api_router.get("/summary")
async def dashboard_summary(
    user_email: str = Depends(get_current_user),
):
    """
    Lightweight JSON summary.
    Useful for charts / async UI widgets.
    """
    is_admin_user = is_admin(user_email)

    if is_admin_user:
        return {
            "role": "Admin",
            "total_reports": reports_collection.count_documents({}),
            "analyzed_reports": reports_collection.count_documents(
                {"status": "Analyzed"}
            ),
            "active_users": users_collection.count_documents(
                {"status": "Active"}
            ),
            "assistant_queries": conversations_collection.count_documents({}),
        }

    user_reports = list(
        reports_collection.find({"user_email": user_email})
    )

    return {
        "role": "User",
        "my_reports": len(user_reports),
        "my_analyzed_reports": len(
            [r for r in user_reports if r.get("status") == "Analyzed"]
        ),
        "my_pending_reports": len(
            [r for r in user_reports if r.get("status") != "Analyzed"]
        ),
    }


# =====================================================
# AI ASSISTANT UI
# =====================================================
@ui_router.get("/assistant", response_class=HTMLResponse, include_in_schema=False)
async def assistant_page(
    request: Request,
    user_email: str = Depends(get_current_user),
):
    """
    AI Medical Assistant chat interface.
    Accessible to logged-in users only.
    """

    return templates.TemplateResponse(
        "assistant.html",
        {
            "request": request,
            "active_page": "assistant",
            "is_admin": is_admin(user_email),
        }
    )

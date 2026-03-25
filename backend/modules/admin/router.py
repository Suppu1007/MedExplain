from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse
from datetime import datetime

from core.config import users_collection, role_history_collection
from core.dependencies import admin_required, get_current_user
from utils.email_utils import send_role_change_email
from main import templates


# =====================================================
# ROUTER SETUP
# =====================================================
ui_router = APIRouter(
    tags=["Admin (UI)"],
    dependencies=[Depends(admin_required)],
)

api_router = APIRouter(
    prefix="/api/admin",
    tags=["Admin (API)"],
    dependencies=[Depends(admin_required)],
)

# =====================================================
# UI: USER & ACCESS MANAGEMENT
# =====================================================
@ui_router.get("/admin/users", include_in_schema=False)
async def admin_users_page(
    request: Request,
    admin_email: str = Depends(get_current_user),
):
    """
    Displays all platform users with roles & status.
    Used for governance and access control.
    """
    users = list(users_collection.find({}, {"password": 0}))
    flash = request.cookies.get("flash")

    # Calculate System Metrics for the View
    total_users = users_collection.count_documents({})
    active_users = users_collection.count_documents({"status": "Active"})
    pending_users = users_collection.count_documents({"status": "Pending"})

    response = templates.TemplateResponse(
        "admin_users.html",
        {
            "request": request,
            "users": users,
            "active_page": "users",
            "is_admin": True,
            "flash": flash,
            "metrics": {
                "total": total_users,
                "active": active_users,
                "pending": pending_users
            }
        },
    )

    if flash:
        response.delete_cookie("flash")

    return response


# =====================================================
# UI: UPDATE USER ROLE (WITH AUDIT + EMAIL)
# =====================================================
@ui_router.post("/admin/users/update-role", include_in_schema=False)
async def update_user_role(
    user_email_target: str = Form(...),
    new_role: str = Form(...),
    admin_email: str = Depends(get_current_user),
):
    """
    Updates user role.
    Records audit history and notifies user via email.
    """

    user = users_collection.find_one({"email": user_email_target})
    if not user:
        resp = RedirectResponse("/admin/users", status_code=303)
        resp.set_cookie("flash", "User not found", max_age=4)
        return resp

    old_role = user.get("role", "User")

    users_collection.update_one(
        {"email": user_email_target},
        {"$set": {"role": new_role}},
    )

    # ---- AUDIT + EMAIL (only if changed) ----
    if old_role != new_role:
        role_history_collection.insert_one(
            {
                "target_user": user_email_target,
                "changed_by": admin_email,
                "old_role": old_role,
                "new_role": new_role,
                "timestamp": datetime.utcnow(),
            }
        )

        try:
            send_role_change_email(
                to_email=user_email_target,
                username=user.get("name", user_email_target),
                old_role=old_role,
                new_role=new_role,
                changed_by=admin_email,
            )
        except Exception as e:
            print("Role change email failed:", e)

    resp = RedirectResponse("/admin/users", status_code=303)
    resp.set_cookie("flash", "Role updated successfully", max_age=4)
    return resp


# =====================================================
# UI: USER STATUS UPDATE (ACTIVE / DISABLED)
# =====================================================
@ui_router.post("/admin/users/update-status", include_in_schema=False)
async def update_user_status(
    email: str = Form(...),
    status: str = Form(...),
    admin_email: str = Depends(get_current_user),
):
    """
    Enables or disables user access.
    Used for security & compliance.
    """

    user = users_collection.find_one({"email": email})
    if not user:
        resp = RedirectResponse("/admin/users", status_code=303)
        resp.set_cookie("flash", "User not found", max_age=4)
        return resp

    users_collection.update_one(
        {"email": email},
        {"$set": {"status": status}},
    )

    resp = RedirectResponse("/admin/users", status_code=303)
    resp.set_cookie("flash", "User status updated", max_age=4)
    return resp


# =====================================================
# UI: DELETE USER (PERMANENT)
# =====================================================
@ui_router.post("/admin/users/delete", include_in_schema=False)
async def delete_user(
    email: str = Form(...),
    admin_email: str = Depends(get_current_user),
):
    """
    Permanently deletes a user.
    """
    if email == admin_email:
         resp = RedirectResponse("/admin/users", status_code=303)
         resp.set_cookie("flash", "You cannot delete yourself!", max_age=4)
         return resp
         
    users_collection.delete_one({"email": email})
    # Optional: Cascading delete of reports/history could go here.

    resp = RedirectResponse("/admin/users", status_code=303)
    resp.set_cookie("flash", "User deleted permanently.", max_age=4)
    return resp


# =====================================================
# UI: ROLE CHANGE AUDIT HISTORY
# =====================================================
@ui_router.get("/admin/role-history", include_in_schema=False)
async def role_history_page(
    request: Request,
    admin_email: str = Depends(get_current_user),
):
    """
    Displays role change history.
    This is part of MedExplain's governance & audit trail.
    """

    history = list(
        role_history_collection
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
    )

    # Format timestamps for UI
    for item in history:
        if isinstance(item.get("timestamp"), datetime):
            item["timestamp"] = item["timestamp"].strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

    flash = request.cookies.get("flash")

    response = templates.TemplateResponse(
        "role_history.html",
        {
            "request": request,
            "history": history,
            "active_page": "role_history",
            "is_admin": True,
            "flash": flash,
        },
    )

    if flash:
        response.delete_cookie("flash")

    return response


# =====================================================
# API: LIST USERS (JSON)
# =====================================================
@api_router.get("/users")
async def api_list_users():
    """
    Returns all users (excluding passwords).
    Used by dashboards or admin tooling.
    """
    return {
        "users": list(
            users_collection.find({}, {"password": 0})
        )
    }


# =====================================================
# API: ROLE HISTORY (JSON)
# =====================================================
@api_router.get("/role-history")
async def api_role_history():
    """
    Returns role change audit history.
    """
    history = list(
        role_history_collection
        .find({}, {"_id": 0})
        .sort("timestamp", -1)
    )
    return {"history": history}

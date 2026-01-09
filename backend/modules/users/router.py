from fastapi import APIRouter, Depends, HTTPException, Form
from app.core.config import users_collection
from app.core.dependencies import admin_required


# =====================================================
# ROUTER: ADMIN-ONLY USER MANAGEMENT
# =====================================================
api_router = APIRouter(
    prefix="/api/users",
    tags=["Users (Admin API)"],
    dependencies=[Depends(admin_required)],
)


# =====================================================
# LIST ALL USERS
# =====================================================
@api_router.get("/")
async def list_users():
    """
    Returns all registered users.
    Password hashes are excluded.
    """
    users = list(users_collection.find({}, {"password": 0}))
    return {"users": users}


# =====================================================
# GET USER BY EMAIL
# =====================================================
@api_router.get("/{email}")
async def get_user(email: str):
    """
    Fetch a single user by email.
    """
    user = users_collection.find_one(
        {"email": email},
        {"password": 0},
    )

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"user": user}


# =====================================================
# UPDATE USER STATUS
# =====================================================
@api_router.post("/status")
async def update_status(
    email: str = Form(...),
    status: str = Form(...),
):
    """
    Update user account status.
    Typical values: Active, Disabled, Pending
    """
    allowed_status = {"Active", "Disabled", "Pending"}
    if status not in allowed_status:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Allowed: {allowed_status}",
        )

    result = users_collection.update_one(
        {"email": email},
        {"$set": {"status": status}},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "message": "User status updated successfully",
        "email": email,
        "status": status,
    }

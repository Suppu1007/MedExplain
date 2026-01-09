from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from datetime import datetime

from app.main import templates
from app.core.config import users_collection
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    validate_password,
)
from app.core.dependencies import get_current_user, is_admin
from app.utils.email_utils import (
    send_account_created_email,
    send_reset_password_email,
)

# =====================================================
# HELPERS
# =====================================================
def flash_redirect(url: str, message: str):
    response = RedirectResponse(url, status_code=303)
    response.set_cookie(
        "flash",
        message,
        max_age=5,
        httponly=True,
        samesite="strict",
    )
    return response


def redirect_user(email: str):
    return "/dashboard" if is_admin(email) else "/home"


# =====================================================
# ROUTERS
# =====================================================
ui_router = APIRouter(tags=["Auth"])
api_router = APIRouter(prefix="/api/auth", tags=["Auth"])


# =====================================================
# LANDING
# =====================================================
@ui_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing(request: Request):
    token = request.cookies.get("access_token")
    email = decode_token(token) if token else None

    if email:
        return RedirectResponse(redirect_user(email), status_code=303)

    flash = request.cookies.get("flash")
    response = templates.TemplateResponse(
        "landing.html",
        {"request": request, "flash": flash},
    )

    if flash:
        response.delete_cookie("flash")

    return response


# =====================================================
# LOGIN
# =====================================================
@ui_router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    flash = request.cookies.get("flash")
    response = templates.TemplateResponse(
        "login.html",
        {"request": request, "flash": flash},
    )

    if flash:
        response.delete_cookie("flash")

    return response


@ui_router.post("/login", include_in_schema=False)
async def login(
    email: str = Form(...),
    password: str = Form(...),
):
    email = email.strip().lower()
    user = users_collection.find_one({"email": email})

    if not user or not verify_password(password, user["password"]):
        return flash_redirect("/login", "Invalid email or password")

    token = create_access_token(email)

    response = RedirectResponse(
        redirect_user(email),
        status_code=303,
    )
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="strict",
    )
    response.set_cookie("flash", "Login successful!", max_age=3)

    return response


# =====================================================
# SIGNUP
# =====================================================
@ui_router.get("/signup", response_class=HTMLResponse, include_in_schema=False)
async def signup_page(request: Request):
    flash = request.cookies.get("flash")
    response = templates.TemplateResponse(
        "signup.html",
        {"request": request, "flash": flash},
    )

    if flash:
        response.delete_cookie("flash")

    return response


@ui_router.post("/signup", include_in_schema=False)
async def signup(
    fullname: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    fullname = fullname.strip()
    email = email.strip().lower()

    if password != confirm_password:
        return flash_redirect("/signup", "Passwords do not match")

    if not validate_password(password):
        return flash_redirect("/signup", "Weak password")

    if users_collection.find_one({"email": email}):
        return flash_redirect("/signup", "Email already registered")

    users_collection.insert_one({
        "name": fullname,
        "email": email,
        "password": hash_password(password),
        "role": "User",
        "status": "Active",
        "created_at": datetime.utcnow(),
    })

    send_account_created_email(email, fullname, password)

    return flash_redirect("/login", "Signup successful! Please login")


# =====================================================
# LOGOUT
# =====================================================
@ui_router.get("/logout", include_in_schema=False)
async def logout():
    response = RedirectResponse("/", status_code=303)
    response.delete_cookie("access_token")
    response.set_cookie("flash", "Logged out successfully", max_age=3)
    return response


# =====================================================
# FORGOT PASSWORD
# =====================================================
@ui_router.get("/forgot-password", response_class=HTMLResponse, include_in_schema=False)
async def forgot_password_page(request: Request):
    flash = request.cookies.get("flash")
    response = templates.TemplateResponse(
        "forgot_password.html",
        {"request": request, "flash": flash},
    )

    if flash:
        response.delete_cookie("flash")

    return response


@ui_router.post("/forgot-password", include_in_schema=False)
async def forgot_password(email: str = Form(...)):
    email = email.strip().lower()
    user = users_collection.find_one({"email": email})

    if not user:
        return flash_redirect("/forgot-password", "Email not found")

    reset_token = create_access_token(email)
    reset_link = f"http://localhost:8000/reset-password?token={reset_token}"

    send_reset_password_email(email, user["name"], reset_link)

    return flash_redirect("/login", "Reset link sent to your email")


# =====================================================
# RESET PASSWORD
# =====================================================
@ui_router.get("/reset-password", response_class=HTMLResponse, include_in_schema=False)
async def reset_password_page(request: Request, token: str):
    email = decode_token(token)

    if not email:
        return flash_redirect("/forgot-password", "Invalid or expired reset link")

    return templates.TemplateResponse(
        "reset_password.html",
        {"request": request, "token": token},
    )


@ui_router.post("/reset-password", include_in_schema=False)
async def reset_password(
    token: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    email = decode_token(token)

    if not email:
        return flash_redirect("/forgot-password", "Invalid or expired reset link")

    if password != confirm_password:
        return flash_redirect(
            f"/reset-password?token={token}",
            "Passwords do not match",
        )

    if not validate_password(password):
        return flash_redirect(
            f"/reset-password?token={token}",
            "Weak password",
        )

    users_collection.update_one(
        {"email": email},
        {"$set": {"password": hash_password(password)}},
    )

    return flash_redirect("/login", "Password reset successful!")


# =====================================================
# AUTH API
# =====================================================
@api_router.get("/me")
async def me(user_email: str = Depends(get_current_user)):
    user = users_collection.find_one(
        {"email": user_email},
        {"password": 0},
    )
    return {"user": user}

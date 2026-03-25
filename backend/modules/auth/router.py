from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from datetime import datetime

from core.templates import templates
from core.config import users_collection
from core.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    validate_password,
)
from core.dependencies import get_current_user, is_admin
from utils.email_utils import (
    send_account_created_email,
    send_reset_password_email,
)
from core.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
)
import httpx
import json

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
        samesite="lax",
    )
    return response


async def redirect_user(email: str):
    # Redirect everyone to Home (Launchpad) first
    return "/home"


# =====================================================
# ROUTERS
# =====================================================
ui_router = APIRouter(tags=["Auth"])
api_router = APIRouter(prefix="/api/auth", tags=["Auth"])


# =====================================================
# =====================================================
# LANDING
# =====================================================
@ui_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing(request: Request):
    token = request.cookies.get("access_token")
    email = decode_token(token) if token else None

    if email:
        target = await redirect_user(email)
        return RedirectResponse(target, status_code=303)

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
        await redirect_user(email),
        status_code=303,
    )
    response.set_cookie(
        "access_token",
        token,
        httponly=True,
        samesite="lax",
        path="/",
    )
    response.set_cookie("flash", "Login successful!", max_age=3, path="/")

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
    response.delete_cookie("access_token", path="/")
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
async def forgot_password(request: Request, email: str = Form(...)):
    email = email.strip().lower()
    user = users_collection.find_one({"email": email})

    if not user:
        return flash_redirect("/forgot-password", "Email not found")

    reset_token = create_access_token(email)
    # Use relative path or detect host from request if possible, 
    # but for now, we'll keep it simple and fix the port if needed.
    # Ideally, this should come from an environment variable.
    base_url = str(request.base_url).rstrip("/")
    reset_link = f"{base_url}/reset-password?token={reset_token}"

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
# GOOGLE SSO
# =====================================================

@ui_router.get("/auth/google")
async def google_login():
    if not GOOGLE_CLIENT_ID:
        return flash_redirect("/login", "Google SSO not configured")
        
    scope = "openid email profile"
    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={GOOGLE_REDIRECT_URI}&"
        f"response_type=code&"
        f"scope={scope}"
    )
    return RedirectResponse(url)


@ui_router.get("/auth/google/callback")
async def google_callback(request: Request, code: str = None):
    if not code:
        return flash_redirect("/login", "Google login failed")

    # Exchange code for token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        print(f"DEBUG: Exchanging code for token. Redirect URI: {GOOGLE_REDIRECT_URI}")
        resp = await client.post(token_url, data=data)
        if resp.status_code != 200:
            print(f"DEBUG: Token exchange failed. Status: {resp.status_code}, Body: {resp.text}")
            return flash_redirect("/login", "Google token exchange failed")
        
        token_data = resp.json()
        id_token = token_data.get("id_token")
        print(f"DEBUG: Token exchange successful. Got ID Token.")
        
        # Get user info
        userinfo_url = f"https://www.googleapis.com/oauth2/v3/tokeninfo?id_token={id_token}"
        user_resp = await client.get(userinfo_url)
        if user_resp.status_code != 200:
            print(f"DEBUG: Failed to get user info. Status: {user_resp.status_code}, Body: {user_resp.text}")
            return flash_redirect("/login", "Failed to get Google user info")
            
        user_info = user_resp.json()
        email = user_info.get("email")
        name = user_info.get("name")
        print(f"DEBUG: Got user email: {email}")
        
        if not email:
            return flash_redirect("/login", "Email not provided by Google")

        # Check if user exists
        user = users_collection.find_one({"email": email})
        if not user:
            # Create new user
            users_collection.insert_one({
                "name": name,
                "email": email,
                "password": "[GOOG_SSO]", # Placeholder
                "role": "User",
                "status": "Active",
                "created_at": datetime.utcnow(),
            })

        # Login user
        token = create_access_token(email)
        response = RedirectResponse(await redirect_user(email), status_code=303)
        response.set_cookie(
            "access_token",
            token,
            httponly=True,
            samesite="lax",
            path="/",
        )
        response.set_cookie("flash", f"Welcome back, {name}!", max_age=3, path="/")
        return response


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

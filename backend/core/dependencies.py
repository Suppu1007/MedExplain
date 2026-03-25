from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from typing import Optional
from core.security import decode_token
from core.config import users_collection

# Define OAuth2 scheme (optional, mainly for Swagger UI support)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

async def get_current_user(request: Request) -> str:
    """
    Dependency to get the current authenticated user's email.
    Checks 'Authorization' header first, then 'access_token' cookie.
    """
    token = None
    
    # 1. Check Authorization Header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    
    # 2. Check Cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = decode_token(token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Verify user exists in DB
    user = users_collection.find_one({"email": email})
    if not user:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
         
    return email

async def get_current_user_obj(email: str = Depends(get_current_user)):
    """Returns the full user object."""
    user = users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

async def is_admin(email: str) -> bool:
    """Helper function to check if a user is an admin."""
    user = users_collection.find_one({"email": email})
    return user and user.get("role") == "Admin"

async def admin_required(email: str = Depends(get_current_user)):
    """Dependency to ensure the user is an admin."""
    if not await is_admin(email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return email

# Alias for compatibility
is_admin_by_email = is_admin

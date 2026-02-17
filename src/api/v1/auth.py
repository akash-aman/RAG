"""
Authentication API routes.
POST /api/v1/auth/register — create a new user
POST /api/v1/auth/login    — returns JWT token
GET  /api/v1/auth/me       — returns current user info
GET  /api/v1/auth/users    — list all users (admin only)
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from src.core.security import create_access_token, get_current_user, require_role, Role
from src.models.schemas import (
    AuthRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.models.database import get_user_store

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201,
    summary="Register a new user",
)
async def register(request: RegisterRequest):
    """
    Create a new user account.

    - **username**: 3-50 characters
    - **password**: minimum 6 characters
    - **org_id**: organization ID (default: "default")

    Returns the created user info.
    """
    store = get_user_store()
    try:
        user = store.create_user(
            username=request.username,
            password=request.password,
            org_id=request.org_id,
        )
        return UserResponse(
            username=user.username,
            role=user.role.value,
            org_id=user.org_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post("/login", response_model=TokenResponse, summary="Login")
async def login(request: AuthRequest):
    """
    Authenticate and receive a JWT access token.

    Use the token in the **Authorize** button above as: `Bearer <token>`
    """
    store = get_user_store()
    user = store.authenticate(request.username, request.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role.value,
            "org_id": user.org_id,
            "user_id": user.user_id,
        }
    )

    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse, summary="Current user info")
async def get_me(user: dict = Depends(get_current_user)):
    """Return info about the currently authenticated user."""
    return UserResponse(
        username=user.get("sub", ""),
        role=user.get("role", "user"),
        org_id=user.get("org_id", "default"),
    )


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="List all users (admin)",
)
async def list_users(
    user: dict = Depends(require_role(Role.ADMIN)),
):
    """List all registered users. **Admin only.**"""
    store = get_user_store()
    return [
        UserResponse(
            username=u.username,
            role=u.role.value,
            org_id=u.org_id,
        )
        for u in store.list_users()
    ]

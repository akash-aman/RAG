"""
Security module: JWT authentication, API key validation, RBAC.
"""

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.core.config import get_settings

# ── Password Hashing ────────────────────────────────────────────────────

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Roles ────────────────────────────────────────────────────────────────

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    SERVICE = "service"


# ── JWT Tokens ───────────────────────────────────────────────────────────

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token."""
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT token. Raises HTTPException on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependency ───────────────────────────────────────────────────

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """FastAPI dependency that extracts the current user from the JWT."""
    payload = decode_access_token(credentials.credentials)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )
    return payload


def require_role(*allowed_roles: Role):
    """
    Return a FastAPI dependency that ensures the current user has one of
    the allowed roles.

    Usage:
        @router.get("/admin-only", dependencies=[Depends(require_role(Role.ADMIN))])
    """
    async def _check(user: dict = Depends(get_current_user)):
        user_role = user.get("role", "user")
        if user_role not in [r.value for r in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user
    return _check


# ── Input Sanitization ──────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "disregard all prior",
    "forget everything",
    "you are now",
    "system prompt",
    "ignore all instructions",
]


def sanitize_query(query: str) -> str:
    """
    Basic prompt-injection sanitization.
    Strips known injection patterns and limits length.
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must not be empty",
        )
    # Length limit
    if len(query) > 10_000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query exceeds maximum allowed length (10 000 chars)",
        )
    lowered = query.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query contains disallowed content",
            )
    return query.strip()

"""
Simple in-memory user store for development.
Replace with SQLAlchemy / PostgreSQL in production.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.core.security import hash_password, verify_password, Role


@dataclass
class User:
    username: str
    hashed_password: str
    role: Role = Role.USER
    org_id: str = "default"
    user_id: str = ""

    def __post_init__(self):
        if not self.user_id:
            self.user_id = self.username


class UserStore:
    """Thread-safe(ish) in-memory user registry."""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._seed_defaults()

    def _seed_defaults(self):
        """Create default admin and test user accounts."""
        self.create_user("admin", "admin123", role=Role.ADMIN, org_id="system")
        self.create_user("user", "user123", role=Role.USER, org_id="default")

    def create_user(
        self,
        username: str,
        password: str,
        role: Role = Role.USER,
        org_id: str = "default",
    ) -> User:
        if username in self._users:
            raise ValueError(f"User '{username}' already exists")
        user = User(
            username=username,
            hashed_password=hash_password(password),
            role=role,
            org_id=org_id,
        )
        self._users[username] = user
        return user

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self._users.get(username)
        if user and verify_password(password, user.hashed_password):
            return user
        return None

    def get_user(self, username: str) -> Optional[User]:
        return self._users.get(username)

    def list_users(self) -> list:
        """Return all users."""
        return list(self._users.values())


# Singleton
_store: Optional[UserStore] = None


def get_user_store() -> UserStore:
    global _store
    if _store is None:
        _store = UserStore()
    return _store

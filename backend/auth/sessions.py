"""JWT session helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

import jwt


class SessionManager:
    def __init__(
        self,
        *,
        secret: str,
        cookie_name: str,
        access_ttl_minutes: int,
        refresh_ttl_days: int,
        secure_cookie: bool,
    ) -> None:
        self.secret = secret
        self.cookie_name = cookie_name
        self.access_delta = timedelta(minutes=access_ttl_minutes)
        self.refresh_delta = timedelta(days=refresh_ttl_days)
        self.secure_cookie = secure_cookie

    @staticmethod
    def new_session_token() -> str:
        return str(uuid4())

    def encode(self, *, user_id: str, session_token: str) -> str:
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "jti": session_token,
            "iat": int(now.timestamp()),
            "exp": int((now + self.access_delta).timestamp()),
        }
        return jwt.encode(payload, self.secret, algorithm="HS256")

    def decode(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, self.secret, algorithms=["HS256"])
        except jwt.PyJWTError:
            return None

    def cookie_kwargs(self) -> Dict[str, Any]:
        return {
            "httponly": True,
            "secure": self.secure_cookie,
            "samesite": "lax",
            "path": "/",
        }

    def refresh_expiration(self) -> datetime:
        return datetime.now(timezone.utc) + self.refresh_delta

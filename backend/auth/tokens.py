"""Helpers for issuing verification/reset tokens."""

from __future__ import annotations

import enum
import hashlib
import hmac
import secrets


class TokenPurpose(str, enum.Enum):
    VERIFICATION = "verification"
    PASSWORD_RESET = "password_reset"


class TokenService:
    """Issue random URL-safe tokens and hash them for storage."""

    def __init__(self, *, length: int = 32) -> None:
        self.length = length

    def new_token(self) -> str:
        return secrets.token_urlsafe(self.length)

    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def matches(provided: str, stored_hash: str) -> bool:
        candidate = TokenService.hash_token(provided)
        return hmac.compare_digest(candidate, stored_hash)

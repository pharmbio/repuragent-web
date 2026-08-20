'''High-level authentication workflows.'''

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from email_validator import EmailNotValidError, validate_email

from app.config import (
    AUTH_PEPPER,
    AUTH_REFRESH_EXPIRES_DAYS,
    RESET_TOKEN_TTL_HOURS,
    VERIFICATION_TOKEN_TTL_HOURS,
)
from .emailer import EmailService
from .passwords import PasswordHasher
from .repository import AuthRepository, UserRecord
from .sessions import SessionManager
from .tokens import TokenPurpose, TokenService


class AuthService:
    def __init__(self) -> None:
        self.repo = AuthRepository()
        self.hasher = PasswordHasher(pepper=AUTH_PEPPER)
        self.tokens = TokenService()
        self.email = EmailService()
        self.sessions = SessionManager(refresh_ttl_days=AUTH_REFRESH_EXPIRES_DAYS)

    async def register_user(self, email: str, password: str) -> UserRecord:
        normalized = self._validate_email(email)
        existing = await self.repo.get_user_by_email(normalized)
        if existing:
            raise ValueError("Email already registered")
        hashed = self.hasher.hash(password)
        user = await self.repo.create_user(normalized, hashed)
        await self._send_verification(user)
        return user

    async def login(self, email: str, password: str) -> UserRecord:
        normalized = self._validate_email(email)
        user = await self.repo.get_user_by_email(normalized)
        if not user or not self.hasher.verify(user.password_hash, password):
            raise ValueError("Invalid credentials")
        if not user.is_verified:
            raise ValueError("Email not verified")
        return user

    async def _send_verification(self, user: UserRecord) -> None:
        token = self.tokens.new_token()
        expires = datetime.now(timezone.utc) + timedelta(hours=VERIFICATION_TOKEN_TTL_HOURS)
        await self.repo.record_token(
            user_id=user.id,
            purpose=TokenPurpose.VERIFICATION,
            token_hash=self.tokens.hash_token(token),
            expires_at=expires,
        )
        self.email.send_verification(user.email, token)

    async def verify_email(self, token: str) -> Optional[UUID]:
        match = await self.repo.consume_token(
            purpose=TokenPurpose.VERIFICATION,
            token_hash=self.tokens.hash_token(token),
        )
        if not match:
            return None
        user_id = match["user_id"]
        await self.repo.mark_user_verified(user_id)
        return user_id

    async def send_password_reset(self, email: str) -> None:
        normalized = self._validate_email(email)
        user = await self.repo.get_user_by_email(normalized)
        if not user or not user.is_verified:
            return
        token = self.tokens.new_token()
        expires = datetime.now(timezone.utc) + timedelta(hours=RESET_TOKEN_TTL_HOURS)
        await self.repo.record_token(
            user_id=user.id,
            purpose=TokenPurpose.PASSWORD_RESET,
            token_hash=self.tokens.hash_token(token),
            expires_at=expires,
        )
        self.email.send_password_reset(user.email, token)

    async def reset_password(self, token: str, new_password: str) -> bool:
        match = await self.repo.consume_token(
            purpose=TokenPurpose.PASSWORD_RESET,
            token_hash=self.tokens.hash_token(token),
        )
        if not match:
            return False
        await self.repo.update_password(match["user_id"], self.hasher.hash(new_password))
        return True

    async def create_session(self, user_id: UUID) -> str:
        '''Issue a session token and record it, returning the token.

        Parameters:
        ---------
        user_id (UUID): the account to open a session for.

        Returns:
        ----------
        token (str): the session token, recorded in Postgres so it survives a restart.
        '''

        token = self.sessions.new_session_token()
        await self.repo.create_session(
            user_id=user_id,
            session_token=token,
            expires_at=self.sessions.refresh_expiration(),
        )
        return token

    async def restore_session(self, session_token: str) -> Optional[UserRecord]:
        '''The user behind a session token, or None if it is invalid or expired.

        Called on every download, so a link keeps working only while the session that
        minted it does — a valid signature alone is not authorization.

        Parameters:
        ---------
        session_token (str): the token presented by the browser.

        Returns:
        ----------
        user (UserRecord): the account behind the token, or None when it is invalid or expired.
        '''

        if not session_token:
            return None
        session = await self.repo.get_session(session_token)
        if not session:
            return None
        user_id = session.get("user_id")
        if not user_id:
            return None
        user = await self.repo.get_user_by_id(user_id)
        if not user or not user.is_verified:
            # An account that was un-verified after signing in should not keep a
            # working session.
            await self.repo.revoke_session(session_token)
            return None
        return user

    async def logout(self, session_token: Optional[str]) -> None:
        if session_token:
            await self.repo.revoke_session(session_token)

    def _validate_email(self, email: str) -> str:
        try:
            return validate_email(email, check_deliverability=False).email
        except EmailNotValidError as exc:
            raise ValueError(str(exc)) from exc

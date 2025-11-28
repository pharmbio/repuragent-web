"""Database access helpers for authentication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from psycopg.rows import dict_row

from app.config import logger
from backend.db import get_async_pool
from .tokens import TokenPurpose


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@dataclass
class UserRecord:
    id: UUID
    email: str
    password_hash: str
    is_verified: bool
    last_login: Optional[datetime]


class AuthRepository:
    """Execute auth-related queries using the shared pool."""

    async def create_user(self, email: str, password_hash: str) -> UserRecord:
        pool = await get_async_pool()
        normalized = _normalize_email(email)
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    INSERT INTO users (email, password_hash)
                    VALUES (%s, %s)
                    RETURNING id, email, password_hash, is_verified, last_login
                    """,
                    (normalized, password_hash),
                )
                row = await cur.fetchone()
        return UserRecord(**row)

    async def get_user_by_email(self, email: str) -> Optional[UserRecord]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, email, password_hash, is_verified, last_login
                    FROM users
                    WHERE email = %s
                    LIMIT 1
                    """,
                    (_normalize_email(email),),
                )
                row = await cur.fetchone()
                return UserRecord(**row) if row else None

    async def get_user_by_id(self, user_id: UUID) -> Optional[UserRecord]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, email, password_hash, is_verified, last_login
                    FROM users
                    WHERE id = %s
                    """,
                    (user_id,),
                )
                row = await cur.fetchone()
                return UserRecord(**row) if row else None

    async def mark_user_verified(self, user_id: UUID) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE users
                    SET is_verified = TRUE, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (user_id,),
                )

    async def update_password(self, user_id: UUID, password_hash: str) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE users
                    SET password_hash = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (password_hash, user_id),
                )

    async def record_token(
        self,
        *,
        user_id: UUID,
        purpose: TokenPurpose,
        token_hash: str,
        expires_at: datetime,
    ) -> UUID:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO auth_tokens (user_id, purpose, token_hash, expires_at)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (user_id, purpose.value, token_hash, expires_at),
                )
                row = await cur.fetchone()
        return row["id"] if row else None

    async def consume_token(
        self, *, purpose: TokenPurpose, token_hash: str
    ) -> Optional[Dict[str, Any]]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    UPDATE auth_tokens
                    SET used_at = NOW()
                    WHERE purpose = %s
                      AND token_hash = %s
                      AND used_at IS NULL
                      AND expires_at > NOW()
                    RETURNING id, user_id, purpose, expires_at
                    """,
                    (purpose.value, token_hash),
                )
                return await cur.fetchone()

    async def create_session(
        self, *, user_id: UUID, session_token: str, expires_at: datetime
    ) -> UUID:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO auth_sessions (user_id, session_token, expires_at)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (user_id, session_token, expires_at),
                )
                row = await cur.fetchone()
        return row["id"] if row else None

    async def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT *
                    FROM auth_sessions
                    WHERE session_token = %s
                      AND revoked_at IS NULL
                      AND expires_at > NOW()
                    LIMIT 1
                    """,
                    (session_token,),
                )
                return await cur.fetchone()

    async def revoke_session(self, session_token: str) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE auth_sessions
                    SET revoked_at = NOW()
                    WHERE session_token = %s
                    """,
                    (session_token,),
                )

    async def upsert_thread(
        self,
        *,
        user_id: UUID,
        thread_id: str,
        title: str,
    ) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_threads (user_id, thread_id, title)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (thread_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        updated_at = NOW()
                    WHERE user_threads.user_id = %s
                    """,
                    (user_id, thread_id, title, user_id),
                )

    async def list_threads(self, user_id: UUID) -> list[Dict[str, Any]]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT thread_id, title, created_at, updated_at
                    FROM user_threads
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
                return await cur.fetchall()

    async def delete_thread(self, user_id: UUID, thread_id: str) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM user_threads
                    WHERE user_id = %s AND thread_id = %s
                    """,
                    (user_id, thread_id),
                )

    async def add_file(
        self,
        *,
        user_id: UUID,
        thread_id: str,
        storage_path: str,
        original_name: str,
        mime_type: Optional[str],
        checksum: Optional[str],
        expires_at: datetime,
    ) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO user_files (
                        user_id, thread_id, storage_path, original_name, mime_type, checksum, expires_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        thread_id,
                        storage_path,
                        original_name,
                        mime_type,
                        checksum,
                        expires_at,
                    ),
                )

    async def list_files(self, user_id: UUID, thread_id: str) -> list[Dict[str, Any]]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, storage_path, original_name, mime_type, checksum, created_at, expires_at
                    FROM user_files
                    WHERE user_id = %s AND thread_id = %s AND deleted_at IS NULL
                    ORDER BY created_at DESC
                    """,
                    (user_id, thread_id),
                )
                return await cur.fetchall()

    async def mark_file_deleted(self, user_id: UUID, file_id: UUID) -> None:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE user_files
                    SET deleted_at = NOW()
                    WHERE id = %s AND user_id = %s
                    """,
                    (file_id, user_id),
                )

    async def fetch_expired_files(self, limit: int = 200) -> list[Dict[str, Any]]:
        pool = await get_async_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT id, user_id, thread_id, storage_path
                    FROM user_files
                    WHERE deleted_at IS NULL
                      AND expires_at <= NOW()
                    ORDER BY expires_at ASC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return await cur.fetchall()

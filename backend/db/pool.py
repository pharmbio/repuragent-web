"""Shared async PostgreSQL connection pool."""

from __future__ import annotations

import asyncio
from typing import Optional

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

from app.config import DATABASE_URL, logger

_pool: Optional[AsyncConnectionPool] = None
_pool_lock: Optional[asyncio.Lock] = None


async def get_async_pool() -> AsyncConnectionPool:
    """Return a singleton async connection pool."""

    global _pool
    if _pool and not _pool.closed:
        return _pool

    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()

    async with _pool_lock:
        if _pool and not _pool.closed:
            return _pool

        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is required for auth features")

        conninfo = DATABASE_URL
        if "application_name" not in conninfo:
            separator = "&" if "?" in conninfo else "?"
            conninfo = f"{conninfo}{separator}application_name=repuragent_auth"

        kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }

        pool = AsyncConnectionPool(
            conninfo=conninfo,
            kwargs=kwargs,
            min_size=2,
            max_size=10,
            max_idle=300.0,
            max_lifetime=3600.0,
            timeout=30.0,
            reconnect_timeout=300.0,
            open=False,
        )

        await pool.open()
        _pool = pool
        logger.info("Authentication pool initialized")
        return _pool


async def close_async_pool() -> None:
    """Close the shared pool when the app shuts down."""

    global _pool
    if _pool and not _pool.closed:
        await _pool.close()
        logger.info("Authentication pool closed")
    _pool = None

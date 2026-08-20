'''The LangGraph PostgreSQL checkpointer.

A process-wide singleton over the shared pool. This is what makes conversations
resumable and what makes the plan-approval `interrupt()` work: the paused state
lives in Postgres, not in the browser session.
'''

from __future__ import annotations

import asyncio
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import logger

from .pool import get_async_pool

_checkpointer: Optional[AsyncPostgresSaver] = None
_setup_completed = False
_lock: Optional[asyncio.Lock] = None


async def get_postgres_checkpointer() -> AsyncPostgresSaver:
    global _checkpointer, _setup_completed, _lock

    if _checkpointer is not None:
        return _checkpointer

    if _lock is None:
        _lock = asyncio.Lock()

    async with _lock:
        if _checkpointer is not None:
            return _checkpointer

        pool = await get_async_pool()
        checkpointer = AsyncPostgresSaver(pool)

        if not _setup_completed:
            try:
                await checkpointer.setup()
            except Exception as exc:
                # Concurrent workers race to create the same tables on first boot.
                message = str(exc).lower()
                if not any(
                    keyword in message
                    for keyword in ("already exists", "duplicate", "relation", "table")
                ):
                    raise
                logger.info("LangGraph checkpoint tables already exist")
            _setup_completed = True

        _checkpointer = checkpointer
        logger.info("PostgreSQL checkpointer initialized")
        return _checkpointer


async def check_postgres_connection() -> bool:
    '''Diagnostic: is the checkpointer's pool actually reachable?

    Returns:
    ----------
    reachable (boolean): True when the checkpointer's pool answers, for use as a startup diagnostic.
    '''

    try:
        checkpointer = await get_postgres_checkpointer()
        pool = getattr(checkpointer, "conn", None)
        if pool is None:
            raise RuntimeError("Checkpointer has no connection pool")
        async with pool.connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT 1")
                await cursor.fetchone()
        return True
    except Exception as exc:
        logger.error("PostgreSQL connection check failed: %s", exc)
        return False


async def close_postgres_checkpointer() -> None:
    '''Forget the cached checkpointer so a fresh pool can back a new one.'''

    global _checkpointer, _setup_completed
    _checkpointer = None
    _setup_completed = False

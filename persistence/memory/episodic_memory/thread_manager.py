import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from app.config import DATABASE_URL, logger
from backend.auth.repository import AuthRepository
from backend.db.pool import get_async_pool

_repo = AuthRepository()


def _ensure_uuid(user_id: str) -> Optional[UUID]:
    try:
        return UUID(str(user_id))
    except (ValueError, TypeError):
        logger.error("Invalid user_id provided for thread operation: %s", user_id)
        return None


def _run_sync(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run()" in str(exc):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        raise


async def aload_thread_ids(user_id: Optional[str]) -> List[Dict]:
    if not user_id:
        return []
    user_uuid = _ensure_uuid(user_id)
    if not user_uuid:
        return []
    rows = await _repo.list_threads(user_uuid)
    formatted = []
    for row in rows:
        created_at = row.get("created_at")
        title = row.get("title")
        if isinstance(created_at, datetime):
            created_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_str = str(created_at)
        formatted.append(
            {
                "thread_id": row.get("thread_id"),
                "title": title or f"Conversation {created_str}",
                "created_at": created_str,
                "updated_at": row.get("updated_at"),
            }
        )
    return formatted


def load_thread_ids(user_id: Optional[str] = None) -> List[Dict]:
    return _run_sync(aload_thread_ids(user_id))


async def add_thread_id(user_id: str, thread_id: str, title: Optional[str] = None) -> None:
    user_uuid = _ensure_uuid(user_id)
    if not user_uuid:
        return
    resolved_title = title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    await _repo.upsert_thread(user_id=user_uuid, thread_id=thread_id, title=resolved_title)


async def delete_thread_from_postgres(thread_id: str, *, user_id: Optional[str]) -> bool:
    """
    Delete checkpoint rows for a thread, scoped to the owning user namespace.

    We only allow deletion when the thread_id is namespaced with the user's ID
    to avoid accidental cross-user cleanup if an attacker guesses another thread_id.
    """
    max_retries = 3
    retry_delay = 1

    database_url = DATABASE_URL
    if not database_url:
        logger.warning("DATABASE_URL not set, skipping PostgreSQL deletion")
        return False

    if user_id and not thread_id.startswith(f"{user_id}:"):
        logger.warning("Refusing to delete thread %s for mismatched user %s", thread_id, user_id)
        return False

    for attempt in range(max_retries):
        try:
            pool = await get_async_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cursor:
                    # Delete from checkpoint_writes table first (due to potential foreign key constraints)
                    await cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
                    writes_deleted = cursor.rowcount

                    # Delete from checkpoint_blobs table
                    await cursor.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
                    blobs_deleted = cursor.rowcount

                    # Delete from checkpoints table
                    await cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
                    checkpoints_deleted = cursor.rowcount

            logger.info(
                f"Deleted {checkpoints_deleted} checkpoints, {writes_deleted} checkpoint_writes, "
                f"and {blobs_deleted} checkpoint_blobs for thread {thread_id} from PostgreSQL"
            )
            return True

        except Exception as e:
            error_msg = str(e).lower()

            # Check if it's a connection/SSL error that we can retry
            if any(
                keyword in error_msg
                for keyword in ["ssl", "connection", "closed unexpectedly", "timeout", "operational"]
            ) and attempt < max_retries - 1:
                logger.warning("PostgreSQL deletion attempt %s failed for thread %s: %s", attempt + 1, thread_id, e)
                logger.info("Retrying in %s seconds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue
            else:
                logger.error("Error deleting thread %s from PostgreSQL after %s attempts: %s", thread_id, attempt + 1, e)
                return False

    return False

async def remove_thread_id(user_id: str, thread_id: str) -> None:
    """Remove a thread ID from persistent storage and backing database."""
    user_uuid = _ensure_uuid(user_id)
    if not user_uuid:
        return
    await _repo.delete_thread(user_uuid, thread_id)
    await delete_thread_from_postgres(thread_id, user_id=user_id)


async def update_thread_title(user_id: str, thread_id: str, new_title: str) -> None:
    user_uuid = _ensure_uuid(user_id)
    if not user_uuid:
        return
    await _repo.upsert_thread(user_id=user_uuid, thread_id=thread_id, title=new_title)


def generate_new_thread_id(user_id: Optional[str] = None) -> str:
    """Generate a new unique thread ID, optionally namespaced by user."""
    suffix = str(uuid.uuid4())
    if user_id:
        return f"{user_id}:{suffix}"
    return suffix

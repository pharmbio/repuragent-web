'''Conversations in Postgres: the list, the titles, the rendered timeline.

Thin async wrapper over `AuthRepository`, so the UI layer never writes SQL and the
`(user_id, thread_id)` scoping is applied in exactly one place.

A thread id is `"{user_id}:{uuid4}"`. The prefix is what lets a filesystem path be
derived from a thread id without a second lookup, and it is stripped from the
folder name so the id does not appear twice in the path.
'''

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.config import DEFAULT_CONVERSATION_TITLE, logger
from app.state import ConversationMeta
from backend.auth.repository import AuthRepository

_repo = AuthRepository()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _to_uuid(user_id: str) -> UUID:
    return UUID(str(user_id))


def new_thread_id(user_id: str) -> str:
    return f"{user_id}:{uuid4()}"


def _row_to_meta(row: Dict[str, Any], user_id: str) -> ConversationMeta:
    return ConversationMeta(
        thread_id=row["thread_id"],
        title=row.get("title") or DEFAULT_CONVERSATION_TITLE,
        created_at=_to_iso(row.get("created_at")),
        updated_at=_to_iso(row.get("updated_at")),
        user_id=user_id,
    )


async def load_threads(user_id: str) -> List[ConversationMeta]:
    await _repo.ensure_schema()
    rows = await _repo.list_threads(_to_uuid(user_id))
    return [_row_to_meta(row, user_id) for row in rows]


async def create_thread(user_id: str, title: Optional[str] = None) -> ConversationMeta:
    await _repo.ensure_schema()
    resolved_title = (title or DEFAULT_CONVERSATION_TITLE).strip() or DEFAULT_CONVERSATION_TITLE
    thread_id = new_thread_id(user_id)
    await _repo.upsert_thread(user_id=_to_uuid(user_id), thread_id=thread_id, title=resolved_title)
    stamp = _now().isoformat()
    return ConversationMeta(
        thread_id=thread_id,
        title=resolved_title,
        created_at=stamp,
        updated_at=stamp,
        user_id=user_id,
    )


async def update_thread_title(user_id: str, thread_id: str, title: str) -> None:
    await _repo.ensure_schema()
    await _repo.update_thread_title(
        _to_uuid(user_id), thread_id, (title or "").strip() or DEFAULT_CONVERSATION_TITLE
    )


async def delete_thread(user_id: str, thread_id: str) -> None:
    await _repo.ensure_schema()
    await _repo.delete_thread(_to_uuid(user_id), thread_id)


async def load_timeline(user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
    await _repo.ensure_schema()
    try:
        return await _repo.get_thread_timeline(_to_uuid(user_id), thread_id)
    except Exception as exc:  # noqa: BLE001 - a missing timeline is not fatal
        logger.warning("Unable to load timeline for %s: %s", thread_id, exc)
        return None


async def save_timeline(user_id: str, thread_id: str, timeline: Dict[str, Any]) -> None:
    await _repo.ensure_schema()
    user_uuid = _to_uuid(user_id)
    # A thread can be written to before it has a row when a run starts on a
    # conversation created in another process.
    if not await _repo.get_thread(user_uuid, thread_id):
        await _repo.upsert_thread(
            user_id=user_uuid, thread_id=thread_id, title=DEFAULT_CONVERSATION_TITLE
        )
    try:
        await _repo.update_thread_timeline(user_uuid, thread_id, timeline or {})
    except Exception as exc:  # noqa: BLE001 - never lose a run over a snapshot write
        logger.warning("Unable to save timeline for %s: %s", thread_id, exc)


__all__ = [
    "create_thread",
    "delete_thread",
    "load_threads",
    "load_timeline",
    "new_thread_id",
    "save_timeline",
    "update_thread_title",
]

'''The read-only demo conversations.

`persistence/memory/demo_threads.json` names published case studies whose results
ship with the repository, so a new visitor can read a real run before creating an
account. They appear in every user's sidebar, after their own conversations, and
they cannot be written to — sending into one would append to a shared transcript
everybody sees.

A demo entry may point its files at a different `(user, thread)` than its own id,
which is how one committed results directory is shared by all viewers.
'''

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from app.config import DEMO_THREADS_FILE, logger


@lru_cache(maxsize=1)
def load_demo_threads() -> tuple[Dict[str, Any], ...]:
    if not DEMO_THREADS_FILE.exists():
        return ()
    try:
        raw = json.loads(DEMO_THREADS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to load demo thread metadata: %s", exc)
        return ()
    if not isinstance(raw, list):
        return ()

    entries: List[Dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict) or not entry.get("thread_id"):
            continue
        entries.append(
            {
                "thread_id": entry["thread_id"],
                "title": entry.get("title") or "Demo conversation",
                "created_at": entry.get("created_at"),
                "is_demo": True,
                "user_id": entry.get("user_id"),
                "results_user_id": entry.get("results_user_id"),
                "results_thread_id": entry.get("results_thread_id"),
            }
        )
    return tuple(entries)


def clear_cache() -> None:
    load_demo_threads.cache_clear()


def combine(user_threads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''The user's own conversations first, then the demos in display order.

    Parameters:
    ---------
    user_threads (list): the signed-in user's own conversations.

    Returns:
    ----------
    threads (list): their conversations first, then the read-only demos in display order.
    '''

    demos = load_demo_threads()
    demo_ids = {entry["thread_id"] for entry in demos}
    combined = [dict(thread) for thread in user_threads if thread.get("thread_id") not in demo_ids]

    # A demo may also exist as a real row (it was run by someone); keep whatever
    # metadata that row has, then let the demo entry override it.
    persisted = {
        thread.get("thread_id"): thread for thread in user_threads if thread.get("thread_id") in demo_ids
    }
    for entry in reversed(demos):
        merged = dict(persisted.get(entry["thread_id"], {}))
        merged.update(entry)
        merged["is_demo"] = True
        combined.append(merged)
    return combined


def results_scope(meta: Optional[Dict[str, Any]]) -> Optional[tuple[str, str]]:
    '''Where a demo thread's files actually live, as `(user_id, thread_id)`.

    Parameters:
    ---------
    meta (dict): a thread's metadata, or None.

    Returns:
    ----------
    scope (tuple): `(user_id, thread_id)` naming where a demo thread's committed files actually live, or None when it is not a demo.
    '''

    if not meta or not meta.get("is_demo"):
        return None
    user_id = meta.get("results_user_id")
    thread_id = meta.get("results_thread_id") or meta.get("thread_id")
    if not user_id or not thread_id:
        return None
    return str(user_id), str(thread_id)


__all__ = ["clear_cache", "combine", "load_demo_threads", "results_scope"]

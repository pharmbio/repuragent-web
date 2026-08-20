'''Persisting and restoring the rendered timeline.

The transcript the user sees is stored as a **snapshot of rendered blocks**, not as
raw LangGraph messages, so reopening a conversation restores the exact agent and
tool entries — including which tool calls succeeded — without replaying graph state
through the renderer again.

`DetachedTimelineWriter` covers the case where the run keeps going while the user is
looking at another conversation. Every event used to cost a load, a full rebuild and
a save; a long unwatched run turned into hundreds of those round trips.
'''

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.conversation_store import load_timeline, save_timeline
from app.state import UIState
from app.ui.chat_timeline import (
    export_timeline_snapshot,
    rebuild_from_plain_messages,
    rebuild_from_timeline_snapshot,
    reset_chat_messages,
)

TIMELINE_STATE_VERSION = 1

# How long a detached run may buffer changes before writing them.
DETACHED_FLUSH_SECONDS = 2.0


def serialize_timeline_state(state: UIState) -> Dict[str, Any]:
    return {
        "timeline_state_version": TIMELINE_STATE_VERSION,
        "timeline_snapshot": export_timeline_snapshot(state),
        "processed_message_ids": sorted(
            str(value) for value in state.processed_message_ids if value
        ),
        "processed_tools_ids": sorted(str(value) for value in state.processed_tools_ids if value),
    }


def extract_timeline_snapshot(payload: Any) -> Any:
    if isinstance(payload, dict) and "timeline_snapshot" in payload:
        return payload["timeline_snapshot"]
    return payload


def restore_timeline_processing_state(state: UIState, payload: Any) -> None:
    '''Restore the de-duplication sets that belong with a snapshot.

    Without them a rebuilt timeline re-ingests messages it has already rendered:
    "have I seen this message id" is the only thing stopping a replayed chunk from
    being appended twice.

    Parameters:
    ---------
    state (UIState): the state to populate.
    payload (Any): the persisted snapshot.
    '''

    if not isinstance(payload, dict) or "timeline_snapshot" not in payload:
        return
    state.processed_message_ids = {
        str(value) for value in (payload.get("processed_message_ids") or []) if value
    }
    state.processed_tools_ids = {
        str(value) for value in (payload.get("processed_tools_ids") or []) if value
    }


def apply_snapshot(state: UIState, payload: Any) -> bool:
    '''Rebuild a state's timeline from a persisted payload.

    Parameters:
    ---------
    state (UIState): the state to rebuild in place.
    payload (Any): the persisted timeline payload.

    Returns:
    ----------
    applied (boolean): True when the snapshot was usable, so reopening a conversation restores the exact tool entries without replaying graph state.
    '''

    snapshot = extract_timeline_snapshot(payload)
    rebuilt = False
    if isinstance(snapshot, dict):
        rebuilt = rebuild_from_timeline_snapshot(state, snapshot)
    elif isinstance(snapshot, list):
        rebuild_from_plain_messages(state, snapshot)
        rebuilt = True
    if not rebuilt:
        reset_chat_messages(state)
    restore_timeline_processing_state(state, payload)
    return rebuilt


async def persist(thread_id: Optional[str], state: UIState) -> None:
    if not thread_id or not state.user_id:
        return
    await save_timeline(state.user_id, thread_id, serialize_timeline_state(state))


async def load_detached(user_id: str, thread_id: str) -> UIState:
    '''A standalone state carrying only this thread's rendered timeline.

    Parameters:
    ---------
    user_id (str): owner of the conversation.
    thread_id (str): the conversation to load.

    Returns:
    ----------
    state (UIState): a standalone state carrying only that thread's rendered timeline, for a run the user is not watching.
    '''

    detached = UIState()
    detached.user_id = user_id
    apply_snapshot(detached, await load_timeline(user_id, thread_id))
    return detached


class DetachedTimelineWriter:
    '''Buffers timeline updates for a run whose thread nobody is watching.'''

    def __init__(self, user_id: Optional[str], thread_id: str) -> None:
        self._user_id = user_id
        self._thread_id = thread_id
        self._state: Optional[UIState] = None
        self._dirty = False
        self._last_flush = 0.0

    async def state(self) -> UIState:
        if self._state is None:
            self._state = await load_detached(self._user_id or "", self._thread_id)
            self._last_flush = time.monotonic()
        return self._state

    def mark_dirty(self) -> None:
        self._dirty = True

    async def maybe_flush(self, *, force: bool = False) -> None:
        if not self._dirty or self._state is None or not self._user_id:
            return
        now = time.monotonic()
        if not force and now - self._last_flush < DETACHED_FLUSH_SECONDS:
            return
        await save_timeline(self._user_id, self._thread_id, serialize_timeline_state(self._state))
        self._dirty = False
        self._last_flush = now

    def discard(self) -> None:
        '''Drop the buffer after the viewer reattached and took over.'''

        self._state = None
        self._dirty = False


__all__ = [
    "DETACHED_FLUSH_SECONDS",
    "DetachedTimelineWriter",
    "apply_snapshot",
    "load_detached",
    "persist",
    "serialize_timeline_state",
]

'''Sign-in, conversation switching, uploads, and episodic-memory extraction.

Everything here is a Gradio handler or something a handler calls directly.
'''

from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from app import demo_threads, timeline_store
from app.config import DEFAULT_CONVERSATION_TITLE, PASSWORD_MIN_LENGTH, logger
from app.conversation_store import create_thread, delete_thread, load_threads, load_timeline
from app.downloads import is_data_path
from app.files import (
    clear_thread_uploads,
    delete_thread_data,
    hash_file,
    list_upload_files,
    refresh_thread_files,
    save_uploaded_file,
)
from app.langgraph_runner import read_pending_approval
from app.state import UIState
from app.ui.chat_timeline import reset_chat_messages
from app.ui.conversation_panel import (
    conversation_panel_update,
    invalidate_panel_cache,
    thread_to_dict,
)
from app.ui.progress_panel import progress_update
from app.ui.projection import auth_message, render, render_auth
from backend.auth.service import AuthService

AUTH_SERVICE = AuthService()


def initialize_state() -> UIState:
    return UIState()


def _validate_password_strength(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
    if password.isdigit() or password.isalpha():
        raise ValueError("Password must include both letters and numbers")


def reset_user_state(state: UIState) -> None:
    state.user_id = None
    state.user_email = None
    state.is_authenticated = False
    state.is_verified = False
    state.auth_error = None
    state.session_token = None
    state.thread_ids = []
    state.current_thread_id = None
    state.selected_thread_id = None
    state.thread_files.clear()
    state.uploaded_files = []
    state.current_app_config = None
    state.stop_signals = {}
    state.running_threads = set()
    state.pending_approval = None
    state.stale_threads = set()
    state.last_run_at = {}
    state.processed_message_ids = set()
    state.processed_tools_ids = set()
    # Nothing has been sent to the new, signed-out panels yet, so the next render
    # must emit them rather than compare against the previous user's markup.
    invalidate_panel_cache(state)
    reset_chat_messages(state)


# --- Conversations ------------------------------------------------------------


async def refresh_conversation(state: UIState, thread_id: str) -> None:
    '''Load one conversation into view: timeline, files, approval state.

    Parameters:
    ---------
    state (UIState): the state to load into.
    thread_id (str): the conversation to bring into view.
    '''

    state.current_thread_id = thread_id
    state.stale_threads.discard(thread_id)
    state.processed_message_ids = set()
    state.processed_tools_ids = set()

    payload = await load_timeline(state.user_id, thread_id) if state.user_id else None
    timeline_store.apply_snapshot(state, payload)

    state.ensure_thread_storage(thread_id)
    refresh_thread_files(state, thread_id)
    state.uploaded_files = [
        record for record in state.thread_files.get(thread_id, []) if is_data_path(record.path)
    ]

    # Restore the approval gate from the graph rather than clearing it, so a
    # conversation paused for plan review is still resumable — and still says so —
    # after a thread switch or a page reload.
    state.pending_approval = None if state.is_demo_thread(thread_id) else await read_pending_approval(thread_id)


async def sync_user_threads(state: UIState, ensure_one: bool = True) -> None:
    if not state.user_id:
        state.thread_ids = demo_threads.combine([])
        state.current_thread_id = None
        state.selected_thread_id = None
        state.thread_files = {}
        state.uploaded_files = []
        reset_chat_messages(state)
        return

    threads = await load_threads(state.user_id)
    if not threads and ensure_one:
        await create_thread(state.user_id, DEFAULT_CONVERSATION_TITLE)
        threads = await load_threads(state.user_id)

    state.thread_ids = demo_threads.combine([thread_to_dict(meta) for meta in threads])
    valid_ids = {thread["thread_id"] for thread in state.thread_ids}
    for thread_id in valid_ids:
        state.ensure_thread_storage(thread_id)
    state.stale_threads = {tid for tid in state.stale_threads if tid in valid_ids}

    if state.current_thread_id not in valid_ids:
        # Prefer the user's own newest conversation over a demo.
        own = [
            thread["thread_id"] for thread in state.thread_ids if not thread.get("is_demo")
        ]
        state.current_thread_id = own[0] if own else (
            state.thread_ids[0]["thread_id"] if state.thread_ids else None
        )
    if state.selected_thread_id not in valid_ids:
        state.selected_thread_id = state.current_thread_id

    # Only the conversation actually on screen is scanned: walking every thread's
    # directories here put an O(conversations) filesystem crawl on the login path.
    if state.current_thread_id:
        await refresh_conversation(state, state.current_thread_id)
    else:
        reset_chat_messages(state)
        state.uploaded_files = []


async def activate_thread(thread_id: Optional[str], state: UIState):
    if not thread_id or thread_id not in {thread["thread_id"] for thread in state.thread_ids}:
        return render(state)
    state.selected_thread_id = thread_id
    await refresh_conversation(state, thread_id)
    state.current_app_config = None
    # Switching conversations clears the composer: text drafted for one thread reads
    # as a mistake once another is on screen.
    return render(state, clear_input=True)


async def new_task(state: UIState):
    if not state.user_id:
        state.auth_error = auth_message("Please sign in first.", success=False)
        return render(state)
    meta = await create_thread(state.user_id, DEFAULT_CONVERSATION_TITLE)
    state.current_thread_id = meta.thread_id
    state.selected_thread_id = meta.thread_id
    state.thread_ids = demo_threads.combine(
        [thread_to_dict(meta)] + [t for t in state.thread_ids if not t.get("is_demo")]
    )
    state.thread_files[meta.thread_id] = []
    state.stale_threads.discard(meta.thread_id)
    state.uploaded_files = []
    reset_chat_messages(state)
    state.pending_approval = None
    invalidate_panel_cache(state)
    return render(state, clear_input=True)


async def _delete_thread_action(thread_id: Optional[str], state: UIState):
    if not thread_id or not state.user_id:
        return render(state)
    if state.is_demo_thread(thread_id):
        gr.Warning("Example conversations are shared and cannot be deleted.")
        return render(state)
    own = [thread for thread in state.thread_ids if not thread.get("is_demo")]
    if len(own) <= 1:
        gr.Info("Keep at least one conversation — start a new task first.")
        return render(state)
    await delete_thread(state.user_id, thread_id)
    delete_thread_data(state.user_id, thread_id)
    invalidate_panel_cache(state)
    await sync_user_threads(state, ensure_one=True)
    return render(state)


# --- Handlers -----------------------------------------------------------------


async def on_app_load():
    state = initialize_state()
    await sync_user_threads(state, ensure_one=False)
    return (*render_auth(state, clear_input=True), gr.update(value=""))


async def on_new_task(state: UIState):
    return await new_task(state or initialize_state())


async def on_conversation_action(action_payload: str, state: UIState):
    '''Handle a click routed through the sidebar's hidden action bus.

    Parameters:
    ---------
    action_payload (str): the encoded click, routed through the sidebar's hidden action bus.
    state (UIState): the state to act on.

    Returns:
    ----------
    updates (tuple): the render after handling the click.
    '''

    state = state or initialize_state()
    payload = (action_payload or "").strip()
    if not payload:
        return (*render(state), gr.update(value=""))
    try:
        action = json.loads(payload)
    except json.JSONDecodeError:
        return (*render(state), gr.update(value=""))

    action_type = action.get("type")
    thread_id = action.get("thread_id")
    if action_type == "delete":
        result = await _delete_thread_action(thread_id, state)
    elif action_type == "activate":
        result = await activate_thread(thread_id, state)
    else:
        result = render(state)
    # Clear the bus so the same click is not replayed on the next change event.
    return (*result, gr.update(value=""))


async def on_register(email: str, password: str, confirm: str, state: UIState):
    state = state or initialize_state()
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        state.auth_error = auth_message("Email and password are required.", success=False)
        return render_auth(state)
    if password != (confirm or ""):
        state.auth_error = auth_message("Passwords do not match.", success=False)
        return render_auth(state)
    try:
        _validate_password_strength(password)
        await AUTH_SERVICE.register_user(email, password)
        state.auth_error = auth_message(
            "Account created. Check your inbox for the verification link, then sign in.",
            success=True,
        )
    except Exception as exc:  # noqa: BLE001 - shown to the user verbatim
        state.auth_error = auth_message(str(exc), success=False)
    return render_auth(state)


async def on_login(email: str, password: str, state: UIState):
    state = state or initialize_state()
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        state.auth_error = auth_message("Email and password are required.", success=False)
        return render_auth(state)
    try:
        user = await AUTH_SERVICE.login(email, password)
        state.user_id = str(user.id)
        state.user_email = user.email
        state.is_authenticated = True
        state.is_verified = True
        state.session_token = await AUTH_SERVICE.create_session(user.id)
        state.auth_error = None
        invalidate_panel_cache(state)
        await sync_user_threads(state)
    except Exception as exc:  # noqa: BLE001 - shown to the user verbatim
        reset_user_state(state)
        state.auth_error = auth_message(str(exc), success=False)
    return render_auth(state)


async def on_logout(state: UIState):
    state = state or initialize_state()
    with_token = state.session_token
    reset_user_state(state)
    if with_token:
        await AUTH_SERVICE.logout(with_token)
    state.auth_error = auth_message("Signed out.", success=True)
    await sync_user_threads(state, ensure_one=False)
    return render_auth(state)


async def on_request_password_reset(email: str, state: UIState):
    state = state or initialize_state()
    email = (email or "").strip()
    if not email:
        state.auth_error = auth_message("Enter the email address for your account.", success=False)
        return render_auth(state)
    try:
        await AUTH_SERVICE.send_password_reset(email)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Password reset request failed: %s", exc)
    # Deliberately identical whether or not the address exists, so this cannot be
    # used to discover which addresses have accounts.
    state.auth_error = auth_message(
        "If that address has an account, a reset link is on its way.", success=True
    )
    return render_auth(state)


# --- Files --------------------------------------------------------------------


async def on_files_uploaded(files, state: UIState):
    state = state or initialize_state()
    if not files or not state.user_id or not state.current_thread_id:
        return state, conversation_panel_update(state)
    if state.is_demo_thread(state.current_thread_id):
        gr.Warning('Example conversations are read-only. Click "New task" to upload your own data.')
        return state, conversation_panel_update(state)

    existing = {
        hash_file(path)
        for path in list_upload_files(state.current_thread_id, user_id=state.user_id)
    }
    for file_object in files:
        destination, digest = save_uploaded_file(
            file_object, user_id=state.user_id, thread_id=state.current_thread_id
        )
        if digest in existing:
            # Uploading the same file twice would give the agents two paths to the
            # same data and no way to know they match.
            destination.unlink(missing_ok=True)
            continue
        existing.add(digest)
    refresh_thread_files(state, state.current_thread_id)
    return state, conversation_panel_update(state)


async def on_clear_files(state: UIState):
    state = state or initialize_state()
    if not state.user_id or not state.current_thread_id:
        return state, conversation_panel_update(state)
    if state.is_demo_thread(state.current_thread_id):
        return state, conversation_panel_update(state)
    clear_thread_uploads(state.user_id, state.current_thread_id)
    refresh_thread_files(state, state.current_thread_id)
    return state, conversation_panel_update(state)


async def on_periodic_file_refresh(state: UIState):
    '''Keep the sidebar and the plan panel current while a run is in flight.

    Gated on there being a run: this ticks once a second for every connected
    browser, and unconditionally walking the conversation's directories meant idle
    sessions paid for a filesystem crawl forever.

    The plan panel refreshes on the same tick, so a step that resolves during a long
    stretch of tool calls shows up promptly rather than at the end.

    Parameters:
    ---------
    state (UIState): the state to refresh.

    Returns:
    ----------
    updates (tuple): the sidebar and plan panel, sent only when they changed.
    '''

    if state is None or not state.current_thread_id or not state.running_threads:
        return state, gr.skip(), gr.skip()
    files_changed = refresh_thread_files(state, state.current_thread_id)
    return (
        state,
        conversation_panel_update(state) if files_changed else gr.skip(),
        progress_update(state),
    )


# --- Episodic memory ----------------------------------------------------------


def on_toggle_learning(use_learning: bool, state: UIState):
    state = state or initialize_state()
    state.use_episodic_learning = bool(use_learning)
    return state


def on_extract_learning(state: UIState) -> str:
    '''Record how this conversation was planned, as precedent for future planning.

    Deliberately manual. Extraction reads the whole conversation and calls a model,
    and only the user knows whether a run went well enough to be worth learning from.

    Parameters:
    ---------
    state (UIState): the state holding the conversation to learn from.

    Returns:
    ----------
    message (str): what was recorded, kept as precedent for future planning.
    '''

    if state is None or not state.user_id:
        return "Sign in first."
    thread_id = state.current_thread_id
    if not thread_id:
        return "No conversation selected."
    if state.is_demo_thread(thread_id):
        return "Example conversations are already part of the shipped memory."
    try:
        from core.agents.context import clear_episodic_cache
        from persistence.memory.episodic_memory.episodic_learning import get_orchestrator

        result = get_orchestrator().extract_current_conversation(thread_id)
        # The planner's examples are cached per request text; a new episode must be
        # visible to the next plan.
        clear_episodic_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Episode extraction failed: %s", exc)
        return f"Could not extract: {exc}"

    if result.get("success") and result.get("episodes_extracted"):
        return result.get("message") or "Saved. Future plans will use this as precedent."
    return result.get("message") or "Nothing worth recording from this conversation yet."


__all__ = [
    "AUTH_SERVICE",
    "initialize_state",
    "on_app_load",
    "on_clear_files",
    "on_conversation_action",
    "on_extract_learning",
    "on_files_uploaded",
    "on_login",
    "on_logout",
    "on_new_task",
    "on_periodic_file_refresh",
    "on_register",
    "on_request_password_reset",
    "on_toggle_learning",
    "reset_user_state",
    "sync_user_threads",
]

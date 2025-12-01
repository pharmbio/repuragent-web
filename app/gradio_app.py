from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import shutil
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import gradio as gr
from gradio.themes.utils import colors
from fastapi import APIRouter, FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from app.app_config import AppRunConfig
from app.config import (
    APP_TITLE,
    LOGO_PATH,
    RESULT_RETENTION_DAYS,
    DEMO_THREADS_FILE,
    UI_QUEUE_MAX_SIZE,
    UI_CONCURRENCY_LIMIT,
    FILE_DOWNLOAD_SECRET,
    FILE_DOWNLOAD_TOKEN_TTL_SECONDS,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    logger,
)
from app.langgraph_runner import build_stream_input, stream_langgraph_events, app_session
from app.state import FileRecord, UIState
from app.ui.chat_timeline import (
    append_user_message,
    process_chunk,
    rebuild_from_plain_messages,
    rebuild_from_raw_messages,
)
from backend.memory.episodic_memory.conversation import (
    create_new_conversation,
    load_conversation,
)
from backend.memory.episodic_memory.episodic_learning import get_orchestrator
from backend.memory.episodic_memory.thread_manager import (
    aload_thread_ids,
    remove_thread_id,
    update_thread_title,
)
from backend.auth.repository import AuthRepository
from backend.auth.service import AuthService
from backend.utils.output_paths import (
    get_results_root,
    list_task_files,
    remove_task_dir,
    set_current_task_id,
    reset_current_task_id,
    set_current_user_id,
    reset_current_user_id,
)
from backend.utils.storage_paths import get_data_root


DATA_DIR = get_data_root()

RESULTS_DIR = get_results_root()
ALLOWED_DOWNLOAD_ROOTS = (DATA_DIR.resolve(), RESULTS_DIR.resolve())
DOWNLOAD_ROUTE = "/api/files/download"
_DOWNLOAD_SECRET = (FILE_DOWNLOAD_SECRET or "repuragent-download").encode("utf-8")

EPISODIC_ORCHESTRATOR = None
AUTH_SERVICE = AuthService()
AUTH_ROUTER = APIRouter()
FILES_ROUTER = APIRouter(prefix="/api/files")
AUTH_REPOSITORY = AuthRepository()

INTRO_MARKDOWN = (
    """Hello! I'm **Repuragent** - your AI Agent for Drug Repurposing. My team includes:

    - **Planning Agent:** Decomposes given task into sub-tasks using knowledge from Standard Operating Procedures (SOPs) and biomedical literatures. 
    - **Supervisor Agent:** Keeps track and coordinates agent's plan. 
    - **Prediction Agent:** Makes ADMET predictions using pre-trained models.
    - **Research Agent:** Retrieves relevant Standard Operating Procedures (SOPs), biomedical data from multiple database, and knowledge graph analysis.
    - **Data Agent:** Performs data manipulation, preprocessing, and analysis.
    - **Report Agent:** Summarizes agent workflow and wrtie final report. 

    How can I assist you today?"""
)

INTRO_SKIP_TEXTS = {INTRO_MARKDOWN.strip()}
PASSWORD_MIN_LENGTH = 8


PRIMARY_FERN = colors.Color(
    c50="#dbeee5",
    c100="#cfe3d9",
    c200="#bad4c7",
    c300="#9fc3b2",
    c400="#78a78f",
    c500="#3f7f6e",
    c600="#1f5c55",
    c700="#184842",
    c800="#10322d",
    c900="#0a211f",
    c950="#05110f",
    name="repuragent_primary_green",
)

SECONDARY_SAGE = colors.Color(
    c50="#edf6f2",
    c100="#dfeee6",
    c200="#c8dfd3",
    c300="#b2d1c0",
    c400="#95bfa9",
    c500="#79ad92",
    c600="#5e967c",
    c700="#4b7761",
    c800="#365646",
    c900="#233a2f",
    c950="#142019",
    name="repuragent_secondary_green",
)

REPURAGENT_THEME = (
    gr.themes.Default(
        primary_hue=PRIMARY_FERN,
        secondary_hue=SECONDARY_SAGE,
        neutral_hue=colors.gray,
    ).set(
        color_accent="*primary_600",
        color_accent_soft="#dbeee5",
        color_accent_soft_dark="*primary_700",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_text_color="#f6fbf8",
        button_primary_text_color_hover="#f6fbf8",
    )
)


def _load_demo_threads() -> List[Dict]:
    """Load demo thread metadata from shared short-term storage."""
    if not DEMO_THREADS_FILE.exists():
        return []
    try:
        raw_data = json.loads(DEMO_THREADS_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to load demo thread metadata: %s", exc)
        return []
    if not isinstance(raw_data, list):
        return []
    demo_threads: List[Dict] = []
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue
        thread_id = entry.get("thread_id")
        if not thread_id:
            continue
        title = entry.get("title") or "Demo conversation"
        demo_threads.append(
            {
                "thread_id": thread_id,
                "title": title,
                "created_at": entry.get("created_at"),
                "is_demo": True,
            }
        )
    return demo_threads


def _combine_user_and_demo_threads(user_threads: List[Dict], demo_threads: Optional[List[Dict]] = None) -> List[Dict]:
    """Append demo threads to the user's thread list without duplicating IDs."""
    resolved_demo = demo_threads if demo_threads is not None else _load_demo_threads()
    combined = list(user_threads)
    seen_ids = {thread.get("thread_id") for thread in user_threads}
    for demo in resolved_demo:
        tid = demo.get("thread_id")
        if not tid or tid in seen_ids:
            continue
        combined.append(demo)
    return combined


def _thread_meta(state: UIState, thread_id: Optional[str]) -> Optional[Dict]:
    if not thread_id:
        return None
    for thread in state.thread_ids:
        if thread.get("thread_id") == thread_id:
            return thread
    return None


def _is_demo_thread(state: UIState, thread_id: Optional[str]) -> bool:
    meta = _thread_meta(state, thread_id)
    return bool(meta and meta.get("is_demo"))


def _validate_password_strength(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
    if password.isdigit() or password.isalpha():
        raise ValueError("Password must include both letters and numbers")


def _auth_message(text: str, success: bool = True) -> str:
    prefix = "✅" if success else "⚠️"
    return f"{prefix} {text}"


def _logo_html() -> str:
    """Embed the logo as inline HTML to avoid Gradio's image toolbar."""
    logo_path = Path(LOGO_PATH)
    if not logo_path.exists():
        return ""
    data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    mime, _ = mimetypes.guess_type(str(logo_path))
    mime = mime or "image/png"
    return f'<img src="data:{mime};base64,{data}" alt="{APP_TITLE} logo" class="app-logo-img" />'


def _get_orchestrator():
    global EPISODIC_ORCHESTRATOR
    if EPISODIC_ORCHESTRATOR is None:
        EPISODIC_ORCHESTRATOR = get_orchestrator()
    return EPISODIC_ORCHESTRATOR


def _sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip() or "file"


def _hash_file(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _auth_guard(state: UIState) -> Optional[str]:
    if not state.is_authenticated:
        return "🔒 Please log in to use Repuragent."
    if not state.is_verified:
        return "📧 Check your inbox to verify your email before continuing."
    return None


def _auth_status_text(state: UIState) -> str:
    if not state.is_authenticated:
        return "🔒 Please log in to start a session."
    if not state.is_verified:
        return "📧 Awaiting email verification."
    return f"✅ Signed in as {state.user_email}"


def _guard_and_warn(state: UIState) -> Optional[str]:
    message = _auth_guard(state)
    if message:
        gr.Warning(message)
    return message


def _logout_visibility(state: Optional[UIState]):
    visible = bool(state and state.is_authenticated)
    return gr.update(visible=visible)


def _as_uuid(value: Optional[str]) -> Optional[UUID]:
    try:
        return UUID(str(value))
    except (ValueError, TypeError):
        return None


def _simple_html_page(title: str, message: str) -> str:
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f6f6f6; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }}
        .card {{ background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); max-width: 420px; }}
        h1 {{ margin-top: 0; font-size: 1.5rem; }}
        p {{ line-height: 1.5; }}
        a {{ color: #2563eb; text-decoration: none; }}
    </style>
</head>
<body>
    <div class='card'>
        <h1>{title}</h1>
        <p>{message}</p>
        <p><a href='/'>Return to Repuragent</a></p>
    </div>
</body>
</html>"""


def _reset_form_html(token: str, message: str = "", success: bool = False) -> str:
    alert = ""
    if message:
        color = "#16a34a" if success else "#dc2626"
        alert = f"<p style='color:{color}; font-weight:600;'>{message}</p>"
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='utf-8' />
    <title>Reset password</title>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f0f4f8; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }}
        form {{ background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); width: 360px; }}
        label {{ display: block; margin-top: 1rem; font-weight: 600; }}
        input {{ width: 100%; padding: 0.6rem; margin-top: 0.35rem; border-radius: 8px; border: 1px solid #cbd5f5; }}
        button {{ margin-top: 1.25rem; width: 100%; padding: 0.75rem; border: none; background: #2563eb; color: #fff; border-radius: 8px; font-size: 1rem; cursor: pointer; }}
    </style>
</head>
<body>
    <form method='post'>
        <h1>Reset password</h1>
        {alert}
        <input type='hidden' name='token' value='{token}' />
        <label>New password
            <input type='password' name='password' required />
        </label>
        <label>Confirm password
            <input type='password' name='confirm' required />
        </label>
        <button type='submit'>Update password</button>
    </form>
</body>
</html>"""


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _is_allowed_download_path(path: Path) -> bool:
    for root in ALLOWED_DOWNLOAD_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _encode_download_token(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(_DOWNLOAD_SECRET, body, hashlib.sha256).digest()
    return f"{_urlsafe_b64encode(body)}.{_urlsafe_b64encode(signature)}"


def _decode_download_token(token: str) -> Dict[str, Any]:
    try:
        body_part, sig_part = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Malformed download token") from exc
    body = _urlsafe_b64decode(body_part)
    provided_sig = _urlsafe_b64decode(sig_part)
    expected_sig = hmac.new(_DOWNLOAD_SECRET, body, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(status_code=403, detail="Invalid download token")
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Corrupted download token") from exc
    expires_at = int(payload.get("exp", 0))
    if not expires_at or expires_at < int(time.time()):
        raise HTTPException(status_code=401, detail="Download link expired")
    return payload


def _safe_resolve(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _build_download_payload(
    record: FileRecord,
    thread_id: str,
    *,
    user_id: Optional[str],
    is_demo: bool,
) -> Optional[Dict[str, Any]]:
    if not record.path:
        return None
    resolved_path = _safe_resolve(record.path)
    if not _is_allowed_download_path(resolved_path):
        return None
    issued_at = int(time.time())
    payload: Dict[str, Any] = {
        "path": str(resolved_path),
        "thread_id": thread_id,
        "name": record.name,
        "exp": issued_at + FILE_DOWNLOAD_TOKEN_TTL_SECONDS,
        "ts": issued_at,
        "demo": bool(is_demo),
    }
    if record.hash:
        payload["hash"] = record.hash
    if user_id and not is_demo:
        payload["user_id"] = user_id
    return payload


def _thread_folder_name(user_id: Optional[str], thread_id: Optional[str]) -> str:
    """Derive folder name for uploaded artifacts, stripping redundant user prefixes."""
    if not thread_id:
        return "unassigned"
    if user_id and thread_id.startswith(f"{user_id}:"):
        suffix = thread_id.split(":", 1)[1]
        return suffix or thread_id
    return thread_id


def _thread_data_root(user_id: Optional[str], thread_id: Optional[str]) -> Path:
    """Return the canonical data directory for a specific user/thread pair."""
    user_segment = user_id or "anonymous"
    thread_segment = _thread_folder_name(user_id, thread_id)
    return DATA_DIR / user_segment / thread_segment


def _save_uploaded_file(
    uploaded_file,
    *,
    user_id: Optional[str],
    thread_id: Optional[str],
) -> Tuple[Path, str]:
    """Persist uploaded file to the data directory."""
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_name = getattr(uploaded_file, "orig_name", None) or os.path.basename(uploaded_file.name)
    filename, ext = os.path.splitext(orig_name)
    safe_name = _sanitize_filename(filename)
    final_name = f"{safe_name}_{timestamp}{ext}"
    destination_root = _thread_data_root(user_id, thread_id)
    destination = destination_root / final_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(uploaded_file.name, destination)
    return destination, _hash_file(destination)


def _prune_user_dir(path: Path) -> None:
    """Remove user-specific data directory if empty."""
    if path.exists() and path != DATA_DIR:
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                pass
        except OSError:
            pass


def _delete_thread_data(user_id: Optional[str], thread_id: Optional[str]) -> None:
    """Remove uploaded data artifacts for a thread (best effort)."""
    if not user_id:
        return
    canonical_dir = _thread_data_root(user_id, thread_id)
    legacy_dir = DATA_DIR / (user_id or "anonymous") / (thread_id or "unassigned")
    targets = {canonical_dir, legacy_dir}
    for target_dir in targets:
        shutil.rmtree(target_dir, ignore_errors=True)
        _prune_user_dir(target_dir.parent)


_thread_locks: Dict[str, asyncio.Lock] = {}
_thread_locks_guard = asyncio.Lock()


async def _get_thread_lock(thread_id: str) -> asyncio.Lock:
    async with _thread_locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[thread_id] = lock
        return lock


@asynccontextmanager
async def _thread_execution_lock(thread_id: Optional[str]):
    if not thread_id:
        yield
        return
    lock = await _get_thread_lock(thread_id)
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _reset_user_state(state: UIState) -> None:
    state.user_id = None
    state.user_email = None
    state.is_authenticated = False
    state.is_verified = False
    state.auth_error = None
    state.pending_reset_token = None
    state.thread_ids = []
    state.current_thread_id = None
    state.messages = []
    state.message_lookup.clear()
    state.agent_blocks.clear()
    state.last_agent_block_id = None
    state.tool_call_block_lookup.clear()
    state.thread_files.clear()
    state.uploaded_files = []
    state.current_app_config = None
    state.message_seq = 0
    state.processed_message_ids = set()
    state.processed_tools_ids = set()
    state.processed_content_hashes = set()
    state.waiting_for_approval = False
    state.approval_interrupted = False


MAX_VISIBLE_FILES = 50
FILE_LIST_REFRESH_INTERVAL_SECONDS = 1.0


def _render_thread_files(state: UIState, thread_id: str) -> str:
    files = state.thread_files.get(thread_id, [])
    if not files:
        return "<p class='conversation-card__empty'>No output files yet.</p>"
    items: List[str] = []
    limited_files = files[:MAX_VISIBLE_FILES]
    is_demo_thread = _is_demo_thread(state, thread_id)
    for record in limited_files:
        path_obj = Path(record.path) if record.path else None
        rel_display = str(path_obj) if path_obj else record.name
        payload = _build_download_payload(
            record,
            thread_id,
            user_id=state.user_id,
            is_demo=is_demo_thread,
        )
        if payload:
            token = _encode_download_token(payload)
            href = f"{DOWNLOAD_ROUTE}?token={token}"
            name_markup = (
                "<a class='conversation-card__file-link' href='{href}' "
                "target='_blank' rel='noopener' data-download-link='{token}' "
                "data-file-name='{download_name}' download='{download_name}'>"
                "{label}</a>"
            ).format(
                href=escape(href, quote=True),
                token=escape(token, quote=True),
                label=escape(record.name),
                download_name=escape(record.name, quote=True),
            )
        else:
            name_markup = "<span class='conversation-card__file-name'>{}</span>".format(
                escape(record.name)
            )
        items.append(
            "<li class='conversation-card__file-item' title='{rel}'>"
            "{name}"
            "</li>".format(
                rel=escape(rel_display),
                name=name_markup,
            )
        )
    remaining = len(files) - MAX_VISIBLE_FILES
    more_indicator = ""
    if remaining > 0:
        more_indicator = f"<li class='conversation-card__file-more'>+{remaining} more…</li>"
    return (
        "<div class='conversation-card__files-container'>"
        "<ul class='conversation-card__files'>{}</ul>"
        "{}"
        "</div>"
    ).format("".join(items), more_indicator)


def _conversation_panel_markup(state: UIState) -> str:
    cards: List[str] = [
        "<div class='conversation-list__container' id='conversation-list-root'>",
        "<div class='conversation-list__header'>Conversation</div>",
    ]
    if not state.thread_ids:
        cards.append("<p class='conversation-card__empty'>No conversations yet.</p></div>")
        return "\n".join(cards)
    for thread in state.thread_ids:
        thread_id = thread["thread_id"]
        is_active = thread_id == state.current_thread_id
        raw_title = thread.get("title") or "Conversation"
        if thread.get("is_demo"):
            raw_title = f"{raw_title} · Demo"
        title = escape(raw_title)
        file_block = _render_thread_files(state, thread_id)
        delete_button = ""
        if not thread.get("is_demo"):
            delete_button = (
                "<button type='button' class='conversation-card__delete' "
                "data-delete-thread='{tid}' data-confirm-message='Delete this conversation?'>🗑️</button>"
            ).format(tid=escape(thread_id))
        cards.append(
            "<details class='conversation-card {active}' data-thread-id='{tid}' {open_attr}>"
            "<summary>"
            "<div class='conversation-card__title-row'>"
            "<span class='conversation-card__chevron' aria-hidden='true'></span>"
            "<span class='conversation-card__title'>{title}</span>"
            "{delete_button}"
            "</div>"
            "</summary>"
            "<div class='conversation-card__body'>{files}</div>"
            "</details>".format(
                active="is-active" if is_active else "",
                tid=escape(thread_id),
                open_attr="open" if is_active else "",
                title=title,
                delete_button=delete_button,
                files=file_block,
            )
        )
    cards.append("</div>")
    return "\n".join(cards)


def _conversation_panel_update(state: UIState):
    return gr.update(value=_conversation_panel_markup(state))


async def on_register(email: str, password: str, confirm: str, state: UIState):
    if state is None:
        state = _initialize_state()
    email = (email or "").strip()
    password = password or ""
    confirm = confirm or ""
    if not email or not password:
        return state, _auth_message("Email and password are required", success=False), _logout_visibility(state)
    if password != confirm:
        return state, _auth_message("Passwords do not match", success=False), _logout_visibility(state)
    try:
        _validate_password_strength(password)
        await AUTH_SERVICE.register_user(email, password)
        return (
            state,
            _auth_message("Registration successful! Please verify your email to continue."),
            _logout_visibility(state),
        )
    except ValueError as exc:
        return state, _auth_message(str(exc), success=False), _logout_visibility(state)


async def on_login(email: str, password: str, state: UIState):
    if state is None:
        state = _initialize_state()
    email = (email or "").strip()
    password = password or ""
    if not email or not password:
        return (
            state,
            _auth_message("Email and password are required", success=False),
            _logout_visibility(state),
            _conversation_panel_update(state),
            list(state.messages),
        )
    try:
        user = await AUTH_SERVICE.login(email, password)
        state.user_id = str(user.id)
        state.user_email = user.email
        state.is_authenticated = True
        state.is_verified = True
        state.auth_error = None
        await _sync_user_threads(state)
        return (
            state,
            _auth_message(f"Welcome, {user.email}!"),
            _logout_visibility(state),
            _conversation_panel_update(state),
            list(state.messages),
        )
    except ValueError as exc:
        _reset_user_state(state)
        return (
            state,
            _auth_message(str(exc), success=False),
            _logout_visibility(state),
            _conversation_panel_update(state),
            list(state.messages),
        )


async def on_logout(state: UIState):
    if state is None:
        state = _initialize_state()
    _reset_user_state(state)
    return (
        state,
        _auth_message("Logged out."),
        _logout_visibility(state),
        _conversation_panel_update(state),
        list(state.messages),
    )


async def on_request_password_reset(email: str, state: UIState):
    if state is None:
        state = _initialize_state()
    target_email = (email or state.user_email or "").strip()
    if not target_email:
        return state, _auth_message("Enter your account email", success=False), _logout_visibility(state)
    await AUTH_SERVICE.send_password_reset(target_email)
    return (
        state,
        _auth_message("If the account exists, a reset email is on the way."),
        _logout_visibility(state),
    )


@AUTH_ROUTER.get("/verify", response_class=HTMLResponse)
async def verify_email_route(token: str):
    if not token:
        return HTMLResponse(_simple_html_page("Verification", "Missing token."), status_code=400)
    user_id = await AUTH_SERVICE.verify_email(token)
    if not user_id:
        return HTMLResponse(
            _simple_html_page("Verification", "Link is invalid or has expired."),
            status_code=400,
        )
    return HTMLResponse(_simple_html_page("Verification", "Email verified! You can return to the app."))


@AUTH_ROUTER.get("/reset-password", response_class=HTMLResponse)
async def reset_form(token: str):
    if not token:
        return HTMLResponse(_simple_html_page("Reset password", "Missing token."), status_code=400)
    return HTMLResponse(_reset_form_html(token))


@AUTH_ROUTER.post("/reset-password", response_class=HTMLResponse)
async def reset_submit(token: str = Form(...), password: str = Form(...), confirm: str = Form(...)):
    if password != confirm:
        return HTMLResponse(_reset_form_html(token, "Passwords do not match", success=False), status_code=400)
    try:
        _validate_password_strength(password)
    except ValueError as exc:
        return HTMLResponse(_reset_form_html(token, str(exc), success=False), status_code=400)
    success = await AUTH_SERVICE.reset_password(token, password)
    if not success:
        return HTMLResponse(_reset_form_html(token, "Invalid or expired token", success=False), status_code=400)
    return HTMLResponse(_reset_form_html("", "Password updated successfully.", success=True))


@FILES_ROUTER.get("/download")
async def download_file(token: str):
    payload = _decode_download_token(token)
    path_value = payload.get("path")
    if not path_value:
        raise HTTPException(status_code=400, detail="Missing file path")
    resolved_path = _safe_resolve(path_value)
    if not _is_allowed_download_path(resolved_path):
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    filename = payload.get("name") or resolved_path.name
    mime, _ = mimetypes.guess_type(filename)
    return FileResponse(resolved_path, filename=filename, media_type=mime or "application/octet-stream")


_CONVERSATION_SCRIPT = """
<script>
(function() {
    function findBus() {
        const el = document.getElementById("conversation-action-bus");
        if (!el) {
            return null;
        }
        if (el.matches && el.matches("textarea, input")) {
            return el;
        }
        return el.querySelector ? el.querySelector("textarea, input") : null;
    }

    function sendAction(payload) {
        const bus = findBus();
        if (!bus) {
            return;
        }
        const enriched = Object.assign({ ts: Date.now() }, payload || {});
        bus.value = JSON.stringify(enriched);
        bus.dispatchEvent(new Event("input", { bubbles: true }));
        bus.dispatchEvent(new Event("change", { bubbles: true }));
    }

    function filenameFromDisposition(disposition) {
        if (!disposition) {
            return "";
        }
        const utf8Match = /filename\\*=UTF-8''([^;]+)/i.exec(disposition);
        if (utf8Match && utf8Match[1]) {
            try {
                return decodeURIComponent(utf8Match[1]);
            } catch (error) {
                return utf8Match[1];
            }
        }
        const basicMatch = /filename="?([^";]+)"?/i.exec(disposition);
        if (basicMatch && basicMatch[1]) {
            return basicMatch[1];
        }
        return "";
    }

    async function triggerDownload(anchor) {
        const url = anchor.getAttribute("href");
        if (!url) {
            return;
        }
        anchor.dataset.downloading = "1";
        try {
            const response = await fetch(url, { credentials: "same-origin" });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const blob = await response.blob();
            const headerName = response.headers.get("content-disposition");
            const inferred = filenameFromDisposition(headerName);
            const preferred = anchor.getAttribute("data-file-name") || "";
            const filename = preferred || inferred || anchor.textContent.trim() || "download";
            const blobUrl = window.URL.createObjectURL(blob);
            const temp = document.createElement("a");
            temp.href = blobUrl;
            temp.download = filename;
            document.body.appendChild(temp);
            temp.click();
            window.setTimeout(() => {
                document.body.removeChild(temp);
                window.URL.revokeObjectURL(blobUrl);
            }, 0);
        } catch (error) {
            console.error("Download failed", error);
            window.alert("Unable to download file. Please try again.");
            window.open(url, "_blank", "noopener");
        } finally {
            delete anchor.dataset.downloading;
        }
    }

    function bindHandlers() {
        const root = document.getElementById("conversation-list-root");
        const bus = findBus();
        if (!root || !bus) {
            return;
        }

        root.querySelectorAll("summary").forEach((summary) => {
            if (summary.dataset.repBound === "1") {
                return;
            }
            summary.dataset.repBound = "1";
            summary.addEventListener("click", (event) => {
                if (event.target && event.target.closest("[data-delete-thread]")) {
                    return;
                }
                const parent = summary.closest("details");
                if (!parent) {
                    return;
                }
                const threadId = parent.getAttribute("data-thread-id");
                if (!threadId) {
                    return;
                }
                sendAction({ type: "activate", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-delete-thread]").forEach((button) => {
            if (button.dataset.repBound === "1") {
                return;
            }
            button.dataset.repBound = "1";
            button.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                const threadId = button.getAttribute("data-delete-thread");
                if (!threadId) {
                    return;
                }
                const confirmMessage = button.getAttribute("data-confirm-message");
                if (confirmMessage && !window.confirm(confirmMessage)) {
                    return;
                }
                sendAction({ type: "delete", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-download-link]").forEach((link) => {
            if (link.dataset.repDownloadBound === "1") {
                return;
            }
            link.dataset.repDownloadBound = "1";
            link.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                if (link.dataset.downloading === "1") {
                    return;
                }
                triggerDownload(link);
            });
        });
    }

    function ensureReady() {
        if (!document.getElementById("conversation-list-root") || !findBus()) {
            window.requestAnimationFrame(ensureReady);
            return;
        }
        bindHandlers();
    }

    ensureReady();
    if (window.__repConversationObserver) {
        window.__repConversationObserver.disconnect();
    }
    const observer = new MutationObserver(() => {
        window.requestAnimationFrame(bindHandlers);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.__repConversationObserver = observer;
})();
</script>
"""


async def _refresh_conversation(state: UIState, thread_id: str) -> None:
    app_config = AppRunConfig(user_request=None, use_episodic_learning=False)
    async with app_session(app_config) as app:
        convo = await load_conversation(thread_id, app)
    state.current_thread_id = thread_id
    state.processed_message_ids = set()
    raw_messages = convo.get("raw_messages") or []
    if raw_messages:
        rebuild_from_raw_messages(state, raw_messages)
    else:
        rebuild_from_plain_messages(state, convo.get("messages", []))
    state.processed_message_ids = convo.get("processed_message_ids", set())
    state.processed_content_hashes = set()
    state.ensure_thread_storage(thread_id)
    state.uploaded_files = list(state.thread_files.get(thread_id, []))


def _initialize_state() -> UIState:
    state = UIState()
    _reset_user_state(state)
    state.thread_ids = []
    state.current_thread_id = None
    return state


async def _sync_user_threads(state: UIState, ensure_one: bool = True) -> None:
    if not state.user_id:
        demo_threads = _load_demo_threads()
        state.thread_ids = list(demo_threads)
        state.thread_files = {}
        _hydrate_demo_thread_files(state, demo_threads)
        for thread in state.thread_ids:
            state.ensure_thread_storage(thread["thread_id"])
        valid_ids = {t["thread_id"] for t in state.thread_ids}
        if state.current_thread_id not in valid_ids:
            state.current_thread_id = demo_threads[0]["thread_id"] if demo_threads else None
        state.uploaded_files = list(state.thread_files.get(state.current_thread_id or "", []))
        return
    user_threads = await aload_thread_ids(state.user_id)
    demo_threads = _load_demo_threads()
    if not user_threads and ensure_one and not demo_threads:
        new_conv = await create_new_conversation(state.user_id)
        user_threads = await aload_thread_ids(state.user_id)
        state.current_thread_id = new_conv["thread_id"]
    state.thread_ids = _combine_user_and_demo_threads(user_threads, demo_threads)
    state.thread_files = {}
    await _hydrate_thread_files(state, [t["thread_id"] for t in user_threads])
    _hydrate_demo_thread_files(state, demo_threads)
    for thread in state.thread_ids:
        state.ensure_thread_storage(thread["thread_id"])
    valid_ids = {t["thread_id"] for t in state.thread_ids}
    if state.current_thread_id not in valid_ids:
        if user_threads:
            state.current_thread_id = user_threads[0]["thread_id"]
        elif state.thread_ids:
            state.current_thread_id = state.thread_ids[0]["thread_id"]
        else:
            state.current_thread_id = None
    if state.current_thread_id:
        await _refresh_conversation(state, state.current_thread_id)
        state.uploaded_files = list(state.thread_files.get(state.current_thread_id, []))
    else:
        state.uploaded_files = []


async def _hydrate_thread_files(state: UIState, thread_ids: List[str]) -> Set[str]:
    """Refresh tracked files for the provided thread IDs. Returns the set of threads that changed."""
    user_uuid = _as_uuid(state.user_id)
    if not user_uuid or not thread_ids:
        return set()
    changed_threads: Set[str] = set()
    for tid in thread_ids:
        current_rows = await AUTH_REPOSITORY.list_files(user_uuid, tid)
        recorded_paths = {row.get("storage_path") for row in current_rows}
        disk_files = list_task_files(tid, user_id=state.user_id)
        previous_records = list(state.thread_files.get(tid, []))
        for file_path in disk_files:
            file_str = str(file_path)
            if file_str in recorded_paths:
                continue
            checksum = _hash_file(file_path)
            await _record_file_metadata(state, tid, file_path, checksum)
        rows = await AUTH_REPOSITORY.list_files(user_uuid, tid)
        records: List[FileRecord] = []
        for row in rows:
            storage_path = row.get("storage_path")
            name = row.get("original_name") or (Path(storage_path).name if storage_path else "file")
            records.append(
                FileRecord(
                    path=storage_path,
                    hash=row.get("checksum"),
                    name=name,
                )
            )
        if records != previous_records:
            changed_threads.add(tid)
        state.thread_files[tid] = records
    return changed_threads


def _hydrate_demo_thread_files(state: UIState, demo_threads: List[Dict]) -> None:
    """Populate file listings for demo threads by reading disk artifacts directly."""
    if not demo_threads:
        return
    for thread in demo_threads:
        thread_id = thread.get("thread_id")
        if not thread_id:
            continue
        disk_files = list_task_files(thread_id, user_id=None)
        records: List[FileRecord] = []
        for path in disk_files:
            records.append(
                FileRecord(
                    path=str(path),
                    hash=None,
                    name=path.name,
                )
            )
        state.thread_files[thread_id] = records


async def _refresh_thread_files_for(state: UIState, thread_id: Optional[str]) -> bool:
    """Re-sync files for a single thread and report whether anything changed."""
    if state is None or not thread_id or _is_demo_thread(state, thread_id):
        return False
    changed_threads = await _hydrate_thread_files(state, [thread_id])
    if thread_id in changed_threads:
        state.uploaded_files = list(state.thread_files.get(thread_id, []))
        return True
    return False


async def _record_file_metadata(state: UIState, thread_id: str, path: Path, checksum: str) -> None:
    user_uuid = _as_uuid(state.user_id)
    if not user_uuid:
        return
    mime, _ = mimetypes.guess_type(str(path))
    expires_at = datetime.now(timezone.utc) + timedelta(days=RESULT_RETENTION_DAYS)
    await AUTH_REPOSITORY.add_file(
        user_id=user_uuid,
        thread_id=thread_id,
        storage_path=str(path),
        original_name=path.name,
        mime_type=mime,
        checksum=checksum,
        expires_at=expires_at,
    )


async def on_app_load():
    state = _initialize_state()
    approve_update = gr.update(visible=state.waiting_for_approval)
    auth_status = _auth_status_text(state)
    logout_update = _logout_visibility(state)
    return (
        state,
        auth_status,
        logout_update,
        _conversation_panel_markup(state),
        list(state.messages),
        state.use_episodic_learning,
        gr.update(value=""),
        gr.update(value=""),
    )


def on_toggle_learning(use_learning: bool, state: UIState):
    # Handle case where state is None
    if state is None:
        state = _initialize_state()
        
    state.use_episodic_learning = bool(use_learning)
    return state


async def _activate_thread(thread_id: Optional[str], state: UIState):
    if _guard_and_warn(state):
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    if not thread_id or thread_id not in {t["thread_id"] for t in state.thread_ids}:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    await _refresh_conversation(state, thread_id)
    state.waiting_for_approval = False
    state.current_app_config = None
    state.approval_interrupted = False
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


async def on_new_task(state: UIState):
    # Handle case where state is None
    if state is None:
        state = _initialize_state()
    if _guard_and_warn(state) or not state.user_id:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )

    new_conv = await create_new_conversation(state.user_id)
    state.current_thread_id = new_conv["thread_id"]
    rebuild_from_plain_messages(state, new_conv["messages"])
    state.processed_content_hashes = set()
    state.waiting_for_approval = False
    state.approval_interrupted = False
    state.current_app_config = None
    state.thread_files[new_conv["thread_id"]] = []
    state.uploaded_files = []
    state.processed_message_ids = set()
    await _sync_user_threads(state, ensure_one=False)
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


async def _delete_thread(thread_id: Optional[str], state: UIState):
    if _guard_and_warn(state):
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    if not thread_id or len(state.thread_ids) <= 1 or not state.user_id:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    if _is_demo_thread(state, thread_id):
        gr.Warning("The demo conversation is read-only and cannot be deleted.")
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
    )
    await remove_thread_id(state.user_id, thread_id)
    remove_task_dir(thread_id, user_id=state.user_id)
    _delete_thread_data(state.user_id, thread_id)
    user_threads = await aload_thread_ids(state.user_id)
    demo_threads = _load_demo_threads()
    state.thread_ids = _combine_user_and_demo_threads(user_threads, demo_threads)
    await _hydrate_thread_files(state, [t["thread_id"] for t in user_threads])
    _hydrate_demo_thread_files(state, demo_threads)
    for thread in state.thread_ids:
        state.ensure_thread_storage(thread["thread_id"])
    state.thread_files.pop(thread_id, None)
    if state.current_thread_id == thread_id and state.thread_ids:
        state.current_thread_id = state.thread_ids[-1]["thread_id"]
        await _refresh_conversation(state, state.current_thread_id)
    state.waiting_for_approval = False
    state.approval_interrupted = False
    state.current_app_config = None
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


async def on_conversation_action(action_payload: str, state: UIState):
    # Handle case where state is None
    if state is None:
        state = _initialize_state()
    if _guard_and_warn(state):
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
            gr.update(value=""),
        )
        
    payload = (action_payload or "").strip()
    if not payload:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
            gr.update(value=""),
        )
    try:
        action = json.loads(payload)
    except json.JSONDecodeError:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
            gr.update(value=""),
        )
    action_type = action.get("type")
    thread_id = action.get("thread_id")
    if action_type == "delete":
        result = await _delete_thread(thread_id, state)
    elif action_type == "activate":
        result = await _activate_thread(thread_id, state)
    else:
        result = (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    return (*result, gr.update(value=""))


async def on_files_uploaded(files, state: UIState):
    if state is None:
        state = _initialize_state()
    if _guard_and_warn(state):
        return state, _conversation_panel_update(state)
    if not files or not state.user_id:
        return state, _conversation_panel_update(state)
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _conversation_panel_update(state)
    if _is_demo_thread(state, current_thread):
        gr.Warning('Uploads are disabled for the demo conversation. Click "New Task" to start your own.')
        return state, _conversation_panel_update(state)
    state.ensure_thread_storage(current_thread)
    existing_hashes = {record.hash for record in state.thread_files.get(current_thread, []) if record.hash}
    for file_obj in files:
        destination, file_hash = _save_uploaded_file(
            file_obj,
            user_id=state.user_id,
            thread_id=current_thread,
        )
        if file_hash in existing_hashes:
            destination.unlink(missing_ok=True)
            continue
        await _record_file_metadata(state, current_thread, destination, file_hash)
        existing_hashes.add(file_hash)
    changed_threads = await _hydrate_thread_files(state, [current_thread])
    if current_thread in changed_threads:
        state.uploaded_files = list(state.thread_files.get(current_thread, []))
    return state, _conversation_panel_update(state)


def on_clear_files(state: UIState):
    # Handle case where state is None
    if state is None:
        state = _initialize_state()
    if _guard_and_warn(state):
        return state, _conversation_panel_update(state)
        
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _conversation_panel_update(state)
    if _is_demo_thread(state, current_thread):
        gr.Warning('The demo conversation does not support file management.')
        return state, _conversation_panel_update(state)
    state.thread_files[current_thread] = []
    state.uploaded_files = []
    return state, _conversation_panel_update(state)


async def on_periodic_file_refresh(state: UIState):
    if state is None:
        return state, gr.update()
    thread_id = state.current_thread_id
    if not thread_id:
        return state, gr.update()
    async with _thread_execution_lock(thread_id):
        updated = await _refresh_thread_files_for(state, thread_id)
    if not updated:
        return state, gr.update()
    return state, _conversation_panel_update(state)


def _append_file_paths(prompt: str, state: UIState) -> str:
    files = state.uploaded_files
    if not files:
        return prompt
    if len(files) == 1:
        return f"{prompt}\n\nUploaded file: {files[0].path}"
    addition = "\n\nUploaded files:\n" + "\n".join(f"- {file.path}" for file in files)
    return prompt + addition


async def _run_user_message_internal(prompt: str, state: UIState, *, approve_signal: Optional[str] = None):
    if _guard_and_warn(state):
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
        return
    if _is_demo_thread(state, state.current_thread_id):
        gr.Warning('The demo conversation is read-only. Click "New Task" to start your own conversation.')
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
        return
    prompt = (prompt or "").strip()
    if not prompt and not approve_signal:
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
        return

    if approve_signal:
        state.waiting_for_approval = False
        state.approval_interrupted = False
        app_config = state.current_app_config or AppRunConfig(
            user_request=None,
            use_episodic_learning=state.use_episodic_learning,
        )
        stream_input = build_stream_input(approve_signal, resume=True)
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
    else:
        final_prompt = _append_file_paths(prompt, state)
        append_user_message(state, prompt)
        user_messages = [m for m in state.messages if m.role == "user"]
        if len(user_messages) == 1 and state.current_thread_id and state.user_id:
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            await update_thread_title(state.user_id, state.current_thread_id, title)
            user_threads = await aload_thread_ids(state.user_id)
            state.thread_ids = _combine_user_and_demo_threads(user_threads)
        app_config = AppRunConfig(
            user_request=prompt if state.use_episodic_learning else None,
            use_episodic_learning=state.use_episodic_learning,
        )
        state.current_app_config = app_config
        resume = state.waiting_for_approval
        state.waiting_for_approval = False
        state.approval_interrupted = False
        stream_input = build_stream_input(prompt if resume else final_prompt, resume=resume)
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )

    state.current_app_config = app_config

    context_task_token = set_current_task_id(state.current_thread_id)
    context_user_token = set_current_user_id(state.user_id)
    try:
        stream_iter = stream_langgraph_events(
            app_config,
            stream_input,
            state.current_thread_id,
            user_id=state.user_id,
            check_for_interrupts=True,
        )
        stream_task = asyncio.create_task(stream_iter.__anext__())
        active_thread_id = state.current_thread_id if state else None
        watch_thread_id = (
            active_thread_id
            if active_thread_id and state.user_id and not _is_demo_thread(state, active_thread_id)
            else None
        )
        poll_task = (
            asyncio.create_task(asyncio.sleep(FILE_LIST_REFRESH_INTERVAL_SECONDS))
            if watch_thread_id
            else None
        )
        try:
            while stream_task:
                wait_tasks = [stream_task]
                if poll_task:
                    wait_tasks.append(poll_task)
                done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if poll_task and poll_task in done:
                    poll_task = asyncio.create_task(asyncio.sleep(FILE_LIST_REFRESH_INTERVAL_SECONDS))
                    refreshed = await _refresh_thread_files_for(state, watch_thread_id)
                    if refreshed:
                        yield (
                            state,
                            list(state.messages),
                            gr.update(value=""),
                            _conversation_panel_update(state),
                        )
                if stream_task in done:
                    try:
                        event_type, payload = stream_task.result()
                    except StopAsyncIteration:
                        stream_task = None
                        break
                    if event_type == "chunk":
                        additions = process_chunk(state, payload)
                        if additions:
                            yield (
                                state,
                                list(state.messages),
                                gr.update(value=""),
                                _conversation_panel_update(state),
                            )
                    elif event_type == "complete":
                        state.waiting_for_approval = bool(payload)
                        state.approval_interrupted = bool(payload)
                        yield (
                            state,
                            list(state.messages),
                            gr.update(value=""),
                            _conversation_panel_update(state),
                        )
                        if await _refresh_thread_files_for(state, watch_thread_id):
                            yield (
                                state,
                                list(state.messages),
                                gr.update(value=""),
                                _conversation_panel_update(state),
                            )
                    stream_task = asyncio.create_task(stream_iter.__anext__())
        finally:
            if poll_task:
                poll_task.cancel()
                with suppress(asyncio.CancelledError):
                    await poll_task

        if await _refresh_thread_files_for(state, state.current_thread_id):
            yield (
                state,
                list(state.messages),
                gr.update(value=""),
                _conversation_panel_update(state),
            )
    finally:
        reset_current_task_id(context_task_token)
        reset_current_user_id(context_user_token)


async def _run_user_message(prompt: str, state: UIState, *, approve_signal: Optional[str] = None):
    thread_id = state.current_thread_id
    async with _thread_execution_lock(thread_id):
        async for update in _run_user_message_internal(prompt, state, approve_signal=approve_signal):
            yield update


async def on_send_message(prompt: str, state: UIState):
    # Handle case where state is None
    if state is None:
        state = _initialize_state()
        
    async for update in _run_user_message(prompt, state):
        yield update


def on_extract_learning(state: UIState):
    # Handle case where state is None
    if state is None:
        return "⚠️ No active session."
    if _guard_and_warn(state):
        return "🔒 Please log in first."
        
    orchestrator = _get_orchestrator()
    if not state.current_thread_id:
        return "⚠️ No active thread."
    result = orchestrator.extract_current_conversation(state.current_thread_id)
    if result.get("success") and result.get("episodes_extracted", 0):
        return result.get("message", "✅ Pattern extracted!")
    return result.get("message", "No patterns extracted.")


def build_demo():
    extra_css = """
    :root {
        --app-font: "Inter", "Helvetica Neue", Arial, sans-serif;
    }
    body,
    .gradio-container,
    .gradio-container * {
        font-family: var(--app-font) !important;
    }
    .gradio-container {
        max-width: 1280px;
        width: 95vw;
        margin: 0 auto !important;
        padding-top: 1.25rem;
    }
    #app-header {
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.85rem;
    }
    #app-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
    }
    #app-logo .app-logo-img {
        width: 90px;
        height: 90px;
        object-fit: contain;
        display: block;
    }
    #app-title {
        margin: 0 !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
    }
    #app-title .app-title-text {
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
    }
    #intro-text {
        margin: 0 0 0.75rem 0 !important;
        padding: 0;
    }
    #layout-row {
        gap: 1rem;
        align-items: flex-start;
    }
    #conversation-column {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
    }
    #sidebar-column {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
    }
    #chatbot-panel {
        font-size: 1rem;
        line-height: 1.5;
    }
    #chatbot-panel .prose,
    #chatbot-panel .prose p {
        font-size: inherit !important;
        line-height: inherit !important;
    }
    #chatbot-panel .bot-message *,
    #chatbot-panel .message.bot *,
    #chatbot-panel [data-testid*="assistant"],
    #chatbot-panel [data-testid*="assistant"] * {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    #chatbot-panel .user-message *,
    #chatbot-panel .message.user *,
    #chatbot-panel [data-testid*="user"],
    #chatbot-panel [data-testid*="user"] * {
        font-size: 1rem !important;
    }
    #conversation-action-bus {
        display: none !important;
    }
    details.tool-block {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 12px;
        background: #f9fafb;
        margin: 10px 0;
    }
    details.tool-block summary {
        font-weight: 600;
        color: #374151;
        cursor: pointer;
    }
    details.tool-block pre {
        margin: 8px 0 0 0;
        font-size: 0.95rem;
        background: #f4f6fb;
        padding: 12px;
        border-radius: 8px;
        overflow-x: auto;
        white-space: pre-wrap;
        font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
    }
    .tool-code-block {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 14px 16px;
        margin-top: 10px;
        overflow-x: auto;
    }
    .tool-code-label {
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        font-weight: 600;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .tool-code-block pre {
        margin: 0;
        font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #111827;
        background: transparent;
        white-space: pre;
    }
    #conversation-list {
        margin-top: 0.5rem;
        font-family: inherit;
        width: 100%;
        display: block;
    }
    #conversation-list,
    #conversation-list > div,
    #conversation-list-root {
        width: 100%;
        box-sizing: border-box;
    }
    #conversation-list-root {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        background: #fff;
        overflow: hidden;
        box-shadow: 0 1px 2px rgb(15 23 42 / 0.04);
    }
    .conversation-list__header {
        font-weight: 600;
        padding: 0.75rem 0.85rem;
        border-bottom: 1px solid #e5e7eb;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.85rem;
        color: #4b5563;
        background: #f9fafb;
    }
    details.conversation-card {
        border-bottom: 1px solid #f3f4f6;
    }
    details.conversation-card:last-child {
        border-bottom: none;
    }
    details.conversation-card summary {
        list-style: none;
        padding: 0.6rem 0.85rem;
        cursor: pointer;
        background: transparent;
        transition: background 0.2s ease, color 0.2s ease;
    }
    details.conversation-card summary::-webkit-details-marker {
        display: none;
    }
    details.conversation-card.is-active summary {
        background: #eef2ff;
        color: #111827;
    }
    .conversation-card__title-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .conversation-card__title {
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        flex: 1;
    }
    .conversation-card__chevron {
        width: 12px;
        height: 12px;
        border-right: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
        transform: rotate(45deg);
        transition: transform 0.2s ease;
    }
    details.conversation-card[open] .conversation-card__chevron {
        transform: rotate(-135deg);
    }
    .conversation-card__delete {
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 0.1rem 0.35rem;
        font-size: 0.8rem;
        background: #fff;
        cursor: pointer;
        color: #4b5563;
        transition: background 0.2s ease, color 0.2s ease;
    }
    .conversation-card__delete:hover {
        background: #f3f4f6;
        color: #111827;
    }
    .conversation-card__body {
        background: #f9fafb;
        padding: 0.5rem 0.85rem 0.85rem;
    }
    .conversation-card__files-container {
        max-height: 180px;
        overflow-y: auto;
        padding-right: 0.25rem;
    }
    .conversation-card__files {
        list-style: none;
        margin: 0;
        padding: 0;
    }
    .conversation-card__file-item {
        font-size: 0.82rem;
        padding: 0 0;
        color: #374151;
    }
    .conversation-card__file-name {
        font-weight: 500;
    }
    .conversation-card__file-link {
        font-weight: 600;
        color: #1d4ed8;
        text-decoration: none;
    }
    .conversation-card__file-link:hover,
    .conversation-card__file-link:focus {
        text-decoration: underline;
    }
    .conversation-card__file-more {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    .conversation-card__empty {
        font-size: 0.82rem;
        color: #6b7280;
        margin: 0;
    }
    """
    with gr.Blocks(
        title=APP_TITLE,
        theme=REPURAGENT_THEME,
        css=extra_css,
        head=_CONVERSATION_SCRIPT,
    ) as demo:
        state = gr.State()

        with gr.Row(elem_id="app-header"):
            logo_markup = _logo_html()
            if logo_markup:
                with gr.Column(scale=0, min_width=96):
                    gr.HTML(logo_markup, elem_id="app-logo")
            with gr.Column(scale=1):
                gr.HTML(f"<div class='app-title-text'>{APP_TITLE}</div>", elem_id="app-title")

        with gr.Row(elem_id="layout-row"):
            with gr.Column(scale=1, min_width=280, elem_id="sidebar-column"):
                auth_status_md = gr.Markdown(value="🔒 Please log in to start.")
                with gr.Tabs():
                    with gr.Tab("Login"):
                        login_email = gr.Textbox(label="Email", placeholder="you@example.com")
                        login_password = gr.Textbox(label="Password", type="password")
                        login_btn = gr.Button("Log in", variant="primary")
                    with gr.Tab("Register"):
                        register_email = gr.Textbox(label="Email", placeholder="you@example.com")
                        register_password = gr.Textbox(label="Password", type="password")
                        register_confirm = gr.Textbox(label="Confirm Password", type="password")
                        register_btn = gr.Button("Create account")
                    with gr.Tab("Forgot password"):
                        reset_request_email = gr.Textbox(label="Email", placeholder="you@example.com")
                        reset_request_btn = gr.Button("Send reset link")
                logout_btn = gr.Button("Logout", visible=False)

                conversation_list = gr.HTML(
                    value="",
                    elem_id="conversation-list",
                    min_height=10,
                    container=False,
                )
                conversation_action_bus = gr.Textbox(
                    value="",
                    show_label=False,
                    elem_id="conversation-action-bus",
                )
                file_refresh_timer = gr.Timer(
                    value=FILE_LIST_REFRESH_INTERVAL_SECONDS,
                    active=True,
                    render=False,
                )
                new_task_btn = gr.Button("New Task")

                file_upload = gr.File(label="Upload files", file_count="multiple", file_types=["file"])
                clear_files_btn = gr.Button("Clear Files")
                use_learning = gr.Checkbox(label="Use Episodic Learning", value=True)
                extract_btn = gr.Button("📚 Extract Learning")
                learning_status = gr.Markdown()

            with gr.Column(scale=4, elem_id="conversation-column"):
                gr.Markdown(INTRO_MARKDOWN, elem_id="intro-text")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    type="messages",
                    elem_id="chatbot-panel",
                )
                user_input = gr.Textbox(label="Your message", lines=3)
                send_btn = gr.Button("Send", variant="primary")

        demo.load(
            on_app_load,
            inputs=None,
            outputs=[
                state,
                auth_status_md,
                logout_btn,
                conversation_list,
                chatbot,
                use_learning,
                user_input,
                conversation_action_bus,
            ],
        )

        use_learning.change(
            on_toggle_learning,
            inputs=[use_learning, state],
            outputs=state,
        )

        conversation_action_bus.change(
            on_conversation_action,
            inputs=[conversation_action_bus, state],
            outputs=[state, conversation_list, chatbot, user_input, conversation_action_bus],
        )

        file_refresh_timer.tick(
            on_periodic_file_refresh,
            inputs=[state],
            outputs=[state, conversation_list],
            trigger_mode="always_last",
        )

        login_btn.click(
            on_login,
            inputs=[login_email, login_password, state],
            outputs=[state, auth_status_md, logout_btn, conversation_list, chatbot],
        )

        register_btn.click(
            on_register,
            inputs=[register_email, register_password, register_confirm, state],
            outputs=[state, auth_status_md, logout_btn],
        )

        reset_request_btn.click(
            on_request_password_reset,
            inputs=[reset_request_email, state],
            outputs=[state, auth_status_md, logout_btn],
        )

        logout_btn.click(
            on_logout,
            inputs=state,
            outputs=[state, auth_status_md, logout_btn, conversation_list, chatbot],
        )

        new_task_btn.click(
            on_new_task,
            inputs=state,
            outputs=[state, conversation_list, chatbot, user_input],
        )

        file_upload.upload(
            on_files_uploaded,
            inputs=[file_upload, state],
            outputs=[state, conversation_list],
        )

        clear_files_btn.click(
            on_clear_files,
            inputs=state,
            outputs=[state, conversation_list],
        )

        send_btn.click(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input, conversation_list],
        )
        user_input.submit(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input, conversation_list],
        )

        extract_btn.click(
            on_extract_learning,
            inputs=state,
            outputs=learning_status,
        )

    return demo


def launch():
    demo = build_demo().queue(
        max_size=UI_QUEUE_MAX_SIZE,
        default_concurrency_limit=UI_CONCURRENCY_LIMIT,
    )
    fastapi_app = FastAPI()
    fastapi_app.include_router(AUTH_ROUTER)
    fastapi_app.include_router(FILES_ROUTER)

    from backend.utils.retention import retention_worker  # local import to avoid cycle

    fastapi_app.add_event_handler("startup", retention_worker.start)
    fastapi_app.add_event_handler("shutdown", retention_worker.stop)

    application = gr.mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(application, host=GRADIO_SERVER_NAME, port=GRADIO_SERVER_PORT, log_level="info")

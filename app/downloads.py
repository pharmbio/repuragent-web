'''Signed, expiring download links, and the route that serves them.

A link is an HMAC-signed token, and the route re-validates the session and the
thread scope before serving — so a forged or replayed token still cannot reach
another user's files.

Tokens are minted inside a **time bucket** rather than at the exact second. That
sounds like a detail and is not: the sidebar embeds a token per file, so a
per-second `exp` made the markup change every second even when the file list had
not, which replaced the panel's DOM and threw away the user's scroll position on
every streamed event. The TTL is sized so quantisation cannot shorten a link below
its former fixed lifetime (`TTL - BUCKET = 600 s`).
'''

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import (
    DATA_ROOT,
    DOWNLOAD_TOKEN_BUCKET_SECONDS,
    DOWNLOAD_TOKEN_SECRET,
    DOWNLOAD_TTL_SECONDS,
    RESULTS_ROOT,
)
from app.state import FileRecord
from backend.auth.service import AuthService
from backend.utils.output_paths import conversation_output_root
from backend.utils.storage_paths import thread_data_root

FILES_ROUTER = APIRouter(prefix="/api/files")
DOWNLOAD_ROUTE = "/api/files/download"

ALLOWED_DOWNLOAD_ROOTS = (Path(DATA_ROOT).resolve(), Path(RESULTS_ROOT).resolve())

_auth_service = AuthService()


def safe_resolve(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def is_allowed_download_path(path: Path) -> bool:
    for root in ALLOWED_DOWNLOAD_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def is_data_path(path_value: str) -> bool:
    '''True when the path is an upload rather than an agent-produced artifact.

    Parameters:
    ---------
    path_value (str): the file's path.

    Returns:
    ----------
    is_upload (boolean): True when it is an upload rather than an agent-produced artifact.
    '''

    try:
        safe_resolve(path_value).relative_to(Path(DATA_ROOT).resolve())
        return True
    except ValueError:
        return False


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def encode_download_token(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(DOWNLOAD_TOKEN_SECRET, body, hashlib.sha256).digest()
    return f"{_b64encode(body)}.{_b64encode(signature)}"


def decode_download_token(token: str) -> Dict[str, Any]:
    # Everything up to the signature check runs on attacker-controlled input: bad
    # padding, non-UTF-8 bytes and non-object JSON must all be 4xx, not a 500.
    try:
        body_part, signature_part = token.split(".", 1)
        body = _b64decode(body_part)
        provided = _b64decode(signature_part)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Malformed download token") from exc

    expected = hmac.new(DOWNLOAD_TOKEN_SECRET, body, hashlib.sha256).digest()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=403, detail="Invalid download token")

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Malformed download token") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Malformed download token")

    try:
        expires_at = int(payload.get("exp", 0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Malformed download token") from exc
    if not expires_at or expires_at < int(time.time()):
        raise HTTPException(status_code=401, detail="Download link expired")
    return payload


def build_download_payload(
    record: FileRecord,
    thread_id: str,
    *,
    user_id: Optional[str],
    session_token: Optional[str],
    scope_user_id: Optional[str] = None,
    scope_thread_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    '''The token contents for one file, or None when it must not be offered.

    `scope_*` override which `(user, thread)` directory the file is validated
    against, which is how a demo conversation can serve files that live under the
    account that produced them.

    Parameters:
    ---------
    record (FileRecord): the file being offered.
    thread_id (str): the conversation the link is rendered in.
    user_id (str): the signed-in user, who must own the file.
    session_token (str): their session, bound into the token.
    scope_user_id (str): the owning user when the files live outside the viewer's own scope, as for a demo.
    scope_thread_id (str): the owning thread in that same case.

    Returns:
    ----------
    payload (dict): the signed token's contents, or None when the file must not be offered at all.
    '''

    if not record.path or not user_id or not session_token:
        return None
    resolved = safe_resolve(record.path)
    if not is_allowed_download_path(resolved):
        return None
    # Quantised, so re-rendering the sidebar reproduces a byte-identical token.
    issued_at = (int(time.time()) // DOWNLOAD_TOKEN_BUCKET_SECONDS) * DOWNLOAD_TOKEN_BUCKET_SECONDS
    return {
        "path": str(resolved),
        "thread_id": thread_id,
        "name": record.name,
        "exp": issued_at + DOWNLOAD_TTL_SECONDS,
        "ts": issued_at,
        "user_id": user_id,
        "session_token": session_token,
        "scope_user_id": scope_user_id or user_id,
        "scope_thread_id": scope_thread_id or thread_id,
    }


async def _validate_download_access(payload: Dict[str, Any], resolved_path: Path) -> None:
    user_id = payload.get("user_id")
    session_token = payload.get("session_token")
    thread_id = payload.get("thread_id")
    if not user_id or not session_token or not thread_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # A valid signature proves the token was minted here; it does not prove the
    # requester is still signed in, so the session is checked again.
    restored = await _auth_service.restore_session(session_token)
    if not restored or str(restored.id) != str(user_id):
        raise HTTPException(status_code=403, detail="Access denied")

    scope_user = payload.get("scope_user_id") or user_id
    scope_thread = payload.get("scope_thread_id") or thread_id
    allowed = [
        thread_data_root(scope_thread, user_id=scope_user, create=False).resolve(),
        conversation_output_root(scope_thread, user_id=scope_user).resolve(),
    ]
    for directory in allowed:
        try:
            resolved_path.relative_to(directory)
            return
        except ValueError:
            continue
    raise HTTPException(status_code=403, detail="Access denied")


@FILES_ROUTER.get("/download")
async def download_file(token: str):
    payload = decode_download_token(token)
    path_value = payload.get("path")
    if not path_value:
        raise HTTPException(status_code=400, detail="Missing file path")
    resolved = safe_resolve(path_value)
    if not is_allowed_download_path(resolved):
        raise HTTPException(status_code=403, detail="Access denied")
    await _validate_download_access(payload, resolved)
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    filename = payload.get("name") or resolved.name
    mime, _ = mimetypes.guess_type(filename)
    return FileResponse(resolved, filename=filename, media_type=mime or "application/octet-stream")


__all__ = [
    "DOWNLOAD_ROUTE",
    "FILES_ROUTER",
    "build_download_payload",
    "decode_download_token",
    "encode_download_token",
    "is_allowed_download_path",
    "is_data_path",
    "safe_resolve",
]

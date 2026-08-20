'''Uploads and the sidebar's file list.

Only the **active** conversation is ever scanned. The sidebar renders a file list
for the active card alone — clicking any other card activates that thread and
re-renders — so building them all put an O(conversations) filesystem crawl on the
login path and on the first message of every new conversation.
'''

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from app import demo_threads
from app.downloads import is_data_path
from app.state import FileRecord, UIState
from backend.utils.output_paths import list_task_files, remove_task_dir
from backend.utils.storage_paths import thread_data_root


def sanitize_filename(name: str) -> str:
    cleaned = "".join(char for char in name if char.isalnum() or char in (" ", ".", "_", "-")).strip()
    return cleaned or "file"


def hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_upload_files(thread_id: str, *, user_id: Optional[str]) -> List[Path]:
    root = thread_data_root(thread_id, user_id=user_id, create=False)
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file()]

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=_mtime, reverse=True)
    return files


def scan_thread_files(thread_id: str, *, user_id: Optional[str]) -> List[FileRecord]:
    '''Every file a conversation owns — uploads and outputs — newest first.

    Parameters:
    ---------
    thread_id (str): the conversation to scan.
    user_id (str): its owner.

    Returns:
    ----------
    records (list): every file it owns, uploads and outputs together, newest first.
    '''

    combined: List[Path] = []
    seen: set[Path] = set()
    for path in list_upload_files(thread_id, user_id=user_id) + list_task_files(
        thread_id, user_id=user_id
    ):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        combined.append(path)

    records: List[FileRecord] = []
    for path in combined:
        try:
            modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            # The agent may replace or delete a file between listing and stat.
            continue
        records.append(
            FileRecord(path=str(path), hash=None, name=path.name, uploaded_at=modified_at)
        )
    return records


def save_uploaded_file(
    uploaded_file,
    *,
    user_id: Optional[str],
    thread_id: Optional[str],
) -> Tuple[Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original = getattr(uploaded_file, "orig_name", None) or os.path.basename(uploaded_file.name)
    stem, extension = os.path.splitext(original)
    destination = thread_data_root(thread_id, user_id=user_id, create=True) / (
        f"{sanitize_filename(stem)}_{timestamp}{extension}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(uploaded_file.name, destination)
    return destination, hash_file(destination)


def delete_thread_data(user_id: Optional[str], thread_id: Optional[str]) -> None:
    if not user_id or not thread_id:
        return
    shutil.rmtree(thread_data_root(thread_id, user_id=user_id, create=False), ignore_errors=True)
    remove_task_dir(thread_id, user_id=user_id)


def clear_thread_uploads(user_id: str, thread_id: str) -> None:
    for path in list_upload_files(thread_id, user_id=user_id):
        path.unlink(missing_ok=True)
    shutil.rmtree(thread_data_root(thread_id, user_id=user_id, create=False), ignore_errors=True)


def _scan_scope(state: UIState, thread_id: str) -> Tuple[Optional[str], str]:
    '''Which `(user, thread)` directory holds this conversation's files.

    A demo conversation's files live under the account that produced them, so it
    reads from there rather than from the viewer's own empty directory.

    Parameters:
    ---------
    state (UIState): the current UI state.
    thread_id (str): the conversation being displayed.

    Returns:
    ----------
    scope (tuple): the `(user, thread)` directory holding its files, which differs from the viewer for a demo.
    '''

    demo_scope = demo_threads.results_scope(state.thread_meta(thread_id))
    if demo_scope:
        return demo_scope
    return state.user_id, thread_id


def refresh_thread_files(state: UIState, thread_id: Optional[str]) -> bool:
    '''Rescan one thread's files. True when the list changed.

    Parameters:
    ---------
    state (UIState): the state to update in place.
    thread_id (str): the conversation to rescan, or None.

    Returns:
    ----------
    changed (boolean): True when the list differs, so the sidebar is only re-sent when it must be.
    '''

    if not thread_id or not state.user_id:
        return False
    scan_user, scan_thread = _scan_scope(state, thread_id)
    previous = list(state.thread_files.get(thread_id, []))
    current = scan_thread_files(scan_thread, user_id=scan_user)
    state.thread_files[thread_id] = current
    if thread_id == state.current_thread_id:
        state.uploaded_files = [record for record in current if is_data_path(record.path)]
    return current != previous


__all__ = [
    "clear_thread_uploads",
    "delete_thread_data",
    "hash_file",
    "list_upload_files",
    "refresh_thread_files",
    "sanitize_filename",
    "save_uploaded_file",
    "scan_thread_files",
]

'''Where persisted data lives on disk.

Three managed roots, all per-user and per-conversation below the top level:

    DATA_ROOT/<user>/<thread>/      uploads
    RESULTS_ROOT/<user>/<thread>/   agent outputs (see output_paths)
    MEMORY_ROOT/                    vector stores, demo metadata
'''

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import DATA_ROOT, MEMORY_ROOT, RESULTS_ROOT


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


@lru_cache(maxsize=1)
def get_memory_root() -> Path:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return MEMORY_ROOT


@lru_cache(maxsize=1)
def get_results_root() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def thread_folder_name(thread_id: Optional[str], user_id: Optional[str]) -> str:
    '''Folder name for a thread, with the redundant `<user_id>:` prefix stripped.

    Thread ids are `"{user_id}:{uuid4}"`; nesting that under the user's own
    directory would repeat the id in the path.

    Parameters:
    ---------
    thread_id (str): the thread id, normally `<user_id>:<uuid4>`.
    user_id (str): its owner, whose redundant `<user_id>:` prefix is stripped.

    Returns:
    ----------
    name (str): the folder name for that thread.
    '''

    if not thread_id:
        return "unassigned"
    if user_id and thread_id.startswith(f"{user_id}:"):
        suffix = thread_id.split(":", 1)[1]
        return suffix or thread_id
    return thread_id


def user_data_root(user_id: Optional[str] = None) -> Path:
    path = get_data_root() / (user_id or "anonymous-user")
    path.mkdir(parents=True, exist_ok=True)
    return path


def thread_data_root(
    thread_id: Optional[str] = None,
    *,
    user_id: Optional[str] = None,
    create: bool = True,
) -> Path:
    '''Upload directory for one conversation.

    Falls back to a legacy `<user>/<user_id>:<uuid>` directory when one exists,
    so conversations created before the prefix was stripped keep their files.

    Parameters:
    ---------
    thread_id (str): the conversation, defaulting to the ambient scope.
    user_id (str): its owner, defaulting to the ambient scope.
    create (boolean): create the directory if it is missing.

    Returns:
    ----------
    root (Path): the upload directory for that conversation.
    '''

    resolved_thread_id = thread_id or "unassigned"
    root = user_data_root(user_id)
    folder_name = thread_folder_name(resolved_thread_id, user_id)
    canonical = root / folder_name
    legacy = root / resolved_thread_id

    path = canonical
    if folder_name != resolved_thread_id and not canonical.exists() and legacy.exists():
        path = legacy

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

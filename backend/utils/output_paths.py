"""Utility helpers for managing per-conversation result directories."""

from __future__ import annotations

import contextvars
import shutil
from pathlib import Path
from typing import List, Optional

RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "repuragent_task_id",
    default=None,
)
_user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "repuragent_user_id",
    default=None,
)


def get_results_root() -> Path:
    """Return the root results directory, ensuring it exists."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def set_current_task_id(task_id: Optional[str]):
    """Push the active task/conversation id into a context variable."""
    if task_id is None:
        return None
    return _task_id_var.set(task_id)


def reset_current_task_id(token) -> None:
    """Reset the task context using a token returned from set_current_task_id."""
    if token is None:
        _task_id_var.set(None)
        return
    try:
        _task_id_var.reset(token)
    except ValueError:
        # Token sourced from a different asyncio context; fall back to clearing.
        _task_id_var.set(None)


def get_current_task_id() -> Optional[str]:
    """Get the task id currently bound to this execution context."""
    return _task_id_var.get()


def set_current_user_id(user_id: Optional[str]):
    if user_id is None:
        return None
    return _user_id_var.set(user_id)


def reset_current_user_id(token) -> None:
    if token is None:
        _user_id_var.set(None)
        return
    try:
        _user_id_var.reset(token)
    except ValueError:
        _user_id_var.set(None)


def get_current_user_id() -> Optional[str]:
    return _user_id_var.get()


def _task_folder_name(task_id: Optional[str], user_id: Optional[str]) -> Optional[str]:
    """Derive a filesystem folder name for a task, stripping redundant user prefixes."""
    if not task_id:
        return None
    if user_id and task_id.startswith(f"{user_id}:"):
        _, suffix = task_id.split(":", 1)
        return suffix or task_id
    return task_id


def _user_root(user_id: Optional[str]) -> Path:
    base = get_results_root()
    if user_id:
        base = base / user_id
        base.mkdir(parents=True, exist_ok=True)
    return base


def ensure_task_dir(task_id: Optional[str] = None) -> Path:
    """Return the directory for the provided (or current) task, creating it if needed."""
    tid = task_id or get_current_task_id()
    current_user = get_current_user_id()
    user_root = _user_root(current_user)
    if not tid:
        user_root.mkdir(parents=True, exist_ok=True)
        return user_root
    folder_name = _task_folder_name(tid, current_user) or tid
    path = user_root / folder_name
    legacy_path = user_root / tid
    if folder_name != tid and not path.exists() and legacy_path.exists():
        path = legacy_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_folder(
    preferred_folder: Optional[str] = None,
    *,
    task_id: Optional[str] = None,
) -> Path:
    """
    Resolve an output directory that stays scoped under the results root.

    Args:
        preferred_folder: Optional folder hint (absolute or relative). Relative paths are
            always resolved beneath the results root. Absolute paths that escape the root
            are ignored for safety.
        task_id: Explicit task id override.
    """
    base_dir = ensure_task_dir(task_id)
    if not preferred_folder:
        return base_dir

    candidate = Path(preferred_folder)
    results_root = _user_root(get_current_user_id())

    if not candidate.is_absolute():
        parts = list(candidate.parts)
        root_name = results_root.name
        global_root_name = RESULTS_ROOT.name
        skip_values = {"", ".", root_name, global_root_name}
        while parts and parts[0] in skip_values:
            parts.pop(0)
        if parts:
            candidate = results_root / Path(*parts)
        else:
            candidate = results_root

    try:
        candidate.relative_to(results_root)
    except ValueError:
        # Never allow writes outside of the managed results directory.
        return base_dir

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def task_file_path(
    filename: str,
    *,
    output_folder: Optional[Path | str] = None,
    task_id: Optional[str] = None,
) -> Path:
    """Build a file path inside the active task's directory (or provided folder)."""
    if isinstance(output_folder, str):
        folder_path = resolve_output_folder(output_folder, task_id=task_id)
    elif isinstance(output_folder, Path):
        folder_path = output_folder
    else:
        folder_path = ensure_task_dir(task_id)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / filename


def list_task_files(task_id: str, *, user_id: Optional[str] = None) -> List[Path]:
    """List files that belong to a task, newest first."""
    effective_user = user_id or get_current_user_id()
    user_root = _user_root(effective_user)
    folder_name = _task_folder_name(task_id, effective_user) or task_id
    candidate_dirs = [user_root / folder_name]
    if folder_name != task_id:
        candidate_dirs.append(user_root / task_id)
    directory = next((d for d in candidate_dirs if d.exists()), None)
    if directory is None:
        return []
    files = [path for path in directory.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def remove_task_dir(task_id: str, *, user_id: Optional[str] = None) -> None:
    """Remove every artifact for a task."""
    effective_user = user_id or get_current_user_id()
    user_root = _user_root(effective_user)
    folder_name = _task_folder_name(task_id, effective_user) or task_id
    candidate_dirs = [user_root / folder_name]
    if folder_name != task_id:
        candidate_dirs.append(user_root / task_id)
    for directory in candidate_dirs:
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)

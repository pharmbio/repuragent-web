'''Deleting conversation files once they are past their retention window.

This is a shared deployment, so uploads and generated artifacts do not accumulate
forever. A background sweep removes files under the two managed roots whose mtime is
older than `RESULT_RETENTION_DAYS`, then prunes the directories left empty.

The sweep is **filesystem-based**, which is both simpler and broader than the
`user_files` table it replaced: that table only ever held uploads, so everything the
agents produced — every candidate table, every figure — was never cleaned up at all.
Rows in the table are still drained, so anything recorded by an earlier version is
honoured.

Two things must never be swept, because both are committed to the repository and a
fresh clone needs them:

* the **demo conversations'** results, named by `persistence/memory/demo_threads.json`;
* anything under `DATA_ROOT` that is not a user directory — the SOP corpus and the
  API reference data live there.

The guard is structural rather than a list of names: only
`<root>/<uuid user>/<thread>/…` is eligible, and demo scopes are excluded explicitly.
'''

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple
from uuid import UUID

from app.config import RESULT_RETENTION_DAYS, RETENTION_INTERVAL_SECONDS, logger
from backend.auth.repository import AuthRepository
from backend.utils.output_paths import get_results_root
from backend.utils.storage_paths import get_data_root


def _is_user_directory(path: Path) -> bool:
    '''True for a real per-user directory: its name is a user id (a UUID).

    Everything else at that level is shipped data — `SOP`, `api_related_data` — and is
    left alone.

    Parameters:
    ---------
    path (Path): a directory under one of the managed roots.

    Returns:
    ----------
    is_user_dir (boolean): True when its name is a user id, so a stray folder is never swept.
    '''

    try:
        UUID(path.name)
        return True
    except (ValueError, AttributeError):
        return False


def _protected_scopes() -> Set[Tuple[str, str]]:
    '''`(user_id, thread_folder)` pairs whose files are committed demo material.

    Returns:
    ----------
    scopes (set): `(user_id, thread_folder)` pairs holding committed demo material, which retention must never delete.
    '''

    protected: Set[Tuple[str, str]] = set()
    try:
        from app import demo_threads
        from backend.utils.storage_paths import thread_folder_name

        for entry in demo_threads.load_demo_threads():
            scope = demo_threads.results_scope(entry)
            if not scope:
                continue
            user_id, thread_id = scope
            protected.add((user_id, thread_folder_name(thread_id, user_id)))
            protected.add((user_id, thread_id))
    except Exception as exc:  # noqa: BLE001 - a failure here must not license deletion
        logger.warning("Could not read demo scopes; skipping the sweep: %s", exc)
        raise
    return protected


def _sweepable_conversations(root: Path, protected: Set[Tuple[str, str]]) -> Iterable[Path]:
    if not root.exists():
        return
    for user_directory in sorted(root.iterdir()):
        if not user_directory.is_dir() or not _is_user_directory(user_directory):
            continue
        for conversation in sorted(user_directory.iterdir()):
            if not conversation.is_dir():
                continue
            if (user_directory.name, conversation.name) in protected:
                continue
            yield conversation


def sweep_expired_files(
    *,
    retention_days: int = RESULT_RETENTION_DAYS,
    roots: Optional[Iterable[Path]] = None,
) -> Tuple[int, int]:
    '''Delete expired files under the managed roots. Returns `(files, directories)`.

    `roots` is explicit rather than read from module globals, so a caller — the test
    suite in particular — can never accidentally point this at the live directories.
    A retention window of zero or less disables the sweep entirely, so a deployment
    that wants to keep everything can say so.

    Parameters:
    ---------
    retention_days (int): how long a file may live before it is eligible for deletion.
    roots (list): the directories to sweep, defaulting to the managed roots.

    Returns:
    ----------
    counts (tuple): `(files, directories)` removed.
    '''

    if retention_days <= 0:
        return 0, 0

    try:
        protected = _protected_scopes()
    except Exception:
        return 0, 0

    # Resolved at call time, never bound at import: a module-level constant here made
    # the sweep unpatchable, which is how a test once swept the real directories.
    targets = list(roots) if roots is not None else [get_results_root(), get_data_root()]

    cutoff = time.time() - retention_days * 86400
    removed_files = 0
    removed_dirs = 0

    for root in (Path(target) for target in targets):
        for conversation in _sweepable_conversations(root, protected):
            for path in sorted(conversation.rglob("*"), reverse=True):
                try:
                    if path.is_file():
                        if path.stat().st_mtime < cutoff:
                            path.unlink()
                            removed_files += 1
                    elif path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                        removed_dirs += 1
                except OSError as exc:
                    logger.debug("Could not remove %s: %s", path, exc)
            try:
                if not any(conversation.iterdir()):
                    conversation.rmdir()
                    removed_dirs += 1
            except OSError:
                pass

    if removed_files or removed_dirs:
        logger.info(
            "Retention sweep removed %s file(s) and %s empty directory(ies) older than %s day(s)",
            removed_files,
            removed_dirs,
            retention_days,
        )
    return removed_files, removed_dirs


class RetentionWorker:
    def __init__(
        self,
        *,
        interval_seconds: int = RETENTION_INTERVAL_SECONDS,
        batch_size: int = 200,
        retention_days: int = RESULT_RETENTION_DAYS,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.retention_days = retention_days
        self.repo = AuthRepository()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="retention-worker")
        logger.info("File retention worker started (%s day window)", self.retention_days)

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("File retention worker stopped")

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - best effort, never fatal
                logger.error("Retention job failed: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def run_once(self) -> None:
        # The filesystem sweep is the actual policy; walking the tree is blocking work,
        # so it runs off the event loop.
        await asyncio.to_thread(
            sweep_expired_files,
            retention_days=self.retention_days,
            roots=[get_results_root(), get_data_root()],
        )
        await self._drain_recorded_files()

    async def _drain_recorded_files(self) -> None:
        '''Honour `user_files` rows written by an earlier version of the app.'''

        try:
            expired = await self.repo.fetch_expired_files(limit=self.batch_size)
        except Exception as exc:  # noqa: BLE001 - the table may not exist yet
            logger.debug("Could not read expired file rows: %s", exc)
            return
        for row in expired:
            path = Path(row.get("storage_path", ""))
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Could not delete %s: %s", path, exc)
            try:
                await self.repo.mark_file_deleted(row["user_id"], row["id"])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not mark file row deleted: %s", exc)


retention_worker = RetentionWorker()

__all__ = ["RetentionWorker", "retention_worker", "sweep_expired_files"]

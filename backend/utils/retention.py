"""Background job to clean up expired user files."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from backend.auth.repository import AuthRepository

logger = logging.getLogger(__name__)


class RetentionWorker:
    def __init__(self, *, interval_seconds: int = 1800, batch_size: int = 200) -> None:
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self.repo = AuthRepository()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="retention-worker")
        logger.info("File retention worker started")

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
                await self._cleanup_once()
            except Exception as exc:  # pragma: no cover - best effort
                logger.error("Retention job failed: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def _cleanup_once(self) -> None:
        expired = await self.repo.fetch_expired_files(limit=self.batch_size)
        if not expired:
            return
        for row in expired:
            path = Path(row.get("storage_path", ""))
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Could not delete %s: %s", path, exc)
            await self.repo.mark_file_deleted(row["user_id"], row["id"])


retention_worker = RetentionWorker()


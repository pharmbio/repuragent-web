'''Entry point: mount the UI on FastAPI and serve it.

Everything else lives in its own module — `app/ui/layout.py` builds the widgets,
`app/run_controller.py` drives a run, `app/session.py` handles sign-in and
conversations. This file only wires them to a server, which is why it is short: it
replaced a single 2,733-line module that held the widget tree, the CSS, the run loop,
the auth routes, the download signing and the timeline renderer together.
'''

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.auth_routes import AUTH_ROUTER
from app.config import APP_TITLE, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, logger
from app.downloads import FILES_ROUTER
from app.session import AUTH_SERVICE
from app.ui.layout import build_demo
from backend.db import (
    close_async_pool,
    close_postgres_checkpointer,
    get_async_pool,
    get_postgres_checkpointer,
)
from backend.utils.retention import retention_worker

__all__ = ["build_demo", "create_fastapi_app", "launch"]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    '''Open the database, verify the schema, and start the retention worker.

    Doing this at startup rather than lazily means a misconfigured `DATABASE_URL`
    fails on boot with one clear error, instead of surfacing as a broken sign-in
    later.

    Parameters:
    ---------
    _ (FastAPI): the app, which this hook does not need.

    Returns:
    ----------
    lifespan (async generator): yields once the pool is open, the schema applied and the retention worker started.
    '''

    await get_async_pool()
    await AUTH_SERVICE.repo.ensure_schema()
    await get_postgres_checkpointer()
    await retention_worker.start()
    logger.info("%s ready on %s:%s", APP_TITLE, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT)
    try:
        yield
    finally:
        await retention_worker.stop()
        await close_postgres_checkpointer()
        await close_async_pool()


def create_fastapi_app() -> FastAPI:
    demo = build_demo()
    application = FastAPI(title=APP_TITLE, lifespan=_lifespan)
    # Email verification and password reset arrive as plain page loads, with no
    # Gradio session, so they are routes rather than handlers.
    application.include_router(AUTH_ROUTER)
    application.include_router(FILES_ROUTER)

    mount_kwargs = {"path": "/"}
    if "footer_links" in inspect.signature(gr.mount_gradio_app).parameters:
        mount_kwargs["footer_links"] = ["api", "gradio"]
    return gr.mount_gradio_app(application, demo, **mount_kwargs)


def launch() -> None:
    uvicorn.run(
        create_fastapi_app(),
        host=GRADIO_SERVER_NAME,
        port=GRADIO_SERVER_PORT,
        log_level="info",
    )

'''Session tokens.

A session is an opaque random token stored in Postgres with an expiry, not a signed
JWT. That is deliberate: a row can be revoked immediately, which a self-contained
token cannot, and there is no signing key to keep in step across workers.

The JWT encode/decode and cookie helpers this replaced were never called by anything
that survived — the token has always been held in the Gradio session and validated
against the database on use.
'''

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone


class SessionManager:
    def __init__(self, *, refresh_ttl_days: int) -> None:
        self.refresh_delta = timedelta(days=refresh_ttl_days)

    @staticmethod
    def new_session_token() -> str:
        # 32 bytes of urandom: unguessable, and it is the only credential a download
        # link carries.
        return secrets.token_urlsafe(32)

    def refresh_expiration(self) -> datetime:
        return datetime.now(timezone.utc) + self.refresh_delta


__all__ = ["SessionManager"]

'''Email verification and password reset, as plain HTTP pages.

These are the two flows that cannot live inside Gradio: the user arrives from a link
in an email, with no session and no websocket. They are served as standalone HTML
from the same FastAPI app the UI is mounted on.
'''

from __future__ import annotations

from html import escape

from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse

from app.config import APP_TITLE, PASSWORD_MIN_LENGTH
from backend.auth.service import AuthService

AUTH_ROUTER = APIRouter()
_auth_service = AuthService()

_PAGE_CSS = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        background: #f7f9f8; color: #16211d; margin: 0;
        display: flex; align-items: center; justify-content: center; min-height: 100vh;
    }
    .card {
        background: #fff; border: 1px solid #d8e2dc; padding: 2rem 2.25rem;
        max-width: 28rem; width: 100%;
    }
    h1 { font-size: 1.15rem; margin: 0 0 0.75rem; }
    p { line-height: 1.55; margin: 0 0 1rem; }
    label { display: block; font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.04em; color: #5c6b64; margin: 0.9rem 0 0.3rem; }
    input { width: 100%; padding: 0.6rem 0.7rem; border: 1px solid #b9c9c1; font-size: 1rem; }
    button { margin-top: 1.2rem; width: 100%; padding: 0.7rem; border: 0;
             background: #1f5c55; color: #f6fbf8; font-weight: 700; font-size: 0.95rem; cursor: pointer; }
    button:hover { background: #3f7f6e; }
    .ok { color: #2f8f4e; font-weight: 600; }
    .bad { color: #b3382b; font-weight: 600; }
    a { color: #1f5c55; }
"""


def _page(title: str, body: str, *, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{escape(title)} · {escape(APP_TITLE)}</title>"
        f"<style>{_PAGE_CSS}</style></head>"
        f"<body><main class='card'><h1>{escape(title)}</h1>{body}</main></body></html>",
        status_code=status,
    )


def _message_page(title: str, message: str, *, ok: bool = True, status: int = 200) -> HTMLResponse:
    css_class = "ok" if ok else "bad"
    return _page(
        title,
        f"<p class='{css_class}'>{escape(message)}</p><p><a href='/'>Return to {escape(APP_TITLE)}</a></p>",
        status=status,
    )


def _reset_form(token: str, message: str = "", *, ok: bool = False) -> HTMLResponse:
    notice = f"<p class='{'ok' if ok else 'bad'}'>{escape(message)}</p>" if message else ""
    if ok and not token:
        # Password changed: offering the form again would invite a second attempt
        # with a token that has already been consumed.
        return _page("Password updated", notice + f"<p><a href='/'>Sign in to {escape(APP_TITLE)}</a></p>")
    return _page(
        "Choose a new password",
        notice
        + f"<p>Pick a password of at least {PASSWORD_MIN_LENGTH} characters, with both letters and numbers.</p>"
        + "<form method='post' action='/reset-password'>"
        + f"<input type='hidden' name='token' value='{escape(token, quote=True)}' />"
        + "<label for='password'>New password</label>"
        + "<input id='password' name='password' type='password' required autocomplete='new-password' />"
        + "<label for='confirm'>Confirm password</label>"
        + "<input id='confirm' name='confirm' type='password' required autocomplete='new-password' />"
        + "<button type='submit'>Update password</button>"
        + "</form>",
    )


def _validate_password_strength(password: str) -> None:
    if len(password) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {PASSWORD_MIN_LENGTH} characters long")
    if password.isdigit() or password.isalpha():
        raise ValueError("Password must include both letters and numbers")


@AUTH_ROUTER.get("/verify", response_class=HTMLResponse)
async def verify_email_route(token: str = ""):
    if not token:
        return _message_page("Verification", "This link is missing its token.", ok=False, status=400)
    user_id = await _auth_service.verify_email(token)
    if not user_id:
        return _message_page(
            "Verification",
            "This link is invalid or has expired. Request a new one by signing in.",
            ok=False,
            status=400,
        )
    return _message_page("Verification", "Email verified. You can sign in now.")


@AUTH_ROUTER.get("/reset-password", response_class=HTMLResponse)
async def reset_form(token: str = ""):
    if not token:
        return _message_page("Reset password", "This link is missing its token.", ok=False, status=400)
    return _reset_form(token)


@AUTH_ROUTER.post("/reset-password", response_class=HTMLResponse)
async def reset_submit(
    token: str = Form(...),
    password: str = Form(...),
    confirm: str = Form(...),
):
    if password != confirm:
        return _reset_form(token, "The two passwords do not match.")
    try:
        _validate_password_strength(password)
    except ValueError as exc:
        return _reset_form(token, str(exc))
    if not await _auth_service.reset_password(token, password):
        return _reset_form(token, "This reset link is invalid or has expired.")
    return _reset_form("", "Password updated.", ok=True)


__all__ = ["AUTH_ROUTER"]

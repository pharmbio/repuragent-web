"""Simple email dispatcher for verification/reset flows."""

from __future__ import annotations

import logging

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import ClickTracking, Mail, TrackingSettings
except ImportError:  # pragma: no cover - optional dependency
    SendGridAPIClient = None
    Mail = None
    TrackingSettings = None
    ClickTracking = None

from app.config import EMAIL_BASE_URL, EMAIL_SENDER, SENDGRID_API_KEY

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self) -> None:
        self.base_url = EMAIL_BASE_URL.rstrip("/")
        self.sender = EMAIL_SENDER
        self.api_key = SENDGRID_API_KEY

    def send_verification(self, recipient: str, token: str) -> None:
        link = f"{self.base_url}/verify?token={token}"
        subject = "Verify your Repuragent account"
        body = (
            "Hello,\n\n"
            "Welcome to Repuragent - your AI Agent for Drug Repurposing. Please verify your email address by clicking the link below within 24 hours:\n"
            f"{link}\n\n"
            "If you did not request this account, you can ignore this email.\n\n"
            "Best regards,\n\n"
            "Repuragent."
        )
        self._dispatch(recipient, subject, body)

    def send_password_reset(self, recipient: str, token: str) -> None:
        link = f"{self.base_url}/reset-password?token={token}"
        subject = "Reset your Repuragent password"
        body = (
            "Hello,\n\n"
            "A password reset was requested for your account.\n"
            f"Use the following link within one hour to set a new password:\n{link}\n\n"
            "If you did not request this, you can safely ignore this message."
        )
        self._dispatch(recipient, subject, body)

    def _dispatch(self, recipient: str, subject: str, body: str) -> None:
        if not self.api_key:
            logger.warning("SendGrid API key missing; logging message instead.")
            logger.info("Email to %s\nSubject: %s\n%s", recipient, subject, body)
            return

        self._send_via_sendgrid(recipient, subject, body)

    def _send_via_sendgrid(self, recipient: str, subject: str, body: str) -> None:
        if not SendGridAPIClient or not Mail:
            logger.error("SendGrid SDK not available; cannot send email.")
            return

        message = Mail(
            from_email=self.sender,
            to_emails=recipient,
            subject=subject,
            plain_text_content=body,
        )
        if TrackingSettings and ClickTracking:
            message.tracking_settings = TrackingSettings(
                click_tracking=ClickTracking(enable=False, enable_text=False)
            )

        try:
            client = SendGridAPIClient(self.api_key)
            response = client.send(message)
        except Exception as exc:  # pragma: no cover - network
            logger.error("Failed to send email via SendGrid: %s", exc)
            return

        status = getattr(response, "status_code", None)
        body_bytes = getattr(response, "body", b"")
        if status and status >= 400:
            detail = (
                body_bytes.decode("utf-8", errors="ignore")
                if isinstance(body_bytes, (bytes, bytearray))
                else body_bytes
            )
            logger.error("SendGrid API responded with %s: %s", status, detail)

"""Simple email dispatcher for verification/reset flows."""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional
from urllib.parse import urlparse

import requests

from app.config import (
    EMAIL_BASE_URL,
    EMAIL_PROVIDER_API_KEY,
    EMAIL_PROVIDER_API_URL,
    EMAIL_PROVIDER_USERNAME,
    EMAIL_SENDER,
)

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self) -> None:
        self.base_url = EMAIL_BASE_URL.rstrip("/")
        self.sender = EMAIL_SENDER
        self.api_url = EMAIL_PROVIDER_API_URL
        self.api_key = EMAIL_PROVIDER_API_KEY
        self.username = EMAIL_PROVIDER_USERNAME or self.sender

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
        if not self.api_url:
            logger.warning("Email provider not configured; logging message instead.")
            logger.info("Email to %s\nSubject: %s\n%s", recipient, subject, body)
            return

        parsed = urlparse(self.api_url)
        scheme = (parsed.scheme or "").lower()
        if scheme in {"smtp", "smtps"}:
            self._send_via_smtp(parsed, recipient, subject, body)
            return

        if not self.api_key:
            logger.warning("Email API key missing; logging message instead.")
            logger.info("Email to %s\nSubject: %s\n%s", recipient, subject, body)
            return

        payload = {
            "from": self.sender,
            "to": recipient,
            "subject": subject,
            "text": body,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to send email: %s", exc)

    def _send_via_smtp(self, parsed, recipient: str, subject: str, body: str) -> None:
        host = parsed.hostname
        if not host:
            logger.error("SMTP URL missing hostname: %s", self.api_url)
            return
        port = parsed.port or (465 if parsed.scheme.lower() == "smtps" else 587)
        username = parsed.username or self.username
        password = parsed.password or self.api_key
        if not password:
            logger.error("SMTP password/API key missing; cannot send email")
            return

        msg = EmailMessage()
        msg["From"] = self.sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        try:
            if parsed.scheme.lower() == "smtps" or port == 465:
                with smtplib.SMTP_SSL(host, port) as smtp:
                    smtp.login(username, password)
                    smtp.send_message(msg)
            else:
                with smtplib.SMTP(host, port) as smtp:
                    smtp.starttls()
                    smtp.login(username, password)
                    smtp.send_message(msg)
        except Exception as exc:  # pragma: no cover - network
            logger.error("Failed to send email via SMTP: %s", exc)

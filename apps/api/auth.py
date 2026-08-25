"""Small HMAC-signed bearer tokens for the single shared application account."""

import base64
import hashlib
import hmac
import time

from retrieval.config import settings


TOKEN_TTL_SECONDS = 90 * 24 * 60 * 60


def create_token(username: str) -> str:
    """Create a compact, signed token that expires after a long-lived session."""
    expires_at = int(time.time()) + TOKEN_TTL_SECONDS
    payload = f"{username}:{expires_at}"
    encoded_payload = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    signature = hmac.new(
        settings.app_secret_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{encoded_payload}.{signature}"


def _decode(token: str | None) -> tuple[str, int] | None:
    """Return (username, expires_at) if the token is correctly signed and unexpired."""
    if not token or "." not in token:
        return None
    encoded_payload, signature = token.rsplit(".", 1)
    try:
        padding = "=" * (-len(encoded_payload) % 4)
        payload = base64.urlsafe_b64decode((encoded_payload + padding).encode("ascii")).decode("utf-8")
        username, expires_at_text = payload.rsplit(":", 1)
        expires_at = int(expires_at_text)
    except (ValueError, TypeError, UnicodeDecodeError, base64.binascii.Error):
        return None
    if not username or expires_at < int(time.time()):
        return None
    expected_signature = hmac.new(
        settings.app_secret_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        return None
    return username, expires_at


def verify_token(token: str | None) -> bool:
    """Return whether a token is correctly signed and has not expired."""
    return _decode(token) is not None


def get_username(token: str | None) -> str | None:
    """Return the username embedded in a valid token, or None if invalid/expired."""
    decoded = _decode(token)
    return decoded[0] if decoded else None

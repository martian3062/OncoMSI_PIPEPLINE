import hashlib
import hmac
import json
from urllib import request

from django.conf import settings


def trigger_n8n_run_created(payload: dict) -> bool:
    if not settings.N8N_ENABLED:
        return False
    webhook_url = payload.get("n8n_webhook_url")
    if not webhook_url:
        return False

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-MSI-Signature": _signature(body),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _signature(body: bytes) -> str:
    secret = settings.N8N_WEBHOOK_SECRET.encode("utf-8")
    if not secret:
        return ""
    return hmac.new(secret, body, hashlib.sha256).hexdigest()

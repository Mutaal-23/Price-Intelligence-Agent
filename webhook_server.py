"""Bundled local webhook receiver.

Makes "Approve & Dispatch" a real dispatch even without an external WEBHOOK_URL:
the graph POSTs the approved payload to this server on http://127.0.0.1:WEBHOOK_PORT,
which stores it in webhook_events.json (also retrievable via GET /dispatch).

Point WEBHOOK_URL at any real endpoint (Slack/Teams/Pipedream/webhook.site) to
forward approved actions there instead of (or in addition to) this receiver.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "127.0.0.1")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8504"))
EVENTS_FILE = Path(__file__).parent / "webhook_events.json"

_lock = threading.Lock()
_started = False
_server = None


def _load() -> list[dict]:
    if EVENTS_FILE.exists():
        try:
            return json.loads(EVENTS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("empty body")
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:  # noqa: BLE001
            self._send(400, b'{"status":"bad_request"}')
            return
        event = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "path": self.path,
            "payload": payload,
        }
        entries = _load()
        entries.append(event)
        with _lock:
            EVENTS_FILE.write_text(json.dumps(entries, indent=2, default=str))
        self._send(200, b'{"status":"ok","stored":true}')

    def do_GET(self) -> None:
        self._send(200, json.dumps(_load(), indent=2, default=str).encode())

    def log_message(self, *args) -> None:  # silence request logging
        pass


def ensure_started() -> str | None:
    """Start the receiver once; returns the dispatch URL or None if unavailable."""
    global _started, _server
    if _started and _server is not None:
        return f"http://{WEBHOOK_HOST}:{WEBHOOK_PORT}/dispatch"
    try:
        _server = ThreadingHTTPServer((WEBHOOK_HOST, WEBHOOK_PORT), _Handler)
        threading.Thread(target=_server.serve_forever, daemon=True, name="webhook-receiver").start()
        _started = True
        return f"http://{WEBHOOK_HOST}:{WEBHOOK_PORT}/dispatch"
    except OSError:
        return None


def last_event() -> dict | None:
    entries = _load()
    return entries[-1] if entries else None


if __name__ == "__main__":
    url = ensure_started()
    print(f"Webhook receiver listening on {url}")
    print(f"Events stored in {EVENTS_FILE}")
    import time

    while True:
        time.sleep(60)
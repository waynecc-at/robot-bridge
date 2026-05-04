"""OpenClaw Channel Service — HTTP callback alternative to WebSocket bridge.

OpenClaw calls back to an HTTP endpoint with reply.response.
Less preferred than bridge.py (WebSocket) due to polling/port management.

Merged from xiaozhi-hermes with adapted imports.
"""
import asyncio
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from ..config import config


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP callback handler for OpenClaw reply responses."""
    server: "_CallbackHttpServer"

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("OpenClaw callback: " + format % args)

    def _write_json(self, status_code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        if self.path != self.server.callback_path:
            self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid Content-Length"})
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Invalid JSON"})
            return

        request_id = str(payload.get("requestId", "")).strip()
        if not request_id:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing requestId"})
            return

        self.server.service.handle_callback(payload)
        self._write_json(HTTPStatus.OK, {"ok": True})


class _CallbackHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        callback_path: str,
        service: "OpenClawChannelService",
    ) -> None:
        super().__init__(server_address, _CallbackHandler)
        self.callback_path = callback_path
        self.service = service


class OpenClawChannelService:
    """HTTP channel service for OpenClaw reply callback."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server: Optional[_CallbackHttpServer] = None
        self._thread: Optional[threading.Thread] = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._early_callbacks: dict[str, dict[str, Any]] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return

        self._loop = asyncio.get_running_loop()
        self._server = _CallbackHttpServer(
            ("127.0.0.1", 0),  # bind random port
            "/openclaw/callback",
            self,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openclaw-callback",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        port = self._server.server_address[1]
        logger.info(f"OpenClaw callback listening on 127.0.0.1:{port}")

    async def stop(self) -> None:
        if not self._started:
            return
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        self._started = False

        if server is not None:
            await asyncio.to_thread(server.shutdown)
            await asyncio.to_thread(server.server_close)
        if thread is not None:
            await asyncio.to_thread(thread.join, 1.0)

        pending = list(self._pending.values())
        self._pending.clear()
        self._early_callbacks.clear()
        for future in pending:
            if not future.done():
                future.cancel()

    def handle_callback(self, payload: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        request_id = str(payload.get("requestId", "")).strip()
        if not request_id:
            return

        def _deliver() -> None:
            future = self._pending.pop(request_id, None)
            if future is None:
                self._early_callbacks[request_id] = payload
                return
            if not future.done():
                future.set_result(payload)

        loop.call_soon_threadsafe(_deliver)

    async def request_reply(
        self,
        user_text: str,
        from_id: str,
        conversation_id: str,
        sender_name: Optional[str] = None,
    ) -> str:
        if not self._started:
            await self.start()

        port = self._server.server_address[1]
        body = {
            "text": user_text,
            "from": from_id,
            "senderName": sender_name or from_id,
            "conversationId": conversation_id,
            "callbackUrl": f"http://127.0.0.1:{port}/openclaw/callback",
        }

        loop = asyncio.get_running_loop()
        response = await asyncio.to_thread(
            self._post_openclaw, body
        )
        request_id = str(response.get("requestId", "")).strip()
        if not request_id:
            raise RuntimeError(f"OpenClaw missing requestId: {response}")

        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        early = self._early_callbacks.pop(request_id, None)
        if early is not None:
            future.set_result(early)
        else:
            self._pending[request_id] = future

        timeout = getattr(config.hermes, 'timeout', 60)
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
        except Exception:
            pending = self._pending.get(request_id)
            if pending is future:
                self._pending.pop(request_id, None)
            raise

        if not result.get("ok", False):
            raise RuntimeError(str(result.get("error", "OpenClaw callback error")))
        return str(result.get("reply", "")).strip()

    def _post_openclaw(self, body: dict[str, Any]) -> dict[str, Any]:
        from ..config import config
        url = getattr(config.hermes, 'openclaw_channel_url', None)
        if not url:
            raise RuntimeError("OpenClaw channel URL not configured")
        payload = json.dumps(body).encode("utf-8")
        request = Request(url=url, data=payload, headers={
            "Content-Type": "application/json",
        }, method="POST")
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenClaw HTTP {exc.code}: {detail}")
        except URLError as exc:
            raise RuntimeError(f"OpenClaw connection error: {exc}")


_service: Optional[OpenClawChannelService] = None


def get_openclaw_channel_service() -> OpenClawChannelService:
    global _service
    if _service is None:
        _service = OpenClawChannelService()
    return _service

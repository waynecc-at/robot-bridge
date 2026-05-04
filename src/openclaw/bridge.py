"""OpenClaw Bridge Service — WebSocket bridge to OpenClaw agent.

OpenClaw plugin (myapp-channel) connects as a WebSocket client.
This service receives reply.request from robot-bridge, forwards to OpenClaw,
and returns reply.response.

Merged from xiaozhi-hermes with adapted imports and config.
"""
import asyncio
import json
import uuid
from contextlib import suppress
from typing import Any, Optional

from loguru import logger
from websockets.exceptions import ConnectionClosed

from ..config import config


class OpenClawBridgeService:
    """WebSocket bridge to OpenClaw agent runtime."""

    def __init__(self) -> None:
        self._connection = None
        self._connection_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._connected_event = asyncio.Event()

    def is_connected(self) -> bool:
        return self._connection is not None

    async def wait_until_connected(self, timeout: float) -> bool:
        if self.is_connected():
            return True
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def handle_connection(self, websocket) -> None:
        """Accept an incoming OpenClaw bridge connection."""
        async with self._connection_lock:
            previous = self._connection
            self._connection = websocket
            self._connected_event.set()

        if previous is not None and previous is not websocket:
            with suppress(Exception):
                await previous.close(code=1000, reason="Replaced by new bridge")

        logger.info("OpenClaw bridge connected")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    raw = message.decode("utf-8", errors="ignore")
                else:
                    raw = str(message)
                await self._handle_message(raw)
        except ConnectionClosed:
            logger.info("OpenClaw bridge disconnected")
        finally:
            async with self._connection_lock:
                if self._connection is websocket:
                    self._connection = None
                    self._connected_event.clear()

    async def _handle_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("OpenClaw bridge ignored invalid JSON")
            return

        message_type = str(payload.get("type", "")).strip()
        if message_type in {"bridge.hello", "bridge.pong"}:
            return
        if message_type == "bridge.ping":
            await self._send_json({"type": "bridge.pong"})
            return
        if message_type != "reply.response":
            logger.debug(f"OpenClaw bridge ignored message type: {message_type}")
            return

        request_id = str(payload.get("requestId", "")).strip()
        if not request_id:
            return

        logger.info(
            f"OpenClaw bridge received reply.response "
            f"request_id={request_id} ok={payload.get('ok', False)}"
        )
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.set_result(payload)

    async def request_reply(
        self,
        user_text: str,
        from_id: str,
        conversation_id: str,
        sender_name: Optional[str] = None,
    ) -> str:
        """Send a reply request to OpenClaw and wait for response."""
        timeout = getattr(config.hermes, 'timeout', 60)
        if self._connection is None:
            logger.info("OpenClaw bridge waiting for connection before sending request")
            connected = await self.wait_until_connected(
                timeout=min(timeout, 10)
            )
            if not connected:
                raise RuntimeError("OpenClaw bridge is not connected")

        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future

        payload = {
            "type": "reply.request",
            "requestId": request_id,
            "text": user_text,
            "from": from_id,
            "senderName": sender_name or from_id,
            "conversationId": conversation_id,
        }

        try:
            logger.info(
                f"OpenClaw bridge sending reply.request "
                f"request_id={request_id} from={from_id}"
            )
            await self._send_json(payload)
            response = await asyncio.wait_for(future, timeout=timeout)
        finally:
            pending = self._pending.get(request_id)
            if pending is future:
                self._pending.pop(request_id, None)

        if not response.get("ok", False):
            raise RuntimeError(
                str(response.get("error", "OpenClaw bridge reply error"))
            )

        return str(response.get("reply", "")).strip()

    async def _send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            if self._connection is None:
                raise RuntimeError("OpenClaw bridge is not connected")
            await self._connection.send(json.dumps(payload))


_service: Optional[OpenClawBridgeService] = None


def get_openclaw_bridge_service() -> OpenClawBridgeService:
    global _service
    if _service is None:
        _service = OpenClawBridgeService()
    return _service

"""WebSocket handler for robot connections"""
import json
import time
import base64
from typing import Optional
from dataclasses import dataclass
from loguru import logger

from .asr_service import asr_service
from .hermes_client import HermesClient
from .tts_service import tts_service


@dataclass
class RobotSession:
    """Represents a connected robot device"""
    device_id: str
    session_id: str
    websocket: object
    last_activity: float = 0

    is_listening: bool = False
    current_text: str = ""
    hermes: Optional[HermesClient] = None


class RobotWebSocketHandler:
    """Handles WebSocket connections from robot devices"""

    def __init__(self):
        self.sessions: dict[str, RobotSession] = {}
        self.hermes: Optional[HermesClient] = None

    async def set_hermes_client(self, client: HermesClient):
        self.hermes = client

    async def cleanup_stale_sessions(self, ttl: float = 300):
        now = time.time()
        stale = [
            sid for sid, s in self.sessions.items()
            if now - s.last_activity > ttl
        ]
        for sid in stale:
            del self.sessions[sid]
            logger.info(f"[WS] Stale session cleaned up: {sid}")

    async def _handle_message(self, session: RobotSession, raw_message: str | bytes):
        session.last_activity = time.time()
        try:
            if isinstance(raw_message, bytes):
                message = self._parse_binary_message(raw_message)
            else:
                message = json.loads(raw_message)

            msg_type = message.get("type", "unknown")
            logger.debug(f"[WS] Message type={msg_type} from {session.session_id}")

            if msg_type == "text":
                await self._handle_text_message(session, message)
            elif msg_type == "audio":
                await self._handle_audio_message(session, message)
            elif msg_type == "vad":
                await self._handle_vad_message(session, message)
            elif msg_type == "ping":
                await self._handle_ping(session, message)
            else:
                logger.warning(f"[WS] Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"[WS] Invalid JSON: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"[WS] Message handling error: {e}")

    def _parse_binary_message(self, data: bytes) -> dict:
        msg_type = data[0] if data else 0
        audio_data = data[1:] if len(data) > 1 else b""
        return {
            "type": "audio",
            "data": base64.b64encode(audio_data).decode(),
            "format": "pcm"
        }

    async def _handle_text_message(self, session: RobotSession, message: dict):
        text = message.get("text", "")
        logger.info(f"[WS] Text input: {text}")
        session.current_text = text
        await self._process_and_respond(session, text)

    async def _handle_audio_message(self, session: RobotSession, message: dict):
        audio_b64 = message.get("data", "")
        if not audio_b64:
            logger.warning("[WS] Empty audio data received")
            return

        audio_bytes = base64.b64decode(audio_b64)
        logger.info(f"[WS] Audio received: {len(audio_bytes)} bytes")

        await self._send_message(session.websocket, {
            "type": "status",
            "message": "listening",
            "action": "listening"
        })

        text = await asr_service.transcribe(audio_bytes)
        if not text:
            await self._send_message(session.websocket, {
                "type": "error",
                "message": "Speech not recognized"
            })
            return

        session.current_text = text
        await self._process_and_respond(session, text)

    async def _handle_vad_message(self, session: RobotSession, message: dict):
        state = message.get("state", "unknown")
        logger.debug(f"[WS] VAD state: {state}")
        if state == "speaking":
            session.is_listening = True
        elif state in ("silence", "end"):
            session.is_listening = False
            if session.current_text:
                await self._process_and_respond(session, session.current_text)

    async def _handle_ping(self, session: RobotSession, message: dict):
        await self._send_message(session.websocket, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })

    async def _process_and_respond(self, session: RobotSession, text: str):
        if not self.hermes:
            logger.error("[WS] Hermes client not available")
            await self._send_message(session.websocket, {
                "type": "error",
                "message": "Hermes gateway not connected"
            })
            return

        try:
            await self._send_message(session.websocket, {
                "type": "status",
                "message": "thinking",
                "action": "thinking"
            })

            logger.info(f"[WS] Processing: {text}")
            response = await self.hermes.chat(
                message=text,
                session_id=session.session_id,
            )

            response_text = response.text
            logger.info(f"[WS] Hermes response: {response_text[:100]}...")

            await self._send_message(session.websocket, {
                "type": "response_text",
                "text": response_text,
                "session_id": session.session_id
            })

            await self._send_message(session.websocket, {
                "type": "status",
                "message": "synthesizing",
                "action": "speaking"
            })

            audio_chunks = []
            async for chunk in tts_service.synthesize_stream(response_text):
                audio_chunks.append(chunk)
                await self._send_message(session.websocket, {
                    "type": "tts_audio",
                    "data": base64.b64encode(chunk).decode(),
                    "final": False
                })

            await self._send_message(session.websocket, {
                "type": "tts_audio",
                "data": "",
                "final": True
            })

            logger.info(f"[WS] TTS complete, streamed {len(audio_chunks)} chunks")

        except Exception as e:
            logger.error(f"[WS] Processing error: {e}")
            await self._send_message(session.websocket, {
                "type": "error",
                "message": f"Error: {str(e)}"
            })

    async def _send_message(self, websocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[WS] Send error: {e}")

    async def broadcast(self, message: dict):
        for session in self.sessions.values():
            try:
                await self._send_message(session.websocket, message)
            except Exception:
                logger.warning(f"[WS] Broadcast failed for session {session.session_id}")


ws_handler = RobotWebSocketHandler()

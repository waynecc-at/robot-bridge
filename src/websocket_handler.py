"""WebSocket handler for robot connections with concurrent LLM+TTS pipeline.

Optimizations v0.2:
- Concurrent pipeline: LLM stream reader + TTS worker via asyncio.Queue
- LLM sentence N+1 generation overlaps with TTS playback of sentence N
- Thread pool offloaded ASR/TTS (via service layer)
- Turn-id barge-in checking preserved at all yield points
"""
import asyncio
import json
import re
import struct
import time
import base64
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger
import opuslib

from .asr_service import asr_service
from .config import config
from .hermes_client import HermesClient
from .metrics import metrics
from .profile_store import profile_store
from .text_utils import sanitize_for_tts
from .tts_service import tts_service
from .vision_service import vision_service


@dataclass
class RobotSession:
    """Represents a connected robot device with turn management"""
    device_id: str
    session_id: str
    websocket: object
    last_activity: float = field(default_factory=time.time)

    is_listening: bool = False
    current_text: str = ""
    hermes: Optional[HermesClient] = None

    # Barge-in turn management
    turn_id: int = 0
    pending_task: Optional[asyncio.Task] = None

    # Audio buffer for VAD-based barge-in
    audio_buffer: list[bytes] = field(default_factory=list)
    # Opus decoder (one per session, maintains state across packets)
    opus_decoder: Optional[object] = None
    # End-of-speech timer for StackChan binary protocol
    eos_task: Optional[asyncio.Task] = None
    eos_timeout: float = 1.5  # seconds of silence to trigger end-of-speech

    # Vision / person tracking
    current_person: Optional[str] = None    # name of recognized person
    person_profiles: dict[str, dict] = field(default_factory=dict)  # multi-user: name → {name, relationship, preferences}

    def cancel_current_turn(self):
        """Cancel the active turn so a new one can start"""
        self.turn_id += 1
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()
            metrics.turns_cancelled += 1
            logger.info(f"[WS] Turn cancelled: new turn_id={self.turn_id}")


def _split_sentences(buffer: str) -> list[str]:
    """Split text buffer into complete sentences at Chinese punctuation."""
    sentences = []
    while True:
        m = re.search(r"(.+?[。！？\n])", buffer)
        if not m:
            break
        sentences.append(m.group(1).strip())
        buffer = buffer[m.end():]
    return sentences, buffer


class RobotWebSocketHandler:
    """Handles WebSocket connections from robot devices with concurrent pipeline"""

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
            session = self.sessions[sid]
            session.cancel_current_turn()
            del self.sessions[sid]
            logger.info(f"[WS] Stale session cleaned up: {sid}")

    async def _handle_message(self, session: RobotSession, raw_message: str | bytes):
        session.last_activity = time.time()
        try:
            if isinstance(raw_message, bytes):
                message = self._parse_binary_message(raw_message)
            else:
                message = json.loads(raw_message)

            # XiaoZhi protocol handshake
            if "protocolVersion" in message:
                await self._handle_hello(session, message)
                return

            msg_type = message.get("type", "unknown")
            metrics.record_ws_message(msg_type)
            logger.debug(f"[WS] Message type={msg_type} from {session.session_id}")

            if msg_type == "hello":
                await self._handle_hello(session, message)
            elif msg_type == "listen":
                await self._handle_listen(session, message)
            elif msg_type == "text":
                await self._handle_text_message(session, message)
            elif msg_type == "audio":
                await self._handle_audio_message(session, message)
            elif msg_type == "vad":
                await self._handle_vad_message(session, message)
            elif msg_type == "vision_frame":
                await self._handle_vision_frame(session, message)
            elif msg_type == "ping":
                await self._handle_ping(session, message)
            else:
                logger.warning(f"[WS] Unknown message type: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"[WS] Invalid JSON: {raw_message[:100]}")
        except Exception as e:
            logger.error(f"[WS] Message handling error: {e}")

    # StackChan binary protocol constants
    STACKCHAN_OPUS = 0x01
    STACKCHAN_JPEG = 0x02
    STACKCHAN_HEARTBEAT_PONG = 0x11
    STACKCHAN_DECLINE_CALL = 0x0A
    STACKCHAN_ACCEPT_CALL = 0x0B
    STACKCHAN_END_CALL = 0x0C

    def _parse_binary_message(self, data: bytes) -> dict:
        """Try StackChan binary protocol [1B type][4B big-endian len][N B payload].
        If the header looks invalid (absurd length), treat as raw Opus audio."""
        if len(data) < 5:
            logger.warning(f"[WS] Binary message too short: {len(data)} bytes")
            return {"type": "audio", "data": base64.b64encode(data).decode(), "format": "opus"}

        msg_type = data[0]
        payload_len = struct.unpack(">I", data[1:5])[0]

        # Sanity check: if payload length exceeds remaining data, no header present
        if payload_len > len(data) - 5:
            logger.debug(f"[WS] Raw binary, no StackChan header: {len(data)}B, first_byte=0x{msg_type:02X}")
            return {"type": "audio", "data": base64.b64encode(data).decode(), "format": "opus"}

        payload = data[5:5 + payload_len]

        if msg_type == self.STACKCHAN_OPUS:
            return {"type": "audio", "data": base64.b64encode(payload).decode(), "format": "opus"}
        elif msg_type == self.STACKCHAN_JPEG:
            return {"type": "vision_frame", "data": base64.b64encode(payload).decode()}
        elif msg_type == self.STACKCHAN_HEARTBEAT_PONG:
            return {"type": "heartbeat_pong"}
        else:
            logger.debug(f"[WS] Binary type 0x{msg_type:02X}, payload={payload_len}B, first_bytes={payload[:16].hex()}")
            return {"type": "binary_other", "code": msg_type}


    # ── message handlers ──────────────────────────────────────

    async def _handle_text_message(self, session: RobotSession, message: dict):
        text = message.get("text", "")
        logger.info(f"[WS] Text input: {text}")
        session.current_text = text
        self._start_turn(session, text)

    async def _handle_audio_message(self, session: RobotSession, message: dict):
        audio_b64 = message.get("data", "")
        audio_format = message.get("format", "pcm")
        if not audio_b64:
            logger.warning("[WS] Empty audio data received")
            return

        audio_bytes = base64.b64decode(audio_b64)

        # Decode Opus to PCM if needed
        if audio_format == "opus":
            try:
                audio_bytes = self._decode_opus(session, audio_bytes)
            except Exception as e:
                logger.error(f"[WS] Opus decode error: {e}")
                return

        # During VAD listening: buffer chunks, reset end-of-speech timer
        if session.is_listening:
            session.audio_buffer.append(audio_bytes)
            self._reset_eos_timer(session)
            return

        # Immediate transcription (non-listening mode)
        logger.info(f"[WS] Audio received: {len(audio_bytes)} bytes")
        await self._send_message(session.websocket, {
            "type": "status",
            "message": "listening",
            "action": "listening",
            "turn": session.turn_id,
        })
        text = await asr_service.transcribe(audio_bytes)
        if not text:
            await self._send_message(session.websocket, {
                "type": "error",
                "message": "Speech not recognized",
            })
            return
        session.current_text = text
        self._start_turn(session, text)

    def _decode_opus(self, session: RobotSession, opus_data: bytes) -> bytes:
        """Decode a single Opus packet to 16-bit PCM. Creates decoder on first call."""
        if session.opus_decoder is None:
            session.opus_decoder = opuslib.Decoder(16000, 1)
        return session.opus_decoder.decode(opus_data, frame_size=960)

    def _reset_eos_timer(self, session: RobotSession):
        """Reset the end-of-speech timer. When it fires, process buffered audio."""
        if session.eos_task and not session.eos_task.done():
            session.eos_task.cancel()
        session.eos_task = asyncio.create_task(
            self._eos_timeout_handler(session),
            name=f"eos-{session.session_id}",
        )

    async def _eos_timeout_handler(self, session: RobotSession):
        """Wait for silence timeout, then process buffered audio as complete utterance."""
        try:
            await asyncio.sleep(session.eos_timeout)
            if session.audio_buffer and session.is_listening:
                chunk_count = len(session.audio_buffer)
                combined = b"".join(session.audio_buffer)
                session.audio_buffer.clear()
                logger.info(f"[WS] End-of-speech: {len(combined)} bytes PCM from {chunk_count} chunks")
                text = await asr_service.transcribe(combined)
                if text:
                    session.current_text = text
                    logger.info(f"[WS] ASR result: {text[:60]}")
                    self._start_turn(session, text)
                else:
                    logger.info("[WS] ASR: speech not recognized")
                    await self._send_stackchan_text(session, "抱歉，我没听清")
        except asyncio.CancelledError:
            pass

    async def _handle_vad_message(self, session: RobotSession, message: dict):
        state = message.get("state", "unknown")
        logger.info(f"[WS] VAD state: {state} from {session.session_id}")

        if state == "start":
            session.cancel_current_turn()
            session.audio_buffer.clear()
            session.is_listening = True
            await self._send_message(session.websocket, {
                "type": "interrupt",
                "turn": session.turn_id,
            })
            logger.info(f"[WS] Barge-in: interrupt sent, new turn_id={session.turn_id}")

            # Fire-and-forget LLM pre-warm while user is still speaking
            # (reduces TTFT when ASR completes and LLM turn starts)
            if self.hermes:
                asyncio.create_task(self.hermes.ensure_warm())

        elif state == "end":
            session.is_listening = False
            if session.audio_buffer:
                # Transcribe buffered audio chunks as one combined clip
                logger.info(f"[WS] Transcribing {len(session.audio_buffer)} buffered chunks")
                combined = b"".join(session.audio_buffer)
                text = await asr_service.transcribe(combined)
                session.audio_buffer.clear()
                if text:
                    session.current_text = text
                    self._start_turn(session, text)
            elif session.current_text:
                self._start_turn(session, session.current_text)

        elif state == "speaking":
            pass

    async def _handle_vision_frame(self, session: RobotSession, message: dict):
        """Process a JPEG frame from ESP32 camera (base64-encoded)."""
        if not vision_service.enabled:
            return

        jpeg_b64 = message.get("data", "")
        if not jpeg_b64:
            return

        jpeg_bytes = base64.b64decode(jpeg_b64)
        results = await vision_service.process_frame(jpeg_bytes)

        # Multi-user: merge all recognized persons into session
        if results:
            seen_names = set()
            for det in results:
                name = det.get("name", "unknown")
                if name == "unknown":
                    continue
                seen_names.add(name)
                if name in session.person_profiles:
                    continue  # already tracked
                profile = profile_store.get_profile(name)
                session.person_profiles[name] = {
                    "name": name,
                    "relationship": profile.relationship if profile else "",
                    "preferences": profile.preferences if profile else "",
                }
                await self._send_message(session.websocket, {
                    "type": "person_detected",
                    "name": name,
                    "relationship": session.person_profiles[name].get("relationship", ""),
                })
                logger.info(f"[WS] Person '{name}' entered frame")

            # Remove persons no longer in frame
            departed = [
                n for n in session.person_profiles
                if n not in seen_names
            ]
            for name in departed:
                logger.info(f"[WS] Person '{name}' left frame")
                del session.person_profiles[name]

            session.current_person = next(iter(session.person_profiles), None)
        else:
            # No faces at all
            if session.person_profiles:
                session.person_profiles.clear()
                session.current_person = None
                logger.info("[WS] All persons left frame")

    async def _handle_hello(self, session: RobotSession, message: dict):
        """Respond to XiaoZhi protocol handshake."""
        proto_version = message.get("protocolVersion", "2024-11-05")
        server_info = message.get("serverInfo", {})
        device_name = server_info.get("name", "unknown")
        device_version = server_info.get("version", "unknown")
        logger.info(
            f"[WS] XiaoZhi hello from {device_name} v{device_version} "
            f"(protocol={proto_version})"
        )
        await self._send_message(session.websocket, {
            "type": "hello",
            "transport": "websocket",
            "session_id": session.session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60,
            },
        })

    async def _handle_listen(self, session: RobotSession, message: dict):
        """Acknowledge listen state from StackChan (wake-word listening mode)."""
        state = message.get("state", "start")
        logger.info(f"[WS] Listen state: {state} from {session.session_id}")

        if state == "start":
            session.is_listening = True
            session.audio_buffer.clear()
            session.cancel_current_turn()
        elif state in ("end", "stop"):
            session.is_listening = False
            if session.audio_buffer:
                combined = b"".join(session.audio_buffer)
                session.audio_buffer.clear()
                logger.info(f"[WS] Explicit end-of-speech: {len(combined)} bytes PCM")
                text = await asr_service.transcribe(combined)
                if text:
                    session.current_text = text
                    self._start_turn(session, text)
            if session.eos_task and not session.eos_task.done():
                session.eos_task.cancel()

        await self._send_message(session.websocket, {
            "type": "listen_ack",
            "state": state,
            "session_id": session.session_id,
        })

    async def _handle_ping(self, session: RobotSession, message: dict):
        await self._send_message(session.websocket, {
            "type": "pong",
            "timestamp": message.get("timestamp"),
        })

    # ── turn management ───────────────────────────────────────

    def _start_turn(self, session: RobotSession, text: str):
        """Cancel any active turn and launch a new one"""
        session.cancel_current_turn()
        turn_id = session.turn_id

        task = asyncio.create_task(
            self._process_turn(session, text, turn_id),
            name=f"turn-{turn_id}-{session.session_id}",
        )
        session.pending_task = task
        metrics.turns_total += 1
        logger.info(f"[WS] Turn {turn_id} started: {text[:60]}")

    async def _thinking_heartbeat(self, session: RobotSession, turn_id: int):
        """Send periodic status updates during long LLM waits.

        Fires at 4s, 8s, 15s if first token hasn't arrived yet.
        Cancelled by llm_reader when first token arrives.
        """
        try:
            await asyncio.sleep(4)
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "status", "message": "deep thinking",
                    "action": "thinking", "turn": turn_id,
                })
            await asyncio.sleep(4)
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "status", "message": "still working",
                    "action": "thinking", "turn": turn_id,
                })
            await asyncio.sleep(7)
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "status", "message": "almost ready",
                    "action": "thinking", "turn": turn_id,
                })
        except asyncio.CancelledError:
            pass

    def _build_system_prompt(self, session: RobotSession) -> str:
        """Build system prompt with multi-user profile injection."""
        base = config.robot.system_prompt
        if not session.person_profiles:
            return base

        lines = [base, "\n当前在场用户信息："]
        for name, p in session.person_profiles.items():
            rel = p.get("relationship", "")
            pref = p.get("preferences", "")
            info = f"- {name}"
            if rel:
                info += f"（{rel}）"
            if pref:
                info += f"，偏好：{pref}"
            lines.append(info)

        lines.append("根据当前用户身份调整回应。")
        return "\n".join(lines)

    async def _process_turn(self, session: RobotSession, text: str, turn_id: int):
        """Simplified pipeline: LLM stream → collect response → send via StackChan TextMessage."""
        if not self.hermes:
            logger.error("[WS] Hermes client not available")
            await self._send_stackchan_text(session, "抱歉，AI 服务未连接")
            return

        try:
            logger.info(f"[WS] Turn {turn_id} processing: {text[:60]}")

            # Collect full LLM response
            buffer = ""
            try:
                async for token in self.hermes.chat_stream(
                    message=text,
                    session_id=session.session_id,
                    system_prompt=self._build_system_prompt(session),
                ):
                    if turn_id != session.turn_id:
                        logger.info(f"[WS] Turn {turn_id} preempted")
                        return
                    clean = sanitize_for_tts(token)
                    if clean:
                        buffer += clean
            except Exception as e:
                logger.error(f"[WS] LLM stream error: {e}")

            if turn_id != session.turn_id:
                return

            response = buffer.strip()
            if response:
                logger.info(f"[WS] Turn {turn_id} response: {response[:60]}")
                await self._send_stackchan_text(session, response)
            else:
                await self._send_stackchan_text(session, "嗯？")

        except asyncio.CancelledError:
            logger.info(f"[WS] Turn {turn_id} cancelled")
        except Exception as e:
            logger.error(f"[WS] Turn {turn_id} error: {e}")
            if turn_id == session.turn_id:
                await self._send_stackchan_text(session, f"出错了：{e}")

    # ── utilities ─────────────────────────────────────────────

    async def _send_message(self, websocket, message: dict):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"[WS] Send error: {e}")

    async def _send_stackchan_binary(self, websocket, data_type: int, payload: bytes = b""):
        """Send a StackChan binary protocol packet: [1B type][4B big-endian len][N B payload]"""
        header = bytes([data_type]) + struct.pack(">I", len(payload))
        try:
            await websocket.send_bytes(header + payload)
        except Exception as e:
            logger.error(f"[WS] StackChan send error: {e}")

    async def _send_stackchan_text(self, session: RobotSession, content: str, name: str = "叁叁"):
        """Send a text response via StackChan TextMessage (0x07)."""
        payload = json.dumps({"name": name, "content": content}, ensure_ascii=False).encode('utf-8')
        await self._send_stackchan_binary(session.websocket, 0x07, payload)
        logger.info(f"[WS] StackChan text sent: {content[:40]}")

    async def broadcast(self, message: dict):
        for session in self.sessions.values():
            try:
                await self._send_message(session.websocket, message)
            except Exception:
                logger.warning(f"[WS] Broadcast failed for session {session.session_id}")


ws_handler = RobotWebSocketHandler()
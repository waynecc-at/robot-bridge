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
import time
import base64
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

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
    last_activity: float = 0

    is_listening: bool = False
    current_text: str = ""
    hermes: Optional[HermesClient] = None

    # Barge-in turn management
    turn_id: int = 0
    pending_task: Optional[asyncio.Task] = None

    # Audio buffer for VAD-based barge-in
    audio_buffer: list[bytes] = field(default_factory=list)

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

            msg_type = message.get("type", "unknown")
            metrics.record_ws_message(msg_type)
            logger.debug(f"[WS] Message type={msg_type} from {session.session_id}")

            if msg_type == "text":
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

    def _parse_binary_message(self, data: bytes) -> dict:
        msg_type = data[0] if data else 0
        audio_data = data[1:] if len(data) > 1 else b""
        return {
            "type": "audio",
            "data": base64.b64encode(audio_data).decode(),
            "format": "pcm"
        }

    # ── message handlers ──────────────────────────────────────

    async def _handle_text_message(self, session: RobotSession, message: dict):
        text = message.get("text", "")
        logger.info(f"[WS] Text input: {text}")
        session.current_text = text
        self._start_turn(session, text)

    async def _handle_audio_message(self, session: RobotSession, message: dict):
        audio_b64 = message.get("data", "")
        if not audio_b64:
            logger.warning("[WS] Empty audio data received")
            return

        audio_bytes = base64.b64decode(audio_b64)

        # During VAD listening: buffer chunks for batch transcription on VAD end
        if session.is_listening:
            session.audio_buffer.append(audio_bytes)
            return

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
        """Concurrent pipeline: LLM stream → sentence queue → TTS worker.

        LLM streaming and TTS run in parallel via asyncio.Queue.
        Turn timeout (120s), TTS error resilience, barge-in fade-out.
        """
        if not self.hermes:
            logger.error("[WS] Hermes client not available")
            await self._send_message(session.websocket, {
                "type": "error",
                "message": "Hermes gateway not connected",
            })
            return

        heartbeat_task: Optional[asyncio.Task] = None

        try:
            await self._send_message(session.websocket, {
                "type": "status",
                "message": "understanding",
                "action": "thinking",
                "turn": turn_id,
            })

            logger.info(f"[WS] Turn {turn_id} processing: {text[:60]}")

            # Queue coordinates LLM reader ↔ TTS worker
            sentence_queue: asyncio.Queue = asyncio.Queue()

            # Thinking heartbeat: background task, cancelled on first token
            heartbeat_task = asyncio.create_task(
                self._thinking_heartbeat(session, turn_id)
            )

            async def llm_reader():
                """Stream LLM tokens → sanitize → typewriter → push sentences to queue."""
                nonlocal heartbeat_task
                buffer = ""
                first_token_sent = False
                try:
                    async for token in self.hermes.chat_stream(
                        message=text,
                        session_id=session.session_id,
                        system_prompt=self._build_system_prompt(session),
                    ):
                        if turn_id != session.turn_id:
                            logger.info(f"[WS] Turn {turn_id} preempted (LLM reader)")
                            return

                        # Cancel heartbeat on first token
                        if not first_token_sent:
                            first_token_sent = True
                            heartbeat_task.cancel()

                        # Sanitize each token
                        clean_token = sanitize_for_tts(token)
                        if not clean_token:
                            continue

                        buffer += clean_token

                        # Typewriter effect: per-token
                        await self._send_message(session.websocket, {
                            "type": "response_text",
                            "text": clean_token,
                            "partial": True,
                            "turn": turn_id,
                        })

                        if turn_id != session.turn_id:
                            return

                        # Split complete sentences
                        sentences, buffer = _split_sentences(buffer)
                        for sentence in sentences:
                            if sentence:
                                await self._send_message(session.websocket, {
                                    "type": "response_text",
                                    "text": sentence,
                                    "partial": False,
                                    "turn": turn_id,
                                })
                                await sentence_queue.put(sentence)
                finally:
                    if not first_token_sent:
                        heartbeat_task.cancel()

                # Flush remaining buffer
                if buffer.strip():
                    sentence = buffer.strip()
                    await self._send_message(session.websocket, {
                        "type": "response_text",
                        "text": sentence,
                        "partial": False,
                        "turn": turn_id,
                    })
                    await sentence_queue.put(sentence)

                # Signal end of LLM output
                await sentence_queue.put(None)

            async def tts_worker():
                """Consume sentences → TTS synthesize → stream audio."""
                sentence_count = 0
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        break

                    if turn_id != session.turn_id:
                        # Barge-in fade-out: send final marker so robot stops cleanly
                        await self._send_message(session.websocket, {
                            "type": "tts_audio", "data": "",
                            "final": True, "turn": session.turn_id,
                        })
                        return

                    sentence_count += 1
                    logger.info(
                        f"[WS] Turn {turn_id} sentence {sentence_count}: {sentence[:60]}"
                    )

                    await self._send_message(session.websocket, {
                        "type": "status",
                        "message": "speaking",
                        "action": "speaking",
                        "turn": turn_id,
                    })

                    # TTS with error resilience — don't crash pipeline
                    metrics.tts_requests += 1
                    try:
                        async for chunk in tts_service.synthesize_stream(sentence):
                            if turn_id != session.turn_id:
                                await self._send_message(session.websocket, {
                                    "type": "tts_audio", "data": "",
                                    "final": True, "turn": session.turn_id,
                                })
                                return
                            # Header text frame, then audio data as binary frame
                            await self._send_message(session.websocket, {
                                "type": "tts_audio",
                                "format": "wav",
                                "final": False,
                                "turn": turn_id,
                            })
                            metrics.tts_audio_bytes += len(chunk)
                            await session.websocket.send_bytes(chunk)
                    except Exception as e:
                        metrics.tts_errors += 1
                        logger.warning(f"[WS] TTS failed for sentence, skipping: {e}")
                        continue

                # Final marker
                if turn_id == session.turn_id:
                    await self._send_message(session.websocket, {
                        "type": "tts_audio",
                        "data": "",
                        "final": True,
                        "turn": turn_id,
                    })
                    logger.info(f"[WS] Turn {turn_id} complete: {sentence_count} sentences")

            # Run both tasks with turn-level timeout
            await asyncio.wait_for(
                asyncio.gather(llm_reader(), tts_worker()),
                timeout=120,
            )

        except asyncio.TimeoutError:
            logger.warning(f"[WS] Turn {turn_id} timed out (120s)")
            if heartbeat_task:
                heartbeat_task.cancel()
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "status", "message": "timeout",
                    "action": "idle", "turn": turn_id,
                })
        except asyncio.CancelledError:
            logger.info(f"[WS] Turn {turn_id} cancelled (CancelledError)")
            if heartbeat_task:
                heartbeat_task.cancel()
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "interrupt",
                    "turn": session.turn_id,
                })
        except Exception as e:
            logger.error(f"[WS] Turn {turn_id} processing error: {e}")
            if heartbeat_task:
                heartbeat_task.cancel()
            if turn_id == session.turn_id:
                await self._send_message(session.websocket, {
                    "type": "error",
                    "message": f"Error: {str(e)}",
                })

    # ── utilities ─────────────────────────────────────────────

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
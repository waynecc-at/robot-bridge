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
from dataclasses import dataclass
from loguru import logger

from .asr_service import asr_service
from .config import config
from .hermes_client import HermesClient
from .text_utils import sanitize_for_tts
from .tts_service import tts_service


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

    def cancel_current_turn(self):
        """Cancel the active turn so a new one can start"""
        self.turn_id += 1
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()
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
            await self._send_message(session.websocket, {
                "type": "interrupt",
                "turn": session.turn_id,
            })
            logger.info(f"[WS] Barge-in: interrupt sent, new turn_id={session.turn_id}")

        elif state == "end":
            session.is_listening = False
            if session.current_text:
                self._start_turn(session, session.current_text)

        elif state == "speaking":
            pass

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
                        system_prompt=config.robot.system_prompt,
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
                    try:
                        async for chunk in tts_service.synthesize_stream(sentence):
                            if turn_id != session.turn_id:
                                await self._send_message(session.websocket, {
                                    "type": "tts_audio", "data": "",
                                    "final": True, "turn": session.turn_id,
                                })
                                return
                            await self._send_message(session.websocket, {
                                "type": "tts_audio",
                                "format": "wav",
                                "data": base64.b64encode(chunk).decode(),
                                "final": False,
                                "turn": turn_id,
                            })
                    except Exception as e:
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
"""WebSocket handler — XiaoZhi protocol server-side for StackChan ESP32 devices.

Protocol (xiaozhi-esp32):
  Client hello: {"type":"hello","version":1,"transport":"websocket","audio_params":{...}}
  Server hello: {"type":"hello","session_id":"...","audio_params":{...}}
  Listen start: {"type":"listen","state":"start","mode":"auto"}
  Listen stop:  {"type":"listen","state":"stop"}  (VAD silence trigger)
  Raw Opus audio in binary WS frames while listening (version 1: no header)

  Server → Device:
    {"type":"stt","text":"..."}        — display recognised speech
    {"type":"llm","emotion":"neutral"} — set facial emotion
    {"type":"tts","state":"start"}     — begin TTS playback
    <binary Opus frames>               — TTS audio
    {"type":"tts","state":"stop"}      — end TTS playback
"""
import asyncio
import io
import json
import time
from typing import Optional
from dataclasses import dataclass, field
from loguru import logger

import opuslib

from .asr_service import asr_service
from .config import config
from .face_registry import face_registry
from .face_tracker import face_tracker
from .metrics import metrics
from .profile_store import profile_store
from .tts_service import tts_service
from .vision_service import vision_service


OPUS_SR = 16000       # Opus sample rate
OPUS_CHANNELS = 1
OPUS_FRAME_MS = 60    # frame duration
FRAME_SAMPLES = OPUS_SR * OPUS_FRAME_MS // 1000  # 960
FRAME_BYTES = FRAME_SAMPLES * 2                   # 1920


@dataclass
class RobotSession:
    device_id: str
    session_id: str
    websocket: object
    last_activity: float = field(default_factory=time.time)
    last_asr_text: str = ""  # Latest ASR result, read by MCP listen() tool
    messages: list = field(default_factory=list)  # Conversation history for LLM

    is_listening: bool = False
    current_text: str = ""

    turn_id: int = 0
    pending_task: Optional[asyncio.Task] = None
    audio_buffer: list[bytes] = field(default_factory=list)
    opus_decoder: Optional[object] = None
    eos_task: Optional[asyncio.Task] = None

    # multi-user vision
    current_person: Optional[str] = None
    person_profiles: dict[str, dict] = field(default_factory=dict)
    unknown_face_pending: bool = False    # set when face_registry signals a new person
    last_conversation: str = ""           # user+assistant text from latest turn
    _led_task: Optional[asyncio.Task] = None
    _tts_stop: Optional[asyncio.Event] = None  # set to cancel in-progress TTS stream

    # Idle mode — when True, auto-listen is suppressed (user said goodbye)
    idle_mode: bool = False

    def cancel_current_turn(self):
        self.turn_id += 1
        # Cancel in-progress TTS stream
        if self._tts_stop:
            self._tts_stop.set()
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()
            metrics.turns_cancelled += 1
            logger.info(f"[WS] Turn cancelled: new turn_id={self.turn_id}")


class RobotWebSocketHandler:
    def __init__(self):
        self.sessions: dict[str, RobotSession] = {}
        # pending response carried across reconnects (keyed by device MAC)
        self._pending: dict[str, str] = {}
        # Webhook URL for Hermes Agent (replaces Agent Driver callbacks)
        self._webhook_url = "http://127.0.0.1:8644/webhooks/stackchan"

    async def cleanup_stale_sessions(self, ttl: float = 600):
        now = time.time()
        stale = [sid for sid, s in self.sessions.items() if now - s.last_activity > ttl]
        for sid in stale:
            self.sessions[sid].cancel_current_turn()
            del self.sessions[sid]
            logger.info(f"[WS] Stale session cleaned: {sid}")

    # ── message dispatch ──────────────────────────────────────

    async def _handle_message(self, session: RobotSession, raw: str | bytes):
        session.last_activity = time.time()
        try:
            if isinstance(raw, bytes):
                # Check for binary DataType::Jpeg frame (0x02 + 4-byte BE length)
                if len(raw) >= 5 and raw[0] == 0x02:
                    await self._handle_jpeg_frame(session, raw)
                    return
                msg = {"type": "audio", "data": raw, "format": "opus"}
            else:
                msg = json.loads(raw)

            # XiaoZhi hello (has "version" or "type":"hello" + "audio_params")
            if msg.get("type") == "hello":
                await self._handle_hello(session, msg)
                return
            if msg.get("type") == "listen":
                await self._handle_listen(session, msg)
                return
            if msg.get("type") == "audio":
                await self._handle_audio(session, msg)
                return
            if msg.get("type") == "abort":
                logger.info(f"[WS] Abort: {msg.get('reason','')}")
                session.cancel_current_turn()
                self._stop_rainbow(session)
                return
            if msg.get("type") == "mcp":
                await self._handle_mcp_response(session, msg)
                return
            if msg.get("type") == "vision_frame":
                await self._handle_vision_frame(session, msg)
                return
            if msg.get("type") == "ping":
                await self._send_json(session.websocket, {"type": "pong"})
                return
            logger.warning(f"[WS] Unknown message type: {msg.get('type','?')}")
        except json.JSONDecodeError:
            logger.error(f"[WS] Invalid JSON: {raw[:100] if isinstance(raw, str) else raw[:20].hex()}")
        except Exception as e:
            logger.error(f"[WS] Message handling error: {e}")

    # ── XiaoZhi protocol handlers ─────────────────────────────

    async def _handle_hello(self, session: RobotSession, msg: dict):
        version = msg.get("version", 1)
        transport = msg.get("transport", "websocket")
        audio = msg.get("audio_params", {})
        logger.info(f"[WS] XiaoZhi hello v{version} {transport} "
                    f"sr={audio.get('sample_rate')} ch={audio.get('channels')}")

        # carry pending response from previous session
        pending = self._pending.pop(session.device_id, None)

        await self._send_json(session.websocket, {
            "type": "hello",
            "session_id": session.session_id,
            "transport": "websocket",
            "audio_params": {
                "format": "opus",
                "sample_rate": OPUS_SR,
                "channels": OPUS_CHANNELS,
                "frame_duration": OPUS_FRAME_MS,
            },
        })
        # Start camera stream for face detection/tracking
        asyncio.create_task(self._start_camera(session))

        if pending:
            asyncio.create_task(self._deliver_pending(session, pending))

    async def _deliver_pending(self, session: RobotSession, text: str):
        """Send a previously-computed response to a reconnected device."""
        await asyncio.sleep(0.5)  # let hello processing settle
        logger.info(f"[WS] Delivering pending: {text[:40]}")
        await self._send_json(session.websocket, {
            "type": "stt", "text": text, "session_id": session.session_id,
        })
        await self._send_json(session.websocket, {
            "type": "llm", "emotion": "neutral", "session_id": session.session_id,
        })
        await self._tts_and_send(session, text)

    async def _handle_listen(self, session: RobotSession, msg: dict):
        state = msg.get("state", "start")
        mode = msg.get("mode", "manual")
        logger.info(f"[WS] Listen {state} mode={mode}")

        if state == "start":
            if session.idle_mode:
                logger.info("[WS] Idle mode — ignoring auto-listen")
                return
            # Don't start listening during TTS — wait for TTS to finish first.
            # Prevents green LED from overwriting blue LED mid-speech.
            if session._tts_stop and not session._tts_stop.is_set():
                return
            session.is_listening = True
            session.audio_buffer.clear()
            if session.eos_task and not session.eos_task.done():
                session.eos_task.cancel()
            self._stop_rainbow(session)
            self._set_led(session, 0, 168, 0)  # Green = listening
        elif state == "stop":
            session.is_listening = False
            self._set_led(session, 0, 0, 0)  # LED off
        elif state == "detect":
            logger.info(f"[WS] Wake word: {msg.get('text','')}")
            # Exit idle mode on wake word
            if session.idle_mode:
                session.idle_mode = False
                logger.info("[WS] Exiting idle mode")
            # Flash green to confirm wake — hold for 1.5s
            self._stop_rainbow(session)
            self._set_led(session, 0, 168, 0)
            # Schedule off (rainbow will override if speech follows)
            asyncio.create_task(self._led_off_after(session, 1.8))
            # Notify Hermes webhook of wake word
            asyncio.create_task(self._post_webhook(session, "叁叁"))

    async def _handle_audio(self, session: RobotSession, msg: dict):
        """Raw Opus audio in binary frames — decode, buffer all.
        Only reset EOS timer on non-silence chunks (DTX sends silence packets).
        If speech arrives while TTS is playing, barge-in (interrupt TTS)."""
        if not session.is_listening:
            # Barge-in: if TTS is playing and user speaks, interrupt
            if hasattr(session, '_tts_stop') and session._tts_stop and not session._tts_stop.is_set():
                opus_data = msg["data"] if isinstance(msg["data"], bytes) else msg.get("data")
                if opus_data:
                    try:
                        if isinstance(opus_data, str):
                            import base64
                            opus_data = base64.b64decode(opus_data)
                        pcm = self._opus_decode(session, opus_data)
                        silent, rms = _is_silence(pcm)
                        if not silent:
                            logger.info(f"[WS] Barge-in detected (RMS={rms:.0f}) — interrupting TTS")
                            session._tts_stop.set()
                            session.cancel_current_turn()
                            session.is_listening = True
                            session.audio_buffer.clear()
                            session.last_asr_text = ""
                            self._reset_eos(session)
                    except Exception:
                        pass
                return
            # No TTS, but audio is arriving — auto-start listening.
            # Covers the gap between ASR completion and next listen start where
            # ESP32 auto-listen hasn't fired yet but the user is already speaking.
            opus_data = msg["data"] if isinstance(msg["data"], bytes) else msg.get("data")
            if opus_data:
                try:
                    if isinstance(opus_data, str):
                        import base64
                        opus_data = base64.b64decode(opus_data)
                    pcm = self._opus_decode(session, opus_data)
                    silent, rms = _is_silence(pcm)
                    if not silent:
                        logger.info(f"[WS] Pre-listen audio (RMS={rms:.0f}) — starting early")
                        session.is_listening = True
                        session.audio_buffer.clear()
                        session.last_asr_text = ""
                        session.audio_buffer.append(pcm)
                        self._reset_eos(session)
                except Exception:
                    pass
            return
        opus_data = msg["data"] if isinstance(msg["data"], bytes) else msg.get("data")
        if not opus_data:
            return
        if isinstance(opus_data, str):
            import base64
            opus_data = base64.b64decode(opus_data)
        try:
            pcm = self._opus_decode(session, opus_data)
            session.audio_buffer.append(pcm)
            # Only reset EOS if this chunk contains actual speech
            silent, rms = _is_silence(pcm)
            if not silent:
                self._reset_eos(session)
            # Throttled log every 30 chunks
            if len(session.audio_buffer) % 30 == 0:
                logger.info(f"[WS] Audio #{len(session.audio_buffer)} chunks, "
                           f"RMS={rms:.0f} silent={silent}, "
                           f"total={sum(len(c) for c in session.audio_buffer)}B")
        except Exception as e:
            logger.error(f"[WS] Opus decode: {e}")

    async def _handle_jpeg_frame(self, session: RobotSession, raw: bytes):
        """Decode binary DataType::Jpeg frame from ESP32 and route to vision."""
        import base64
        # Binary format: [0x02][len:4 BE][jpeg payload]
        if len(raw) < 5:
            return
        jpeg_data = raw[5:]
        if not jpeg_data:
            return
        # Route to vision_frame handler using base64 wrapper
        msg = {
            "type": "vision_frame",
            "data": base64.b64encode(jpeg_data).decode(),
        }
        await self._handle_vision_frame(session, msg)

    async def _start_camera(self, session: RobotSession):
        """Send StartCameraStream binary command to ESP32."""
        # DataType::StartCameraStream = 0x05, no payload
        packet = bytes([0x05, 0x00, 0x00, 0x00, 0x00])
        try:
            await session.websocket.send_bytes(packet)
            logger.info("[WS] Camera stream started")
        except Exception as e:
            logger.error(f"[WS] Failed to start camera: {e}")

    def _reset_eos(self, session: RobotSession):
        """Start/reset end-of-speech timer. Fires 1.5s after last speech chunk."""
        if session.eos_task and not session.eos_task.done():
            session.eos_task.cancel()
        session.eos_task = asyncio.create_task(
            self._eos_handler(session),
            name=f"eos-{session.session_id}",
        )

    # ── vision frame handling ──────────────────────────────────

    async def _handle_vision_frame(self, session: RobotSession, msg: dict):
        """Process JPEG frame/crop from ESP32: detect + recognize faces.
        Also runs face_tracker to keep the camera centered on the speaker.
        """
        import base64

        data_b64 = msg.get("data", "")
        if not data_b64:
            return
        try:
            jpeg = base64.b64decode(data_b64)
        except Exception:
            return

        is_face_crop = msg.get("face", False)

        if is_face_crop:
            import cv2
            import numpy as np
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if crop is None:
                return
            profile = profile_store.recognize(crop)
            bx, by, bw, bh = msg.get("bx", 0), msg.get("by", 0), msg.get("bw", 0), msg.get("bh", 0)
            person_name = profile.name if profile else ""
            self._apply_vision_result(session, profile, crop, (bx, by, bw, bh), person_name,
                                      (320, 240))
            return

        # Full frame — server-side detect + recognize
        fw = msg.get("width", 320)
        fh = msg.get("height", 240)
        results = await vision_service.process_frame(jpeg)
        for r in results:
            profile = r.get("profile")
            bbox = r.get("bbox", (0, 0, 0, 0))
            person_name = profile.name if profile else ""
            self._apply_vision_result(session, profile, r.get("crop"), bbox, person_name,
                                      (fw, fh))

    def _apply_vision_result(self, session: RobotSession, profile, crop,
                             bbox: tuple, person_name: str, frame_size: tuple):
        """Update session state + run face tracker for camera centering."""
        if profile is not None:
            session.person_profiles[profile.name] = {
                "name": profile.name,
                "relationship": profile.relationship,
                "preferences": profile.preferences,
            }
            if session.current_person != profile.name:
                session.current_person = profile.name
                session.unknown_face_pending = False
                logger.info(f"[Vision] Recognized: {profile.name} ({profile.relationship})")
        elif not session.current_person:
            triggered = face_registry.on_unknown_face(crop, bbox)
            if triggered:
                session.unknown_face_pending = True
                logger.info("[Vision] Unknown face flag set — LLM will be prompted")

        # Face tracking: gently turn toward the speaker
        cmd = face_tracker.update(
            bbox, frame_size,
            person_name=person_name,
            current_person=session.current_person or "",
        )
        if cmd:
            asyncio.create_task(self._send_mcp_async(session, "self.robot.set_head_angles", {
                "yaw": cmd[0], "pitch": cmd[1], "speed": cmd[2],
            }))

    async def _eos_handler(self, session: RobotSession):
        """After 1.5s of silence: run ASR, POST result to Hermes webhook."""
        try:
            await asyncio.sleep(1.5)
            if session.is_listening and session.audio_buffer:
                pcm_data = b"".join(session.audio_buffer)
                logger.info(f"[WS] EOS — processing {len(pcm_data)}B from {len(session.audio_buffer)} chunks")
                text = await asr_service.transcribe(pcm_data)
                if text.strip():
                    logger.info(f"[WS] ASR: {text}")
                    session.last_asr_text = text
                    # Send STT display to ESP32
                    await self._send_json(session.websocket, {
                        "type": "stt", "text": text, "session_id": session.session_id,
                    })
                    # POST to Hermes webhook — Hermes Agent handles the rest
                    asyncio.create_task(self._post_webhook(session, text))
                else:
                    logger.info("[WS] ASR: no speech detected")
                session.is_listening = False
        except asyncio.CancelledError:
            pass

    async def _post_webhook(self, session: RobotSession, text: str):
        """POST ASR text to Hermes webhook. Hermes Agent drives the response."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(
                    "http://127.0.0.1:8644/webhooks/stackchan",
                    json={
                        "text": text,
                        "session_id": session.session_id,
                        "person": session.current_person or "",
                    },
                    headers={"X-Webhook-Signature": "INSECURE_NO_AUTH"},
                )
            logger.info(f"[Webhook] Sent: {text[:40]}")
        except Exception as e:
            logger.error(f"[Webhook] Failed: {e}")

    # ── TTS streaming ──────────────────────────────────────────

    async def _tts_and_send(self, session: RobotSession, text: str):
        """TTS → scipy resample → Opus encode → stream to device.

        Reuses one Opus encoder for all sentences to avoid state-reset artifacts.
        """
        import re
        sentences = _split_sentences(text)

        session._tts_stop = asyncio.Event()  # fresh cancel event for this turn

        try:
            await self._send_json(session.websocket, {
                "type": "tts", "state": "start", "session_id": session.session_id,
            })
            self._stop_rainbow(session)
            self._set_led(session, 0, 0, 168)  # Blue = speaking
        except Exception:
            return

        # Single Opus encoder — reused across sentences to avoid reset artifacts
        encoder = opuslib.Encoder(OPUS_SR, OPUS_CHANNELS, 'voip')
        encoder.bitrate = 48000
        encoder.complexity = 10

        frame_count = 0
        try:
            for i, sentence in enumerate(sentences):
                if session._tts_stop and session._tts_stop.is_set():
                    logger.info("[WS] TTS stopped mid-stream")
                    return

                frame_count += await self._tts_send_one(session, sentence, encoder)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[WS] TTS stream error: {e}")
        finally:
            self._set_led(session, 0, 0, 0)
            try:
                await self._send_json(session.websocket, {
                    "type": "tts", "state": "stop", "session_id": session.session_id,
                })
            except Exception:
                pass
            logger.info(f"[WS] TTS sent: {frame_count} Opus frames, {len(sentences)} sentences")

    async def _tts_send_one(self, session: RobotSession, text: str, encoder) -> int:
        """Synthesize one sentence and stream Opus frames. Returns frame count."""
        if not text.strip():
            return 0

        pcm_bytes = await tts_service.synthesize_pcm(text)
        if not pcm_bytes:
            return 0

        count = 0
        t0 = time.monotonic()
        pos = 0
        while pos + FRAME_BYTES <= len(pcm_bytes):
            if session._tts_stop and session._tts_stop.is_set():
                return count
            frame = pcm_bytes[pos:pos + FRAME_BYTES]
            pos += FRAME_BYTES
            opus = encoder.encode(frame, FRAME_SAMPLES)
            try:
                await session.websocket.send_bytes(opus)
                count += 1
            except Exception:
                return count
            # Frame budget pacing: each frame is 60ms. Sleep only the
            # remaining budget so actual interval stays ≤60ms. Prevents
            # both ESP32 buffer underrun (too slow) and overflow (too fast).
            elapsed = time.monotonic() - t0
            target = count * (OPUS_FRAME_MS / 1000.0)
            slack = target - elapsed
            if slack > 0.002:  # Only sleep if >2ms remaining
                await asyncio.sleep(slack)

        # Drop partial final frame (<60ms, inaudible)
        return count

    async def _handle_mcp_response(self, session: RobotSession, msg: dict):
        """Resolve MCP future when ESP32 sends back a response."""
        payload = msg.get("payload", {})
        mcp_id = payload.get("id")
        if mcp_id and hasattr(session, '_mcp_futures') and mcp_id in session._mcp_futures:
            result = payload.get("result", payload.get("error", "ok"))
            session._mcp_futures[mcp_id].set_result(json.dumps(result, ensure_ascii=False))
            session._mcp_futures.pop(mcp_id, None)
            logger.info(f"[WS] MCP response id={mcp_id}")

    async def _send_mcp_async(self, session: RobotSession, name: str, args: dict):
        """Fire-and-forget MCP to ESP32 — used for face tracking (no response needed)."""
        mcp_id = int(time.time() * 1000) % 100000
        payload = {
            "jsonrpc": "2.0",
            "id": mcp_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        try:
            await session.websocket.send_text(json.dumps({
                "type": "mcp",
                "session_id": session.session_id,
                "payload": payload,
            }, ensure_ascii=False))
        except Exception:
            pass

    async def _mcp_call(self, session: RobotSession, name: str, args: dict) -> str:
        """Send MCP JSON-RPC to ESP32 and wait for response."""
        # Pause face tracker when LLM deliberately moves the head
        if name == "self.robot.set_head_angles":
            face_tracker.pause(4.0)

        mcp_id = int(time.time() * 1000) % 100000
        payload = {
            "jsonrpc": "2.0",
            "id": mcp_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        msg = {
            "type": "mcp",
            "session_id": session.session_id,
            "payload": payload,
        }

        # Create an event to wait for the response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        if not hasattr(session, '_mcp_futures'):
            session._mcp_futures = {}
        session._mcp_futures[mcp_id] = future

        await session.websocket.send_text(json.dumps(msg, ensure_ascii=False))
        logger.info(f"[WS] MCP sent: {name}({args}) id={mcp_id}")

        try:
            result = await asyncio.wait_for(future, timeout=10.0)
            return result
        except asyncio.TimeoutError:
            session._mcp_futures.pop(mcp_id, None)
            return json.dumps({"error": "timeout"})

    # ── LED status indicators ─────────────────────────────────

    _RAINBOW = [
        (168, 0, 0), (168, 84, 0), (168, 168, 0),
        (0, 168, 0), (0, 168, 168), (0, 0, 168), (84, 0, 168),
    ]

    def _set_led(self, session: RobotSession, r: int, g: int, b: int):
        """Fire-and-forget LED color."""
        asyncio.create_task(self._send_mcp_async(
            session, "self.robot.set_led_color",
            {"red": r, "green": g, "blue": b},
        ))

    async def _led_off_after(self, session: RobotSession, delay: float):
        """Turn LED off after delay (unless overridden by later LED commands)."""
        await asyncio.sleep(delay)
        self._set_led(session, 0, 0, 0)

    def _start_rainbow(self, session: RobotSession):
        """Start color-chasing LED animation (cancels any previous)."""
        self._stop_rainbow(session)
        session._led_task = asyncio.create_task(
            self._rainbow_loop(session),
            name=f"led-{session.session_id}",
        )

    async def _rainbow_loop(self, session: RobotSession):
        """Background: cycle LED through rainbow colors."""
        i = 0
        try:
            while True:
                r, g, b = self._RAINBOW[i % len(self._RAINBOW)]
                try:
                    await self._send_mcp_async(
                        session, "self.robot.set_led_color",
                        {"red": r, "green": g, "blue": b},
                    )
                except Exception:
                    pass
                i += 1
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            pass

    def _stop_rainbow(self, session: RobotSession):
        """Stop rainbow animation."""
        if session._led_task and not session._led_task.done():
            session._led_task.cancel()
            session._led_task = None

    # ── utilities ──────────────────────────────────────────────

    def _opus_decode(self, session: RobotSession, data: bytes) -> bytes:
        if session.opus_decoder is None:
            session.opus_decoder = opuslib.Decoder(OPUS_SR, OPUS_CHANNELS)
        return session.opus_decoder.decode(data, FRAME_SAMPLES)

    async def _send_json(self, ws, obj: dict):
        try:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            logger.error(f"[WS] send error: {e}")

    def _save_tts_debug(self, frames: list[bytes]):
        """Decode Opus frames back to WAV and save for quality verification."""
        try:
            dec = opuslib.Decoder(OPUS_SR, OPUS_CHANNELS)
            pcm = b""
            for f in frames:
                pcm += dec.decode(f, FRAME_SAMPLES)
            with wave.open("/tmp/tts_debug.wav", "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(OPUS_SR)
                w.writeframes(pcm)
            logger.info(f"[WS] Debug WAV saved: {len(pcm)}B PCM")
        except Exception as e:
            logger.error(f"[WS] Debug save error: {e}")

    async def process_on_disconnect(self, session: RobotSession):
        """Fallback: cleanup on device disconnect."""
        if session.is_listening and session.audio_buffer:
            logger.info("[WS] Disconnect during listen — discarding buffered audio")
        # Cancel any in-flight TTS tasks
        session.cancel_current_turn()


# ── helpers ────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split Chinese text into sentences by punctuation, keeping separator."""
    import re
    parts = re.split(r'(?<=[。！？!?\n])', text)
    result = [p.strip() for p in parts if p.strip()]
    return result if result else [text]


def _is_silence(pcm: bytes, threshold: int = 250) -> tuple[bool, float]:
    """Check if 16-bit mono PCM is silence. Returns (is_silent, rms)."""
    import array
    if len(pcm) < 2:
        return True, 0.0
    samples = array.array('h', pcm)
    if len(samples) == 0:
        return True, 0.0
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return rms < threshold, rms


ws_handler = RobotWebSocketHandler()

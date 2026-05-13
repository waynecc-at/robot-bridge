"""HTTP API Server for Robot Bridge"""
import asyncio
import base64
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel

from .asr_service import asr_service
from .config import config
from .hermes_client import hermes_client
from .metrics import metrics
from .tts_service import tts_service
from .vision_service import vision_service


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = True


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class WebSocketMessage(BaseModel):
    type: str
    text: Optional[str] = None
    data: Optional[str] = None


# Lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    from .websocket_handler import ws_handler

    logger.info("Robot Bridge API starting...")

    # Shared thread pool for CPU-bound inference (ASR + TTS)
    executor = ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="inference",
    )
    asr_service.set_executor(executor)
    tts_service.set_executor(executor)

    await hermes_client.__aenter__()
    await asr_service.start()
    await tts_service.start()

    # Start vision service (no-op if disabled in config)
    if config.vision.enabled:
        vision_service.enabled = True
        logger.info("[API] Vision service enabled")

    # Background task to clean up stale WebSocket sessions every 5 minutes
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)
            await ws_handler.cleanup_stale_sessions()

    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    await hermes_client.close()
    executor.shutdown(wait=True)
    logger.info("Robot Bridge API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Robot Bridge API",
    description="API for StackChan Robot Bridge",
    version="0.1.0",
    lifespan=lifespan,
)


# ============= Health & Info Endpoints =============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    async with hermes_client as hermes:
        hermes_healthy = await hermes.check_health()
    
    return {
        "status": "healthy",
        "service": "robot-bridge",
        "version": "0.1.0",
        "hermes_connected": hermes_healthy
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Robot Bridge",
        "version": "0.1.0",
        "endpoints": {
            "chat": "/api/chat",
            "tts": "/api/tts",
            "tts_stream": "/api/tts/stream",
            "websocket": "/ws/robot",
            "voices": "/api/voices",
        }
    }


@app.get("/api/metrics")
async def api_metrics():
    """Pipeline metrics in JSON format."""
    return metrics.json()


@app.get("/api/metrics/prometheus")
async def api_metrics_prometheus():
    """Pipeline metrics in Prometheus text format."""
    return Response(
        content=metrics.prometheus(),
        media_type="text/plain; charset=utf-8",
    )


# ============= Chat Endpoints =============

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.stream:
        return StreamingResponse(
            stream_chat_response(request.message, request.session_id),
            media_type="application/json"
        )

    async with hermes_client as hermes:
        logger.info(f"[API] Chat request: {request.message[:50]}...")
        response = await hermes.chat(
            message=request.message,
            session_id=request.session_id,
            system_prompt=config.robot.system_prompt,
        )
        return JSONResponse({
            "text": response.text,
            "session_id": response.session_id,
        })


async def stream_chat_response(message: str, session_id: Optional[str]):
    """Stream chat response with TTS audio (WAV format)"""
    async with hermes_client as hermes:
        # Get Hermes response
        response = await hermes.chat(
            message=message,
            session_id=session_id,
            system_prompt=config.robot.system_prompt,
        )

        response_text = response.text

        # Send text first
        yield json.dumps({
            "type": "text",
            "text": response_text,
            "session_id": response.session_id
        }) + "\n"

        # Then stream TTS audio (WAV chunks)
        async for chunk in tts_service.synthesize_stream(response_text):
            yield json.dumps({
                "type": "audio",
                "format": "wav",
                "data": base64.b64encode(chunk).decode(),
                "final": False
            }) + "\n"

        # Final marker
        yield json.dumps({
            "type": "audio",
            "data": "",
            "final": True
        }) + "\n"


# ============= TTS Endpoints =============

@app.post("/api/tts")
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize text to speech, return audio file
    """
    logger.info(f"[API] TTS request: {request.text[:50]}...")
    
    audio = await tts_service.synthesize(
        text=request.text,
        voice=request.voice,
    )
    
    return StreamingResponse(
        iter([audio]),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename=tts_{uuid.uuid4().hex[:8]}.wav"
        }
    )


@app.post("/api/tts/stream")
async def synthesize_speech_stream(request: TTSRequest):
    """
    Synthesize text to speech, stream audio chunks
    """
    logger.info(f"[API] TTS stream request: {request.text[:50]}...")
    
    async def generate():
        async for chunk in tts_service.synthesize_stream(
            text=request.text,
            voice=request.voice,
        ):
            yield chunk
    
    return StreamingResponse(
        generate(),
        media_type="audio/wav"
    )


@app.get("/api/voices")
async def list_voices(language: Optional[str] = None):
    """
    List available TTS voices
    """
    voices = await tts_service.list_voices(language)
    
    # Format for easier consumption
    formatted = []
    for voice in voices:
        formatted.append({
            "name": voice["Name"],
            "short_name": voice["ShortName"],
            "language": voice["Locale"],
            "gender": voice.get("Gender", "Unknown"),
        })
    
    return {
        "voices": formatted,
        "count": len(formatted)
    }


# ============= Profile / Vision Endpoints =============

class ProfileRegisterRequest(BaseModel):
    name: str
    relationship: str = ""
    preferences: str = ""

@app.post("/api/profile/register")
async def register_profile(request: ProfileRegisterRequest):
    """Register a person profile (face training not included)."""
    from .profile_store import profile_store

    profile_store.save_profile(
        name=request.name,
        relationship=request.relationship,
        preferences=request.preferences,
    )
    return {"status": "ok", "name": request.name}


@app.get("/api/profiles")
async def list_profiles():
    """List all registered person profiles."""
    from .profile_store import profile_store
    profiles = []
    for p in profile_store.get_all_profiles():
        profiles.append({
            "name": p.name,
            "relationship": p.relationship,
            "preferences": p.preferences,
            "sample_count": p.sample_count,
            "last_seen": p.last_seen,
        })
    return {"profiles": profiles}


@app.delete("/api/profile/{name}")
async def delete_profile(name: str):
    """Delete a person profile."""
    from .profile_store import profile_store
    ok = profile_store.remove_person(name)
    return {"status": "ok" if ok else "not_found"}


# ============= WebSocket Endpoint =============

WS_IDLE_TIMEOUT = 300  # seconds without activity → goodbye + close

@app.websocket("/ws/robot")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ESP32 connections.

    Continuous conversation mode: once connected, no wake word needed.
    Idle for 5 minutes → sends goodbye message and closes.
    ESP32 should return to low-power wake-word listening on close.

    Protocol:
    - Send: {"type": "text", "text": "Hello"}
    - Receive: {"type": "response_text", "text": "Hi there!"}
    - Receive: {"type": "tts_audio", "data": "base64...", "final": false}
    """
    from .websocket_handler import ws_handler

    await websocket.accept()
    logger.info("[WS] WebSocket client connected")

    await ws_handler.set_hermes_client(hermes_client)

    session_id = str(uuid.uuid4())[:8]
    session = RobotSession(websocket=websocket, session_id=session_id)
    ws_handler.sessions[session_id] = session

    try:
        while True:
            idle = time.time() - session.last_activity
            if idle > WS_IDLE_TIMEOUT:
                logger.info(f"[WS] Session {session_id} idle timeout ({idle:.0f}s)")
                await ws_handler._send_message(websocket, {
                    "type": "goodbye", "reason": "idle_timeout",
                })
                await websocket.close(1000, "idle_timeout")
                break

            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=5)
            except asyncio.TimeoutError:
                # No data for 5s — loop back to check idle timeout
                continue

            session.last_activity = time.time()

            if "text" in msg:
                data = msg["text"]
                logger.debug(f"[WS] Received text: {data[:100]}...")
                await ws_handler._handle_message(session, data)
            elif "bytes" in msg:
                data = msg["bytes"]
                logger.debug(f"[WS] Received binary: {len(data)} bytes")
                await ws_handler._handle_message(session, data)
            elif msg.get("type") == "websocket.disconnect":
                logger.info("[WS] WebSocket client disconnected")
                break
            else:
                logger.warning(f"[WS] Unknown message format: {list(msg.keys())}")

    except WebSocketDisconnect:
        logger.info("[WS] WebSocket client disconnected")
    except Exception as e:
        logger.opt(exception=True).error(f"[WS] Error: {e}")
    finally:
        ws_handler.sessions.pop(session_id, None)


class RobotSession:
    """Helper class for HTTP WebSocket handling, compatible with handler's session protocol"""
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.device_id = session_id
        self.turn_id = 0
        self.pending_task = None
        self.last_activity = time.time()
        self.is_listening = False
        self.current_text = ""
        self.hermes = None
        self.opus_decoder = None
        self.audio_buffer: list[bytes] = []
        self.eos_task = None
        self.eos_timeout = 1.5
        self.person_profiles: dict[str, dict] = {}
        self.current_person: Optional[str] = None

    def cancel_current_turn(self):
        self.turn_id += 1
        if self.pending_task and not self.pending_task.done():
            self.pending_task.cancel()

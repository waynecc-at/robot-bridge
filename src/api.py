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
from .metrics import metrics
from .tts_service import tts_service
from .vision_service import vision_service


# Pydantic models
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
    from .websocket_handler import ws_handler, RobotSession

    logger.info("Robot Bridge API starting...")

    # Shared thread pool for CPU-bound inference (ASR + TTS)
    executor = ThreadPoolExecutor(
        max_workers=3,
        thread_name_prefix="inference",
    )
    asr_service.set_executor(executor)
    tts_service.set_executor(executor)

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
    return {
        "status": "healthy",
        "service": "robot-bridge",
        "version": "0.1.0",
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Robot Bridge",
        "version": "0.1.0",
        "endpoints": {
            "tts": "/api/tts",
            "tts_stream": "/api/tts/stream",
            "websocket": "/ws/robot",
            "voices": "/api/voices",
        }
    }


@app.api_route("/ota", methods=["GET", "POST"])
async def ota_config():
    """OTA configuration endpoint for ESP32 firmware.

    ESP32 fetches this on boot to get the WebSocket URL and server time.
    Replaces the temporary /tmp/ota_server.py script on port 8884.
    """
    import datetime
    return {
        "websocket": {
            "url": "ws://192.168.50.32:8081/ws/robot",
            "version": 1,
        },
        "server_time": {
            "timestamp": int(datetime.datetime.now().timestamp() * 1000),
            "timezone_offset": 480,  # UTC+8 minutes
        },
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


# ============= Internal API (MCP fallback for separate process) =============
from pydantic import BaseModel as PydanticBase

class InternalSpeakReq(PydanticBase):
    text: str
    emotion: str = "neutral"

class InternalLedReq(PydanticBase):
    red: int = 0
    green: int = 0
    blue: int = 0
    rainbow: bool = False

class InternalLookReq(PydanticBase):
    yaw: int = 0
    pitch: int = 30
    speed: int = 200

class InternalEmoteReq(PydanticBase):
    emotion: str = "neutral"

@app.post("/internal/speak")
async def internal_speak(req: InternalSpeakReq):
    from .websocket_handler import ws_handler
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    asyncio.create_task(ws_handler._tts_and_send(session, req.text))
    return {"status": "ok"}

@app.post("/internal/led")
async def internal_led(req: InternalLedReq):
    from .websocket_handler import ws_handler
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    if req.rainbow:
        ws_handler._start_rainbow(session)
    else:
        ws_handler._stop_rainbow(session)
        ws_handler._set_led(session, req.red, req.green, req.blue)
    return {"status": "ok"}

@app.post("/internal/look")
async def internal_look(req: InternalLookReq):
    from .websocket_handler import ws_handler
    from .face_tracker import face_tracker
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    face_tracker.pause(4.0)
    asyncio.create_task(ws_handler._send_mcp_async(
        session, "self.robot.set_head_angles",
        {"yaw": req.yaw, "pitch": req.pitch, "speed": req.speed},
    ))
    return {"status": "ok"}

@app.post("/internal/emote")
async def internal_emote(req: InternalEmoteReq):
    from .websocket_handler import ws_handler
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    await ws_handler._send_json(session.websocket, {
        "type": "llm", "emotion": req.emotion,
        "session_id": session.session_id,
    })
    return {"status": "ok"}

@app.post("/internal/listen")
async def internal_listen():
    from .websocket_handler import ws_handler
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    session.is_listening = True
    session.audio_buffer.clear()
    session.last_asr_text = ""
    await ws_handler._send_json(session.websocket, {
        "type": "listen", "state": "start",
        "session_id": session.session_id,
    })
    return {"status": "ok"}

@app.post("/internal/idle")
async def internal_idle():
    from .websocket_handler import ws_handler
    session = _get_active_ws_session(ws_handler)
    if not session: return {"error": "No ESP32 connected"}
    session.cancel_current_turn()
    ws_handler._stop_rainbow(session)
    ws_handler._set_led(session, 0, 0, 0)
    return {"status": "ok"}

def _get_active_ws_session(ws_handler):
    if not ws_handler.sessions: return None
    sessions = sorted(ws_handler.sessions.values(),
                      key=lambda s: getattr(s, 'last_activity', 0), reverse=True)
    return sessions[0] if sessions else None


# ============= MCP Endpoint =============

from .mcp_router import router as mcp_router
app.include_router(mcp_router)


# ============= WebSocket Endpoint =============

WS_IDLE_TIMEOUT = 600  # seconds without activity

@app.websocket("/ws/robot")
async def websocket_endpoint(websocket: WebSocket):
    """XiaoZhi-protocol WebSocket endpoint for ESP32 StackChan devices."""
    from .websocket_handler import ws_handler, RobotSession

    await websocket.accept()
    logger.info("[WS] WebSocket client connected")

    # Register MCP server with WebSocket handler (for forwarding tool calls to ESP32)
    from .mcp_server import tools as mcp_tools
    mcp_tools.set_ws_handler(ws_handler)


    # ASR → Hermes webhook (:8644). Agent Driver is deleted.

    # Use Device-Id header (ESP32 MAC) as stable identity for cross-session pending responses
    device_id = websocket.headers.get("device-id", "") or str(uuid.uuid4())[:12]
    session_id = str(uuid.uuid4())[:8]
    session = RobotSession(websocket=websocket, session_id=session_id, device_id=device_id)
    logger.info(f"[WS] Device {device_id} session {session_id}")
    ws_handler.sessions[session_id] = session

    try:
        while True:
            idle = time.time() - session.last_activity
            if idle > WS_IDLE_TIMEOUT:
                logger.info(f"[WS] Session {session_id} idle timeout ({idle:.0f}s)")
                break

            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=5)
            except asyncio.TimeoutError:
                continue

            session.last_activity = time.time()

            if "text" in msg:
                await ws_handler._handle_message(session, msg["text"])
            elif "bytes" in msg:
                await ws_handler._handle_message(session, msg["bytes"])
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
        # Cleanup on disconnect
        try:
            await ws_handler.process_on_disconnect(session)
        except Exception:
            pass
        ws_handler.sessions.pop(session_id, None)



    # RobotSession is imported from websocket_handler (single source of truth)

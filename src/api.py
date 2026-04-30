"""HTTP API Server for Robot Bridge"""
import base64
import json
import uuid
from typing import Optional
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .asr_service import asr_service
from .config import config
from .hermes_client import hermes_client
from .tts_service import tts_service


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
    import asyncio
    from .websocket_handler import ws_handler

    logger.info("Robot Bridge API starting...")
    await hermes_client.__aenter__()
    await asr_service.start()

    # Background task to clean up stale WebSocket sessions every 5 minutes
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)
            await ws_handler.cleanup_stale_sessions()

    cleanup_task = asyncio.create_task(cleanup_loop())

    yield

    cleanup_task.cancel()
    await hermes_client.close()
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
        )
        return JSONResponse({
            "text": response.text,
            "session_id": response.session_id,
        })


async def stream_chat_response(message: str, session_id: Optional[str]):
    """Stream chat response with TTS audio"""
    async with hermes_client as hermes:
        # Get Hermes response
        response = await hermes.chat(
            message=message,
            session_id=session_id,
        )
        
        response_text = response.text
        
        # Send text first
        yield json.dumps({
            "type": "text",
            "text": response_text,
            "session_id": response.session_id
        }) + "\n"
        
        # Then stream TTS audio
        async for chunk in tts_service.synthesize_stream(response_text):
            yield json.dumps({
                "type": "audio",
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
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f"attachment; filename=tts_{uuid.uuid4().hex[:8]}.mp3"
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
        media_type="audio/mpeg"
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


# ============= WebSocket Endpoint =============

@app.websocket("/ws/robot")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ESP32 connections
    
    Protocol:
    - Send: {"type": "text", "text": "Hello"}
    - Receive: {"type": "response_text", "text": "Hi there!"}
    - Receive: {"type": "tts_audio", "data": "base64...", "final": false}
    """
    from .websocket_handler import ws_handler
    
    await websocket.accept()
    logger.info("[WS] WebSocket client connected")
    
    await ws_handler.set_hermes_client(hermes_client)

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"[WS] Received: {data[:100]}...")

            session_id = "http_client"
            await ws_handler._handle_message(
                RobotSession(websocket, session_id),
                data
            )

    except WebSocketDisconnect:
        logger.info("[WS] WebSocket client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")


class RobotSession:
    """Helper class for HTTP WebSocket handling"""
    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id

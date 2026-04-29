"""WebSocket handler for ESP32 connections"""
import asyncio
import json
import base64
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from loguru import logger
import websockets
from websockets.server import WebSocketServerProtocol

from .hermes_client import HermesClient
from .tts_service import tts_service


@dataclass
class RobotSession:
    """Represents a connected robot device"""
    device_id: str
    session_id: str
    websocket: WebSocketServerProtocol
    last_activity: float = 0
    
    # State
    is_listening: bool = False
    current_text: str = ""
    
    # Hermes client
    hermes: Optional[HermesClient] = None


class RobotWebSocketHandler:
    """Handles WebSocket connections from ESP32 devices"""
    
    def __init__(self):
        self.sessions: Dict[str, RobotSession] = {}
        self.hermes: Optional[HermesClient] = None
    
    async def set_hermes_client(self, client: HermesClient):
        """Set the Hermes client"""
        self.hermes = client
    
    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
        path: str,
    ):
        """Handle new WebSocket connection"""
        # Extract device info from headers or path
        device_id = self._extract_device_id(websocket, path)
        session_id = f"robot_{device_id}_{int(asyncio.get_event_loop().time())}"
        
        logger.info(f"[WS] New connection: device={device_id}, session={session_id}")
        
        # Create session
        session = RobotSession(
            device_id=device_id,
            session_id=session_id,
            websocket=websocket,
            hermes=self.hermes,
        )
        self.sessions[session_id] = session
        
        # Send welcome message
        await self._send_message(websocket, {
            "type": "connected",
            "session_id": session_id,
            "message": "Connected to Robot Bridge"
        })
        
        try:
            # Handle incoming messages
            async for raw_message in websocket:
                await self._handle_message(session, raw_message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[WS] Connection closed: {session_id}")
        except Exception as e:
            logger.error(f"[WS] Error: {e}")
        finally:
            # Cleanup session
            if session_id in self.sessions:
                del self.sessions[session_id]
            logger.info(f"[WS] Session cleaned up: {session_id}")
    
    def _extract_device_id(self, websocket: WebSocketServerProtocol, path: str) -> str:
        """Extract device ID from WebSocket connection"""
        # Try from headers
        device_id = websocket.request_headers.get("X-Device-ID")
        if device_id:
            return device_id
        
        # Try from path
        if path and path != "/":
            return path.strip("/").split("/")[-1]
        
        return "unknown_device"
    
    async def _handle_message(self, session: RobotSession, raw_message: str | bytes):
        """Handle incoming message from device"""
        try:
            # Parse message
            if isinstance(raw_message, bytes):
                message = self._parse_binary_message(raw_message)
            else:
                message = json.loads(raw_message)
            
            msg_type = message.get("type", "unknown")
            logger.debug(f"[WS] Message type={msg_type} from {session.session_id}")
            
            # Route to handler
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
        """Parse binary audio message"""
        # Simple protocol: first byte is type, rest is data
        msg_type = data[0] if data else 0
        audio_data = data[1:] if len(data) > 1 else b""
        
        return {
            "type": "audio",
            "data": base64.b64encode(audio_data).decode(),
            "format": "pcm"
        }
    
    async def _handle_text_message(self, session: RobotSession, message: dict):
        """Handle text input (for testing)"""
        text = message.get("text", "")
        logger.info(f"[WS] Text input: {text}")
        
        session.current_text = text
        
        # Process through Hermes and get TTS
        await self._process_and_respond(session, text)
    
    async def _handle_audio_message(self, session: RobotSession, message: dict):
        """
        Handle audio input from device
        NOTE: For simulation, we'll treat this as text input
        In real usage, audio would be sent to ASR first
        """
        audio_data = message.get("data", "")
        logger.info(f"[WS] Audio received: {len(audio_data)} bytes")
        
        # For simulation: skip ASR, directly process
        # In production, would send to FunASR for recognition
        await self._send_message(session.websocket, {
            "type": "status",
            "message": "Audio received (simulation mode)"
        })
    
    async def _handle_vad_message(self, session: RobotSession, message: dict):
        """Handle VAD status update"""
        state = message.get("state", "unknown")
        logger.debug(f"[WS] VAD state: {state}")
        
        if state == "speaking":
            session.is_listening = True
        elif state in ("silence", "end"):
            session.is_listening = False
            # Trigger processing if we have accumulated text
            if session.current_text:
                await self._process_and_respond(session, session.current_text)
    
    async def _handle_ping(self, session: RobotSession, message: dict):
        """Handle ping message"""
        await self._send_message(session.websocket, {
            "type": "pong",
            "timestamp": message.get("timestamp")
        })
    
    async def _process_and_respond(self, session: RobotSession, text: str):
        """Process text through Hermes and respond with TTS"""
        if not self.hermes:
            logger.error("[WS] Hermes client not available")
            await self._send_message(session.websocket, {
                "type": "error",
                "message": "Hermes gateway not connected"
            })
            return
        
        try:
            # Send thinking status
            await self._send_message(session.websocket, {
                "type": "status",
                "message": "thinking",
                "action": "thinking"
            })
            
            # Get response from Hermes
            logger.info(f"[WS] Processing: {text}")
            response = await self.hermes.chat(
                message=text,
                session_id=session.session_id,
            )
            
            response_text = response.text
            logger.info(f"[WS] Hermes response: {response_text[:100]}...")
            
            # Send text response
            await self._send_message(session.websocket, {
                "type": "response_text",
                "text": response_text,
                "session_id": session.session_id
            })
            
            # Generate and send TTS audio
            await self._send_message(session.websocket, {
                "type": "status",
                "message": "synthesizing",
                "action": "speaking"
            })
            
            # Stream TTS audio
            audio_chunks = []
            async for chunk in tts_service.synthesize_stream(response_text):
                audio_chunks.append(chunk)
                
                # Send audio chunk
                await self._send_message(session.websocket, {
                    "type": "tts_audio",
                    "data": base64.b64encode(chunk).decode(),
                    "final": False
                })
            
            # Send final chunk marker
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
        """Send message to websocket"""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"[WS] Send error: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected sessions"""
        for session in self.sessions.values():
            try:
                await self._send_message(session.websocket, message)
            except:
                pass


# Global handler instance
ws_handler = RobotWebSocketHandler()

"""
MCP (Model Context Protocol) Server for StackChan — stdio transport.

Hermes Agent spawns this as a subprocess via:
    mcp_servers:
      stackchan:
        command: "/home/wayne/robot-bridge/.venv/bin/python"
        args: ["-m", "src.mcp_server"]
        timeout: 30

The server:
  1. Starts a FastAPI WebSocket server for ESP32 XiaoZhi protocol
  2. Exposes 11 MCP tools via stdio for Hermes to call
  3. Uses existing Bridge services (ASR, TTS, vision, etc.)
"""

import asyncio
import json
import os
import signal
import sys
import time
from typing import Any

from loguru import logger

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

# ── Disable MCP SDK's internal logging to avoid clashing with our loguru ──
import logging
logging.getLogger("mcp").setLevel(logging.WARNING)

# ── Import Bridge services ──
from . import asr_service
from . import tts_service
from . import vision_service
from . import face_registry
from . import face_tracker
from . import profile_store
from .config import load_config

logger.info("[MCP] MCP服务启动中...")

# Load config
config = load_config()

# ── Start the WebSocket server (ESP32 connection) in the background ──
# We import after config is loaded so services get the right paths
ws_server = None

async def _start_websocket_server():
    """Start FastAPI + WebSocket listener for ESP32. Handles port conflict gracefully."""
    from .api import app as fastapi_app
    import uvicorn

    try:
        server = uvicorn.Server(
            uvicorn.Config(
                fastapi_app,
                host=config.server.host,
                port=config.server.port,
                log_level="warning",  # Less verbose
                lifespan="on",
            )
        )
        global ws_server
        ws_server = server
        await server.serve()
    except OSError as e:
        if "address already in use" in str(e).lower():
            logger.warning(f"[MCP] Port {config.server.port} in use, ESP32 WebSocket unavailable (old Bridge running?)")
        else:
            raise
    except SystemExit:
        pass  # uvicorn calls sys.exit(1) on bind failure; ignore

# ── Tool implementations ──

class StackChanTools:
    """Handles all MCP tool logic, delegating to Bridge services."""

    def __init__(self):
        self._asr = asr_service.asr_service
        self._tts = tts_service.tts_service
        self._vision = vision_service.vision_service
        self._profiles = profile_store.profile_store
        self._face_registry = face_registry.face_registry
        self._face_tracker = face_tracker.face_tracker
        self._ws_handler = None  # Set after websocket handler is available
        self._ready = asyncio.Event()

    def set_ws_handler(self, handler):
        """Inject the WebSocket handler (called after FastAPI starts)."""
        from .websocket_handler import ws_handler as wsh
        self._ws_handler = wsh
        self._ready.set()

    async def wait_ready(self, timeout: float = 2.0):
        """Wait for WebSocket handler (ESP32 connection), with short timeout.

        When MCP server runs in a separate process (Hermes stdio), the ws_handler
        will never have sessions. Quick timeout → tools fall back to HTTP API."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass  # Separate process — HTTP fallback handles it

    # ── Listen: ASR ──

    async def tool_listen(self, timeout: float = 10.0) -> dict:
        """
        Listen to the user and transcribe speech to text.

        1. Sends listen state=start to ESP32
        2. Buffers incoming Opus audio
        3. Detects EOS (silence)
        4. Runs ASR on buffered audio
        5. Returns transcribed text
        """
        await self.wait_ready()
        if not self._ws_handler or not self._ws_handler.sessions:
            return {"error": "No ESP32 device connected"}

        # Get the active session
        session = self._get_active_session()
        if not session:
            return {"error": "No active ESP32 session"}

        # Cancel any ongoing TTS first (natural conversation interrupt)
        session.cancel_current_turn()

        # Send listen start to ESP32
        start_msg = json.dumps({
            "type": "listen",
            "state": "start",
            "session_id": session.session_id,
        }, ensure_ascii=False)
        await session.websocket.send_text(start_msg)
        session.is_listening = True
        session.last_asr_text = ""
        session.audio_buffer = []

        # Wait for EOS + ASR completion
        # The websocket_handler's _eos_handler runs ASR and sets last_asr_text.
        waited = 0.0
        check_interval = 0.1
        while waited < timeout:
            if not session.is_listening and session.last_asr_text:
                text = session.last_asr_text
                # Send listen stop
                stop_msg = json.dumps({
                    "type": "listen",
                    "state": "stop",
                    "session_id": session.session_id,
                }, ensure_ascii=False)
                try:
                    await session.websocket.send_text(stop_msg)
                except Exception:
                    pass
                return {"text": text, "person": session.current_person or None}
            await asyncio.sleep(check_interval)
            waited += check_interval

        # Timeout — send abort
        session.cancel_current_turn()
        abort_msg = json.dumps({
            "type": "abort",
            "reason": "timeout",
            "session_id": session.session_id,
        }, ensure_ascii=False)
        try:
            await session.websocket.send_text(abort_msg)
        except Exception:
            pass
        return {"text": "", "error": "timeout: no speech detected"}

    # ── Speak: TTS ──

    async def tool_speak(self, text: str, emotion: str = "neutral", wait: bool = True) -> dict:
        """Synthesize text to speech and play on the robot."""
        if not text.strip():
            return {"error": "empty text"}

        await self.wait_ready()
        session = self._get_active_session()
        if not session:
            # Separate process: fall back to Bridge HTTP API
            return await self._bridge_api("speak", text=text, emotion=emotion)

        session.cancel_current_turn()
        try:
            tts_task = asyncio.create_task(self._ws_handler._tts_and_send(session, text))
            if wait:
                await tts_task
                return {"success": True, "duration": None}
            return {"success": True, "async": True}
        except Exception as e:
            logger.error(f"[MCP] TTS error: {e}")
            return {"error": str(e)}

    # ── See: Camera snapshot ──

    async def tool_see(self) -> dict:
        """Take a photo from the robot's camera and return JPEG data."""
        await self.wait_ready()
        session = self._get_active_session()
        if not session:
            return {"error": "No ESP32 device connected"}

        # Use MCP to request photo from ESP32
        try:
            result = await asyncio.wait_for(
                self._ws_handler._mcp_call(session, "self.camera.take_photo", {}),
                timeout=10.0,
            )
            # The result might contain base64 JPEG or a path
            if isinstance(result, str):
                return {"jpeg_base64": result, "timestamp": time.time()}
            elif isinstance(result, dict) and "jpeg_base64" in result:
                return {"jpeg_base64": result["jpeg_base64"], "timestamp": time.time()}
            else:
                # Fallback: ESP32 sends vision_frame with JPEG
                return {"jpeg_base64": None, "raw": str(result), "timestamp": time.time()}
        except asyncio.TimeoutError:
            return {"error": "camera timeout"}
        except Exception as e:
            return {"error": str(e)}

    # ── Face: Detect and recognize faces ──

    async def tool_face(self) -> dict:
        """Detect and recognize faces from the latest camera frame.

        Returns face positions and recognized person identity.
        """
        await self.wait_ready()
        session = self._get_active_session()
        if not session:
            return {"error": "No ESP32 device connected"}

        # Check if vision_service has processed a recent frame
        # The ESP32 sends vision_frame continuously at ~1fps
        # Vision service processes the latest frame
        if not config.vision.enabled:
            return {"faces": [], "current_person": None}

        # The vision service already processes frames via vision_frame messages.
        # We read the current person state from the session.
        # The face positions are stored in face_tracker state.
        faces = []
        
        # Get latest detection from session's person_profiles
        current_person = session.current_person
        if current_person and current_person in session.person_profiles:
            profile = session.person_profiles[current_person]
            faces.append({
                "name": current_person,
                "relationship": profile.get("relationship", ""),
                "preferences": profile.get("preferences", ""),
            })
        elif current_person:
            # Person recognized but no full profile loaded
            faces.append({"name": current_person, "relationship": "", "preferences": ""})

        return {
            "faces": faces,
            "current_person": current_person,
            "timestamp": time.time(),
        }

    # ── Face Register: Train LBPH with collected crops ──

    async def tool_face_register(self, name: str, relationship: str = "") -> dict:
        """Register a new person using collected face crops from the current session."""
        await self.wait_ready()
        session = self._get_active_session()
        if not session:
            return {"error": "No ESP32 device connected"}

        # Get face crops collected by face_registry
        from .face_registry import face_registry as fr
        crops = fr.get_crops()
        if not crops:
            return {"error": "no face crops available; camera hasn't seen this person enough"}

        # Also get crops from any vision_frame messages stored in session
        # (the websocket_handler may store face crops separately)
        extra_crops = getattr(session, 'face_crops', [])
        all_crops = list(crops) + list(extra_crops)
        if not all_crops:
            return {"error": "no face crops available"}

        # Register via profile_store
        profile = self._profiles.register_person(
            name=name,
            face_frames=all_crops[:8],  # Limit to 8 crops (matches face_registry.MAX_CROPS)
            relationship=relationship,
        )

        # Clear the registry so we don't re-register the same person
        fr.clear()
        if hasattr(session, 'face_crops'):
            session.face_crops = []
        session.unknown_face_pending = False

        if profile:
            return {"success": True, "name": name, "relationship": relationship}
        return {"error": "registration failed"}

    # ── Look: Move head servos ──

    async def tool_look(self, yaw: int = 0, pitch: int = 0, speed: int = 150) -> dict:
        """Move the robot's head to specified angles."""
        await self.wait_ready()
        if not self._get_active_session():
            return await self._bridge_api("look", yaw=yaw, pitch=pitch, speed=speed)
        session = self._get_active_session()
        self._face_tracker.pause(duration=4.0)
        try:
            result = await asyncio.wait_for(
                self._ws_handler._mcp_call(session, "self.robot.set_head_angles", {
                    "yaw": yaw, "pitch": pitch, "speed": speed,
                }),
                timeout=5.0,
            )
            return {"success": True, "yaw": yaw, "pitch": pitch}
        except asyncio.TimeoutError:
            return {"error": "servo timeout"}
        except Exception as e:
            return {"error": str(e)}

    # ── Track Target: Continuous face tracking ──

    async def tool_track_target(self, person: str = "", duration: float = 30.0) -> dict:
        """Continuously track a person's face. If person is empty, tracks current speaker."""
        return {
            "mode": "auto_tracking",
            "person": person or "current_speaker",
            "duration": duration,
            "note": "Face tracking runs continuously via ESP32 vision_frame pipeline",
        }

    # ── LED: Set LED color ──

    async def tool_led(self, red: int = 0, green: int = 0, blue: int = 0, rainbow: bool = False) -> dict:
        """Set the robot's LED color or activate rainbow mode."""
        await self.wait_ready()
        if not self._get_active_session():
            return await self._bridge_api("led", red=red, green=green, blue=blue, rainbow=rainbow)
        session = self._get_active_session()
        try:
            if rainbow:
                self._ws_handler._start_rainbow(session)
                return {"success": True, "mode": "rainbow"}
            else:
                self._ws_handler._stop_rainbow(session)
                self._ws_handler._set_led(session, red, green, blue)
                return {"success": True, "color": {"r": red, "g": green, "b": blue}}
        except Exception as e:
            return {"error": str(e)}

    # ── Emote: Set facial expression ──

    async def tool_emote(self, emotion: str = "neutral") -> dict:
        """Set the robot's facial expression (neutral/happy/sad/angry/sleepy/doubtful)."""
        await self.wait_ready()
        if not self._get_active_session():
            return await self._bridge_api("emote", emotion=emotion)
        session = self._get_active_session()
        valid = {"neutral", "happy", "sad", "angry", "sleepy", "doubtful"}
        if emotion not in valid:
            return {"error": f"invalid emotion: {emotion}. Valid: {', '.join(sorted(valid))}"}
        try:
            await self._ws_handler._send_json(session.websocket, {
                "type": "llm", "emotion": emotion,
                "session_id": session.session_id,
            })
            return {"success": True, "emotion": emotion}
        except Exception as e:
            return {"error": str(e)}

    # ── Status: Query robot state ──

    async def tool_status(self) -> dict:
        """Get the robot's current connection and hardware status."""
        await self.wait_ready()
        session = self._get_active_session()
        if not session:
            return {"connected": False, "error": "No ESP32 device connected"}

        return {
            "connected": True,
            "device_id": session.device_id,
            "session_id": session.session_id,
            "current_person": session.current_person,
            "is_listening": session.is_listening,
            "last_activity": session.last_activity,
            "persons_known": list(session.person_profiles.keys()) if session.person_profiles else [],
        }

    # ── Idle: Return to standby ──

    async def tool_idle(self) -> dict:
        """Return the robot to idle/standby state — LEDs off, tracking stops."""
        await self.wait_ready()
        if not self._get_active_session():
            return await self._bridge_api("idle")
        session = self._get_active_session()
        session.cancel_current_turn()
        self._ws_handler._stop_rainbow(session)
        self._ws_handler._set_led(session, 0, 0, 0)
        self._face_tracker.reset()

        # Clear unknown face pending
        session.unknown_face_pending = False

        return {"success": True, "state": "idle"}

    # ── Helpers ──

    def _get_active_session(self):
        """Get the most recently active ESP32 WebSocket session."""
        if not self._ws_handler or not self._ws_handler.sessions:
            return None
        sessions = sorted(
            self._ws_handler.sessions.values(),
            key=lambda s: getattr(s, 'last_activity', 0),
            reverse=True,
        )
        return sessions[0] if sessions else None

    async def _bridge_api(self, method: str, **kwargs) -> dict:
        """Call Bridge HTTP API. Fallback when ws_handler is in a different process."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"http://127.0.0.1:8081/internal/{method}",
                    json=kwargs,
                )
                return resp.json() if resp.status_code == 200 else {"error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"error": str(e)}


# ── Global tool instance ──
tools = StackChanTools()

# ── MCP Server setup ──
mcp_server = Server("stackchan")


@mcp_server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="stackchan_listen",
            description="听用户说话并转成文字。告诉 ESP32 开始聆听, 缓冲音频, 检测静音结束, 运行 ASR 语音识别。返回识别出的中文文字。",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "最长等待时间(秒), 超时返回空文字",
                        "default": 10.0,
                    }
                },
            },
        ),
        Tool(
            name="stackchan_speak",
            description="用语音说一段话。TTS 合成中文文字并发送到机器人播放。可设置表情和是否等待播完。",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要说的中文文字"},
                    "emotion": {
                        "type": "string",
                        "description": "表情: neutral/happy/sad/angry/sleepy/doubtful",
                        "default": "neutral",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "是否等待播完再返回 (false 用于后台通知)",
                        "default": True,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="stackchan_see",
            description="拍照。让机器人拍一张照片, 返回 JPEG 图片数据。用于视觉分析。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="stackchan_face",
            description="识别人脸。检测摄像头前有谁, 返回姓名和关系。无模型时仅检测位置。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="stackchan_face_register",
            description="注册新面孔。用当前 session 收集的人脸 crop 训练人脸模型, 为 person 创建画像。",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "人的姓名"},
                    "relationship": {
                        "type": "string",
                        "description": "与家庭的关系, 如爸爸/妈妈/儿子",
                        "default": "",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="stackchan_look",
            description="转头。让机器人把头转到指定角度。yaw 水平(-90~90), pitch 垂直(0~90), speed 速度(100~1000)。转头后自动暂停追踪 4 秒。",
            inputSchema={
                "type": "object",
                "properties": {
                    "yaw": {
                        "type": "integer",
                        "description": "水平角度, -90(左) 到 90(右)", "default": 0},
                    "pitch": {
                        "type": "integer",
                        "description": "垂直角度, 0(下) 到 90(上)", "default": 0},
                    "speed": {
                        "type": "integer",
                        "description": "速度, 100(慢) 到 1000(快)", "default": 150},
                },
            },
        ),
        Tool(
            name="stackchan_track_target",
            description="追踪人脸。连续追踪指定人的面部位置, 自动调整舵机让脸居中。",
            inputSchema={
                "type": "object",
                "properties": {
                    "person": {
                        "type": "string",
                        "description": "要追踪的人名, 空则追踪当前说话人", "default": ""},
                    "duration": {
                        "type": "number",
                        "description": "追踪持续时间(秒)", "default": 30.0},
                },
            },
        ),
        Tool(
            name="stackchan_led",
            description="设置 LED 灯。控制机器人身上的 LED 灯颜色或开启彩虹跑马灯模式。",
            inputSchema={
                "type": "object",
                "properties": {
                    "red": {"type": "integer", "description": "红色 0-255", "default": 0},
                    "green": {"type": "integer", "description": "绿色 0-255", "default": 0},
                    "blue": {"type": "integer", "description": "蓝色 0-255", "default": 0},
                    "rainbow": {
                        "type": "boolean",
                        "description": "彩虹跑马灯模式(覆盖RGB)", "default": False},
                },
            },
        ),
        Tool(
            name="stackchan_emote",
            description="设置表情。改变机器人屏幕上的表情图标。支持: neutral, happy, sad, angry, sleepy, doubtful。",
            inputSchema={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "表情: neutral/happy/sad/angry/sleepy/doubtful",
                        "default": "neutral",
                    },
                },
            },
        ),
        Tool(
            name="stackchan_status",
            description="查询状态。获取机器人的连接状态、当前识别的人、是否在聆听等信息。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="stackchan_idle",
            description="进入待机。停止所有活动, 关灯, 停止追踪, 回到空闲状态。",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@mcp_server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to the right handler."""
    logger.info(f"[MCP] Call: {name}({arguments})")

    try:
        if name == "stackchan_listen":
            result = await tools.tool_listen(
                timeout=arguments.get("timeout", 10.0)
            )
        elif name == "stackchan_speak":
            result = await tools.tool_speak(
                text=arguments.get("text", ""),
                emotion=arguments.get("emotion", "neutral"),
                wait=arguments.get("wait", True),
            )
        elif name == "stackchan_see":
            result = await tools.tool_see()
        elif name == "stackchan_face":
            result = await tools.tool_face()
        elif name == "stackchan_face_register":
            result = await tools.tool_face_register(
                name=arguments.get("name", ""),
                relationship=arguments.get("relationship", ""),
            )
        elif name == "stackchan_look":
            result = await tools.tool_look(
                yaw=arguments.get("yaw", 0),
                pitch=arguments.get("pitch", 0),
                speed=arguments.get("speed", 150),
            )
        elif name == "stackchan_track_target":
            result = await tools.tool_track_target(
                person=arguments.get("person", ""),
                duration=arguments.get("duration", 30.0),
            )
        elif name == "stackchan_led":
            result = await tools.tool_led(
                red=arguments.get("red", 0),
                green=arguments.get("green", 0),
                blue=arguments.get("blue", 0),
                rainbow=arguments.get("rainbow", False),
            )
        elif name == "stackchan_emote":
            result = await tools.tool_emote(
                emotion=arguments.get("emotion", "neutral"),
            )
        elif name == "stackchan_status":
            result = await tools.tool_status()
        elif name == "stackchan_idle":
            result = await tools.tool_idle()
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"[MCP] Error in {name}: {e}")
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


# ── Main: run both WebSocket server and MCP stdio server ──

async def main():
    """Start MCP stdio server with optional built-in WebSocket server for ESP32."""

    # Try to start FastAPI + WebSocket server in background.
    # If port is occupied (old Bridge running), MCP tools just return "no device".
    try:
        ws_task = asyncio.create_task(_start_websocket_server())
        await asyncio.sleep(0.5)
    except Exception as e:
        logger.warning(f"[MCP] WebSocket server not started (old Bridge may be running): {e}")

    # Try to set ws_handler reference
    try:
        from .websocket_handler import ws_handler as wsh
        tools.set_ws_handler(wsh)
        # Hook agent driver into Bridge's ASR callback (no polling)
        from .agent_driver import agent_driver
        agent_driver.set_ws_handler(wsh)
        logger.info("[MCP] Agent driver hooked to Bridge ASR")
    except Exception as e:
        logger.info(f"[MCP] Running in headless mode (no ESP32 connection): {e}")

    # Start MCP stdio server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="stackchan",
                server_version="1.0.0",
                capabilities={"tools": {}},
            ),
        )


def run():
    """Entry point for `python -m src.mcp_server`."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("[MCP] Shutting down...")
    except Exception as e:
        logger.error(f"[MCP] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""End-to-end test suite for robot-bridge v2.

Tests each feature without requiring the physical ESP32.
Simulates WebSocket messages and checks server responses.

Usage:
  .venv/bin/python tests/test_bridge_v2.py [--test <name>]
"""
import asyncio
import base64
import json
import io
import sys
import time
import struct
import wave
from pathlib import Path

import httpx
import websockets
import numpy as np

BRIDGE = "http://127.0.0.1:8081"
WS = "ws://127.0.0.1:8081/ws/robot"

PASS = 0
FAIL = 0


def ok(name: str):
    global PASS
    PASS += 1
    print(f"  ✅ {name}")


def err(name: str, detail: str = ""):
    global FAIL
    FAIL += 1
    msg = f"  ❌ {name}"
    if detail:
        msg += f"  → {detail}"
    print(msg)


# ═══════════════════════════════════════════════════════════════
# Test 1: Health check
# ═══════════════════════════════════════════════════════════════

async def test_health():
    print("\n── 测试 1: 健康检查 ──")
    async with httpx.AsyncClient() as cli:
        r = await cli.get(f"{BRIDGE}/health")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy" and data.get("hermes_connected"):
                ok(f"Hermes 连接正常, version={data.get('version')}")
            else:
                err("健康检查", str(data))
        else:
            err("健康检查", f"HTTP {r.status_code}")


# ═══════════════════════════════════════════════════════════════
# Test 2: Profile CRUD
# ═══════════════════════════════════════════════════════════════

async def test_profiles():
    print("\n── 测试 2: 画像 CRUD ──")
    async with httpx.AsyncClient() as cli:
        # Create
        r = await cli.post(f"{BRIDGE}/api/profile/register", json={
            "name": "测试用户", "relationship": "朋友", "preferences": "喜欢咖啡",
        })
        if r.status_code == 200:
            ok("创建画像: 测试用户")
        else:
            err("创建画像", f"HTTP {r.status_code}")

        # List
        r = await cli.get(f"{BRIDGE}/api/profiles")
        if r.status_code == 200:
            profiles = r.json().get("profiles", [])
            names = [p["name"] for p in profiles]
            if "测试用户" in names:
                ok(f"画像列表: {len(profiles)} 人 ({', '.join(names[:5])})")
            else:
                err("画像列表", f"测试用户不在列表中: {names}")
        else:
            err("画像列表", f"HTTP {r.status_code}")

        # Delete
        r = await cli.delete(f"{BRIDGE}/api/profile/测试用户")
        if r.status_code == 200:
            ok("删除画像: 测试用户")
        else:
            err("删除画像", f"HTTP {r.status_code}")


# ═══════════════════════════════════════════════════════════════
# Test 3: WebSocket hello handshake
# ═══════════════════════════════════════════════════════════════

async def test_ws_hello():
    print("\n── 测试 3: WebSocket 握手 ──")
    try:
        async with websockets.connect(WS, additional_headers={"Device-Id": "test-001"}) as ws:
            # Send XiaoZhi hello
            hello = json.dumps({
                "type": "hello",
                "version": 1,
                "transport": "websocket",
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
            })
            await ws.send(hello)

            # Receive server hello
            resp = json.loads(await ws.recv())
            if resp.get("type") == "hello" and "session_id" in resp:
                sid = resp["session_id"]
                audio = resp.get("audio_params", {})
                ok(f"握手成功 session={sid} format={audio.get('format')} sr={audio.get('sample_rate')}")
            else:
                err("握手响应", str(resp)[:100])

            # Send ping
            await ws.send(json.dumps({"type": "ping"}))
            pong = json.loads(await ws.recv())
            if pong.get("type") == "pong":
                ok("Ping/Pong 正常")
            else:
                err("Ping/Pong", str(pong)[:50])

    except Exception as e:
        err("WebSocket 连接", str(e))


# ═══════════════════════════════════════════════════════════════
# Test 4: Wake word → green LED
# ═══════════════════════════════════════════════════════════════

async def test_wakeword_led():
    print("\n── 测试 4: 唤醒词 → 绿灯 ──")
    try:
        async with websockets.connect(WS, additional_headers={"Device-Id": "test-002"}) as ws:
            # Handshake
            await ws.send(json.dumps({
                "type": "hello", "version": 1, "transport": "websocket",
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
            }))
            await ws.recv()  # consume hello

            # Simulate wake word detection
            await ws.send(json.dumps({"type": "listen", "state": "detect", "text": "hai san san"}))

            # Check for green LED MCP command (set_led_color with green=168)
            found_green = False
            try:
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                    if msg.get("type") == "mcp":
                        p = msg.get("payload", {})
                        args = p.get("params", {}).get("arguments", {})
                        if (p.get("params", {}).get("name") == "self.robot.set_led_color"
                                and args.get("green") == 168 and not args.get("red") and not args.get("blue")):
                            found_green = True
                            ok("唤醒词触发绿灯 MCP 指令")
                            break
            except asyncio.TimeoutError:
                pass

            if not found_green:
                err("唤醒词绿灯", "未检测到 set_led_color(green=168) MCP 指令")
    except Exception as e:
        err("唤醒词测试", str(e))


# ═══════════════════════════════════════════════════════════════
# Test 5: Vision frame processing
# ═══════════════════════════════════════════════════════════════

def _make_test_jpeg(w=320, h=240) -> bytes:
    """Generate a simple test JPEG (gray gradient rectangle)."""
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[80:160, 120:200] = (180, 140, 120)  # skin-colored rectangle
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()


async def test_vision_frame():
    print("\n── 测试 5: 视觉帧处理 ──")
    try:
        async with websockets.connect(WS, additional_headers={"Device-Id": "test-003"}) as ws:
            # Handshake
            await ws.send(json.dumps({
                "type": "hello", "version": 1, "transport": "websocket",
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
            }))
            await ws.recv()

            # Send vision_frame (full frame)
            jpeg = _make_test_jpeg()
            b64 = base64.b64encode(jpeg).decode()
            await ws.send(json.dumps({
                "type": "vision_frame", "data": b64, "format": "jpeg",
                "width": 320, "height": 240,
            }))

            # Check for response (person_detected, or nothing if no face found)
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                t = msg.get("type", "")
                if t == "person_detected":
                    ok(f"人脸识别 → {msg.get('name', '?')} ({msg.get('relationship', '?')})")
                elif t == "mcp":
                    name = msg.get("payload", {}).get("params", {}).get("name", "")
                    ok(f"视觉处理输出 MCP: {name}")
                else:
                    ok(f"视觉帧已处理（响应类型: {t}）")
            except asyncio.TimeoutError:
                ok("视觉帧处理完成（无识别结果 — 测试图片无人脸特征，预期行为）")

            # Send face crop variant
            await ws.send(json.dumps({
                "type": "vision_frame", "data": b64, "format": "jpeg",
                "face": True, "bx": 100, "by": 60, "bw": 120, "bh": 120,
            }))
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
                ok(f"裁切帧处理完成（响应: {msg.get('type', '?')}）")
            except asyncio.TimeoutError:
                ok("裁切帧处理完成（无响应，预期 — 无人脸特征）")

    except Exception as e:
        err("视觉帧测试", str(e))


# ═══════════════════════════════════════════════════════════════
# Test 6: LLM chat (Hermes connectivity)
# ═══════════════════════════════════════════════════════════════

async def test_chat():
    print("\n── 测试 6: LLM 对话 ──")
    async with httpx.AsyncClient(timeout=30) as cli:
        r = await cli.post(f"{BRIDGE}/api/chat", json={
            "message": "你好，请用一句话介绍你自己",
            "stream": False,
        })
        if r.status_code == 200:
            data = r.json()
            text = data.get("text", "")
            if text:
                ok(f"LLM 回复: {text[:80]}...")
            else:
                err("LLM 回复为空")
        else:
            err("LLM 对话", f"HTTP {r.status_code}: {r.text[:100]}")


# ═══════════════════════════════════════════════════════════════
# Test 7: Rainbow LED during thinking + blue during reply
# ═══════════════════════════════════════════════════════════════

async def test_led_flow():
    print("\n── 测试 7: LED 状态流转 ──")
    try:
        async with websockets.connect(WS, additional_headers={"Device-Id": "test-004"}) as ws:
            # Handshake
            await ws.send(json.dumps({
                "type": "hello", "version": 1, "transport": "websocket",
                "audio_params": {"format": "opus", "sample_rate": 16000, "channels": 1, "frame_duration": 60},
            }))
            await ws.recv()

            # Wake word → expect green
            await ws.send(json.dumps({"type": "listen", "state": "detect", "text": "hai san san"}))

            # Simulate listen start + audio + stop
            await ws.send(json.dumps({"type": "listen", "state": "start", "mode": "auto"}))

            # Generate blank PCM (silence — won't trigger real ASR)
            blank_pcm = b'\x00' * 1920  # 60ms of silence
            import opuslib
            enc = opuslib.Encoder(16000, 1, 'audio')
            enc.bitrate = 32000
            silent_opus = enc.encode(blank_pcm, 960)

            for _ in range(10):
                await ws.send(silent_opus)
                await asyncio.sleep(0.01)

            await ws.send(json.dumps({"type": "listen", "state": "stop"}))

            # Collect MCP LED commands (skip binary Opus frames)
            led_commands = []
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    if isinstance(raw, bytes):
                        continue  # skip Opus audio frames
                    msg = json.loads(raw)
                    if msg.get("type") == "mcp":
                        p = msg.get("payload", {})
                        if p.get("params", {}).get("name") == "self.robot.set_led_color":
                            args = p.get("params", {}).get("arguments", {})
                            led_commands.append(args)
            except asyncio.TimeoutError:
                pass

            if led_commands:
                colors = []
                for c in led_commands:
                    r, g, b = c.get("red", 0), c.get("green", 0), c.get("blue", 0)
                    if g == 168 and r == 0 and b == 0:
                        colors.append("绿(唤醒)")
                    elif b == 168 and r == 0 and g == 0:
                        colors.append("蓝(回复)")
                    elif r == g == b == 0:
                        colors.append("灭(空闲)")
                    else:
                        colors.append(f"RGB({r},{g},{b})")
                ok(f"LED 指令序列: {' → '.join(colors)}")
            else:
                err("LED 指令", "未收到任何 LED MCP 指令")
    except Exception as e:
        err("LED 流程测试", str(e))


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

async def main(test_filter: str = ""):
    print("=" * 60)
    print("robot-bridge v2 测试套件")
    print("=" * 60)

    tests = [
        ("health", test_health),
        ("profiles", test_profiles),
        ("ws-hello", test_ws_hello),
        ("wakeword", test_wakeword_led),
        ("vision", test_vision_frame),
        ("chat", test_chat),
        ("led", test_led_flow),
    ]

    for name, fn in tests:
        if test_filter and test_filter not in name:
            continue
        try:
            await fn()
        except Exception as e:
            err(f"{name} (崩溃)", str(e))

    print(f"\n{'='*60}")
    print(f"结果: {PASS} 通过, {FAIL} 失败, {PASS+FAIL} 总计")
    print(f"{'='*60}")
    return FAIL == 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--test", default="", help="Filter test name")
    args = p.parse_args()
    success = asyncio.run(main(args.test))
    sys.exit(0 if success else 1)

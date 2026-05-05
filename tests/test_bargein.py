"""Three rounds of barge-in tests covering VAD audio buffering,
turn cancellation, and rapid barge-in sequences.

Run: python -m tests.test_bargein
"""
import asyncio
import base64
import json
import time
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.websocket_handler import RobotWebSocketHandler, RobotSession
from src.asr_service import asr_service


class MockWebSocket:
    """Capture outgoing messages for verification."""
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, msg: str):
        self.sent.append(json.loads(msg))

    def get_messages(self, msg_type: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == msg_type]

    def clear(self):
        self.sent.clear()


async def verify(cond: bool, label: str):
    if not cond:
        print(f"  ✗ FAIL: {label}")
        return False
    print(f"  ✓ {label}")
    return True


# ── Round 1: VAD audio buffering ─────────────────────────────

async def test_audio_buffering():
    """Verify audio chunks are buffered during VAD and NOT individually ASR'd."""
    t0 = time.perf_counter()
    print("\n" + "=" * 65)
    print("  Round 1: VAD Audio Buffering & Batch Transcription")
    print("=" * 65)

    ws = MockWebSocket()
    session = RobotSession(device_id="test", session_id="r1", websocket=ws)
    handler = RobotWebSocketHandler()

    ok = True

    # ── 1a. VAD start: buffer cleared, is_listening set ──
    session.audio_buffer.append(b"stale_data")
    await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
    ok &= await verify(session.is_listening, "is_listening=True after VAD start")
    ok &= await verify(len(session.audio_buffer) == 0, "audio_buffer cleared on VAD start")
    ok &= await verify(
        ws.get_messages("interrupt") != [],
        "interrupt sent on VAD start",
    )

    # ── 1b. Audio during VAD → buffered, no status/ASR triggered ──
    chunk1 = b"\x00\x01" * 8000
    chunk2 = b"\x00\x02" * 8000
    b64_1 = base64.b64encode(chunk1).decode()
    b64_2 = base64.b64encode(chunk2).decode()

    await handler._handle_audio_message(session, {"type": "audio", "data": b64_1})
    await handler._handle_audio_message(session, {"type": "audio", "data": b64_2})

    ok &= await verify(len(session.audio_buffer) >= 1, "audio chunks accumulated in buffer")
    ok &= await verify(
        len(ws.get_messages("status")) == 0,
        "no status messages sent for individual audio chunks during VAD",
    )

    # ── 1c. Buffer contains both chunks in order ──
    expected_size = len(chunk1) + len(chunk2)
    total_buffered = sum(len(c) for c in session.audio_buffer)
    ok &= await verify(
        total_buffered == expected_size,
        f"buffer holds all chunks ({total_buffered} == {expected_size})",
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Round 1: {'ALL PASS' if ok else 'SOME FAILED'}  [{elapsed*1000:.0f}ms]")
    return ok, elapsed


# ── Round 2: Turn cancellation during LLM/TTS ────────────────

async def test_turn_cancellation():
    """Verify VAD start cancels an in-flight LLM/TTS turn."""
    t0 = time.perf_counter()
    print("\n" + "=" * 65)
    print("  Round 2: LLM/TTS Barge-in — Turn Cancellation")
    print("=" * 65)

    ws = MockWebSocket()
    session = RobotSession(device_id="test", session_id="r2", websocket=ws)
    handler = RobotWebSocketHandler()

    # Mock Hermes so _process_turn doesn't fast-fail
    async def fake_stream(*args, **kwargs):
        await asyncio.sleep(0.15)
        yield "你好"
        await asyncio.sleep(0.05)
        yield "，"
        yield "我是"
        await asyncio.sleep(0.05)
        yield "机器人。"
    mock_hermes = AsyncMock()
    mock_hermes.chat_stream = fake_stream
    handler.hermes = mock_hermes

    # Mock TTS (tiny fast fake audio)
    import src.tts_service as tts_mod
    original_tts = tts_mod.tts_service.synthesize_stream

    async def fake_tts_stream(*args, **kwargs):
        yield b"\x00\x00\x00\x00" * 100
    tts_mod.tts_service.synthesize_stream = fake_tts_stream

    ok = True

    try:
        # ── 2a. Start a turn ──
        turn_id_before = session.turn_id
        handler._start_turn(session, "测试消息")
        await asyncio.sleep(0.1)

        ok &= await verify(
            session.turn_id > turn_id_before,
            "turn_id incremented after _start_turn",
        )
        ok &= await verify(
            session.pending_task is not None,
            "pending_task created",
        )

        was_running = session.pending_task and not session.pending_task.done()
        ok &= await verify(was_running, "pending_task running (LLM streaming)")

        # ── 2b. VAD start cancels the turn ──
        await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
        await asyncio.sleep(0.05)

        ok &= await verify(
            session.pending_task is None or session.pending_task.done(),
            "pending_task cancelled after VAD start",
        )
        ok &= await verify(
            session.is_listening,
            "is_listening=True after VAD start",
        )
        ok &= await verify(
            session.turn_id == 2,
            "turn_id correctly tracking starts and cancels",
        )

        # ── 2c. Audio chunk buffered after cancelled turn ──
        await handler._handle_audio_message(session, {"type": "audio", "data": "AQID"})
        ok &= await verify(
            len(session.audio_buffer) > 0,
            "audio buffered after cancelled turn (no data loss)",
        )

    finally:
        tts_mod.tts_service.synthesize_stream = original_tts

    elapsed = time.perf_counter() - t0
    print(f"\n  Round 2: {'ALL PASS' if ok else 'SOME FAILED'}  [{elapsed*1000:.0f}ms]")
    return ok, elapsed


# ── Round 3: Rapid barge-in / turn-id safety ─────────────────

async def test_rapid_bargein():
    """Verify rapid VAD sequences don't leak tasks or corrupt turn-id."""
    t0 = time.perf_counter()
    print("\n" + "=" * 65)
    print("  Round 3: Rapid Barge-in Sequences & Turn-id Safety")
    print("=" * 65)

    ws = MockWebSocket()
    session = RobotSession(device_id="test", session_id="r3", websocket=ws)
    handler = RobotWebSocketHandler()
    handler.hermes = None

    # Mock ASR to avoid loading real model
    async def fake_transcribe(audio: bytes) -> str:
        return "测试语音转文字结果"

    mock_asr = patch.object(asr_service, "transcribe", fake_transcribe)
    mock_asr.start()

    ok = True

    # ── 3a. 5 rapid barges ──
    for i in range(5):
        handler._start_turn(session, f"轮次{i}")
        await asyncio.sleep(0.01)
        await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
        await asyncio.sleep(0.01)

    ok &= await verify(
        session.turn_id == 10,
        f"turn_id=10 after 5 rapid cycles (got {session.turn_id})",
    )
    ok &= await verify(
        session.pending_task is None or session.pending_task.done(),
        "no orphan pending_task after rapid barge-in",
    )

    # ── 3b. No leaked asyncio tasks ──
    turn_tasks = [t for t in asyncio.all_tasks() if "turn-" in (t.get_name() or "")]
    ok &= await verify(
        len(turn_tasks) <= 1,
        f"no leaked turn tasks ({len(turn_tasks)} found)",
    )

    # ── 3c. VAD end with buffer ──
    await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
    session.audio_buffer.append(b"\x00\x01" * 8000)
    session.audio_buffer.append(b"\x00\x02" * 8000)

    await handler._handle_vad_message(session, {"type": "vad", "state": "end"})

    ok &= await verify(
        not session.is_listening,
        "is_listening=False after VAD end",
    )

    mock_asr.stop()
    elapsed = time.perf_counter() - t0
    print(f"\n  Round 3: {'ALL PASS' if ok else 'SOME FAILED'}  [{elapsed*1000:.0f}ms]")
    return ok, elapsed


# ── Report ────────────────────────────────────────────────────

async def main():
    print("=" * 65)
    print("  Barge-in Test Suite — 3 Rounds")
    print("=" * 65)

    results = []
    timings = []

    ok, dur = await test_audio_buffering()
    results.append(ok)
    timings.append(dur)

    await asyncio.sleep(0.1)

    ok, dur = await test_turn_cancellation()
    results.append(ok)
    timings.append(dur)

    await asyncio.sleep(0.1)

    ok, dur = await test_rapid_bargein()
    results.append(ok)
    timings.append(dur)

    passed = sum(results)
    total = len(results)
    total_time = sum(timings) * 1000

    print(f"\n{'=' * 65}")
    print(f"  用时明细:")
    for i, (ok, dur) in enumerate(zip(results, timings), 1):
        status = "PASS" if ok else "FAIL"
        print(f"    Round {i}: {dur*1000:.0f}ms  [{status}]")
    print(f"    总计:   {total_time:.0f}ms")
    print(f"  Result: {passed}/{total} rounds passed")
    print(f"{'=' * 65}")
    return all(results)


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)

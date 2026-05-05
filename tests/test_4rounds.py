"""4-round instrumented test: each round counts metrics via MetricsCollector.

Run: HERMES_API_KEY=robot-bridge-key python -m tests.test_4rounds
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import metrics
from src.hermes_client import HermesClient
from src.tts_service import tts_service
from src.config import config

SYSTEM = (
    "你是StackChan语音助手。回复口语化中文，根据问题复杂程度"
    "自然决定回复长度（简单问候不超过30字，解释推荐可到150字）。"
    "不要使用emoji、颜文字或特殊符号。记住对话内容。"
)


def fmt(label: str, val, unit: str = ""):
    print(f"    {label:<20} {val}{unit}")


def print_metrics(round_num: int, label: str):
    """Print current metrics state for a round."""
    j = metrics.json()
    ttft = j["ttft"]
    rtf = j["rtf"]

    print(f"\n  ── [Round {round_num}] {label} ──")
    print(f"  Counters:")
    fmt("Hermes requests",   j["hermes"]["requests"])
    fmt("Hermes errors",     j["hermes"]["errors"])
    fmt("Hermes retries",    j["hermes"]["retries"])
    fmt("TTS requests",      j["tts"]["requests"])
    fmt("TTS errors",        j["tts"]["errors"])
    fmt("TTS audio bytes",   j["tts"]["audio_bytes"])
    fmt("Turns total",       j["turns"]["total"])
    fmt("Turns cancelled",   j["turns"]["cancelled"])
    fmt("Turns errors",      j["turns"]["errors"])
    print(f"  Sliding windows:")
    fmt("TTFT samples",      ttft["samples"])
    fmt("TTFT P50",          f"{ttft['p50']*1000:.0f}",  "ms")
    fmt("TTFT P95",          f"{ttft['p95']*1000:.0f}",  "ms")
    fmt("RTF samples",       rtf["samples"])
    fmt("RTF P50",           rtf["p50"])
    fmt("RTF P95",           rtf["p95"])
    if j["ws_messages"]:
        print(f"  WS messages by type:")
        for t, c in j["ws_messages"].items():
            fmt(f"  {t}", c)
    print()


# ── Round 1: Barge-in & turn management ─────────────────────

async def round1_bargein():
    """Simulate VAD start/end, audio buffering, turn cycles.
    Uses mocks — no real Hermes/TTS needed.
    """
    from unittest.mock import AsyncMock, patch

    from src.websocket_handler import RobotWebSocketHandler, RobotSession
    from src.asr_service import asr_service

    class MockWS:
        def __init__(self):
            self.sent = []
        async def send_text(self, msg: str):
            import json
            self.sent.append(json.loads(msg))
        async def send_bytes(self, data: bytes):
            self.sent.append(("binary", data))

    ws = MockWS()
    session = RobotSession(device_id="test", session_id="r1", websocket=ws)
    handler = RobotWebSocketHandler()
    handler.hermes = None  # exercise error path safely

    # Patch ASR so it returns text without loading real model
    async def fake_transcribe(audio: bytes) -> str:
        return "测试语音转文字"
    patcher = patch.object(asr_service, "transcribe", fake_transcribe)
    patcher.start()

    # 5 VAD cycles with buffered audio
    for i in range(5):
        await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
        session.audio_buffer.append(b"\x00" * 8000)
        await handler._handle_vad_message(session, {"type": "vad", "state": "end"})
        await asyncio.sleep(0.01)

    # 2 rapid barges (cancel during turn)
    handler.hermes = AsyncMock()
    handler.hermes.chat_stream.return_value.__aenter__.return_value = AsyncMock()
    handler.hermes.chat_stream.return_value.__aiter__.return_value = AsyncMock()
    handler.hermes.chat_stream.return_value.__anext__.return_value = "你好"
    handler.hermes.ensure_warm = AsyncMock()

    for _ in range(2):
        handler._start_turn(session, "测试")
        await asyncio.sleep(0.02)
        await handler._handle_vad_message(session, {"type": "vad", "state": "start"})
        await asyncio.sleep(0.01)

    patcher.stop()

    # Force WS message metrics via direct calls
    for t in ("text", "audio", "vad", "ping", "audio", "text"):
        metrics.record_ws_message(t)


# ── Round 2: Hermes LLM only (no TTS) ───────────────────────

async def round2_llm():
    """Real Hermes LLM streaming — measure TTFT, no TTS."""
    client = HermesClient()
    client.api_key = config.hermes.api_key
    await client.__aenter__()
    await asyncio.sleep(0.5)

    session_id = f"test-r2-{int(time.time())}"
    questions = [
        "你好，我叫小明。",
        "我叫什么名字？",
        "给我推荐一本适合我的书。",
    ]

    print(f"  LLM queries: {len(questions)}")
    for q in questions:
        full = ""
        async for chunk in client.chat_stream(
            message=q, session_id=session_id, system_prompt=SYSTEM,
        ):
            full += chunk
        print(f"    \"{q}\" → {len(full.strip())} chars")

    await client.close()


# ── Round 3: TTS only (direct synthesis) ────────────────────

async def round3_tts():
    """Real Sherpa-ONNX TTS — measure RTF, audio bytes."""
    texts = [
        "你好，我是StackChan语音助手。",
        "今天天气真不错。",
        "让我给你推荐一本书，《三体》非常值得一读。",
    ]

    print(f"  TTS queries: {len(texts)}")
    for text in texts:
        t0 = time.perf_counter()
        total_bytes = 0
        async for chunk in tts_service.synthesize_stream(text):
            total_bytes += len(chunk)
        dur = time.perf_counter() - t0
        metrics.record_rtf(dur / (total_bytes / (22050 * 2)) if total_bytes else 0)
        metrics.tts_requests += 1
        metrics.tts_audio_bytes += total_bytes
        print(f"    \"{text[:20]}...\" → {total_bytes} bytes in {dur*1000:.0f}ms")


# ── Round 4: Full pipeline (LLM → TTS) ──────────────────────

async def round4_pipeline():
    """Real Hermes + real TTS — triggers all metrics."""
    client = HermesClient()
    client.api_key = config.hermes.api_key
    await client.__aenter__()
    await asyncio.sleep(0.5)

    session_id = f"test-r4-{int(time.time())}"
    questions = [
        "你好。",
        "给我讲个小知识。",
    ]

    print(f"  Pipeline turns: {len(questions)}")
    for q in questions:
        metrics.turns_total += 1

        # LLM
        metrics.hermes_requests += 1
        full = ""
        ttft_start = time.perf_counter()
        first = True
        async for chunk in client.chat_stream(
            message=q, session_id=session_id, system_prompt=SYSTEM,
        ):
            if first:
                metrics.record_ttft(time.perf_counter() - ttft_start)
                first = False
            full += chunk
        full = full.strip()

        if not full:
            continue

        # TTS
        metrics.tts_requests += 1
        t0 = time.perf_counter()
        total_bytes = 0
        async for chunk in tts_service.synthesize_stream(full):
            total_bytes += len(chunk)
        tts_dur = time.perf_counter() - t0
        audio_secs = total_bytes / (22050 * 2)
        metrics.record_rtf(tts_dur / audio_secs if audio_secs else 0)
        metrics.tts_audio_bytes += total_bytes

        print(f"    \"{q}\" → {len(full)} chars, {total_bytes} bytes audio")

    await client.close()


# ── Main ─────────────────────────────────────────────────────

ROUNDS = [
    ("Barge-in & Turn Management", round1_bargein),
    ("Hermes LLM Streaming",       round2_llm),
    ("Sherpa-ONNX TTS",            round3_tts),
    ("Full Pipeline (LLM + TTS)",  round4_pipeline),
]


async def main():
    print("=" * 68)
    print("  4-Round Instrumented Test — Per-Round Metrics Counting")
    print("=" * 68)

    # Warmup
    print("\n  Warming up TTS...", end=" ", flush=True)
    if not tts_service._ready:
        await tts_service.start()
    t0 = time.perf_counter()
    async for _ in tts_service.synthesize_stream("预热。"):
        pass
    print(f"{1000*(time.perf_counter()-t0):.0f}ms")

    total_t0 = time.perf_counter()

    for i, (label, fn) in enumerate(ROUNDS, 1):
        metrics.reset()
        t_round = time.perf_counter()

        print(f"\n{'─' * 68}")
        print(f"  Round {i}: {label}")
        print(f"{'─' * 68}")

        try:
            await fn()
        except Exception as e:
            metrics.turns_errors += 1
            print(f"  ⚠ Round error: {e}")

        round_ms = 1000 * (time.perf_counter() - t_round)
        print_metrics(i, label)

    total_ms = 1000 * (time.perf_counter() - total_t0)

    print(f"{'=' * 68}")
    print(f"  Total test time: {total_ms:.0f}ms")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    asyncio.run(main())

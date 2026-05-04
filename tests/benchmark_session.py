"""Real benchmark: Hermes server-session vs stateless, both with real TTS.

Measures per-stage latency for 3 conversation rounds:
  LLM TTFT → LLM total → TTS generate → TTS stream

Run: HERMES_API_KEY=robot-bridge-key python -m tests.benchmark_session
"""
import asyncio
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.hermes_client import HermesClient
from src.tts_service import tts_service


SYSTEM = (
    "你是StackChan语音助手。回复口语化中文，不超过50字。"
    "不要使用emoji、颜文字或特殊符号。记住对话内容。"
)

QUESTIONS = [
    "你好，我叫小明，今年25岁。",
    "我叫什么名字？我多大了？",
    "给我推荐一个适合我的周末活动。",
]


async def run_round(round_num: int, question: str, client: HermesClient, session_id: str, label: str):
    ttft = total_llm = tts_time = audio_dur = 0
    full = ""

    # LLM stream
    t0 = time.perf_counter()
    first = False
    async for chunk in client.chat_stream(
        message=question, session_id=session_id, system_prompt=SYSTEM,
    ):
        if not first:
            ttft = 1000 * (time.perf_counter() - t0)
            first = True
        full += chunk
    total_llm = 1000 * (time.perf_counter() - t0)
    full = full.strip()

    # TTS synthesis (real Sherpa-ONNX)
    if full:
        t0 = time.perf_counter()
        result = []
        async for chunk in tts_service.synthesize_stream(full):
            result.append(chunk)
        tts_time = 1000 * (time.perf_counter() - t0)
        audio_dur = sum(len(c) / (22050 * 2) for c in result)

    usage = client._last_usage or {}

    # Report
    print(f"  Round {round_num} [{label}]")
    print(f"    Input:      \"{question[:30]}...\"")
    print(f"    Response:   \"{full}\" ({len(full)} chars)")
    print(f"    LLM TTFT:   {ttft:.0f}ms")
    print(f"    LLM total:  {total_llm:.0f}ms")
    if audio_dur:
        print(f"    TTS:        {tts_time:.0f}ms ({audio_dur:.1f}s audio)  RTF={tts_time/1000/audio_dur:.2f}")
    print(f"    Tokens:     {usage.get('total_tokens', 'N/A')} total")
    print()


async def test_mode(label: str, use_api_key: bool):
    """Run 3 rounds in a given mode with a fresh client."""
    client = HermesClient()
    client.api_key = "robot-bridge-key"
    if not use_api_key:
        client._force_stateless = True

    await client.__aenter__()
    # Ensure warmup completes before timing
    await asyncio.sleep(0.5)

    session_id = f"bench-{label}-{int(time.time())}"
    print(f"\n{'='*65}")
    print(f"  Mode: {'Server Session' if use_api_key else 'Stateless (full history sent)'}")
    print(f"{'='*65}")

    for i, q in enumerate(QUESTIONS, 1):
        await run_round(i, q, client, session_id, label)

    # Print session token tracking
    if session_id in client._sessions:
        store = client._sessions[session_id]
        print(f"  Session tracked: {len(store.messages)} msgs, "
             f"{store._total_tokens} tokens (from API){'' if use_api_key else ' (estimated)'}")
    print()

    await client.close()


async def main():
    print("=" * 65)
    print("  Hermes Session Benchmark: Server Session vs Stateless")
    print("  LLM: Hermes → deepseek-v4-flash  |  TTS: Sherpa-ONNX Matcha")
    print("=" * 65)
    print()

    await tts_service.start()

    # Warmup TTS
    t0 = time.perf_counter()
    async for _ in tts_service.synthesize_stream("预热。"):
        pass
    print(f"  TTS warmed up in {1000*(time.perf_counter()-t0):.0f}ms\n")

    # Test 1: Server-side session (with API key)
    await test_mode("session", use_api_key=True)

    await asyncio.sleep(1)

    # Test 2: Stateless (no API key, full history sent)
    await test_mode("stateless", use_api_key=False)


if __name__ == "__main__":
    asyncio.run(main())

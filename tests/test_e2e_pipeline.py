"""End-to-end pipeline test with streaming and memory verification.

Pipeline: TTS(simulate speech) → ASR → LLM(Hermes, streaming) → TTS
Client-side context management with preventive compression.
"""
import asyncio
import os
import struct
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# Ensure we run from project root (where models/ and configs/ live)
_project_root = Path(__file__).parent.parent
os.chdir(str(_project_root))
sys.path.insert(0, str(_project_root))

# ── conversation rounds ───────────────────────────────────────────────
ROUNDS = [
    "你好，我叫小明，今年25岁，我喜欢爬山。",
    "我刚才说我叫什么名字？多大年纪了？",
    "今天天气真好，根据我的爱好，适合出去玩吗？",
    "你还记得我的名字和年龄吗？我告诉过你的。",
    "那根据我的名字和爱好，帮我推荐一个适合我的周末活动吧。",
]

# ── timing record ────────────────────────────────────────────────────
@dataclass
class RoundTiming:
    round: int
    question: str
    compress_ms: float = 0
    asr_ms: float = 0
    asr_text: str = ""
    llm_first_ms: float = 0       # time to first token
    llm_total_ms: float = 0       # time to complete response
    llm_text: str = ""
    llm_chars: int = 0
    tts_response_ms: float = 0
    tts_duration_s: float = 0
    total_ms: float = 0


# ── main test ─────────────────────────────────────────────────────────
async def run_e2e_test():
    from src.asr_service import asr_service
    from src.config import config
    from src.hermes_client import hermes_client
    from src.tts_service import tts_service

    print("=" * 80)
    print(" Robot Bridge E2E Test — Streaming + Client-side Context")
    print("=" * 80)

    # ── init ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    await hermes_client.__aenter__()
    await asr_service.start()
    await tts_service.start()
    print(f"[init] Ready ({1000*(time.perf_counter()-t0):.0f}ms)\n")

    session_id = str(uuid.uuid4())
    system_prompt = config.robot.system_prompt
    print(f"[config] session={session_id[:8]}...")
    print(f"[config] system_prompt: {system_prompt}\n")

    timings: list[RoundTiming] = []

    # ── 5 rounds ──────────────────────────────────────────────────────
    for i, question in enumerate(ROUNDS):
        t = RoundTiming(round=i + 1, question=question)
        round_start = time.perf_counter()

        print(f"{'─' * 72}")
        print(f" Round {i+1}/5 | {question}")
        print(f"{'─' * 72}")

        # ── step 1: simulate user speech (TTS → PCM 16kHz) ──────────
        t1 = time.perf_counter()
        simulate_wav = await tts_service.synthesize(question)
        # WAV → PCM int16
        pcm_data = simulate_wav[44:]
        # Resample 22050→16000
        if tts_service._sample_rate != 16000:
            ratio = tts_service._sample_rate / 16000
            samples = struct.unpack(f"<{len(pcm_data)//2}h", pcm_data)
            pcm_16k = [samples[min(int(j * ratio), len(samples) - 1)]
                       for j in range(int(len(samples) / ratio))]
            pcm_data = struct.pack(f"<{len(pcm_16k)}h", *pcm_16k)
        tts_sim_ms = 1000 * (time.perf_counter() - t1)
        audio_dur = len(pcm_data) / 32000
        print(f"  [tts-sim] {tts_sim_ms:.0f}ms → PCM {len(pcm_data)} bytes ({audio_dur:.2f}s)")

        # ── step 2: ASR ──────────────────────────────────────────────
        t2 = time.perf_counter()
        asr_text = await asr_service.transcribe(pcm_data)
        t.asr_ms = 1000 * (time.perf_counter() - t2)
        t.asr_text = asr_text or question
        print(f"  [asr]     {t.asr_ms:.0f}ms → \"{t.asr_text[:80]}\"")
        if not asr_text:
            asr_text = question

        # ── step 3: LLM streaming ────────────────────────────────────
        t3 = time.perf_counter()
        first_token = False
        full_text = ""

        async for chunk in hermes_client.chat_stream(
            message=asr_text,
            session_id=session_id,
            system_prompt=system_prompt,
        ):
            if not first_token:
                t.llm_first_ms = 1000 * (time.perf_counter() - t3)
                first_token = True
            full_text += chunk

        t.llm_total_ms = 1000 * (time.perf_counter() - t3)
        t.llm_text = full_text.strip()
        t.llm_chars = len(full_text)
        print(f"  [llm]     first={t.llm_first_ms:.0f}ms total={t.llm_total_ms:.0f}ms "
              f"chars={t.llm_chars} → \"{t.llm_text[:100]}\"")

        # ── step 4: TTS response ─────────────────────────────────────
        t4 = time.perf_counter()
        response_wav = await tts_service.synthesize(t.llm_text if t.llm_text else "嗯")
        t.tts_response_ms = 1000 * (time.perf_counter() - t4)
        if len(response_wav) > 44:
            t.tts_duration_s = (len(response_wav) - 44) / (tts_service._sample_rate * 2)
        print(f"  [tts-resp]{t.tts_response_ms:.0f}ms → {t.tts_duration_s:.2f}s audio")

        # ── total ────────────────────────────────────────────────────
        t.total_ms = 1000 * (time.perf_counter() - round_start)
        timings.append(t)
        print(f"  [total]   {t.total_ms:.0f}ms")

    # ── summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(" TIMING SUMMARY (ms)")
    print("=" * 100)
    hdr = (f"{'#':>2} {'Question':<32} {'ASR':>5} {'LLM1st':>6} {'LLMtot':>6} "
           f"{'TTS':>6} {'Total':>6} {'Chars':>5} {'RTF':>5}")
    print(hdr)
    print("-" * 100)

    for t in timings:
        rtf = t.tts_response_ms / max(1, t.tts_duration_s * 1000)
        print(f"{t.round:>2} {t.question:<32} "
              f"{t.asr_ms:>5.0f} {t.llm_first_ms:>6.0f} {t.llm_total_ms:>6.0f} "
              f"{t.tts_response_ms:>6.0f} {t.total_ms:>6.0f} {t.llm_chars:>5} {rtf:>5.2f}")

    print("-" * 100)
    avg_asr = sum(t.asr_ms for t in timings) / 5
    avg_first = sum(t.llm_first_ms for t in timings) / 5
    avg_llm = sum(t.llm_total_ms for t in timings) / 5
    avg_tts = sum(t.tts_response_ms for t in timings) / 5
    avg_total = sum(t.total_ms for t in timings) / 5
    print(f"{'AVG':>2} {'':32} {avg_asr:>5.0f} {avg_first:>6.0f} {avg_llm:>6.0f} "
          f"{avg_tts:>6.0f} {avg_total:>6.0f}")

    # ── streaming analysis ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" STREAMING ANALYSIS (first token vs full response)")
    print("=" * 70)
    for t in timings:
        gap = t.llm_total_ms - t.llm_first_ms
        print(f"  Round {t.round}: first token at {t.llm_first_ms:.0f}ms, "
              f"remaining {gap:.0f}ms, {t.llm_chars} chars")

    # ── memory verification ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" MEMORY VERIFICATION")
    print("=" * 70)
    session = hermes_client._sessions.get(session_id)
    if session:
        print(f"  Session messages: {len(session.messages)}")
        print(f"  Estimated tokens: {session._total_tokens}")
        print(f"  Compressions: {session._compress_count}")
        if session.summary:
            print(f"  Summary: {session.summary[:120]}")

    memory_checks = [
        ("我叫什么名字？", ["小明"]),
        ("我多大了？", ["25"]),
        ("我喜欢什么运动？", ["爬山", "爬"]),
        ("你一共记住了我哪些信息？", ["小明", "25", "爬山"]),
    ]

    for q, expected in memory_checks:
        full_text = ""
        try:
            async for chunk in hermes_client.chat_stream(
                message=q,
                session_id=session_id,
                system_prompt=system_prompt,
            ):
                full_text += chunk
        except Exception as e:
            full_text = f"ERROR: {e}"

        matches = [w for w in expected if w in full_text]
        status = "✓" if matches else "✗"
        print(f"  {status} Q: {q}")
        print(f"    A: {full_text[:120]}")

    # ── cleanup ──────────────────────────────────────────────────────
    await hermes_client.close()

    print("\n" + "=" * 80)
    print(" TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_e2e_test())

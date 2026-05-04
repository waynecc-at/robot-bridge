"""Real TTS pipeline benchmark — measures actual sherpa-onnx Matcha TTS latency.

Run: python -m tests.benchmark_real_tts
"""
import asyncio
import time
import io
import wave
import struct
import re
from dataclasses import dataclass, field
from typing import AsyncGenerator
from loguru import logger

import sherpa_onnx


# ── Real TTS Engine ─────────────────────────────────────────────

MODEL_DIR = "models/matcha-icefall-zh-baker"

_tts_instance = None
_tts_sample_rate = 22050


def _get_tts():
    global _tts_instance, _tts_sample_rate
    if _tts_instance is not None:
        return _tts_instance
    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=f"{MODEL_DIR}/model-steps-3.onnx",
                vocoder=f"{MODEL_DIR}/vocos-22khz-univ.onnx",
                lexicon=f"{MODEL_DIR}/lexicon.txt",
                tokens=f"{MODEL_DIR}/tokens.txt",
            ),
            num_threads=2,
            debug=False,
        ),
        rule_fsts=f"{MODEL_DIR}/phone.fst,{MODEL_DIR}/date.fst,{MODEL_DIR}/number.fst",
        max_num_sentences=1,
    )
    config.validate()
    _tts_instance = sherpa_onnx.OfflineTts(config)
    _tts_sample_rate = _tts_instance.sample_rate
    return _tts_instance


def _generate_tts(text: str):
    """Synchronous TTS generation."""
    tts = _get_tts()
    gen_config = sherpa_onnx.GenerationConfig()
    gen_config.sid = 0
    gen_config.speed = 1.0
    gen_config.silence_scale = 0.2
    return tts.generate(text, gen_config)


def _samples_to_wav(samples, sample_rate: int) -> bytes:
    """Convert float32 samples to WAV (16-bit PCM) bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = []
        for s in samples:
            s = max(-1.0, min(1.0, s))
            pcm.append(int(s * 32767))
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return buf.getvalue()


def _make_wav_chunk(pcm_chunk: bytes, sr: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_chunk)
    return buf.getvalue()


# ── Session Store (same as production) ─────────────────────────

MAX_TOKENS = 6000
COMPRESS_AT = 0.65
KEEP_RECENT = 6


def _estimate_tokens(text: str) -> int:
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    ascii_chars = len(text) - chinese
    return max(1, int(chinese * 2 + ascii_chars * 0.4))


class SessionStore:
    def __init__(self):
        self.session_id = "benchmark"
        self.messages: list[dict] = []
        self.summary: str = ""
        self._total_tokens: int = 0
        self._compress_count: int = 0
        self._pending_compression = False

    def _add_message(self, role: str, text: str):
        self.messages.append({"role": role, "content": text})
        self._total_tokens += _estimate_tokens(text)


# ── Sentence Splitter ──────────────────────────────────────────

SENTENCE_PATTERN = re.compile(r"(.+?[。！？\n])")


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    sentences = []
    while True:
        m = SENTENCE_PATTERN.search(buffer)
        if not m:
            break
        sentences.append(m.group(1).strip())
        buffer = buffer[m.end():]
    return sentences, buffer


# ── Fake LLM Stream ────────────────────────────────────────────

LLM_TTFT_MS = 800
LLM_TOKEN_MS = 50

TEST_RESPONSES = [
    "今天天气真不错。我们出去走走吧。记得带伞哦。",
    "这个问题很有意思。让我想想怎么回答你。从几个角度来看吧。",
    "好的我记住了。你叫小明今年25岁。爱好是爬山和看书。",
    "根据你刚才说的。我建议可以先试试这个方案。如果不行再换别的。",
    "编程其实不难。关键是多练习。建议你每天写一点代码。慢慢就熟练了。坚持最重要。",
]


async def _fake_llm_stream(text: str) -> AsyncGenerator[str, None]:
    """Simulate streaming LLM response."""
    response = TEST_RESPONSES[hash(text) % len(TEST_RESPONSES)]
    await asyncio.sleep(LLM_TTFT_MS / 1000)
    for char in response:
        await asyncio.sleep(LLM_TOKEN_MS / 1000)
        yield char


# ── Real TTS Streaming (chunked) ───────────────────────────────

PREBUFFER_CHUNK_MS = 50
STREAM_CHUNK_MS = 200


async def _real_tts_stream(text: str) -> AsyncGenerator[bytes, None]:
    """Real sherpa-onnx TTS with chunked streaming."""
    loop = asyncio.get_event_loop()
    tts_result = await loop.run_in_executor(None, _generate_tts, text)

    sr = tts_result.sample_rate
    samples = tts_result.samples
    wav_data = _samples_to_wav(samples, sr)[44:]  # Skip 44-byte WAV header
    bytes_per_sample = 2

    prebuffer_bytes = int(sr * (PREBUFFER_CHUNK_MS / 1000) * bytes_per_sample)
    chunk_bytes = int(sr * (STREAM_CHUNK_MS / 1000) * bytes_per_sample)

    # First chunk: 50ms
    first = wav_data[:prebuffer_bytes]
    if first:
        yield _make_wav_chunk(first, sr)

    # Remaining chunks: 200ms
    offset = prebuffer_bytes
    while offset < len(wav_data):
        chunk = wav_data[offset:offset + chunk_bytes]
        if not chunk:
            break
        yield _make_wav_chunk(chunk, sr)
        offset += chunk_bytes


# ── Concurrent Pipeline ────────────────────────────────────────

async def process_turn(
    text: str,
    session: SessionStore,
    turn_id: int,
) -> dict:
    """Run one turn through the concurrent pipeline and return timing metrics."""
    metrics = {"turn_id": turn_id, "text": text, "phases": {}}
    t_start = time.perf_counter()

    # Add user message
    session._add_message("user", text)

    # Concurrent LLM + TTS
    sentence_queue: asyncio.Queue = asyncio.Queue()
    sentence_times: list[dict] = []

    async def llm_reader():
        buffer = ""
        first_token_received = False
        t_llm_start = time.perf_counter()

        async for token in _fake_llm_stream(text):
            if not first_token_received:
                metrics["phases"]["llm_ttft"] = 1000 * (time.perf_counter() - t_llm_start)
                first_token_received = True
            buffer += token
            sentences, buffer = _split_sentences(buffer)
            for s in sentences:
                await sentence_queue.put(s)

        if buffer.strip():
            await sentence_queue.put(buffer.strip())
        await sentence_queue.put(None)
        metrics["phases"]["llm_total"] = 1000 * (time.perf_counter() - t_llm_start)

    async def tts_worker():
        sentence_count = 0
        total_tts_time = 0
        total_chunks = 0
        first_sentence_start = None

        while True:
            sentence = await sentence_queue.get()
            if sentence is None:
                break

            sentence_count += 1
            t_sent_start = time.perf_counter()
            if first_sentence_start is None:
                first_sentence_start = t_sent_start

            first_chunk_time = None
            chunk_count = 0

            async for chunk in _real_tts_stream(sentence):
                chunk_count += 1
                total_chunks += 1
                if first_chunk_time is None:
                    first_chunk_time = 1000 * (time.perf_counter() - t_sent_start)

            t_sent_end = time.perf_counter()
            sent_time = 1000 * (t_sent_end - t_sent_start)
            total_tts_time += sent_time

            sentence_times.append({
                "index": sentence_count,
                "text": sentence[:30],
                "first_chunk_ms": first_chunk_time,
                "total_ms": sent_time,
                "chunks": chunk_count,
            })

        metrics["phases"]["tts_first_chunk"] = (
            1000 * (first_sentence_start - t_start)
            if first_sentence_start else None
        )
        metrics["phases"]["tts_total"] = total_tts_time
        metrics["phases"]["audio_chunks"] = total_chunks
        metrics["sentence_times"] = sentence_times

    await asyncio.gather(llm_reader(), tts_worker())

    session._add_message("assistant", text)

    metrics["total_ms"] = 1000 * (time.perf_counter() - t_start)
    metrics["session_tokens"] = session._total_tokens
    return metrics


# ── Main Benchmark ─────────────────────────────────────────────

TEST_PROMPTS = [
    "今天天气怎么样？",
    "你能帮我分析一下这个问题吗？",
    "我叫小明，今年25岁，喜欢爬山。",
    "我最近在学习编程，有什么建议吗？",
    "刚才说的你还记得吗？给我讲讲。",
]

REAL_TTS_RESPONSES = [
    "今天天气真不错。我们出去走走吧。记得带伞哦。",
    "这个问题很有意思。让我想想怎么回答你。从几个角度来看吧。",
    "好的我记住了。你叫小明今年25岁。爱好是爬山和看书。",
    "根据你刚才说的。我建议可以先试试这个方案。如果不行再换别的。",
    "编程其实不难。关键是多练习。建议你每天写一点代码。慢慢就熟练了。坚持最重要。",
]


async def run_benchmark():
    """Run 5 rounds with real TTS."""
    logger.remove()
    logger.add(lambda x: None, level="TRACE")

    # Warmup TTS engine
    print("预热 TTS 引擎...")
    _get_tts()
    # Run a warmup generation
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _generate_tts, "预热测试。")
    print("预热完成。\n")

    print("=" * 70)
    print("  Robot Bridge 管道性能基准测试 v0.3 (真实 TTS)")
    print("=" * 70)
    print(f"  TTS 引擎:        Sherpa-ONNX Matcha (matcha-icefall-zh-baker)")
    print(f"  Vocoder:         Vocos (22kHz)")
    print(f"  TTS 采样率:      {_tts_sample_rate} Hz")
    print(f"  LLM 模拟:        TTFT={LLM_TTFT_MS}ms, Token={LLM_TOKEN_MS}ms/token")
    print(f"  分块流式:        首块 {PREBUFFER_CHUNK_MS}ms / 后续 {STREAM_CHUNK_MS}ms")
    print("=" * 70)

    session = SessionStore()

    for round_idx, prompt in enumerate(TEST_PROMPTS):
        response = REAL_TTS_RESPONSES[round_idx]
        print(f"\n  ── 第 {round_idx+1} 轮 ──")
        print(f"  输入: \"{prompt}\"")
        print(f"  回复: \"{response}\"")
        print()

        metrics = await process_turn(prompt, session, round_idx + 1)

        print(f"  耗时分解:")
        print(f"     LLM TTFT:         {metrics['phases'].get('llm_ttft', 0):>8.1f}ms")
        print(f"     LLM 总耗时:       {metrics['phases'].get('llm_total', 0):>8.1f}ms")
        print(f"     TTS 首句首块:     {metrics['phases'].get('tts_first_chunk', 0):>8.1f}ms")
        print(f"     TTS 总耗时:       {metrics['phases'].get('tts_total', 0):>8.1f}ms")
        print(f"     音频块数:         {metrics['phases'].get('audio_chunks', 0)}")
        print(f"  ⏱  总耗时:          {metrics['total_ms']:>8.1f}ms")

        if metrics['sentence_times']:
            for st in metrics['sentence_times']:
                print(f"     句子{st['index']}: \"{st['text']}...\" "
                      f"首块={st['first_chunk_ms']:.0f}ms "
                      f"合计={st['total_ms']:.0f}ms "
                      f"({st['chunks']}块)")

        # Overlap efficiency
        llm_total = metrics['phases'].get('llm_total', 0)
        tts_total = metrics['phases'].get('tts_total', 0)
        total = metrics['total_ms']
        serial_time = llm_total + tts_total
        overlap_saved = serial_time - total
        if serial_time > 0:
            efficiency = (overlap_saved / serial_time) * 100
            print(f"  📈 并发效率: LLM({llm_total:.0f}ms)+TTS({tts_total:.0f}ms)="
                  f"{serial_time:.0f}ms → 实际 {total:.0f}ms = 节省 {efficiency:.0f}%")

        print(f"  会话 token 数: {metrics['session_tokens']}")

    # Summary comparison table
    print("\n" + "=" * 70)
    print("  📋 优化前后对比 (理论值)")
    print("=" * 70)
    print(f"  ┌──────────┬──────────────┬──────────────┬──────────┐")
    print(f"  │ 轮次     │ 优化前(串行)  │ 优化后(并发)  │ 提升     │")
    print(f"  ├──────────┼──────────────┼──────────────┼──────────┤")

    for round_idx, prompt in enumerate(TEST_PROMPTS):
        response = REAL_TTS_RESPONSES[round_idx]
        # Serial = LLM TTFT + LLM tokens + TTS (no overlap)
        tts_time = await _measure_tts_latency(response)
        llm_total = LLM_TTFT_MS + len(response) * LLM_TOKEN_MS
        serial_total = llm_total + tts_time
        # Optimized = max(LLM, TTS) + small overhead
        opt_total = max(llm_total, tts_time) + 50

        improvement = ((serial_total - opt_total) / serial_total * 100)
        print(f"  │ {round_idx+1:<6} │ {serial_total/1000:>7.1f}s      │ {opt_total/1000:>7.1f}s      │ {improvement:>+5.0f}%   │")

    print(f"  └──────────┴──────────────┴──────────────┴──────────┘")

    # Print real TTS performance stats
    print("\n  真实 TTS 性能:")
    for round_idx, response in enumerate(REAL_TTS_RESPONSES):
        tts_time = await _measure_tts_latency(response)
        print(f"    第 {round_idx+1} 轮: \"{response[:20]}...\" → {tts_time:.0f}ms")


async def _measure_tts_latency(text: str) -> float:
    """Measure real TTS generation time in ms."""
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    result = await loop.run_in_executor(None, _generate_tts, text)
    t1 = time.perf_counter()
    return 1000 * (t1 - t0)


if __name__ == "__main__":
    asyncio.run(run_benchmark())

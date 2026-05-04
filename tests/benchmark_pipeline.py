"""End-to-end pipeline benchmark — tests concurrent LLM+TTS pipeline with timing.

This benchmark simulates the full robot-bridge pipeline without requiring
actual ML models (Sherpa-ONNX, FunASR). Uses mock services with realistic
latency profiles to measure:
- Concurrent pipeline performance (LLM stream ↔ TTS worker)
- Deferred compression timing
- Chunked TTS streaming latency
- End-to-end response time per round

Run: python tests/benchmark_pipeline.py
"""
import asyncio
import base64
import json
import re
import time
import io
import wave
import struct
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional
from loguru import logger


# ── Mock Services (simulate real latency) ──────────────────────────

ASR_LATENCY_MS = 500      # Simulated ASR CPU time
TTS_LATENCY_PER_CHAR = 30  # ~30ms per character TTS synthesis
LLM_TTFT_MS = 800         # LLM time-to-first-token
LLM_TOKEN_MS = 50         # ms per token generation
COMPRESS_LATENCY_MS = 2000  # LLM compression round-trip


def _mock_asr(text: str) -> str:
    """Simulate ASR with realistic latency."""
    time.sleep(ASR_LATENCY_MS / 1000)
    return text


def _mock_tts(text: str) -> bytes:
    """Simulate TTS synthesis: return fake WAV bytes with realistic latency."""
    char_count = len(text)
    latency = max(200, char_count * TTS_LATENCY_PER_CHAR)
    time.sleep(latency / 1000)

    # Generate fake audio: 22050Hz, 16-bit, mono
    sample_rate = 22050
    duration = max(0.5, char_count * 0.08)  # ~80ms per char
    num_samples = int(sample_rate * duration)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [int(s * 32767) for s in
                   [0.1] * num_samples]  # constant tone
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


# ── Session Store (same logic as production) ───────────────────────

MAX_TOKENS = 6000
COMPRESS_AT = 0.65
KEEP_RECENT = 6


def _estimate_tokens(text: str) -> int:
    """Improved heuristic: Chinese ~2 tokens, ASCII ~0.4 tokens."""
    chinese = sum(1 for c in text if '一' <= c <= '鿿')
    ascii_chars = len(text) - chinese
    return max(1, int(chinese * 2 + ascii_chars * 0.4))


class SessionStore:
    """Per-session history with deferred compression."""

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

    def needs_compression(self) -> bool:
        threshold = int(MAX_TOKENS * COMPRESS_AT)
        return self._total_tokens > threshold and len(self.messages) > KEEP_RECENT

    def mark_for_compression(self):
        if self.needs_compression() and not self._pending_compression:
            self._pending_compression = True
            return True
        return False

    async def _do_compress(self):
        """Simulate compression (LLM round-trip)."""
        if not self._pending_compression:
            return
        t0 = time.perf_counter()
        logger.info(f"[BENCH] Compression starting (tokens={self._total_tokens})...")
        await asyncio.sleep(COMPRESS_LATENCY_MS / 1000)

        recent = self.messages[-KEEP_RECENT:]
        old = self.messages[:-KEEP_RECENT]
        if old:
            self.summary = "[模拟摘要] 用户与助手进行了多轮对话。"
            self.messages = recent
            self._total_tokens = sum(
                _estimate_tokens(m.get("content", "")) for m in self.messages
            )
            self._compress_count += 1
            elapsed = 1000 * (time.perf_counter() - t0)
            logger.info(f"[BENCH] Compression done in {elapsed:.0f}ms")


# ── Sentence Splitter ────────────────────────────────────────────

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


# ── Fake LLM Stream (simulates Hermes) ──────────────────────────

TEST_RESPONSES = [
    "今天天气真不错。我们出去走走吧。记得带伞哦。",
    "这个问题很有意思。让我想想怎么回答你。从几个角度来看吧。",
    "好的我记住了。你叫小明今年25岁。爱好是爬山和看书。",
    "根据你刚才说的。我建议可以先试试这个方案。如果不行再换别的。",
    "编程其实不难。关键是多练习。建议你每天写一点代码。慢慢就熟练了。坚持最重要。",
]


async def _fake_llm_stream(text: str) -> AsyncGenerator[str, None]:
    """Simulate streaming LLM response with per-token latency."""
    response = TEST_RESPONSES[hash(text) % len(TEST_RESPONSES)]

    # Simulate TTFT (time-to-first-token)
    await asyncio.sleep(LLM_TTFT_MS / 1000)

    # Stream character-by-character as "tokens"
    for char in response:
        await asyncio.sleep(LLM_TOKEN_MS / 1000)
        yield char

    logger.info(f"[BENCH] LLM stream complete ({len(response)} chars)")


# ── TTS Service (same logic as optimized tts_service.py) ─────────

PREBUFFER_CHUNK_MS = 50
STREAM_CHUNK_MS = 200


async def _tts_synthesize_stream(text: str) -> AsyncGenerator[bytes, None]:
    """Simulate chunked TTS: first chunk at 50ms, rest at 200ms."""
    # Simulate TTS compute latency
    await asyncio.to_thread(_mock_tts, text)

    sr = 22050
    char_count = len(text)
    total_duration = max(0.5, char_count * 0.08)
    total_samples = int(sr * total_duration)
    bytes_per_sample = 2

    prebuffer_bytes = int(sr * (PREBUFFER_CHUNK_MS / 1000) * bytes_per_sample)
    chunk_bytes = int(sr * (STREAM_CHUNK_MS / 1000) * bytes_per_sample)

    fake_pcm = b'\x00\x00' * total_samples

    def _make_wav(pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()

    # First chunk: 50ms
    first = fake_pcm[:prebuffer_bytes]
    if first:
        yield _make_wav(first)

    # Remaining chunks: 200ms
    offset = prebuffer_bytes
    while offset < len(fake_pcm):
        chunk = fake_pcm[offset:offset + chunk_bytes]
        if not chunk:
            break
        yield _make_wav(chunk)
        offset += chunk_bytes


# ── Concurrent Pipeline (same logic as optimized websocket_handler) ──

async def process_turn(
    text: str,
    session: SessionStore,
    turn_id: int,
) -> dict:
    """Run one turn through the concurrent pipeline and return timing metrics."""
    metrics = {"turn_id": turn_id, "text": text, "phases": {}}

    t_start = time.perf_counter()

    # Phase 1: Mark for compression (non-blocking)
    t0 = time.perf_counter()
    session.mark_for_compression()
    metrics["phases"]["compress_mark"] = 1000 * (time.perf_counter() - t0)

    # Phase 2: Add user message
    session._add_message("user", text)

    # Phase 3: Concurrent LLM + TTS
    sentence_queue: asyncio.Queue = asyncio.Queue()
    sentence_times: list[dict] = []

    async def llm_reader():
        """Stream LLM tokens, split into sentences, push to queue."""
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
        """Consume sentences from queue, TTS, send audio."""
        sentence_count = 0
        total_tts_time = 0
        total_chunks = 0
        first_sentence_start = None
        last_audio_end = None

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

            async for chunk in _tts_synthesize_stream(sentence):
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

            last_audio_end = t_sent_end

        metrics["phases"]["tts_first_chunk"] = (
            1000 * (first_sentence_start - t_start)
            if first_sentence_start else None
        )
        metrics["phases"]["tts_total"] = total_tts_time
        metrics["phases"]["audio_chunks"] = total_chunks
        metrics["sentence_times"] = sentence_times

    # Run both tasks concurrently
    await asyncio.gather(llm_reader(), tts_worker())

    # Record assistant response
    session._add_message("assistant", text)

    # Phase 4: Compression happens in background (simulate a check)
    if session._pending_compression:
        t0 = time.perf_counter()
        await session._do_compress()
        metrics["phases"]["compress_actual"] = 1000 * (time.perf_counter() - t0)

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


async def run_benchmark():
    """Run 5 rounds through the pipeline with detailed timing."""
    logger.remove()
    logger.add(lambda x: None, level="TRACE")  # silence during benchmark

    print("=" * 70)
    print("  Robot Bridge 管道性能基准测试 v0.2")
    print("=" * 70)
    print(f"  ASR 模拟延迟:    {ASR_LATENCY_MS}ms")
    print(f"  LLM TTFT 模拟:   {LLM_TTFT_MS}ms")
    print(f"  LLM Token 模拟:  {LLM_TOKEN_MS}ms/token")
    print(f"  TTS 模拟延迟:    {TTS_LATENCY_PER_CHAR}ms/char")
    print(f"  Compress 模拟:   {COMPRESS_LATENCY_MS}ms")
    print("=" * 70)

    session = SessionStore()

    # Pre-populate session to trigger compression
    for i in range(10):
        session._add_message("user", f"这是第{i}轮的用户输入内容。")
        session._add_message("assistant", f"这是第{i}轮助手的回复内容。")

    print(f"\n  预填充会话: {len(session.messages)} 条消息, {session._total_tokens} tokens")
    print(f"  压缩阈值: {int(MAX_TOKENS * COMPRESS_AT)} tokens")
    session._total_tokens = 5000  # Force near-threshold
    print(f"  模拟 {session._total_tokens} tokens (触发延迟压缩)")
    print()

    for round_idx, prompt in enumerate(TEST_PROMPTS):
        print(f"  ── 第 {round_idx+1} 轮 ──")
        print(f"  输入: \"{prompt}\"")
        print(f"  模拟回复: \"{TEST_RESPONSES[round_idx]}\"")
        print()

        metrics = await process_turn(prompt, session, round_idx + 1)

        print(f"  📊 耗时分解:")
        print(f"     compress标记:     {metrics['phases'].get('compress_mark', 0):>8.1f}ms (不阻塞)")
        print(f"     LLM TTFT:         {metrics['phases'].get('llm_ttft', 0):>8.1f}ms (首token延迟)")
        print(f"     LLM 总耗时:       {metrics['phases'].get('llm_total', 0):>8.1f}ms")
        print(f"     TTS 首句首块:     {metrics['phases'].get('tts_first_chunk', 0):>8.1f}ms")
        print(f"     TTS 总耗时:       {metrics['phases'].get('tts_total', 0):>8.1f}ms")
        print(f"     音频块数:         {metrics['phases'].get('audio_chunks', 0)}")

        if 'compress_actual' in metrics['phases']:
            print(f"     compress执行:     {metrics['phases']['compress_actual']:>8.1f}ms (后台)")

        print(f"  ⏱  总耗时:          {metrics['total_ms']:>8.1f}ms")

        if metrics['sentence_times']:
            for st in metrics['sentence_times']:
                print(f"     句子{st['index']}: \"{st['text']}...\" "
                      f"首块={st['first_chunk_ms']:.0f}ms "
                      f"合计={st['total_ms']:.0f}ms "
                      f"({st['chunks']}块)")

        # Show overlap efficiency
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
        print()

    # Final summary
    print("=" * 70)
    print("  📋 摘要")
    print("=" * 70)
    print(f"  压缩状态: {session._compress_count} 次压缩, "
          f"摘要长度: {len(session.summary)} chars")
    print(f"  最终 token: {session._total_tokens}")
    print()

    # Compare: with vs without optimizations (theoretical)
    print("  优化前后对比 (理论值):")
    print(f"  ┌──────────┬──────────────┬──────────────┬──────────┐")
    print(f"  │ 轮次     │ 优化前(串行)  │ 优化后(并发)  │ 提升     │")
    print(f"  ├──────────┼──────────────┼──────────────┼──────────┤")

    for round_idx, prompt in enumerate(TEST_PROMPTS):
        # Calculate theoretical serial time (without optimizations)
        response = TEST_RESPONSES[round_idx]
        tts_time_serial = max(200, len(response) * TTS_LATENCY_PER_CHAR)
        llm_total = LLM_TTFT_MS + len(response) * LLM_TOKEN_MS

        serial_total = (
            ASR_LATENCY_MS
            + COMPRESS_LATENCY_MS  # blocking!
            + llm_total
            + tts_time_serial
        )

        # Our optimized time (from benchmark or estimate)
        opt_total = metrics['total_ms'] if round_idx == 0 else (
            ASR_LATENCY_MS
            + max(llm_total, tts_time_serial)  # overlap
            + 50  # overhead
        )

        improvement = ((serial_total - opt_total) / serial_total * 100)

        serial_sec = serial_total / 1000
        opt_sec = opt_total / 1000
        print(f"  │ {round_idx+1:<6} │ {serial_sec:>7.1f}s      │ {opt_sec:>7.1f}s      │ {improvement:>+5.0f}%   │")

    print(f"  └──────────┴──────────────┴──────────────┴──────────┘")
    print()


if __name__ == "__main__":
    asyncio.run(run_benchmark())

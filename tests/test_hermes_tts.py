"""Quick end-to-end test: real Hermes LLM + real Sherpa-ONNX TTS.

Run: python -m tests.test_hermes_tts
"""
import asyncio
import time
import sherpa_onnx
import io
import wave
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.hermes_client import hermes_client

MODEL_DIR = "models/matcha-icefall-zh-baker"
_tts = None


def _get_tts():
    global _tts
    if _tts is not None:
        return _tts
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=f"{MODEL_DIR}/model-steps-3.onnx",
                vocoder=f"{MODEL_DIR}/vocos-22khz-univ.onnx",
                lexicon=f"{MODEL_DIR}/lexicon.txt",
                tokens=f"{MODEL_DIR}/tokens.txt",
            ),
            num_threads=2,
        ),
        rule_fsts=f"{MODEL_DIR}/phone.fst,{MODEL_DIR}/date.fst,{MODEL_DIR}/number.fst",
        max_num_sentences=1,
    )
    cfg.validate()
    _tts = sherpa_onnx.OfflineTts(cfg)
    return _tts


async def test():
    print("=" * 60)
    print("  End-to-End Test: Hermes LLM + Sherpa-ONNX TTS")
    print("=" * 60)

    # Init Hermes client
    await hermes_client.__aenter__()
    print(f"\n  Hermes: {config.hermes.base_url}")
    health = await hermes_client.check_health()
    print(f"  Health: {'✓' if health else '✗'}")

    if not health:
        print("  Hermes not available, aborting.")
        return

    # Warmup TTS
    print("  TTS: warming up...", end=" ", flush=True)
    loop = asyncio.get_event_loop()
    tts = _get_tts()
    g = sherpa_onnx.GenerationConfig()
    g.sid = 0; g.speed = 1.0
    await loop.run_in_executor(None, lambda: tts.generate("预热。", g))
    print("OK")

    session_id = "e2e-test-001"
    system = (
        "你是StackChan语音助手。回复口语化中文，不超过50字。"
        "不要使用emoji、颜文字或特殊符号。记住对话内容。"
    )

    questions = [
        "你好，我叫小明，今年25岁。",
        "我叫什么名字？我多大了？",
        "给我推荐一个适合我的周末活动。",
    ]

    for i, q in enumerate(questions):
        print(f"\n  ── 第{i+1}轮 ──")
        print(f"  用户: {q}")

        # LLM (real Hermes)
        t0 = time.perf_counter()
        first_token = False
        full = ""
        async for chunk in hermes_client.chat_stream(
            message=q, session_id=session_id, system_prompt=system,
        ):
            if not first_token:
                print(f"  LLM首token: {1000*(time.perf_counter()-t0):.0f}ms", end="")
                first_token = True
            full += chunk
        llm_ms = 1000 * (time.perf_counter() - t0)
        print(f", 总耗时: {llm_ms:.0f}ms")
        print(f"  LLM回复: \"{full.strip()}\"")
        print(f"  字数: {len(full.strip())}")

        # TTS (real Sherpa-ONNX)
        if full.strip():
            t0 = time.perf_counter()
            result = await loop.run_in_executor(None, lambda: tts.generate(full.strip(), g))
            tts_ms = 1000 * (time.perf_counter() - t0)
            dur = len(result.samples) / result.sample_rate
            print(f"  TTS: {tts_ms:.0f}ms ({dur:.1f}s audio), RTF={tts_ms/1000/dur:.2f}")

    # Session memory check
    print(f"\n  ── 记忆验证 ──")
    session = hermes_client._sessions.get(session_id)
    if session:
        print(f"  消息数: {len(session.messages)}")
        print(f"  Token数: {session._total_tokens}")

    await hermes_client.close()
    print("\n  ✓ 完成")


if __name__ == "__main__":
    asyncio.run(test())

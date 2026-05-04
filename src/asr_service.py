"""Speech Recognition Service using FunASR SenseVoiceSmall

Optimization:
- CPU-bound inference runs in thread pool via run_in_executor
- Avoids blocking the event loop during ASR processing
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from loguru import logger


class ASRService:
    """FunASR SenseVoiceSmall for speech recognition (CPU)"""

    def __init__(self):
        self._model = None
        self._ready = False
        self._executor: Optional[ThreadPoolExecutor] = None

    def set_executor(self, executor: ThreadPoolExecutor):
        """Inject shared thread pool for CPU-bound inference"""
        self._executor = executor

    async def start(self):
        if self._ready:
            return

        from funasr import AutoModel

        logger.info("[ASR] Loading SenseVoiceSmall model...")
        self._model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_pbar=True,
            device="cpu",
        )
        self._ready = True
        logger.info("[ASR] SenseVoiceSmall loaded")

    async def transcribe(self, audio: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM int16 audio bytes to text (runs in thread pool)"""
        if not self._ready:
            await self.start()

        try:
            audio_np = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            )

            loop = asyncio.get_event_loop()

            def _do_asr():
                return self._model.generate(
                    input=audio_np,
                    language="zh",
                    ban_emo_unk=True,
                )

            if self._executor:
                result = await loop.run_in_executor(self._executor, _do_asr)
            else:
                result = await loop.run_in_executor(None, _do_asr)

            text = self._extract_text(result)
            if text:
                logger.info(f"[ASR] Transcribed: {text[:100]}")
            return text

        except Exception as e:
            logger.error(f"[ASR] Transcription error: {e}")
            return ""

    async def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file (mp3, wav, etc.) to text"""
        if not self._ready:
            await self.start()

        import time
        t0 = time.time()

        loop = asyncio.get_event_loop()

        def _do_asr():
            return self._model.generate(
                input=path,
                language="zh",
                ban_emo_unk=True,
            )

        if self._executor:
            result = await loop.run_in_executor(self._executor, _do_asr)
        else:
            result = await loop.run_in_executor(None, _do_asr)

        text = self._extract_text(result)
        elapsed = time.time() - t0
        if text:
            logger.info(f"[ASR] Transcribed: {text[:100]} ({elapsed:.2f}s)")
        return text

    def _extract_text(self, result) -> str:
        import re

        if isinstance(result, list) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict):
                text = item.get("text", "")
            elif isinstance(item, str):
                text = item
            else:
                return ""

            # Strip SenseVoice metadata tags: <|lang|><|emotion|><|event|><|woitn|>
            text = re.sub(r"<\|[^|]+\|>", "", text).strip()
            return text
        return ""


asr_service = ASRService()

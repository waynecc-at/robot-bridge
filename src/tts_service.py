"""Text-to-Speech Service using Sherpa-ONNX Matcha TTS

Optimizations:
- Thread pool offloading: CPU-bound inference runs in executor, not event loop
- Chunked streaming: splits WAV into 200ms chunks, first chunk at 50ms for instant playback
"""
import asyncio
import base64
import io
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Optional
from loguru import logger

import numpy as np
from scipy import signal as scipy_signal

from .config import config


# Default chunk parameters
PREBUFFER_CHUNK_MS = 50    # First chunk: 50ms for fastest playback start
STREAM_CHUNK_MS = 200      # Subsequent chunks: 200ms each


class TTSService:
    """Local TTS service using Sherpa-ONNX Matcha model"""

    def __init__(self):
        self._tts = None
        self._sample_rate = 22050
        self._ready = False
        self._executor: Optional[ThreadPoolExecutor] = None

    def set_executor(self, executor: ThreadPoolExecutor):
        """Inject shared thread pool for CPU-bound inference"""
        self._executor = executor

    async def start(self):
        """Initialize the Sherpa-ONNX TTS engine"""
        import sherpa_onnx

        model_dir = Path(config.tts.model_dir)
        acoustic_model = str(model_dir / config.tts.acoustic_model)
        vocoder = str(model_dir / config.tts.vocoder)
        tokens = str(model_dir / config.tts.tokens)
        lexicon = str(model_dir / config.tts.lexicon) if config.tts.lexicon else ""
        data_dir = str(model_dir / config.tts.data_dir) if config.tts.data_dir else ""

        # Rule FSTs for numbers, dates, phones
        rule_fsts = ""
        fst_files = []
        for fst_name in config.tts.rule_fsts:
            fst_path = model_dir / fst_name
            if fst_path.exists():
                fst_files.append(str(fst_path))
        if fst_files:
            rule_fsts = ",".join(fst_files)

        logger.info(f"[TTS] Initializing Sherpa-ONNX Matcha TTS...")
        logger.info(f"[TTS]   acoustic_model: {acoustic_model}")
        logger.info(f"[TTS]   vocoder: {vocoder}")
        logger.info(f"[TTS]   tokens: {tokens}")
        logger.info(f"[TTS]   lexicon: {lexicon}")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=acoustic_model,
                    vocoder=vocoder,
                    lexicon=lexicon,
                    tokens=tokens,
                    data_dir=data_dir,
                    noise_scale=config.tts.noise_scale,
                    length_scale=config.tts.length_scale,
                ),
                num_threads=config.tts.num_threads,
                provider="cpu",
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=1,
        )

        if not tts_config.validate():
            raise RuntimeError("Invalid TTS config")

        self._tts = sherpa_onnx.OfflineTts(tts_config)
        self._sample_rate = self._tts.sample_rate
        self._ready = True
        logger.info(f"[TTS] Sherpa-ONNX TTS ready (sample_rate={self._sample_rate})")

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> bytes:
        """Synthesize text to audio, return WAV bytes"""
        if not self._ready:
            raise RuntimeError("TTS service not initialized")

        logger.info(f"[TTS] Synthesizing: {text[:50]}...")

        audio = await self._generate(text)
        wav_bytes = self._samples_to_wav(audio.samples, audio.sample_rate)

        logger.info(f"[TTS] Generated {len(wav_bytes)} bytes of WAV audio "
                    f"(duration: {len(audio.samples)/audio.sample_rate:.2f}s)")
        return wav_bytes

    async def synthesize_pcm(self, text: str) -> bytes:
        """Synthesize text directly to 16kHz int16 PCM.

        Pipeline: float32[22050] → int16 PCM → ffmpeg soxr → int16 PCM[16000]
        Uses ffmpeg soxr for high-quality anti-aliased resampling (lower noise than FFT).
        """
        if not self._ready:
            raise RuntimeError("TTS service not initialized")

        audio = await self._generate(text)
        pcm_bytes = await self._resample_via_ffmpeg(audio.samples, audio.sample_rate)
        return pcm_bytes

    @staticmethod
    async def _resample_via_ffmpeg(samples, src_rate: int, target_rate: int = 16000) -> bytes:
        """Resample via ffmpeg soxr — float32 PCM pipe, resample, int16 PCM out."""
        # Feed float32 PCM directly to ffmpeg (no pre-quantization)
        arr = np.array(samples, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        pcm_in = arr.astype(np.float32).tobytes()

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-f", "f32le", "-ar", str(src_rate), "-ac", "1", "-i", "pipe:0",
            "-resampler", "soxr",
            "-ar", str(target_rate), "-ac", "1",
            "-sample_fmt", "s16", "-f", "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate(pcm_in)
        if proc.returncode == 0:
            return stdout
        # Fallback: if ffmpeg fails, use scipy as backup
        logger.warning("[TTS] ffmpeg soxr unavailable, falling back to scipy")
        from scipy import signal as _scipy_signal
        num = int(len(samples) * target_rate / src_rate)
        fallback = _scipy_signal.resample(arr, num)
        fallback = np.clip(fallback, -1.0, 1.0)
        return (fallback * 32767).astype(np.int16).tobytes()

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio, yield chunked WAV for streaming playback.

        First chunk is PREBUFFER_CHUNK_MS (50ms) for instant playback start.
        Subsequent chunks are STREAM_CHUNK_MS (200ms) each.
        CPU-bound inference runs in thread pool to avoid blocking event loop.
        """
        if not self._ready:
            raise RuntimeError("TTS service not initialized")

        logger.info(f"[TTS] Streaming synthesis: {text[:50]}...")

        # Run CPU-bound inference in thread pool (not event loop)
        audio = await self._generate(text)
        sr = audio.sample_rate
        samples = audio.samples

        total_duration = len(samples) / sr
        logger.info(f"[TTS] Generated {len(samples)} samples ({total_duration:.2f}s)")

        # Build full WAV once, then yield chunked views into it
        wav_bytes = self._samples_to_wav(samples, sr)
        wav_data = wav_bytes[44:]  # Skip WAV header (44 bytes), keep PCM data
        wav_sr = sr
        bytes_per_sample = 2  # 16-bit PCM

        # Chunk size calculations
        prebuffer_bytes = int(wav_sr * (PREBUFFER_CHUNK_MS / 1000) * bytes_per_sample)
        chunk_bytes = int(wav_sr * (STREAM_CHUNK_MS / 1000) * bytes_per_sample)

        # Rebuild minimal WAV header for chunks
        def _make_wav_chunk(pcm_chunk: bytes) -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(wav_sr)
                wf.writeframes(pcm_chunk)
            return buf.getvalue()

        # First chunk: smallest viable for fastest playback start
        first_pcm = wav_data[:prebuffer_bytes]
        if first_pcm:
            yield _make_wav_chunk(first_pcm)

        # Remaining chunks
        offset = prebuffer_bytes
        while offset < len(wav_data):
            chunk_pcm = wav_data[offset:offset + chunk_bytes]
            if not chunk_pcm:
                break
            yield _make_wav_chunk(chunk_pcm)
            offset += chunk_bytes

        logger.info(f"[TTS] Streamed {len(wav_bytes)} bytes in chunks "
                    f"(first={PREBUFFER_CHUNK_MS}ms, chunk={STREAM_CHUNK_MS}ms)")

    async def synthesize_to_base64(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> str:
        """Synthesize and return base64 encoded audio"""
        audio = await self.synthesize(text, voice)
        return base64.b64encode(audio).decode("utf-8")

    async def list_voices(self, language: Optional[str] = None) -> list:
        """List available voices (local TTS has one voice)"""
        return [{
            "Name": "zh-CN-Baker-Matcha",
            "ShortName": "baker",
            "Locale": "zh-CN",
            "Gender": "Female",
        }]

    async def _generate(self, text: str):
        """Generate audio samples in thread pool to avoid blocking event loop"""
        import sherpa_onnx
        loop = asyncio.get_event_loop()
        executor = self._executor

        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = 0
        gen_config.speed = config.tts.speed

        def _do_generate():
            return self._tts.generate(text, gen_config)

        if executor:
            return await loop.run_in_executor(executor, _do_generate)
        else:
            return await loop.run_in_executor(None, _do_generate)

    @staticmethod
    def _samples_to_wav(samples: list, sample_rate: int) -> bytes:
        """Convert float32 samples to WAV (16-bit PCM) bytes"""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # Convert float32 [-1,1] to int16
            pcm = []
            for s in samples:
                s = max(-1.0, min(1.0, s))
                pcm.append(int(s * 32767))
            wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
        return buf.getvalue()


# Global TTS service instance
tts_service = TTSService()

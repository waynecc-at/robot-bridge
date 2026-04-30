"""Text-to-Speech Service using Sherpa-ONNX Matcha TTS"""
import io
import struct
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional
from loguru import logger

from .config import config


class TTSService:
    """Local TTS service using Sherpa-ONNX Matcha model"""

    def __init__(self):
        self._tts = None
        self._sample_rate = 22050
        self._ready = False

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

        audio = self._generate(text)
        wav_bytes = self._samples_to_wav(audio.samples, audio.sample_rate)

        logger.info(f"[TTS] Generated {len(wav_bytes)} bytes of WAV audio "
                    f"(duration: {len(audio.samples)/audio.sample_rate:.2f}s)")
        return wav_bytes

    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio, yield complete WAV as one chunk"""
        if not self._ready:
            raise RuntimeError("TTS service not initialized")

        logger.info(f"[TTS] Streaming synthesis: {text[:50]}...")

        audio = self._generate(text)
        wav_bytes = self._samples_to_wav(audio.samples, audio.sample_rate)

        yield wav_bytes

        logger.info(f"[TTS] Yielded {len(wav_bytes)} bytes of WAV audio")

    async def synthesize_to_base64(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> str:
        """Synthesize and return base64 encoded audio"""
        import base64
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

    def _generate(self, text: str):
        """Generate audio samples for text"""
        gen_config = __import__("sherpa_onnx").GenerationConfig()
        gen_config.sid = 0
        gen_config.speed = config.tts.speed
        return self._tts.generate(text, gen_config)

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

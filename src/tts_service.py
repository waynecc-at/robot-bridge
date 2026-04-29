"""Text-to-Speech Service using Edge TTS"""
import asyncio
import base64
import io
from typing import AsyncGenerator, Optional
from loguru import logger
import edge_tts

from .config import config


class TTSService:
    """Edge TTS streaming service"""
    
    def __init__(self):
        self.voice = config.tts.voice
        self.rate = config.tts.rate
        self.volume = config.tts.volume
        self.output_format = config.tts.output_format
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> bytes:
        """Synthesize text to audio, return full audio bytes"""
        voice = voice or self.voice
        
        logger.info(f"[TTS] Synthesizing: {text[:50]}... with voice {voice}")
        
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
        )
        
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        
        logger.info(f"[TTS] Generated {len(audio_data)} bytes of audio")
        return audio_data
    
    async def synthesize_stream(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize text to audio, yield chunks as they become available"""
        voice = voice or self.voice
        
        logger.info(f"[TTS] Streaming synthesis: {text[:50]}...")
        
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
        )
        
        chunk_count = 0
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunk_count += 1
                yield chunk["data"]
        
        logger.info(f"[TTS] Streamed {chunk_count} audio chunks")
    
    async def synthesize_to_base64(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> str:
        """Synthesize and return base64 encoded audio"""
        audio = await self.synthesize(text, voice)
        return base64.b64encode(audio).decode("utf-8")
    
    async def list_voices(self, language: Optional[str] = None) -> list:
        """List available voices"""
        voices = await edge_tts.list_voices()
        
        if language:
            voices = [v for v in voices if v["Locale"].startswith(language)]
        
        return voices
    
    async def preview_voice(self, text: str, voice: str) -> bytes:
        """Preview a specific voice"""
        return await self.synthesize(text, voice)


# Global TTS service instance
tts_service = TTSService()

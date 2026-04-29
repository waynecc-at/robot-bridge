"""Hermes Gateway API Client"""
import asyncio
import json
import uuid
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field
from loguru import logger
import httpx

from .config import config


@dataclass
class ChatMessage:
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    text: str
    session_id: str
    audio_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class HermesClient:
    """Client for Hermes Gateway API"""
    
    def __init__(self):
        self.base_url = config.hermes.base_url
        self.api_key = config.hermes.api_key
        self.timeout = config.hermes.timeout
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        """Send chat message and get response"""
        if not self._client:
            await self.__aenter__()
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Build messages array
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user", 
            "content": message
        })
        
        payload = {
            "messages": messages,
            "session_id": session_id,
            "stream": False,
        }
        
        logger.info(f"[Hermes] Sending chat request, session: {session_id}")
        
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            # Parse OpenAI format response
            text = ""
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice:
                    text = choice["message"].get("content", "")
                elif "delta" in choice:
                    text = choice["delta"].get("content", "")

            logger.info(f"[Hermes] Got response: {text[:100]}...")

            return ChatResponse(
                text=text,
                session_id=session_id,
                metadata=data
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[Hermes] HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[Hermes] Error: {e}")
            raise
    
    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Send chat message and get streaming response"""
        if not self._client:
            await self.__aenter__()
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "messages": messages,
            "session_id": session_id,
            "stream": True,
        }
        
        logger.info(f"[Hermes] Starting stream chat, session: {session_id}")
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                response.raise_for_status()
                
                full_text = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "text" in data:
                                chunk = data["text"]
                                full_text += chunk
                                yield chunk
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"[Hermes] Stream complete, total: {len(full_text)} chars")
                
        except Exception as e:
            logger.error(f"[Hermes] Stream error: {e}")
            raise
    
    async def get_session_history(self, session_id: str) -> list[ChatMessage]:
        """Get chat history for a session"""
        if not self._client:
            await self.__aenter__()
        
        try:
            response = await self._client.get(
                f"{self.base_url}/v1/session/{session_id}/history"
            )
            response.raise_for_status()
            data = response.json()
            
            messages = []
            for msg in data.get("messages", []):
                messages.append(ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                ))
            
            return messages
            
        except Exception as e:
            logger.error(f"[Hermes] Failed to get history: {e}")
            return []
    
    async def check_health(self) -> bool:
        """Check if Hermes Gateway is healthy"""
        if not self._client:
            await self.__aenter__()
        
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

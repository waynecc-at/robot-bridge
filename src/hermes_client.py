"""Hermes Gateway API Client with client-side context management.

Hermes is a stateless LLM proxy — it does NOT maintain conversation context
server-side. The `session_id` field in the API payload is informational only.
This client maintains the full message history per session and sends it with
every request.

Optimizations v0.2:
- Deferred compression: marks sessions for compression instead of blocking
- Background idle compressor: compresses during idle periods
- Connection pooling with keepalive: httpx.AsyncClient limits configured
- Token estimation: tiktoken fallback with improved heuristic
"""
import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Optional
from dataclasses import dataclass, field
from loguru import logger
import httpx

from .config import config


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    text: str
    session_id: str
    audio_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class SessionStore:
    """Per-session conversation history with deferred compression (non-blocking)."""

    MAX_TOKENS = 6000          # conservative ceiling for 8k context
    COMPRESS_AT = 0.65         # trigger compression mark at 65% full
    KEEP_RECENT = 6            # keep last N messages uncompressed after compression
    IDLE_INTERVAL = 30         # seconds between idle compression sweeps

    def __init__(self, system_prompt: str = ""):
        self.session_id = str(uuid.uuid4())
        self.system_prompt = system_prompt
        self.messages: list[dict] = []
        self.summary: str = ""
        self._total_tokens: int = 0
        self._compress_count: int = 0
        self._pending_compression = False  # marked for background compression

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Token estimation with optional tiktoken, fallback to improved heuristic.

        tiktoken gives exact counts for ~3x better accuracy.
        Fallback: Chinese chars ~2 tokens, ASCII ~0.4 tokens.
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            chinese = sum(1 for c in text if '一' <= c <= '鿿')
            ascii_chars = len(text) - chinese
            return max(1, int(chinese * 2 + ascii_chars * 0.4))

    def add(self, user_text: str, assistant_text: str):
        """Add a complete exchange (user + assistant)."""
        self._add_message("user", user_text)
        self._add_message("assistant", assistant_text)

    def _add_message(self, role: str, text: str):
        self.messages.append({"role": role, "content": text})
        self._total_tokens += self._estimate_tokens(text)

    def needs_compression(self) -> bool:
        threshold = int(self.MAX_TOKENS * self.COMPRESS_AT)
        return self._total_tokens > threshold and len(self.messages) > self.KEEP_RECENT

    def mark_for_compression(self):
        """Non-blocking: mark session for background compression."""
        if self.needs_compression() and not self._pending_compression:
            self._pending_compression = True
            logger.info(f"[Hermes] Session {self.session_id[:8]} marked for compression "
                        f"(tokens={self._total_tokens})")

    def build_messages(self) -> list[dict]:
        """Build the full messages array to send to Hermes."""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        if self.summary:
            msgs.append({"role": "system", "content": f"[历史摘要] {self.summary}"})
        msgs.extend(self.messages)
        return msgs

    async def _do_compress(self, hermes_client: "HermesClient") -> bool:
        """Actually execute compression (called by background task, not response path)."""
        if not self._pending_compression:
            return False
        if not self.needs_compression():
            self._pending_compression = False
            return False

        t0 = time.perf_counter()
        recent = self.messages[-self.KEEP_RECENT:]
        old = self.messages[: -self.KEEP_RECENT]
        if not old:
            self._pending_compression = False
            return False

        history_text = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in old
        )
        compress_prompt = (
            "请用简短中文总结以下对话的关键信息（人名、偏好、重要事实），不超过100字。"
            f"只输出总结，不要其他内容。\n\n{history_text}"
        )

        try:
            response = await hermes_client._raw_chat(
                messages=[
                    {"role": "system", "content": "你是对话摘要助手，输出简洁摘要。"},
                    {"role": "user", "content": compress_prompt},
                ],
                stream=False,
            )
            if response.text.strip():
                self.summary = response.text.strip()
                self.messages = recent
                self._total_tokens = sum(
                    self._estimate_tokens(m.get("content", "")) for m in self.messages
                )
                self._compress_count += 1
                elapsed = 1000 * (time.perf_counter() - t0)
                logger.info(
                    f"[Hermes] Context compressed in {elapsed:.0f}ms "
                    f"(count={self._compress_count}, summary_len={len(self.summary)})"
                )
                return True
        except Exception as e:
            logger.warning(f"[Hermes] Compression failed: {e}")

        self._pending_compression = False
        return False


class HermesClient:
    """Client for Hermes Gateway API with client-side context management.

    Connection pool is managed as a long-lived httpx.AsyncClient with keepalive.
    """

    def __init__(self):
        self.base_url = config.hermes.base_url
        self.api_key = config.hermes.api_key
        self.timeout = config.hermes.timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._sessions: dict[str, SessionStore] = {}
        self._idle_compressor_task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            # Connection pooling with keepalive
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self.timeout,
                    connect=5.0,
                    pool=None,
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                ),
                headers=headers,
            )
            logger.info("[Hermes] Connection pool created (keepalive=5, max=10)")

            # Start background idle compressor
            self._idle_compressor_task = asyncio.create_task(
                self._idle_compressor_loop()
            )
            logger.info("[Hermes] Idle compressor background task started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def close(self):
        """Clean shutdown: cancel compressor, close connection pool."""
        if self._idle_compressor_task:
            self._idle_compressor_task.cancel()
            self._idle_compressor_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("[Hermes] Connection pool closed")

    async def _idle_compressor_loop(self):
        """Background task: compress sessions during idle periods."""
        while True:
            await asyncio.sleep(SessionStore.IDLE_INTERVAL)
            for session_id, store in list(self._sessions.items()):
                if store._pending_compression:
                    logger.info(
                        f"[Hermes] Idle compressor processing session {session_id[:8]}"
                    )
                    try:
                        await store._do_compress(self)
                    except Exception as e:
                        logger.warning(
                            f"[Hermes] Idle compress failed for {session_id[:8]}: {e}"
                        )

    def get_session(self, session_id: Optional[str] = None, system_prompt: str = "") -> SessionStore:
        """Get or create a client-side session store."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        store = SessionStore(system_prompt)
        key = session_id or store.session_id
        self._sessions[key] = store
        return store

    # ── public API ────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        session = self.get_session(session_id, system_prompt or "")

        # Deferred compression: mark for background, don't wait
        session.mark_for_compression()

        # Add user message only (assistant added after response)
        session._add_message("user", message)

        logger.info(
            f"[Hermes] Chat request session={session.session_id[:8]} "
            f"msgs={len(session.messages)} tokens={session._total_tokens}"
        )

        response = await self._raw_chat(
            messages=session.build_messages(),
            stream=False,
        )

        # Record assistant response
        session._add_message("assistant", response.text)

        response.session_id = session.session_id
        return response

    async def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        session = self.get_session(session_id, system_prompt or "")

        # Deferred compression: mark for background, don't wait
        session.mark_for_compression()

        # Add user message (assistant added after streaming completes)
        session._add_message("user", message)

        logger.info(
            f"[Hermes] Stream request session={session.session_id[:8]} "
            f"msgs={len(session.messages)} tokens={session._total_tokens}"
        )

        full_text = ""
        async for chunk in self._raw_chat_stream(
            messages=session.build_messages(),
            session_id=session.session_id,
        ):
            full_text += chunk
            yield chunk

        # Record assistant response
        session._add_message("assistant", full_text)
        logger.info(f"[Hermes] Stream complete: {len(full_text)} chars")

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    # ── internal ──────────────────────────────────────────────────

    async def _raw_chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> ChatResponse:
        payload = {"messages": messages, "stream": stream, "model": config.hermes.model}

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        text = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice:
                text = choice["message"].get("content", "")
            elif "delta" in choice:
                text = choice["delta"].get("content", "")

        return ChatResponse(text=text, session_id="", metadata=data)

    async def _raw_chat_stream(
        self,
        messages: list[dict],
        session_id: str = "",
    ) -> AsyncGenerator[str, None]:
        payload = {"messages": messages, "stream": True, "model": config.hermes.model}

        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"[Hermes] Stream HTTP {response.status_code}: {body[:500]}")
                response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            chunk = delta.get("content", "")
                            if chunk:
                                yield chunk
                    except json.JSONDecodeError:
                        continue


hermes_client = HermesClient()
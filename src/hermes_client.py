"""Hermes Gateway API Client with optional server-side session management.

When API key is configured (HERMES_API_KEY):
- Uses X-Hermes-Session-Id header for server-side context management
- Sends only system_prompt + current user message (not full history)
- Hermes loads conversation history from its own state.db (SQLite)
- Tracks real token usage from API response vs. client estimation

Fallback (no API key):
- Client-side SessionStore maintains full message history
- Sends complete history with every request (legacy mode)

Hermes also handles session lifecycle: max 90 turns per session, auto-reset
after 24h idle, no unbounded growth.
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
from .metrics import metrics


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

    _TTFT_TIMEOUT = 15.0    # seconds; abort if no first token within this
    _MAX_RETRIES = 1        # retry on TTFT timeout (total attempts = _MAX_RETRIES + 1)

    def __init__(self):
        self.base_url = config.hermes.base_url
        self.api_key = config.hermes.api_key
        self.timeout = config.hermes.timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._sessions: dict[str, SessionStore] = {}
        self._idle_compressor_task: Optional[asyncio.Task] = None
        self._warmed_up: bool = False
        self._last_usage: dict = {}
        self._force_stateless: bool = False  # test flag: use full history even with api_key

        # TTFT tracking & adaptive timeout
        self._ttft_window: list[float] = []
        self._last_request_time: float = 0
        self._model_warm_at: float = 0
        self._warmup_lock = asyncio.Lock()

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

            # Fire-and-forget LLM warmup: reduce cold-start TTFT
            asyncio.create_task(self._warmup())
            logger.info("[Hermes] LLM warmup fired (background)")
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

    # ── warmup ────────────────────────────────────────────────

    async def _warmup(self):
        """Fire a quick request to eliminate cold-start delay.

        DeepSeek (cloud) shows ~11s first-token cold start dropping to ~2-5s.
        One brief warmup request triggered fire-and-forget on startup.
        """
        if self._warmed_up:
            return
        try:
            logger.info("[Hermes] Warming up LLM connection...")
            await self._raw_chat(
                messages=[{"role": "user", "content": "你好"}],
                stream=False,
            )
            self._warmed_up = True
            self._model_warm_at = time.time()
            logger.info("[Hermes] LLM warmup complete")
        except Exception as e:
            logger.warning(f"[Hermes] Warmup failed (non-fatal): {e}")

    @property
    def _use_server_session(self) -> bool:
        """Server-side session requires API key configured in Hermes."""
        return bool(self.api_key) and not self._force_stateless

    # ── public API ────────────────────────────────────────────────

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ChatResponse:
        session = self.get_session(session_id, system_prompt or "")

        # Build messages based on session mode
        if self._use_server_session:
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": message})
            extra_headers = {"X-Hermes-Session-Id": session.session_id}
            session._add_message("user", message)
            log_detail = f"server_session={session.session_id[:8]}"
        else:
            session.mark_for_compression()
            session._add_message("user", message)
            msgs = session.build_messages()
            extra_headers = {}
            log_detail = f"msgs={len(session.messages)} tokens={session._total_tokens}"

        logger.info(f"[Hermes] Chat request {log_detail}")

        response = await self._raw_chat(
            messages=msgs,
            stream=False,
            extra_headers=extra_headers,
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

        # Build the messages payload based on session mode
        if self._use_server_session:
            # Server-side session: send only system + current user message.
            # Hermes loads conversation history from its own state.db.
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": message})
            extra_headers = {"X-Hermes-Session-Id": session.session_id}
            # Local tracking only (Hermes manages the real context)
            session._add_message("user", message)
            log_detail = f"server_session={session.session_id[:8]}"
        else:
            # Client-side: send full history (legacy mode)
            session._add_message("user", message)
            msgs = session.build_messages()
            extra_headers = {}
            log_detail = f"msgs={len(session.messages)} tokens={session._total_tokens}"

        logger.info(f"[Hermes] Stream request {log_detail}")

        metrics.hermes_requests += 1

        full_text = ""
        last_err = None
        for attempt in range(self._MAX_RETRIES + 1):
            _request_start = time.perf_counter()
            _first_yield = True
            try:
                async for chunk in self._raw_chat_stream(
                    messages=msgs,
                    session_id=session.session_id,
                    extra_headers=extra_headers,
                ):
                    if _first_yield:
                        _first_yield = False
                        ttft = time.perf_counter() - _request_start
                        self._ttft_window.append(ttft)
                        metrics.record_ttft(ttft)
                        if len(self._ttft_window) > 20:
                            self._ttft_window = self._ttft_window[-20:]
                    full_text += chunk
                    yield chunk
                break  # success
            except (asyncio.TimeoutError, httpx.TimeoutException) as e:
                last_err = e
                logger.warning(f"[Hermes] Stream timeout on attempt {attempt+1}/{self._MAX_RETRIES+1}")
                metrics.hermes_errors += 1
                metrics.hermes_retries += 1
                full_text = ""  # reset partial text
                continue
        else:
            # All attempts exhausted
            logger.error(f"[Hermes] All {self._MAX_RETRIES+1} stream attempts failed")
            # Record partial response if we got anything (shouldn't happen)
            session._add_message("assistant", full_text)
            raise last_err or RuntimeError("Stream failed after retries")

        # Record assistant response (local tracking)
        session._add_message("assistant", full_text)

        # Update token tracking with real usage from API response
        if self._last_usage:
            total = self._last_usage.get("total_tokens", 0)
            if total:
                session._total_tokens = total

        logger.info(f"[Hermes] Stream complete: {len(full_text)} chars, "
                     f"usage={self._last_usage}")

    async def check_health(self) -> bool:
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def _get_adaptive_ttft_timeout(self) -> float:
        """Return P80-based adaptive timeout from recent TTFT window.

        Floor at 8s (covers cold start), cap at _TTFT_TIMEOUT.
        Falls back to _TTFT_TIMEOUT when window has < 3 samples.
        """
        if len(self._ttft_window) < 3:
            return self._TTFT_TIMEOUT
        sorted_win = sorted(self._ttft_window)
        idx = int(len(sorted_win) * 0.8)
        p80 = sorted_win[idx]
        return max(8.0, min(p80 * 1.5, self._TTFT_TIMEOUT))

    async def ensure_warm(self):
        """Pre-warm LLM if idle for >60s (VAD-triggered pre-warm).

        During VAD buffering (user speaking), fires a quick request
        so the model is warm by the time ASR completes and LLM starts.
        Debounced via _warmup_lock.
        """
        if not self._client:
            return
        idle = time.time() - self._model_warm_at
        if idle < 60:
            return
        async with self._warmup_lock:
            idle = time.time() - self._model_warm_at
            if idle < 60:
                return
            try:
                logger.info(f"[Hermes] Pre-warming LLM (idle={idle:.0f}s)...")
                await self._raw_chat(
                    messages=[{"role": "user", "content": "你好"}],
                    stream=False,
                )
                self._model_warm_at = time.time()
                logger.info("[Hermes] Pre-warm complete")
            except Exception as e:
                logger.warning(f"[Hermes] Pre-warm failed (non-fatal): {e}")

    # ── internal ──────────────────────────────────────────────────

    async def _raw_chat(
        self,
        messages: list[dict],
        stream: bool = False,
        extra_headers: Optional[dict] = None,
    ) -> ChatResponse:
        self._last_usage = {}
        payload = {"messages": messages, "stream": stream, "model": config.hermes.model}

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=extra_headers or {},
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

        if "usage" in data:
            self._last_usage = data["usage"]

        return ChatResponse(text=text, session_id="", metadata=data)

    async def _raw_chat_stream(
        self,
        messages: list[dict],
        session_id: str = "",
        extra_headers: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        self._last_usage = {}
        payload = {"messages": messages, "stream": True, "model": config.hermes.model}

        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=extra_headers or {},
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                logger.error(f"[Hermes] Stream HTTP {response.status_code}: {body[:500]}")
                response.raise_for_status()

            ait = response.aiter_lines().__aiter__()
            last_data = None

            # First line with adaptive TTFT timeout — prevents cold-start death spiral
            timeout = self._get_adaptive_ttft_timeout()
            try:
                first_line = await asyncio.wait_for(
                    ait.__anext__(), timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"[Hermes] TTFT timeout ({timeout:.1f}s, "
                             f"window_size={len(self._ttft_window)}) — "
                             f"no data received")
                raise

            # Process first line
            if first_line.startswith("data: "):
                data_str = first_line[6:]
                if data_str == "[DONE]":
                    return
                last_data = data_str
                try:
                    data = json.loads(data_str)
                    for chunk in self._yield_delta_chunks(data):
                        if chunk:
                            yield chunk
                except json.JSONDecodeError:
                    pass
            elif first_line.startswith(":"):
                pass  # SSE comment (keepalive from server)

            # Process remaining lines
            async for line in ait:
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    last_data = data_str  # Track for usage extraction
                    try:
                        data = json.loads(data_str)
                        for chunk in self._yield_delta_chunks(data):
                            if chunk:
                                yield chunk
                    except json.JSONDecodeError:
                        continue

            # Parse usage from last non-DONE chunk
            if last_data:
                try:
                    data = json.loads(last_data)
                    if "usage" in data:
                        self._last_usage = data["usage"]
                except json.JSONDecodeError:
                    pass

            # Track model freshness for pre-warm decisions
            self._model_warm_at = time.time()
            self._last_request_time = time.time()

    @staticmethod
    def _yield_delta_chunks(data: dict):
        """Extract content chunks from a streamed SSE data dict."""
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            yield delta.get("content", "")


hermes_client = HermesClient()
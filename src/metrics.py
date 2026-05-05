"""Lightweight in-memory metrics for observability.

Counters + sliding-window latency histograms.
Exposed as Prometheus-style text at /api/metrics.
"""
import time
from collections import deque


class MetricsCollector:
    """Thread-safe counters and sliding windows for pipeline observability."""

    def __init__(self, window_size: int = 200):
        # Counters
        self.hermes_requests = 0
        self.hermes_errors = 0
        self.hermes_retries = 0
        self.tts_requests = 0
        self.tts_errors = 0
        self.tts_audio_bytes = 0
        self.turns_total = 0
        self.turns_cancelled = 0
        self.turns_errors = 0
        self.ws_messages: dict[str, int] = {}
        self._start_time = time.time()

        # Sliding windows (maxlen keeps bounded memory)
        self._ttft: deque[float] = deque(maxlen=window_size)
        self._rtf: deque[float] = deque(maxlen=window_size)

    # ── Records ────────────────────────────────────────────────

    def record_ttft(self, seconds: float):
        self._ttft.append(seconds)

    def record_rtf(self, value: float):
        self._rtf.append(value)

    def record_ws_message(self, msg_type: str):
        self.ws_messages[msg_type] = self.ws_messages.get(msg_type, 0) + 1

    def reset(self):
        """Zero all counters and clear sliding windows between test rounds."""
        self.hermes_requests = 0
        self.hermes_errors = 0
        self.hermes_retries = 0
        self.tts_requests = 0
        self.tts_errors = 0
        self.tts_audio_bytes = 0
        self.turns_total = 0
        self.turns_cancelled = 0
        self.turns_errors = 0
        self.ws_messages.clear()
        self._ttft.clear()
        self._rtf.clear()
        self._start_time = time.time()

    # ── Format ─────────────────────────────────────────────────

    def json(self) -> dict:
        return {
            "uptime_s": time.time() - self._start_time,
            "hermes": {
                "requests": self.hermes_requests,
                "errors": self.hermes_errors,
                "retries": self.hermes_retries,
            },
            "tts": {
                "requests": self.tts_requests,
                "errors": self.tts_errors,
                "audio_bytes": self.tts_audio_bytes,
            },
            "turns": {
                "total": self.turns_total,
                "cancelled": self.turns_cancelled,
                "errors": self.turns_errors,
            },
            "ttft": self._window_stats(self._ttft),
            "rtf": self._window_stats(self._rtf),
            "ws_messages": dict(self.ws_messages),
        }

    def prometheus(self) -> str:
        j = self.json()
        lines = [
            '# HELP robot_bridge_uptime_seconds Server uptime',
            '# TYPE robot_bridge_uptime_seconds gauge',
            f'robot_bridge_uptime_seconds {j["uptime_s"]}',
            '',
            '# HELP robot_bridge_hermes_requests_total Total Hermes API requests',
            '# TYPE robot_bridge_hermes_requests_total counter',
            f'robot_bridge_hermes_requests_total {j["hermes"]["requests"]}',
            f'robot_bridge_hermes_errors_total {j["hermes"]["errors"]}',
            f'robot_bridge_hermes_retries_total {j["hermes"]["retries"]}',
            '',
            '# HELP robot_bridge_tts_requests_total Total TTS synthesis requests',
            '# TYPE robot_bridge_tts_requests_total counter',
            f'robot_bridge_tts_requests_total {j["tts"]["requests"]}',
            f'robot_bridge_tts_errors_total {j["tts"]["errors"]}',
            f'robot_bridge_tts_audio_bytes_total {j["tts"]["audio_bytes"]}',
            '',
            '# HELP robot_bridge_turns_total Total conversation turns',
            '# TYPE robot_bridge_turns_total counter',
            f'robot_bridge_turns_total {j["turns"]["total"]}',
            f'robot_bridge_turns_cancelled_total {j["turns"]["cancelled"]}',
            f'robot_bridge_turns_errors_total {j["turns"]["errors"]}',
            '',
            '# HELP robot_bridge_ttft_seconds Time to first token',
            '# TYPE robot_bridge_ttft_seconds gauge',
        ]
        stats = j["ttft"]
        for p in ("p50", "p95", "p99"):
            lines.append(f'robot_bridge_ttft_seconds{{quantile="{p}"}} {stats[p]}')
        lines.append(f'robot_bridge_ttft_samples {stats["samples"]}')

        lines.extend([
            '',
            '# HELP robot_bridge_rtf Real-time factor (TTS)',
            '# TYPE robot_bridge_rtf gauge',
        ])
        stats = j["rtf"]
        for p in ("p50", "p95", "p99"):
            lines.append(f'robot_bridge_rtf{{quantile="{p}"}} {stats[p]}')
        lines.append(f'robot_bridge_rtf_samples {stats["samples"]}')

        for msg_type, count in j["ws_messages"].items():
            lines.append(
                f'robot_bridge_ws_messages_total{{type="{msg_type}"}} {count}'
            )

        return "\n".join(lines) + "\n"

    @staticmethod
    def _window_stats(dq: deque) -> dict:
        if not dq:
            return {"p50": 0, "p95": 0, "p99": 0, "samples": 0}
        vals = sorted(dq)
        n = len(vals)
        return {
            "p50": vals[int(n * 0.50)],
            "p95": vals[int(n * 0.95)],
            "p99": vals[int(n * 0.99)],
            "samples": n,
        }


metrics = MetricsCollector()

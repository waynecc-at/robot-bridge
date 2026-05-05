"""Vision service: face detection, recognition, and person tracking.

Processes JPEG frames from StackChan (ESP32 camera), detects and recognizes
faces, and manages person tracking across video frames. When a person is
identified, triggers a callback to update the LLM context.

OpenCV is optional. Without it, the service is a no-op.
"""
import asyncio
import time
from loguru import logger

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    cv2 = None
    np = None

from .config import config
from .profile_store import profile_store


class VisionService:
    """Face detection/recognition pipeline receiving JPEG frames from ESP32.

    Decodes JPEG → detect faces → recognize → returns results.
    Person tracking is handled by the caller (WebSocket handler),
    which maintains multi-user session state.
    """

    def __init__(self):
        self._enabled = config.vision.enabled and HAVE_CV2
        self._last_process_time: float = 0
        self._frame_count: int = 0

        if not HAVE_CV2 and config.vision.enabled:
            logger.warning("[Vision] opencv-python not installed — vision disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val and HAVE_CV2

    async def process_frame(self, jpeg_bytes: bytes) -> list[dict]:
        """Decode JPEG, detect, recognize. Returns [{name, bbox, profile}]."""
        if not self._enabled:
            return []

        now = time.monotonic()
        if now - self._last_process_time < config.vision.detection_interval:
            return []
        self._last_process_time = now

        frame = await asyncio.get_event_loop().run_in_executor(
            None, self._decode_jpeg, jpeg_bytes,
        )
        if frame is None:
            return []

        self._frame_count += 1

        results = await asyncio.get_event_loop().run_in_executor(
            None, profile_store.detect_and_recognize, frame,
        )
        return results

    @staticmethod
    def _decode_jpeg(jpeg_bytes: bytes):
        """Decode JPEG bytes to OpenCV BGR frame."""
        if not HAVE_CV2:
            return None
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"[Vision] JPEG decode failed: {e}")
            return None

vision_service = VisionService()

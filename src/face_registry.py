"""Unknown face tracker — collects crops and signals the LLM.

Thin module. Does NOT handle conversation or registration logic.
Those belong to Hermes Gateway (LLM). The bridge just:
1. Notices when the same unknown person appears consistently
2. Collects face crops for later LBPH training
3. Exposes a flag so the LLM can be prompted to introduce itself

The LLM (via Hermes) handles the rest: natural conversation,
name/relationship extraction, welcome message.
"""
import time
from dataclasses import dataclass, field
from loguru import logger

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


@dataclass
class _Track:
    first_seen: float = 0.0
    last_seen: float = 0.0
    count: int = 0
    crops: list = field(default_factory=list)
    bbox: tuple = ()


class FaceRegistry:
    """Tracks unknown faces. No state machine, no regex — just a signal."""

    DETECTION_THRESHOLD = 3
    GAP_TIMEOUT = 8.0
    MAX_CROPS = 8

    def __init__(self):
        self._track = _Track()
        self._signaled = False   # true after we've told the LLM about this person

    # ── public API ──────────────────────────────────────────────

    def on_unknown_face(self, crop, bbox: tuple) -> bool:
        """Feed a detected-but-unrecognized face. Returns True when the
        same unknown person has been seen enough times to signal the LLM."""
        if not HAVE_CV2:
            return False

        now = time.time()

        if now - self._track.last_seen > self.GAP_TIMEOUT:
            self._track = _Track(first_seen=now)
            self._signaled = False

        if self._track.crops and not self._same_person(crop, self._track.crops[-1]):
            return False

        self._track.last_seen = now
        self._track.count += 1
        self._track.bbox = bbox
        if len(self._track.crops) < self.MAX_CROPS:
            self._track.crops.append(crop.copy())

        if self._track.count >= self.DETECTION_THRESHOLD and not self._signaled:
            self._signaled = True
            logger.info(f"[FaceReg] Unknown person detected ({self._track.count} times)")
            return True

        return False

    def get_crops(self) -> list:
        """Collected face crops for LBPH training after registration."""
        return list(self._track.crops)

    def clear(self):
        """Reset after the person has been registered or left."""
        self._track = _Track()
        self._signaled = False

    def _same_person(self, crop1, crop2) -> bool:
        try:
            h1 = cv2.calcHist([cv2.resize(crop1, (64, 64))], [0, 1, 2],
                              None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            h2 = cv2.calcHist([cv2.resize(crop2, (64, 64))], [0, 1, 2],
                              None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) > 0.6
        except Exception:
            return True


face_registry = FaceRegistry()

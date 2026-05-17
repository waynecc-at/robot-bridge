"""Face-to-servo tracking: keeps the camera centered on the current speaker.

Converts face bounding boxes from vision frames into servo angle commands.
Runs in the bridge for low latency, but only as a gentle reflex — the LLM
can always override with explicit set_head_angles calls.

Multi-person: tracks the recognized person (session.current_person).
Falls back to the largest face if no one is recognized.
"""
import time
from typing import Optional
from loguru import logger


class FaceTracker:
    """Smooth face tracking with dead zone and rate limiting."""

    # Servo limits (match firmware MCP definitions)
    YAW_MIN, YAW_MAX = -90, 90     # safe range (HW max is ±128)
    PITCH_MIN, PITCH_MAX = 10, 70  # comfortable range (HW max is 0-90)

    SMOOTHING = 0.25          # exponential smoothing (lower = gentler)
    DEAD_ZONE = 0.06          # fraction of frame — ignore tiny offsets
    UPDATE_INTERVAL = 0.5     # seconds between servo commands
    MAX_DELTA = 12            # max angle change per update
    TRACKING_SPEED = 120      # slow, natural

    def __init__(self):
        self._smooth_yaw = 0.0
        self._smooth_pitch = 45.0
        self._last_update = 0.0
        self._paused_until = 0.0

    # ── public API ──────────────────────────────────────────────

    def update(self, bbox: tuple, frame_size: tuple,
               person_name: str = "",
               current_person: str = "") -> Optional[tuple]:
        """Feed a face detection. Returns (yaw, pitch, speed) or None.

        bbox = (x, y, w, h) in pixels, frame_size = (width, height).
        Multi-person: only tracks current_person's face.
        """
        now = time.time()
        if now < self._paused_until:
            return None
        if now - self._last_update < self.UPDATE_INTERVAL:
            return None

        # Multi-person priority: track only the recognized speaker
        if current_person:
            if not person_name or person_name != current_person:
                return None  # this face isn't who we're talking to

        fx, fy, fw, fh = bbox
        fw_frame, fh_frame = frame_size
        if fw_frame <= 0 or fh_frame <= 0:
            return None

        # Face center as fraction of frame (0-1)
        face_cx = (fx + fw / 2) / fw_frame
        face_cy = (fy + fh / 2) / fh_frame

        # Offset from center (-0.5 to 0.5)
        ox = face_cx - 0.5
        oy = 0.5 - face_cy  # flip: face above center → positive offset

        # Dead zone
        if abs(ox) < self.DEAD_ZONE and abs(oy) < self.DEAD_ZONE:
            return None

        # Map offset to full servo range, scaled to not hit limits
        target_yaw = ox * (self.YAW_MAX - self.YAW_MIN) * 0.7
        target_pitch = 45 + oy * (self.PITCH_MAX - self.PITCH_MIN) * 0.4

        # Clamp
        target_yaw = max(self.YAW_MIN, min(self.YAW_MAX, target_yaw))
        target_pitch = max(self.PITCH_MIN, min(self.PITCH_MAX, target_pitch))

        # Exponential smoothing + rate limit
        s = self.SMOOTHING
        dyaw = (target_yaw - self._smooth_yaw) * s
        dpitch = (target_pitch - self._smooth_pitch) * s

        dyaw = max(-self.MAX_DELTA, min(self.MAX_DELTA, dyaw))
        dpitch = max(-self.MAX_DELTA, min(self.MAX_DELTA, dpitch))

        self._smooth_yaw += dyaw
        self._smooth_pitch += dpitch

        self._last_update = now
        return (int(self._smooth_yaw), int(self._smooth_pitch), self.TRACKING_SPEED)

    def pause(self, duration: float = 4.0):
        """Pause tracking (e.g. after LLM-initiated head movement)."""
        self._paused_until = time.time() + duration

    def reset(self):
        """Reset smoothed position to center."""
        self._smooth_yaw = 0.0
        self._smooth_pitch = 45.0


face_tracker = FaceTracker()

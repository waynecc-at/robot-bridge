"""WebRTC VAD Python wrapper via ctypes.
Drop-in replacement for the simple RMS-based silence detection.
"""

import ctypes
import ctypes.util
from pathlib import Path

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = Path(__file__).parent / "libwebrtcvad.so"
    if lib_path.exists():
        _LIB = ctypes.CDLL(str(lib_path))
    else:
        path = ctypes.util.find_library("webrtcvad")
        if path:
            _LIB = ctypes.CDLL(path)
        else:
            raise RuntimeError("libwebrtcvad.so not found")
    return _LIB


class WebRtcVad:
    """WebRTC Voice Activity Detector.

    Mode: 0=quality, 1=low bitrate, 2=aggressive, 3=very aggressive.
    Default 2 balances sensitivity vs false positives.
    """

    VALID_RATES = {8000, 16000, 32000, 48000}

    def __init__(self, mode: int = 2):
        lib = _load_lib()
        # WebRtcVad_Create(VadInst** handle) -> int
        # Writes handle through pointer arg, returns 0 on success.
        lib.WebRtcVad_Create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.WebRtcVad_Create.restype = ctypes.c_int

        handle = ctypes.c_void_p()
        err = lib.WebRtcVad_Create(ctypes.byref(handle))
        if err != 0 or not handle:
            raise RuntimeError(
                f"WebRtcVad_Create failed (err={err}, handle={handle})"
            )
        self._handle = handle
        self._lib = lib

        lib.WebRtcVad_Init.argtypes = [ctypes.c_void_p]
        lib.WebRtcVad_Init.restype = ctypes.c_int
        lib.WebRtcVad_Init(self._handle)

        lib.WebRtcVad_set_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.WebRtcVad_set_mode.restype = ctypes.c_int
        lib.WebRtcVad_set_mode(self._handle, mode)
        self._mode = mode

    def set_mode(self, mode: int):
        self._lib.WebRtcVad_set_mode(self._handle, mode)
        self._mode = mode

    def is_speech(self, audio: bytes, sample_rate: int = 16000) -> bool:
        """Check if 16-bit linear PCM audio contains speech.

        For best results, pass 10ms, 20ms, or 30ms frames at 16/32/48kHz.
        Returns True if speech detected.
        """
        if sample_rate not in self.VALID_RATES:
            raise ValueError(f"sample_rate must be one of {self.VALID_RATES}")
        if len(audio) < 2:
            return False
        frame_size = len(audio) // 2
        arr = (ctypes.c_int16 * frame_size).from_buffer_copy(audio)
        result = self._lib.WebRtcVad_Process(
            self._handle, sample_rate, arr, frame_size
        )
        return bool(result)

    def close(self):
        if self._handle:
            self._lib.WebRtcVad_Free.argtypes = [ctypes.c_void_p]
            self._lib.WebRtcVad_Free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


_vad_instance = None


def is_speech(pcm: bytes, sample_rate: int = 16000, mode: int = 2) -> bool:
    global _vad_instance
    if _vad_instance is None:
        _vad_instance = WebRtcVad(mode=mode)
    return _vad_instance.is_speech(pcm, sample_rate)


def is_silence(pcm: bytes, sample_rate: int = 16000,
               threshold_rms: int = 600, vad_mode: int = 2) -> tuple:
    """Combined VAD + RMS check.

    Returns (is_silent, rms_value, vad_active).
    Uses WebRTC VAD first; falls back to RMS on tiny frames.
    """
    import array
    if len(pcm) < 2:
        return True, 0.0, False

    samples = array.array('h', pcm)
    if len(samples) == 0:
        return True, 0.0, False

    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5

    # WebRTC VAD needs at least 10ms frames
    frame_ms = len(pcm) / 2 / sample_rate * 1000
    if frame_ms >= 10:
        vad = is_speech(pcm, sample_rate, mode=vad_mode)
    else:
        vad = False

    silent = not vad and rms < threshold_rms
    return silent, rms, vad

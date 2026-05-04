"""Convert float32 audio samples to WAV bytes (16-bit PCM)."""
import io
import struct
import wave


def samples_to_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    """Convert float32 audio samples to WAV bytes (16-bit PCM, mono)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm = []
        for s in samples:
            s = max(-1.0, min(1.0, s))
            pcm.append(int(s * 32767))
        wf.writeframes(struct.pack(f"<{len(pcm)}h", *pcm))
    return buf.getvalue()

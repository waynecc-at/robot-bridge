"""Audio conversion utilities: Opus, Ogg, WAV format conversion via FFmpeg.

Merged from xiaozhi-hermes with cleaned imports.
"""
import subprocess
import struct
import random
import time
from loguru import logger


# Build Ogg CRC32 table once (non-reflected, poly 0x04C11DB7)
OGG_CRC_TABLE = [0] * 256
for i in range(256):
    r = i << 24
    for _ in range(8):
        if r & 0x80000000:
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
        else:
            r = (r << 1) & 0xFFFFFFFF
    OGG_CRC_TABLE[i] = r


class AudioConverter:
    """Convert between audio formats using FFmpeg and manual Ogg framing."""

    @staticmethod
    def convert_opus_to_wav(opus_data: bytes) -> bytes:
        """Convert raw Opus data to WAV using FFmpeg."""
        try:
            process = subprocess.Popen(
                ['ffmpeg', '-y', '-f', 'opus', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            wav_data, stderr = process.communicate(input=opus_data)
            if process.returncode != 0:
                logger.error(f"FFmpeg conversion failed: {stderr.decode(errors='ignore')}")
                return None
            return wav_data
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install ffmpeg.")
            return None
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    @staticmethod
    def build_ogg_opus_from_frames(
        frames: list[bytes], input_sample_rate: int,
        frame_duration_ms: int, channels: int = 1,
    ) -> bytes:
        """Build a complete Ogg-Opus stream from raw Opus frames."""
        if not frames:
            return b""

        def _ogg_crc(data: bytes) -> int:
            crc = 0
            for b in data:
                crc = ((crc << 8) & 0xFFFFFFFF) ^ OGG_CRC_TABLE[((crc >> 24) & 0xFF) ^ b]
            return crc & 0xFFFFFFFF

        def _ogg_page(header_type: int, granule_pos: int, serial: int,
                       seq: int, segments: list[bytes]) -> bytes:
            segment_table = bytearray()
            body = bytearray()
            for seg in segments:
                size = len(seg)
                full = size // 255
                rem = size % 255
                for _ in range(full):
                    segment_table.append(255)
                segment_table.append(rem)
                body.extend(seg)

            page_header = bytearray()
            page_header.extend(b"OggS")
            page_header.extend(struct.pack("B", 0))
            page_header.extend(struct.pack("B", header_type))
            page_header.extend(struct.pack("<q", granule_pos))
            page_header.extend(struct.pack("<I", serial))
            page_header.extend(struct.pack("<I", seq))
            page_header.extend(struct.pack("<I", 0))  # checksum placeholder
            page_header.extend(struct.pack("B", len(segment_table)))
            page = page_header + segment_table + body
            page[22:26] = b"\x00\x00\x00\x00"
            crc = _ogg_crc(bytes(page))
            page[22:26] = struct.pack("<I", crc)
            return bytes(page)

        # OpusHead per RFC 7845
        opus_head = bytearray()
        opus_head.extend(b"OpusHead")
        opus_head.extend(struct.pack("B", 1))
        opus_head.extend(struct.pack("B", channels))
        opus_head.extend(struct.pack("<H", 0))
        opus_head.extend(struct.pack("<I", 48000))
        opus_head.extend(struct.pack("<H", 0))
        opus_head.extend(struct.pack("B", 0))

        # OpusTags
        vendor = b"xiaozhi-server"
        opus_tags = bytearray()
        opus_tags.extend(b"OpusTags")
        opus_tags.extend(struct.pack("<I", len(vendor)))
        opus_tags.extend(vendor)
        opus_tags.extend(struct.pack("<I", 0))

        serial = random.randint(1, 2**31 - 1)
        seq = 0
        pages = []
        t0 = time.perf_counter()
        pages.append(_ogg_page(0x02, 0, serial, seq, [bytes(opus_head)]))
        seq += 1
        pages.append(_ogg_page(0x00, 0, serial, seq, [bytes(opus_tags)]))
        seq += 1

        samples_per_frame_48k = int(48000 * frame_duration_ms / 1000)
        granule_pos = 0
        current_segments = []
        total_segments_len = 0
        for pkt in frames:
            current_segments.append(pkt)
            total_segments_len += len(pkt)
            granule_pos += samples_per_frame_48k
            if len(current_segments) >= 50 or total_segments_len >= 4096:
                pages.append(_ogg_page(0x00, granule_pos, serial, seq, current_segments))
                seq += 1
                current_segments = []
                total_segments_len = 0

        if current_segments:
            pages.append(_ogg_page(0x04, granule_pos, serial, seq, current_segments))
        else:
            pages.append(_ogg_page(0x04, granule_pos, serial, seq, []))

        out = b"".join(pages)
        t1 = time.perf_counter()
        logger.info(f"Ogg build pages={len(pages)} bytes={len(out)} cost={(t1-t0):.3f}s")
        return out

    @staticmethod
    def convert_opus_frames_to_wav(
        frames: list[bytes], input_sample_rate: int, frame_duration_ms: int,
    ) -> bytes:
        """Convert Opus frames to WAV via intermediate Ogg."""
        ogg_data = AudioConverter.build_ogg_opus_from_frames(
            frames, input_sample_rate, frame_duration_ms,
        )
        if not ogg_data:
            return None
        try:
            process = subprocess.Popen(
                ['ffmpeg', '-y', '-f', 'ogg', '-i', 'pipe:0', '-f', 'wav', 'pipe:1'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            wav_data, stderr = process.communicate(input=ogg_data)
            if process.returncode != 0:
                logger.error(f"FFmpeg Ogg->Wav failed: {stderr.decode(errors='ignore')}")
                return None
            return wav_data
        except FileNotFoundError:
            logger.error("FFmpeg not found.")
            return None
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    @staticmethod
    def convert_wav_to_ogg_opus(wav_data: bytes, sample_rate: int,
                                 frame_duration_ms: int) -> bytes:
        """Convert WAV to Ogg-Opus using FFmpeg."""
        try:
            process = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-f", "wav", "-i", "pipe:0",
                    "-ac", "1", "-ar", str(int(sample_rate)),
                    "-c:a", "libopus", "-application", "voip",
                    "-frame_duration", str(int(frame_duration_ms)),
                    "-f", "ogg", "pipe:1",
                ],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            ogg_data, stderr = process.communicate(input=wav_data)
            if process.returncode != 0:
                logger.error(f"FFmpeg opus encode failed: {stderr.decode(errors='ignore')}")
                return None
            return ogg_data
        except FileNotFoundError:
            logger.error("FFmpeg not found.")
            return None
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return None

    @staticmethod
    def extract_ogg_packets(ogg_data: bytes) -> list[bytes]:
        """Extract individual Opus packets from an Ogg container."""
        packets: list[bytes] = []
        current = bytearray()
        offset = 0
        data_len = len(ogg_data)

        while offset + 27 <= data_len:
            if ogg_data[offset:offset + 4] != b"OggS":
                break
            page_segments = ogg_data[offset + 26]
            seg_table_start = offset + 27
            seg_table_end = seg_table_start + page_segments
            if seg_table_end > data_len:
                break
            segment_table = ogg_data[seg_table_start:seg_table_end]
            body_start = seg_table_end
            body_size = sum(segment_table)
            body_end = body_start + body_size
            if body_end > data_len:
                break
            body = ogg_data[body_start:body_end]
            body_offset = 0
            for seg_len in segment_table:
                if seg_len:
                    current.extend(body[body_offset:body_offset + seg_len])
                body_offset += seg_len
                if seg_len < 255:
                    packets.append(bytes(current))
                    current.clear()
            offset = body_end
        return packets

"""Binary protocol handling for raw Opus/audio packet encode/decode.

Supports multiple protocol versions (1, 2, 3) compatible with xiaozhi protocol.
Merged from xiaozhi-hermes.
"""
import struct
import json
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class AudioPacket:
    payload: bytes
    timestamp: int = 0
    seq: int = 0


class ProtocolHandler:
    """Handle WebSocket binary protocol packaging and unpacking."""

    def __init__(self, version: int = 1):
        self.version = version

    def decode_binary(self, data: bytes) -> Optional[AudioPacket]:
        """Parse binary audio packet according to protocol version."""
        if self.version == 1:
            return AudioPacket(payload=data)

        elif self.version == 2:
            if len(data) < 16:
                return None
            header = struct.unpack('!HHIII', data[:16])
            payload_size = header[4]
            timestamp = header[3]
            return AudioPacket(payload=data[16:16+payload_size], timestamp=timestamp)

        elif self.version == 3:
            if len(data) < 4:
                return None
            header = struct.unpack('!BBH', data[:4])
            payload_size = header[2]
            return AudioPacket(payload=data[4:4+payload_size])

        return None

    def encode_binary(self, payload: bytes, timestamp: int = 0) -> bytes:
        if self.version == 1:
            return payload
        if self.version == 2:
            header = struct.pack(
                "!HHIII",
                int(self.version) & 0xFFFF,
                0,
                0,
                int(timestamp) & 0xFFFFFFFF,
                len(payload) & 0xFFFFFFFF,
            )
            return header + payload
        if self.version == 3:
            header = struct.pack("!BBH", 0, 0, len(payload) & 0xFFFF)
            return header + payload
        return payload

    @staticmethod
    def encode_text(data: dict[str, Any]) -> str:
        return json.dumps(data)

    @staticmethod
    def decode_text(text: str) -> Optional[dict[str, Any]]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

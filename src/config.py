"""Configuration management"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import yaml


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8081
    websocket_path: str = "/ws/robot"
    ping_interval: int = 15
    ping_timeout: int = 10


class HermesConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8642  # Hermes Gateway API port (not xiaozhi-hermes proxy port)
    api_key: str = os.getenv("HERMES_API_KEY", "")
    timeout: int = 60
    first_token_timeout: int = 30  # s, max wait for first stream token
    model: str = "hermes-agent"   # model name sent in API payload

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RobotConfig(BaseModel):
    system_prompt: str = (
        "你是 StackChan，一个语音机器人助手。回复规则："
        "用口语化中文，像朋友聊天；根据问题复杂程度自然决定回复长度"
        "（简单问候确认不超过30字，解释推荐等可到150字）；"
        "不要输出 markdown、列表、内部记忆。"
        "绝对不要使用 emoji、颜文字或任何特殊符号。"
        "只输出纯中文文本和必要的标点。"
        "记住对话中用户告诉你的所有信息。"
    )


class TTSConfig(BaseModel):
    provider: str = "sherpa-onnx"
    # Sherpa-ONNX Matcha model paths (relative to model_dir)
    model_dir: str = "models/matcha-icefall-zh-baker"
    acoustic_model: str = "model-steps-3.onnx"
    vocoder: str = "vocos-22khz-univ.onnx"
    tokens: str = "tokens.txt"
    lexicon: str = "lexicon.txt"
    data_dir: str = ""
    rule_fsts: list[str] = ["phone.fst", "date.fst", "number.fst"]
    # Generation parameters
    noise_scale: float = 0.667
    length_scale: float = 1.0
    speed: float = 1.0
    num_threads: int = 2
    # Chunked streaming
    prebuffer_chunk_ms: int = 50
    stream_chunk_ms: int = 200


class MemoryConfig(BaseModel):
    enabled: bool = True
    provider: str = "hermes"
    idle_compress_interval: int = 30  # seconds between idle compression sweeps
    compress_threshold: float = 0.65  # fraction of MAX_TOKENS to trigger


class VisionConfig(BaseModel):
    enabled: bool = False
    # Face detection
    min_face_size: int = 80         # minimum face px for recognition
    detection_interval: float = 1.0  # seconds between frame processing
    # Face recognition
    recognition_threshold: int = 80  # LBPH confidence threshold (lower = stricter)
    # Profile store
    profile_path: str = "data/profiles"
    # Camera source ('esp32' or 'local')
    source: str = "esp32"
    # Local camera (for testing without StackChan)
    local_cam_id: int = 0


class LogConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/bridge.log"
    rotation: str = "100 MB"
    retention: str = "7 days"


class Config(BaseModel):
    server: ServerConfig = ServerConfig()
    hermes: HermesConfig = HermesConfig()
    robot: RobotConfig = RobotConfig()
    tts: TTSConfig = TTSConfig()
    memory: MemoryConfig = MemoryConfig()
    vision: VisionConfig = VisionConfig()
    log: LogConfig = LogConfig()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    
    config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            return Config(**data)
    
    return Config()


# Global config instance
config = load_config()

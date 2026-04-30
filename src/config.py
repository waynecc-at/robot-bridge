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
    ping_interval: int = 30
    ping_timeout: int = 10


class HermesConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 13001
    api_key: str = os.getenv("HERMES_API_KEY", "")
    timeout: int = 60

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class RobotConfig(BaseModel):
    system_prompt: str = (
        "你是 StackChan，一个语音机器人助手。回复规则："
        "用口语化中文，像朋友聊天；每句不超过15字；"
        "不要输出 markdown、列表或内部记忆；"
        "确认类回复只需一个字。"
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
    num_threads: int = 4


class MemoryConfig(BaseModel):
    enabled: bool = True
    provider: str = "hermes"


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

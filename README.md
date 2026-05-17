# Robot Bridge

StackChan 机器人桥接服务 — 连接 M5Stack/ESP32 与 Hermes Agent，webhook 驱动的语音对话桥。

## 架构

```
ESP32 (XiaoZhi WebSocket)
    │ Opus 音频 / JSON 协议
    ▼
Bridge :8081 (纯执行层，零决策逻辑)
    │ ASR → POST :8644/webhooks/stackchan
    ▼
Hermes Agent (SOUL.md + Skills + Memory + MCP tools)
    │ stackchan_speak/emote/look/led
    ▼
MCP Server → HTTP :8081/internal/* → Bridge → ESP32
```

## 功能

- **WebSocket 全双工** — XiaoZhi 协议，Opus 音频流
- **本地 ASR** — SenseVoiceSmall，中文识别
- **本地 TTS** — Sherpa-ONNX Matcha，ffmpeg soxr 重采样
- **11 个 MCP 工具** — listen/speak/see/face/look/led/emote/idle/track/face_register/status
- **Webhook 驱动** — ASR 完成 POST Hermes，Agent 全权控制机器人
- **face_tracker** — 摄像头人脸追踪，舵机自动跟随
- **face_registry** — 陌生人检测，LBPH 人脸注册
- **Barge-in** — 随时打断 TTS
- **OTA 端点** — `/ota` 提供 ESP32 初始配置

## 快速开始

### 前置

- Python 3.12+
- Hermes Agent (启用 webhook + api_server 平台)
- FunASR / Sherpa-ONNX 模型
- FFmpeg

### 安装

```bash
git clone https://github.com/user/robot-bridge.git
cd robot-bridge
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 配置 Hermes

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  stackchan:
    command: /opt/robot-bridge/.venv/bin/python
    args: ["-m", "src.mcp_server"]
    env:
      PYTHONPATH: /opt/robot-bridge
    timeout: 60

platforms:
  api_server:
    enabled: true
    host: 127.0.0.1
    port: 8642
  webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8644
```

创建 webhook 订阅：

```bash
hermes webhook subscribe stackchan \
  --prompt "你是叁叁，StackChan语音机器人。用户说：{text}。简短回复，纯中文，口语化。用 stackchan_speak 说话。" \
  --deliver log \
  --secret "INSECURE_NO_AUTH"
```

### 启动

```bash
python -m src.main          # Bridge
hermes gateway run --replace # Hermes (Gateway + Webhook)
```

## 项目结构

```
robot-bridge/
├── src/
│   ├── api.py                  # FastAPI + WebSocket + /internal/* + /ota + /mcp
│   ├── websocket_handler.py    # XiaoZhi 协议 + ASR → webhook + TTS + LED
│   ├── mcp_server.py           # MCP stdio server (11 tools, HTTP fallback)
│   ├── mcp_router.py           # HTTP MCP endpoint
│   ├── asr_service.py          # SenseVoiceSmall ASR
│   ├── tts_service.py          # Sherpa-ONNX Matcha TTS
│   ├── vision_service.py       # OpenCV 人脸检测
│   ├── face_tracker.py         # 人脸追踪 → 舵机跟随
│   ├── face_registry.py        # 陌生人检测
│   ├── profile_store.py        # LBPH 人脸识别 + 画像
│   ├── config.py               # Pydantic 配置
│   ├── metrics.py              # Prometheus 指标
│   └── main.py                 # 入口
├── configs/
├── tests/
└── .specify/memory/            # speckit 设计文档
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/ota` | GET/POST | ESP32 OTA 配置 |
| `/ws/robot` | WS | XiaoZhi 协议 |
| `/mcp` | POST | MCP JSON-RPC |
| `/internal/speak` | POST | MCP HTTP 回退 — TTS |
| `/internal/led` | POST | MCP HTTP 回退 — LED |
| `/internal/look` | POST | MCP HTTP 回退 — 舵机 |
| `/internal/emote` | POST | MCP HTTP 回退 — 表情 |
| `/internal/listen` | POST | MCP HTTP 回退 — 聆听 |
| `/internal/idle` | POST | MCP HTTP 回退 — 待机 |

## 许可证

MIT

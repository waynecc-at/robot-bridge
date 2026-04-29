# Robot Bridge

StackChan 机器人的桥接服务，连接 ESP32 设备与本地 Hermes Agent，实现智能语音对话。

## 功能特性

- 🌐 **WebSocket Server**: ESP32/M5Stack 设备连接接口
- 🤖 **Hermes 集成**: 调用本地 Hermes Gateway 进行对话
- 🔊 **Edge TTS**: 流式语音合成，中文支持
- 💾 **长期记忆**: 复用 Hermes 原生 hindsight 记忆系统
- ⚡ **低延迟**: 流式处理，端到端响应快

## 系统架构

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│   StackChan     │ ─────────────────► │  Robot Bridge   │
│   (ESP32)       │ ◄───────────────── │  (本服务)       │
│   + M5Stack     │                    │                 │
└─────────────────┘                    │    ┌─────────┐ │
                                      │    │ Hermes  │ │
                                      │    │ Gateway │ │
                                      │    └────┬────┘ │
                                      └─────────┼───────┘
                                                │
                                         ┌──────┴──────┐
                                         │  Hermes     │
                                         │  Agent      │
                                         │  + Memory   │
                                         └─────────────┘
```

## 快速开始

### 前置要求

- Python 3.10+
- Hermes Gateway (已部署)
- 网络连接 (用于 Edge TTS)

### 1. 安装

```bash
git clone https://github.com/你的用户名/robot-bridge.git
cd robot-bridge

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -e .
```

### 2. 配置

```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑 configs/config.yaml，填入你的 Hermes 地址和 API Key
```

配置项说明:

```yaml
server:
  host: "0.0.0.0"    # 服务监听地址
  port: 8082          # 服务端口

hermes:
  host: "127.0.0.1"   # Hermes Gateway 地址
  port: 8642          # Hermes API 端口
  api_key: "your-key" # Hermes API Key

tts:
  voice: "zh-CN-XiaoxiaoNeural"  # 中文女声
  rate: "+10%"                   # 语速稍微加快
```

### 3. 启动

```bash
# 方式一: 脚本启动
./start.sh

# 方式二: 手动启动
python -m src.main

# 方式三: 后台运行
nohup python -m src.main > logs/bridge.log 2>&1 &
```

### 4. 验证

```bash
curl http://localhost:8082/health
# 应返回: {"status": "healthy", "hermes_connected": true}
```

## API 文档

### HTTP 接口

| 端点 | 方法 | 说明 | 示例 |
|------|------|------|------|
| `/health` | GET | 健康检查 | `curl /health` |
| `/api/chat` | POST | 对话 | `curl -X POST /api/chat -d '{"message": "你好"}'` |
| `/api/tts` | POST | TTS合成 | `curl -X POST /api/tts -d '{"text": "你好"}' -o speech.mp3` |
| `/api/tts/stream` | POST | TTS流式 | 流式返回音频 |
| `/api/voices` | GET | 语音列表 | `curl /api/voices?language=zh` |

### WebSocket 接口

连接地址: `ws://localhost:8082/ws/robot`

**发送消息:**
```json
{"type": "text", "text": "你好"}
```

**接收响应:**
```json
{"type": "response_text", "text": "你好！有什么可以帮你？"}
{"type": "tts_audio", "data": "base64...", "final": false}
{"type": "tts_audio", "data": "", "final": true}
```

## ESP32 固件配置

将 ESP32/M5Stack 的服务器地址指向 Robot Bridge:

```c
// app_config.h
#define SERVER_HOST "你的服务器IP"
#define SERVER_PORT 8082
#define WS_PATH "/ws/robot"
```

## CLI 客户端

交互式对话模式:

```bash
python -m src.cli

# 单次对话
python -m src.cli --chat "你好"

# TTS 试听
python -m src.cli --tts "你好" --play
```

## 项目结构

```
robot-bridge/
├── configs/
│   ├── config.example.yaml   # 配置示例
│   └── config.yaml          # 实际配置 (不提交)
├── src/
│   ├── main.py              # 主入口
│   ├── config.py            # 配置管理
│   ├── api.py               # HTTP API
│   ├── hermes_client.py     # Hermes 客户端
│   ├── tts_service.py       # Edge TTS
│   ├── websocket_handler.py  # WebSocket 处理
│   └── cli.py               # CLI 客户端
├── tests/
│   └── test_bridge.py       # 测试脚本
├── logs/                    # 日志目录
├── .gitignore
├── pyproject.toml
├── README.md
└── start.sh
```

## 与 Hermes Agent 集成

Robot Bridge 依赖本地部署的 Hermes Gateway:

1. 确保 Hermes Gateway 已部署并运行
2. 确认 Hermes 开启 api_server 插件
3. 配置正确的端口 (默认 8642) 和 API Key

Hermes 配置参考:
```yaml
# Hermes config.yaml
platforms:
  api_server:
    enabled: true
    port: 8642
    key: "your-api-key"
```

## 隐私说明

- `configs/config.yaml` 不包含在版本控制中
- API Keys 和敏感配置使用环境变量或单独管理
- 日志文件不包含用户对话内容

## License

MIT

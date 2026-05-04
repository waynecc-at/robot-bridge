# Robot Bridge

StackChan 桌面机器人桥接服务 — 连接 M5Stack / ESP32 设备与本地 Hermes AI，实现全双工语音对话。

## ✨ 功能特性

- **WebSocket 全双工通信** — 对接 StackChan 原厂固件，Opus 音频流实时传输
- **本地 ASR** — FunASR SenseVoiceSmall，中文语音识别，不离家
- **本地 TTS** — Sherpa-ONNX Matcha，中文语音合成，流式分块输出
- **Hermes AI 集成** — 调用本地 Hermes Gateway 进行多轮对话
- **长程记忆** — SessionStore 上下文管理 + 后台空闲压缩（不阻塞响应）
- **OpenClaw 桥接** — 可选 OpenClaw agent 集成（WebSocket bridge / HTTP callback）
- **并发管道** — LLM 流式 + TTS 合成异步重叠，首块 50ms 预缓冲
- **抢占中断** — 支持 barge-in，随时打断机器人当前对话
- **CLI 客户端** — 交互式对话、TTS 试听

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│  StackChan Robot (原厂固件, 不动)                               │
│  ┌─────────────────────────────────────────────────────┐         │
│  │  AppAiAgent                                         │         │
│  │   ├─ ES7210 Mic → Opus(16kHz/60ms) → WebSocket      │         │
│  │   ├─ WS接收TTS → Opus解码 → AW88298播放              │         │
│  │   ├─ LVGL表情 / 舵机运动                              │         │
│  │   └─ 小智协议 (JSON + 二进制帧)                       │         │
│  └─────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                                │ WebSocket
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  robot-bridge (Python 3.10+, FastAPI + uvicorn)                 │
│                                                                  │
│  src/                                                           │
│  ├─ api.py              HTTP API + WebSocket 端点                │
│  ├─ websocket_handler.py ★ 核心: 并发管道 (LLM↔TTS 重叠)         │
│  ├─ hermes_client.py    Hermes 客户端 + 延迟压缩 + 长程记忆       │
│  │                      后台空闲压缩器 (30s 间隔)                  │
│  ├─ tts_service.py      Sherpa-ONNX (线程池 + 分块流式)          │
│  ├─ asr_service.py      FunASR SenseVoiceSmall (线程池卸载)      │
│  ├─ protocol.py         二进制协议 (v1/v2/v3)                     │
│  ├─ audio_converter.py  Opus/Ogg/WAV 格式转换                    │
│  ├─ openclaw/           OpenClaw Agent 桥接插件                  │
│  └─ cli.py              CLI 交互式客户端                          │
└─────────────────────────────────────────────────────────────────┘
                                │ HTTP SSE (chat/completions)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hermes AI (本地 LLM 部署)                                       │
│  输入: system_prompt + summary + 上下文消息                       │
│  输出: streaming tokens                                          │
└─────────────────────────────────────────────────────────────────┘
```

## 优化亮点 (v0.2)

| 改进项 | 之前 | 之后 | 效果 |
|--------|------|------|------|
| TTS/ASR 线程池 | 阻塞事件循环 | `run_in_executor` 卸载 | 不阻塞其他连接 |
| compress() 时机 | 响应前等待 1-3s | 后台空闲压缩 | 节省 ~2s/轮 |
| TTS 流式 | 整句 WAV 一次性发送 | 首块 50ms + 后续 200ms 分块 | 更快开始播放 |
| 管道并行度 | 全串行 (ASR→LLM→TTS) | LLM+TTS 并发 (asyncio.Queue) | 3 句回复省 30-50% |
| 连接池 | 默认 | keepalive=5, max=10 | 减少连接建立 |
| Token 估算 | `len//2` 启发式 | tiktoken 优先 + 改进启发式 | 压缩时机更准 |

## 快速开始

### 前置要求

- Python 3.10+
- 本地部署的 Hermes Gateway（或兼容 OpenAI API 的 LLM 服务）
- FunASR / Sherpa-ONNX 模型文件（[下载说明](#模型下载)）
- FFmpeg（用于音频格式转换）

### 1. 安装

```bash
git clone https://github.com/你的用户名/robot-bridge.git
cd robot-bridge

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置

```bash
cp configs/config.example.yaml configs/config.yaml
# 编辑 config.yaml, 填入你的 Hermes 地址
```

### 3. 模型下载

```bash
# TTS 模型 (Sherpa-ONNX Matcha 中文)
mkdir -p models
# 下载 matcha-icefall-zh-baker 到 models/ 目录
# 参考: https://github.com/k2-fsa/sherpa-onnx/releases

# ASR 模型 (FunASR)
# 首次启动时自动下载 iic/SenseVoiceSmall
```

### 4. 启动

```bash
# 脚本启动
./start.sh

# 或手动
python -m src.main
```

### 5. 验证

```bash
curl http://localhost:8081/health
# 预期: {"status": "healthy", "hermes_connected": true}
```

## API 文档

### HTTP 接口

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/chat` | POST | 对话 (支持流式 SSE) |
| `/api/tts` | POST | TTS 合成, 返回 WAV |
| `/api/tts/stream` | POST | TTS 流式, 分块 WAV |
| `/api/voices` | GET | 语音列表 |

### WebSocket 接口

连接: `ws://<host>:8081/ws/robot`

**机器人 → 服务器:**
```json
{"type": "text", "text": "你好"}
{"type": "audio", "data": "<base64_pcm>", "format": "pcm"}
{"type": "vad", "state": "start"}
{"type": "ping", "timestamp": 1234567890}
```

**服务器 → 机器人:**
```json
{"type": "status", "message": "thinking", "action": "thinking"}
{"type": "response_text", "text": "你好！", "partial": true}
{"type": "tts_audio", "format": "wav", "data": "<base64_chunk>", "final": false}
{"type": "tts_audio", "data": "", "final": true}
{"type": "interrupt", "turn": 3}
```

### CLI 客户端

```bash
# 交互式对话
python -m src.cli

# 单次对话
python -m src.cli --chat "你好"

# TTS 试听
python -m src.cli --tts "你好" --play
```

## 项目结构

```
robot-bridge/
├── src/
│   ├── __init__.py            # 包版本
│   ├── main.py                # 入口 + uvicorn 启动
│   ├── config.py              # Pydantic 配置模型
│   ├── api.py                 # FastAPI 应用 + HTTP/WS 路由
│   ├── websocket_handler.py   # WebSocket 处理 + 并发管道
│   ├── hermes_client.py       # Hermes 客户端 + 延迟压缩
│   ├── tts_service.py         # Sherpa-ONNX TTS (线程池 + 分块)
│   ├── asr_service.py         # FunASR ASR (线程池卸载)
│   ├── protocol.py            # 二进制协议编解码 (v1/v2/v3)
│   ├── audio_converter.py     # Opus/Ogg/WAV 格式转换
│   ├── wav_utils.py           # WAV 字节工具
│   ├── openclaw/
│   │   ├── __init__.py
│   │   ├── bridge.py          # OpenClaw WebSocket 桥接
│   │   └── channel.py         # OpenClaw HTTP 回调
│   └── cli.py                 # CLI 交互式客户端
├── configs/
│   ├── config.example.yaml    # 配置示例
│   └── config.yaml            # 实际配置 (不提交)
├── myapp-channel-0.1.0/       # OpenClaw channel 插件 (TypeScript)
├── tests/
│   ├── test_bridge.py         # 集成测试
│   └── test_e2e_pipeline.py   # 端到端管道测试 + 性能计时
├── pyproject.toml
├── start.sh
├── .gitignore
├── LICENSE
└── README.md
```

## 配置说明

核心配置项 (`configs/config.example.yaml`):

```yaml
server:
  host: "0.0.0.0"
  port: 8081
  ping_interval: 15        # WebSocket ping 间隔 (秒)

hermes:
  host: "127.0.0.1"        # Hermes Gateway 地址
  port: 8642
  api_key: "your-key"      # 或环境变量 HERMES_API_KEY
  timeout: 60

tts:
  model_dir: "models/matcha-icefall-zh-baker"
  num_threads: 2
  prebuffer_chunk_ms: 50   # 首块大小 (越小越快开始发声)
  stream_chunk_ms: 200     # 后续块大小

memory:
  enabled: true
  idle_compress_interval: 30  # 后台压缩间隔 (秒)
  compress_threshold: 0.65    # 触发压缩的 token 阈值比例
```

## 与 Hermes Agent 集成

Robot Bridge 依赖本地 Hermes Gateway 提供的 LLM API 端点。确认 Hermes 已启用 `api_server` 插件并运行：

```yaml
# Hermes config.yaml
platforms:
  api_server:
    enabled: true
    port: 8642
    key: "your-api-key"
```

## OpenClaw Agent 集成 (可选)

若使用 OpenClaw 作为对话引擎而非直接调用 Hermes：

1. 部署 OpenClaw 并安装 `myapp-channel` 插件
2. 配置 `myapp-channel` 连接到 robot-bridge 的 OpenClaw 桥接端点
3. 在代码中调用 `openclaw/bridge.py` 的 `request_reply()` 方法

详情见 [myapp-channel 文档](myapp-channel-0.1.0/myapp-channel/README.md)。

## 性能指标 (参考)

真实 Sherpa-ONNX Matcha TTS (vocos-22kHz) 基准测试结果 (5轮):

| 指标 | 实测值 |
|------|--------|
| TTS 单句合成 | **80-120ms** (短句, ~1.5s音频) |
| TTS 整段回复 | **224-407ms** (3-5句, ~10s音频) |
| TTS 音质 | 22050Hz, 16-bit, 单声道 PCM |
| LLM + TTS 并发总耗时 | **~2.0-2.8s** (LLM模拟 2s + TTS 并行) |
| 分块流式首块 | **50ms** 预缓冲即刻发出 |
| 后续分块间隔 | **200ms** 每块 (~0.9s音频/块) |

优化前后对比:

| 指标 | v0.1 (之前) | v0.2 (优化后) | 提升 |
|------|------------|--------------|------|
| ASR 延迟 | ~500-1500ms (阻塞事件循环) | ~500-1500ms (线程池, 不阻塞) | 并发友好 |
| compress 延迟 | ~2000ms (在响应关键路径上) | ~0ms (后台异步) | 节省 ~2s |
| TTS 首块延迟 | 等整句合成完 (~2-5s) | ~50ms 首块先发 | 更快开始发声 |
| 3 句回复总时间 | ~10-15s (串行) | ~2-3s (并发管道, 实测) | ~60-75% |
| 多机器人并发 | 串行 (阻塞事件循环) | 并行 (线程池) | 多设备可用 |
| TTS 单句 RTF | N/A (未测量) | **0.04-0.05** (生成 < 音频时长) | 远快于实时 |

## 隐私说明

- `configs/config.yaml` 不在版本控制中（已在 .gitignore）
- API Key 从环境变量 `HERMES_API_KEY` 读取，不硬编码
- 模型文件 (`models/`) 不提交
- 音频文件、日志文件不提交

## License

MIT License

## 参考项目

| 项目 | 说明 |
|------|------|
| [StackChan](https://github.com/m5stack/StackChan) | M5Stack 开源桌面机器人 |
| [Hermes Agent](https://github.com/NickYi1990/Hermes) | 本地 AI Agent 框架 |
| [xiaozhi-esp32-server-golang](https://github.com/hackers365/xiaozhi-esp32-server-golang) | 小智 Go 服务端参考实现 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 跨平台 TTS 推理引擎 |
| [FunASR](https://github.com/modelscope/FunASR) | 语音识别工具包 |

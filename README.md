# Robot Bridge

StackChan 桌面机器人桥接服务 — 连接 M5Stack / ESP32 设备与本地 Hermes AI，实现全双工语音对话。

> **仓库整合：** 本仓库已合并原 `xiaozhi-hermes`（已归档）的所有核心组件，包括 OpenClaw 桥接、协议编解码、音频格式转换等。单一仓库统一维护。

## 功能特性

- **WebSocket 全双工通信** — 对接 StackChan 原厂固件，Opus 音频流实时传输
- **本地 ASR** — FunASR SenseVoiceSmall，中文语音识别，不离家
- **本地 TTS** — Sherpa-ONNX Matcha，中文语音合成，流式分块输出（RTF 0.04-0.06）
- **Hermes 服务端会话管理** — 通过 `X-Hermes-Session-Id` 头实现服务端上下文，每次只发 2 条消息（system + user），历史由 Hermes 管理
- **LLM 预热** — 启动时背景预热，消除 DeepSeek 冷启动 ~11s 延迟，TTFT 稳定在 1-2s
- **LLM 超时保护 + 自动重试** — 15s 无首 token 自动断开重连，防止罕见 27s 冷启动
- **思考状态心跳** — LLM 等待期间 4s/8s/15s 递进状态反馈
- **逐 token 打字机效果** — 实时文本流 + 句子完成标记
- **文本净化** — 去除 emoji / markdown / 特殊符号，TTS 友好
- **动态回复长度** — 简单问候 30 字内，复杂解释可到 150 字
- **会话持久化** — 关闭时存盘，重启后恢复长程记忆
- **Token 用量追踪** — 从 API 响应实时获取真实 token 计数
- **并发管道** — LLM 流式 + TTS 合成异步重叠，首块 50ms 预缓冲
- **Barge-in 抢占中断** — 随时打断，带自然收尾淡出
- **视觉智能 (Phase 2)** — 人脸检测识别、运动检测、人物跟随、动态人物画像注入 LLM
- **CLI 客户端** — 交互式对话、TTS 试听

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│  StackChan Robot (原厂固件)                                  │
│  ├─ ES7210 Mic → Opus(16kHz/60ms) → WebSocket               │
│  ├─ WS接收TTS → Opus解码 → AW88298播放                      │
│  ├─ LVGL表情 / 舵机运动                                      │
│  └─ 小智协议 (JSON + 二进制帧)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  robot-bridge (Python 3.10+, FastAPI + uvicorn)              │
│                                                               │
│  src/                                                        │
│  ├─ api.py              HTTP API + WebSocket 端点             │
│  ├─ websocket_handler.py 并发管道 (LLM→TTS 重叠 + 心跳 + 降级) │
│  ├─ hermes_client.py     Hermes 客户端 + Server Session + 预热 │
│  │                       超时重试 + Token 追踪 + 持久化       │
│  ├─ text_utils.py       文本净化 (去 emoji/markdown)          │
│  ├─ tts_service.py       Sherpa-ONNX (线程池 + 分块流式)      │
│  ├─ asr_service.py       FunASR SenseVoiceSmall (线程池)      │
│  ├─ protocol.py          二进制协议 (v1/v2/v3)                │
│  ├─ audio_converter.py   Opus/Ogg/WAV 格式转换                │
│  ├─ openclaw/            OpenClaw Agent 桥接插件              │
│  └─ cli.py               CLI 交互式客户端                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP SSE (chat/completions)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Hermes Agent (本地 LLM 网关)                                 │
│  模式: 服务端会话 (state.db SQLite) + 长期记忆 (holographic)   │
│  模型后端: DeepSeek v4 Flash (或任意 OpenAI 兼容 API)          │
└─────────────────────────────────────────────────────────────┘
```

## 性能指标 (实测)

真实 Hermes LLM (DeepSeek v4 Flash) + Sherpa-ONNX Matcha TTS 基准:

| 指标 | Server Session | Stateless (全量历史) |
|------|:---:|:---:|
| Round 1 TTFT (含 warmup) | **1.3-2.5s** | **1.4-13.7s** |
| Round 2-3 TTFT (warm) | **0.9-1.6s** | **1.3-1.9s** |
| 每次请求消息数 | **2** (system + user) | 随轮次增长 |
| 会话历史管理 | Hermes 端 state.db | 客户端全量发送 |
| TTS RTF | **0.04-0.06** | **0.04-0.06** |

Server Session 模式每次只发 2 条消息，无论会话多长。Stateless 模式需要发送完整历史，消息量线性增长。

## 视觉智能 (Phase 2)

```
┌─────────────────────────────────────────────────────────────┐
│ StackChan (ESP32-S3) — 边缘处理                              │
│                                                              │
│  GC0308 摄像头 (QVGA 320x240)                                 │
│  ├─ 运动检测 (ESP motion_detect, ~15-65 FPS)                 │
│  ├─ 人脸检测 (ESP-WHO, ~3-5 FPS)                             │
│  ├─ JPEG编码 + base64 → WebSocket 上传                       │
│  └─ 舵机PID跟随 (人脸偏移 → 伺服角度)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │ WebSocket: vision_frame (base64 JPEG)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ robot-bridge (Python)                                        │
│                                                               │
│  src/                                                        │
│  ├─ vision_service.py    解码 → 人脸检测 → 识别 → 追踪        │
│  ├─ profile_store.py     人物画像 + LBPH 模型持久化            │
│  └─ websocket_handler.py vision_frame 处理 + 动态prompt注入    │
│                                                               │
│  收到帧 → OpenCV解码 → Haar级联检测 → LBPH识别                 │
│  → 匹配画像 → 更新session.person_profile                       │
│  → 下次LLM请求自动注入: "称呼用户为「小明」。他是你的主人。"      │
└──────────────────────┬──────────────────────────────────────┘
                       │ LLM context with person profile
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Hermes Agent                                                │
│  输入: system_prompt + "称呼用户为「小明」, 他是你的主人"      │
│  输出: "小明你好, 今天想聊点什么？"                             │
└─────────────────────────────────────────────────────────────┘
```

### 人物识别流程

1. **ESP32**: 每秒拍一帧 JPEG → base64 → `{"type":"vision_frame","data":"..."}`
2. **vision_service**: 接收帧 → `asyncio` 线程池解码 → `profile_store.detect_and_recognize()`
3. **profile_store**: Haar级联检测人脸 → LBPH识别 → 匹配已知画像
4. **websocket_handler**: 识别到人物 → 更新 `session.person_profile` → 注入LLM system_prompt
5. **Hermes**: LLM 回复包含人物信息（称呼名字、记住偏好）

### 人物画像注册

通过 API 注册新人物:

```bash
curl -X POST http://localhost:8081/api/profile/register \
  -H "Content-Type: application/json" \
  -d '{"name":"小明","relationship":"主人","preferences":"喜欢户外运动"}'
```

注册时从摄像头捕获 5-10 帧人脸，自动训练 LBPH 识别器。

### 依赖

```bash
pip install opencv-python numpy
```

OpenCV 为可选依赖。不安装时视觉功能自动禁用，不影响语音对话。

## 快速开始

### 前置要求

- Python 3.10+
- 本地 Hermes Agent (启用 api_server 平台)
- FunASR / Sherpa-ONNX 模型文件
- FFmpeg

### 1. 安装

```bash
git clone https://github.com/waynecc-at/robot-bridge.git
cd robot-bridge

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. 配置 Hermes

确认 Hermes 配置中已启用 api_server 并设置 key:

```yaml
# ~/.hermes/config.yaml
platforms:
  api_server:
    enabled: true
    host: 127.0.0.1
    port: 8642
    extra:
      key: your-api-key
```

### 3. 配置 robot-bridge

```bash
cp configs/config.example.yaml configs/config.yaml
export HERMES_API_KEY=your-api-key
```

### 4. 启动

```bash
./start.sh
# 或手动: python -m src.main
```

### 5. 验证

```bash
curl http://localhost:8081/health
# {"status": "healthy", "hermes_connected": true}
```

## 两种会话模式

| | Server Session（推荐） | Stateless（回退） |
|:--|:---|:---|
| 条件 | 设置 `HERMES_API_KEY` | 未设置 key |
| 请求大小 | 始终 2 条消息 | 全量历史，随轮次增长 |
| 历史管理 | Hermes 端 SQLite state.db | 客户端 SessionStore |
| Token 追踪 | API 返回真实用量 | 本地启发式估算 |
| 长期记忆 | Hermes holographic provider | 客户端摘要压缩 |

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
{"type": "status", "message": "understanding", "action": "thinking"}
{"type": "status", "message": "deep thinking", "action": "thinking"}
{"type": "response_text", "text": "你", "partial": true}
{"type": "response_text", "text": "你好！", "partial": false}
{"type": "tts_audio", "format": "wav", "data": "<base64_chunk>", "final": false}
{"type": "tts_audio", "data": "", "final": true}
{"type": "interrupt", "turn": 3}
```

### CLI 客户端

```bash
python -m src.cli                      # 交互式对话
python -m src.cli --chat "你好"         # 单次对话
python -m src.cli --tts "你好" --play   # TTS 试听
```

## 项目结构

```
robot-bridge/
├── src/
│   ├── __init__.py
│   ├── main.py                 # 入口 + uvicorn 启动
│   ├── config.py               # Pydantic 配置
│   ├── api.py                  # FastAPI + WebSocket 路由
│   ├── websocket_handler.py    # 并发管道 + 超时保护 + 心跳 + 视觉
│   ├── hermes_client.py        # Hermes 客户端 + Server Session
│   ├── text_utils.py           # 文本净化
│   ├── tts_service.py          # Sherpa-ONNX TTS
│   ├── asr_service.py          # FunASR ASR
│   ├── vision_service.py       # 人脸检测/识别/追踪 (OpenCV)
│   ├── profile_store.py        # 人物画像 + LBPH 识别器持久化
│   ├── protocol.py             # 二进制协议编解码
│   ├── audio_converter.py      # Opus/Ogg/WAV 转换
│   ├── cli.py                  # CLI 客户端
│   └── openclaw/               # OpenClaw 桥接
├── tests/
│   ├── test_bridge.py
│   ├── test_e2e_pipeline.py
│   ├── test_hermes_tts.py      # 真实 Hermes+TTS E2E 测试
│   └── benchmark_session.py    # Server Session vs Stateless 性能对比
├── configs/
│   └── config.example.yaml
├── pyproject.toml
├── start.sh
├── .gitignore
├── LICENSE
└── README.md
```

## 隐私说明

- `configs/config.yaml` 不在版本控制中（已加入 .gitignore）
- API Key 从环境变量 `HERMES_API_KEY` 读取，不硬编码
- 模型文件 (`models/`) 和数据文件 (`data/`) 不提交
- 配置文件模板不含真实密钥

## License

MIT License

## 参考项目

| 项目 | 说明 |
|------|------|
| [StackChan](https://github.com/m5stack/StackChan) | M5Stack 开源桌面机器人 |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | 本地 AI Agent 框架 |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 跨平台 TTS 推理引擎 |
| [FunASR](https://github.com/modelscope/FunASR) | 语音识别工具包 |

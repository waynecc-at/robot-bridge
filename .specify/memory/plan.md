# Implementation Plan: 叁叁语音机器人 v2 — Hermes-Owns 重构

**Branch**: `main` | **Date**: 2026-05-17 | **Spec**: [spec.md](./spec.md) | **Constitution**: v2.0

**Input**: Feature specification + REFACTOR-PLAN.md analysis

## Summary

StackChan 从"Bridge 调 Hermes API"的胖中间件架构，重构为 webhook 驱动的薄层架构。

**最终架构**：
```
ESP32 → Bridge :8081 (ASR/TTS/Opus/11 MCP 工具)
           │ ASR → POST :8644/webhooks/stackchan
           ▼
  Hermes Agent (SOUL.md + Skills + Memory)
           │ stackchan_speak/emote/look/led
           ▼
  MCP Server → HTTP :8081/internal/* → Bridge → ESP32
```

Bridge 降级为纯执行层（零决策逻辑），Agent Driver 已删除。Hermes Agent 通过 MCP 工具控制机器人全部动作。

## Current Architecture

```
ESP32 StackChan
    │ WebSocket XiaoZhi 协议
    ▼
┌── Bridge (websocket_handler.py) ───────────────┐
│  协议翻译 + 音频管线 + MCP 工具服务器              │
│  ASR → last_asr_text → Hook → Agent Driver     │
│  TTS ← Agent Driver 调 _tts_and_send()         │
│  不做任何LLM/决策逻辑                             │
└──────────────────────┬─────────────────────────┘
                       │ Hook (事件驱动，不轮询)
                       ▼
┌── Agent Driver (agent_driver.py) ──────────────┐
│  Bridge ASR 完成后立即被调用                      │
│  调 Hermes Gateway API (POST :8642/v1/chat...)  │
│  Header: X-Hermes-Session-Id → 上下文 + Memory  │
│  Body: SOUL.md (人格) + 用户文字                 │
│  流式拿回复 → TTS 播报                           │
│  ⚠️ 临时方案，Hermes 原生支持循环后删除             │
└──────────────────────┬─────────────────────────┘
                       │ HTTP API (localhost:8642)
                       ▼
┌── Hermes Gateway API ──────────────────────────┐
│  LLM: DeepSeek V4 Flash                        │
│  Memory: session_id → 对话历史 + fact_store     │
│  SOUL.md 作为 system prompt → 完整人格           │
└────────────────────────────────────────────────┘

┌── Hermes CLI (hermes --profile stackchan) ─────┐
│  MCP 工具: listen/speak/see/look/led/emote...  │
│  手动干预时使用（转头、拍照等）                    │
└────────────────────────────────────────────────┘
```

## Technical Context

**Language/Version**: Python 3.12 (Bridge + Agent Driver)
**Firmware**: C++17 (ESP32-S3, ESP-IDF v5.5.4)

**Key Dependencies**:
- `mcp` Python SDK (MCP server framework)
- `httpx` (Agent Driver → Gateway API calls)
- Hermes Agent v0.13.0+ (native MCP client + Gateway API)

**Removed**: `hermes_client.py` (old Hermes HTTP client)

**Performance Goals**: Wake → first audio <4s, LLM TTFT <2s (streaming)
**Constraints**: Chinese-only output, local ASR/TTS mandatory, no cloud raw camera data

## Constitution Check

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Hermes-Owns, Bridge-Operates | ✅ v2.0 | Agent Driver + Gateway API + SOUL.md. Bridge has zero decision logic |
| II. Chinese-Only | ✅ | SOUL.md 铁律纯中文, TTS Matcha Chinese Baker |
| III. Local-First Latency | ✅ | ASR/TTS/face detection all local |
| IV. Privacy-First | ✅ | Camera data stays local |
| V. Progressive Delivery | ✅ | Phases completed, Agent Driver as temporary bridge |

## Key Design Decisions

### Hook vs Polling for ASR → LLM
- ❌ **Polling**: Check `last_asr_text` every 500ms (wasteful, ~0.5s delay)
- ✅ **Hook**: `_eos_handler` calls `_on_asr_callback` directly (instant, clean)

### Agent Driver vs Hermes CLI subprocess
- ❌ **Hermes CLI**: Spawn `hermes chat -q` per utterance (0.3s process overhead)
- ✅ **Gateway API**: Direct `POST /v1/chat/completions` (no overhead, full Hermes Memory)

### SOUL.md Design (Voice Robot Version)
- **极简短**: 回复 ≤3短句, 10秒可读完
- **纯中文**: 禁止英文/代码/markdown/emoji
- **不暴露思考**: 不说"让我想想", 直接回答
- **时间感知**: Agent Driver 自动注入当前时间到用户消息
- **口语化**: 可以用语气词，但不过度

### TTS Resampling: scipy FFT vs ffmpeg soxr
- ❌ **scipy.signal.resample**: FFT brick-wall filter, 10.5% HF noise, Gibbs ringing → "电流声"
- ✅ **ffmpeg soxr**: Polyphase anti-aliased resampling, 7.2% HF noise, clean output

### Opus Frame Pacing
- ❌ **sleep(0.06) between frames**: 60ms sleep + encode + network = 70-80ms actual interval > 60ms frame duration → ESP32 buffer underrun → pops/clicks
- ✅ **Line-speed send**: ESP32 has 40-frame buffer (2.4s), send all frames as fast as possible, buffer absorbs bursts

### TTS Pipeline (final)
```
float32[22050] → ffmpeg soxr (f32le pipe) → int16 PCM[16000] → Opus (voip, 48kbps) → ESP32
```
Single quantization at pipeline end. One Opus encoder per utterance (reused across sentences).

## Project Structure (Current)

```
/home/wayne/robot-bridge/
├── src/
│   ├── api.py                    # FastAPI WS endpoint (protocol relay)
│   ├── mcp_server.py             # MCP stdio server (11 tools)
│   ├── websocket_handler.py      # STRPED: protocol + audio + MCP relay + ASR hook
│   ├── agent_driver.py           # NEW: ASR hook → LLM → TTS loop (temp)
│   ├── asr_service.py            # SenseVoiceSmall
│   ├── tts_service.py            # Sherpa-ONNX Matcha
│   ├── vision_service.py         # OpenCV face detection
│   ├── face_registry.py          # Unknown face tracking
│   ├── face_tracker.py           # Continuous face tracking
│   ├── profile_store.py          # LBPH + profiles
│   ├── config.py                 # Pydantic config
│   └── metrics.py                # Prometheus metrics
├── src/hermes_client.py          → DELETED
├── src/hermes_cli_bridge.py      → DELETED (unused approach)
├── tests/
│   └── test_bridge_v2.py         # Integration tests
└── .specify/memory/
    ├── constitution.md           → v2.0
    ├── spec.md                   → v2
    ├── plan.md                   → this file
    └── tasks.md                  → 39 tasks

~/.hermes/profiles/stackchan/
├── config.yaml                   # MCP server config
└── SOUL.md                       # Voice robot personality (v2)

~/.hermes/skills/stackchan-family/
└── SKILL.md                      # Family rules template
```

## Remaining Work

### Phase 4: Face Registration Upgrade

Move face registration flow into Agent Driver. Currently face_registry tracks unknown
faces in Bridge but Agent Driver doesn't handle registration prompts.

- Auto-register new faces via LLM-driven conversation
- Store profiles via Hermes Memory (fact_store)

### Phase 5: Proactive Capabilities

- Cron reminders via Hermes cronjob tool
- Breathing LED during idle (send LED commands from Agent Driver)
- Proactive greeting when face detected
- Time-aware greeting variations

### Phase 6: Firmware Tuning

- Wake word: "hai san san" → "sansan" (needs MultiNet re-train + flash)
- Idle motion frequency tuning
- Vision frame rate optimization

### Phase 7 (Future): Function Calling for MCP Tools

Enable Agent Driver to forward LLM tool calls to Bridge MCP tools.
When LLM decides to call `stackchan_look` or `stackchan_see`:
1. Agent Driver receives tool call from Gateway API
2. Forwards to Bridge via WebSocket MCP relay
3. Returns result to LLM → continues conversation

## Key Risks

1. **Gateway API MCP support** — Gateway doesn't load profile-level MCP servers; function calling requires manual tool call forwarding in Agent Driver
2. **Warm startup** — Model loading takes ~5s before ready
3. **Agent Driver lock contention** — Multiple rapid utterances could queue
4. **Hermes restart** — Kills MCP server subprocess, ESP32 disconnects

## Open Items (Deferred)

| Item | Status | Notes |
|------|--------|-------|
| 唤醒词 → "sansan" | ⏳ | 固件 MultiNet 需重训 + 刷写 |
| 已唤醒时不响应唤醒词 | ⏳ | Bridge 端加状态过滤 |
| Gateway 加载 MCP | ⏳ | 需等 Hermes 支持 profile MCP 传入 Gateway |

## 备选功能

| 功能 | 优先级 | 思路 |
|------|--------|------|
| 情绪记忆 | 🔥🔥🔥 | fact_store 存情绪标签 |
| 接着说 | 🔥🔥🔥 | 新 session 捞上次 Memory |
| 关系图谱 | 🔥🔥 | 注册时建立 family tree |
| 主动分享 | 🔥🔥 | 空闲时找话题主动说 |
| 自然微动作 | 🔥 | 聆听时 LED 微闪 |

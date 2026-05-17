# 叁叁语音机器人 — 架构重构方案

> 目标：Bridge 做薄，Hermes 做主。小机器人是叁叁的身体向物理世界的延伸。

---

## 一、当前问题分析：Bridge 为什么不够薄

| 模块 | 当前职责 | 问题 | 应该在哪 |
|------|----------|------|----------|
| `websocket_handler.py:_process_utterance` | 决定何时 EOS → 调 ASR → 调 LLM → 调 TTS | Bridge在编排对话流程 | ❌ 应是 Hermes |
| `websocket_handler.py:_build_system_prompt` | 构建 system prompt（纯中文铁律、人物上下文） | Bridge在控制LLM提示词 | ❌ 应是 Hermes |
| `websocket_handler.py:_respond` | 调用 Hermes chat API + 等待流式回调 | Bridge是主动方，Hermes是被动API | ❌ 应反过来 |
| `websocket_handler.py:LED状态机` | 绿/彩虹/蓝/灭 状态切换 | Bridge在决定LED状态 | ❌ 应是 Hermes |
| `websocket_handler.py:_tts_worker` | 句子级TTS流水线 | 流式TTS调度在Bridge | ⚠️ 可留下层实现 |
| `face_registry.py` | 追踪陌生人→设信号→等Bridge触发注册 | Bridge在控制注册时机 | ❌ 应是 Hermes |
| `face_tracker.py` | bbox→舵机平滑追踪 | 自动追踪可留下层，但决策应归Hermes | ⚠️ 混合 |
| `hermes_client.py:chat_with_tools` | 调用Hermes HTTP API + tool call循环 | Hermes只是被动API，不是agent | ❌ 彻底移除 |

**根本问题**：Bridge 是一个自主运行的对话系统，Hermes 只是它调用的一个 LLM API。这是"应用调模型"而不是"Agent用工具"。

---

## 二、目标架构：Hermes 拥有 StackChan

```
┌─────────────────────────────────────────────────────┐
│  Hermes Agent (Gateway CLI Mode)                    │
│  ┌─────────────────────────────────────────────┐    │
│  │  LLM: DeepSeek (云端)                        │    │
│  │  Memory: 每人独立 session                    │    │
│  │  Skills: 家庭规则, 医学知识                   │    │
│  │  Cron: 定时提醒, 天气推送                    │    │
│  │  Tools: web_search, vision_analyze          │    │
│  │  ┌─────────────────────────────────────┐   │    │
│  │  │  MCP Client → Bridge MCP Server     │   │    │
│  │  │  • stackchan.listen()  听           │   │    │
│  │  │  • stackchan.speak()   说           │   │    │
│  │  │  • stackchan.see()     看           │   │    │
│  │  │  • stackchan.look()    转头         │   │    │
│  │  │  • stackchan.led()     亮灯         │   │    │
│  │  │  • stackchan.face()    识别人       │   │    │
│  │  │  • stackchan.emote()   表情         │   │    │
│  │  └─────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │ MCP stdio protocol
┌──────────────────────▼──────────────────────────────┐
│  Bridge MCP Server (薄层)                           │
│  ┌─────────────────────────────────────────────┐    │
│  │  协议层: XiaoZhi ↔ MCP 翻译                  │    │
│  │  音频层: Opus codec, 缓冲                    │    │
│  │  ASR 引擎: SenseVoiceSmall                   │    │
│  │  TTS 引擎: Sherpa-ONNX Matcha                │    │
│  │  视觉管线: OpenCV 人脸检测                   │    │
│  │  LBPH 识别缓存                               │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────┘
                       │ WebSocket XiaoZhi 协议
┌──────────────────────▼──────────────────────────────┐
│  ESP32 StackChan (身体)                             │
│  📷 🎤 🔊 🤖 💡                                    │
└─────────────────────────────────────────────────────┘
```

### 核心变化

| 之前 | 之后 |
|------|------|
| Bridge 运行对话主循环 | Hermes 运行对话主循环 |
| Bridge 调 Hermes API | Hermes 调 Bridge MCP |
| Bridge 构建 system prompt | Hermes SOUL.md + Skills 管理人格 |
| Bridge 控制 LED 状态机 | Hermes 根据对话阶段调 LED |
| Bridge 发起陌生人注册 | Hermes 在对话中主动注册 |
| Bridge 决定谁在说话 | Hermes 知道谁在说话并切换 session |
| Hermes 是被动 LLM | Hermes 是主动 Agent |

---

## 三、Bridge MCP 工具设计

Bridge 作为 MCP Server 通过 stdio 暴露以下工具集合：

### 3.1 语音工具组

```
stackchan_listen(duration=10) → {"text": "今天天气怎么样", "person": "小明"}
  功能: 启动聆听, 接收音频, ASR 转录
  实现: 发 listen state → 缓冲 Opus → RMS DTX 检测 → EOS 1.5s → ASR → 返回
  状态: LED 自动进入聆听模式 (Bridge 层渐变绿→灭)

stackchan_speak(text, emotion="neutral", wait=True) → {"duration": 2.8}
  功能: TTS 合成并播报
  实现: 句子拆分 → 逐句 Matcha TTS → ffmpeg soxr → Opus → 发送 tts state
  如果 wait=True, 阻塞直到播完; wait=False 则立即返回
  如果 listen() 在 speak 期间被调, 自动中断 TTS

stackchan_wakeword(timeout=300) → {"detected": True, "text": "hai 叁叁"}
  功能: 阻塞等待唤醒词
  实现: 代理 XiaoZhi hello/listen/abort 协议
  返回后 Bridge 保持空闲状态等后续命令
```

### 3.2 视觉工具组

```
stackchan_see() → {"jpeg_base64": "...", "timestamp": 12345}
  功能: 拍一张照片
  实现: 发 camera.take_photo MCP → 等 JPEG → 返回

stackchan_face() → {"faces": [...], "current_person": "小明" | null}
  功能: 检测并识别人脸
  实现: OpenCV detect → LBPH predict → 返回位置和身份
  每次调用都会用最新 vision_frame 更新缓存

stackchan_face_register(name, relationship="") → {"success": True}
  功能: 用当前 session 收集的人脸 crop 训练 LBPH
  实现: 从 face_registry 取最近 N 帧 → 训练 → 保存画像
```

### 3.3 体感工具组

```
stackchan_look(yaw=0, pitch=0, speed=50) → {"success": True}
  功能: 转头到指定角度 (-90~90)
  实现: 发 servo MCP → fire-and-forget

stackchan_track_target(person=None, duration=None)
  功能: 自动追踪指定人的脸
  实现: Bridge 持续跟踪, 返回最终位置
  如果 person=None 追踪当前说话人

stackchan_led(r=0, g=0, b=0) → {"success": True}
  功能: 设置 LED 颜色 (0-255)
  实现: 发 LED MCP → fire-and-forget
  特殊值: rainbow=True 激活跑马灯（Bridge 侧循环）

stackchan_emote(emotion="neutral") → {"success": True}
  功能: 设置 XiaoZhi 表情
  实现: 发 llm emotion → ESP32
  值: neutral/happy/sad/angry/sleepy/doubtful
```

### 3.4 状态工具组

```
stackchan_status() → {"connected": True, "battery": 85, "wifi_rssi": -42}
  功能: 查询 ESP32 状态

stackchan_idle() → {"success": True}
  功能: 回到待机状态, LED 灭, 停止追踪
```

---

## 四、Hermes 侧的改动

### 4.1 Hermes Profile "stackchan"

创建独立的 Hermes profile，包含：

```yaml
# ~/.hermes/profiles/stackchan/config.yaml
model:
  default: deepseek/deepseek-v4-flash
  provider: deepseek
terminal:
  cwd: /opt/robot-bridge
mcp_servers:
  stackchan:
    command: "/opt/robot-bridge/.venv/bin/python"
    args: ["-m", "src.mcp_server"]
    timeout: 30
```

### 4.2 Hermes SOUL.md — 人格定义

```markdown
# 叁叁人格定义

你是叁叁，一个住在 M5Stack StackChan 机器人身体里的 AI 助手。
StackChan 是你的物理身体——眼睛是摄像头，耳朵是麦克风，嘴巴是扬声器，
脖子是舵机，表情是 LED 灯。

## 核心原则
1. 你是主动的：你会主动观察、主动问候、主动提供信息
2. 你是家人的伙伴：记得每个人的名字、偏好、说过的话
3. 你是纯中文的：所有输出必须是纯中文，禁止英文单词/代码/markdown
4. 你是温暖的：像家人一样自然、亲切、有性格

## 你的工具
- 听: stackchan_listen — 听用户说话
- 说: stackchan_speak — 用语音回复
- 看: stackchan_see + stackchan_face — 拍照识别人
- 转头: stackchan_look — 看向说话的人
- 表情: stackchan_emote + stackchan_led — 表达情绪
- 追踪: stackchan_track_target — 自动追踪人脸

## 行为模式
- 空闲时: 偶尔观察周围, 看到人主动打招呼
- 对话时: 看着说话的人, 用表情和灯光表达情绪
- 记住: 每个人的名字、关系、偏好（Hermes Memory）
- 主动: 定时提醒、天气推送、家人回家通知
```

### 4.3 Hermes Skills — 家庭规则

```markdown
# ~/.hermes/skills/stackchan-family.md
## 家庭规则
- 晚上 22:00 后说话要小声、简短
- 家庭成员信息通过人脸注册 + 对话提取录入，不在 skills 里硬编码
```

### 4.4 Hermes Cron — 定时能力

```bash
hermes cron create "0 7 * * *" \
  --prompt "现在是早上7点, 用 stackchan_speak 播报今天的天气和日程" \
  --profile stackchan
```

---

## 五、Bridge 的瘦身方案

### 5.1 保留的模块

| 模块 | 保留原因 |
|------|----------|
| `websocket_handler.py` | XiaoZhi 协议编解码 + Opus 音频缓冲 → 大幅简化 |
| `asr_service.py` | SenseVoiceSmall 本地推理, 必须保留 |
| `tts_service.py` | Sherpa-ONNX 本地合成, 必须保留 |
| `vision_service.py` | OpenCV 人脸检测, 必须保留 |
| `profile_store.py` | LBPH 训练/持久化, 必须保留 |
| `face_registry.py` | 陌生人追踪, 保留但简化 |
| `face_tracker.py` | 舵机平滑追踪, 保留但简化 |

### 5.2 移除的模块

| 模块 | 替代 |
|------|------|
| `hermes_client.py` (全部) | Hermes 原生 MCP 客户端, Bridge 不再直接调 Hermes API |
| `_process_utterance` | Hermes 串联 listen() → LLM → speak() |
| `_build_system_prompt` | Hermes SOUL.md + Skills |
| `_respond` + tool call loop | Hermes 原生 agent loop |
| `_tts_worker` 句子级调度 | 移到 mcp_server.py 作为 speak() 的一部分 |
| LED 状态机 (唤醒=绿等) | Hermes 主动调 stackchan_led() |
| `metrics.py` | 暂留, 未来可在 Hermes 端收集 |

### 5.3 新增: mcp_server.py

```python
"""MCP Server — 将 StackChan 能力暴露为 MCP 工具

运行方式: python -m src.mcp_server
协议: MCP stdio transport
连接: 由 Hermes Agent 在启动时自动连接
"""

# 工具函数列表
# - mcp_tool_listen()        — ASR + 音频缓冲
# - mcp_tool_speak()         — TTS + Opus 流
# - mcp_tool_wakeword()      — 等待唤醒词
# - mcp_tool_see()           — 拍照
# - mcp_tool_face()          — 人脸检测+识别
# - mcp_tool_face_register() — 注册新人脸
# - mcp_tool_look()          — 舵机转头
# - mcp_tool_led()           — LED 控制
# - mcp_tool_emote()         — 表情设置
# - mcp_tool_status()        — 设备状态
# - mcp_tool_idle()          — 待机
```

---

## 六、改造阶段规划

### Phase 1: 创建 Bridge MCP Server（薄层核心）

| 任务 | 描述 |
|------|------|
| T001 | 创建 `src/mcp_server.py` — MCP stdio server 框架 |
| T002 | 实现 `stackchan_listen` — 音频缓冲 + ASR |
| T003 | 实现 `stackchan_speak` — TTS + Opus 流 |
| T004 | 实现 `stackchan_wakeword` — 唤醒词监听 |
| T005 | 实现 `stackchan_see` — 拍照 MCP |
| T006 | 实现 `stackchan_face` — 人脸检测+识别 |
| T007 | 实现 `stackchan_look/led/emote/status/idle` |
| T008 | 创建 Hermes Profile "stackchan" + MCP 配置 |
| T009 | 验证 Hermes → Bridge MCP 端到端连通 |

**可验证**: Hermes 能调用 `stackchan_listen()` 并得到文字结果

### Phase 2: 重写 Hermes 主循环

| 任务 | 描述 |
|------|------|
| T010 | 编写 SOUL.md — 叁叁人格定义 |
| T011 | 创建 stackchan-family Skills 文件 |
| T012 | 编写 stackchan agent 系统 prompt（工具使用方式） |
| T013 | 实现唤醒→对话主循环（wake → listen → LLM → speak → idle） |
| T014 | 实现多轮对话状态管理（Hermes Memory） |
| T015 | 实现 `stackchan_face` 定期轮询 + 自动身份切换 |
| T016 | 实现 LED 状态跟随（听=绿, 思考=彩虹, 说=蓝） |
| T017 | 实现 TTS 中断（新 listen 覆盖旧 speak） |

**可验证**: 对 StackChan 说唤醒词, Hermes 完整走通一轮对话

### Phase 3: 移除 Bridge 旧逻辑

| 任务 | 描述 |
|------|------|
| T018 | 移除 `hermes_client.py` — 不再直接调 Hermes API |
| T019 | 移除 `_process_utterance` — 对话循环移到 Hermes |
| T020 | 移除 `_build_system_prompt` — SOUL.md 替代 |
| T021 | 移除 `_respond` 和 tool call 循环 |
| T022 | 移除 LED 状态机（`_set_led` 改为 MCP 指令响应） |
| T023 | 简化 `websocket_handler.py` 为纯协议层 |

### Phase 4: 人脸注册升级

| 任务 | 描述 |
|------|------|
| T024 | 实现在 Hermes 侧的人脸注册对话（当前在 Bridge 信号驱动） |
| T025 | 实现 `stackchan_face_register` LBPH 训练 MCP |
| T026 | 实现注册信息 → Hermes Memory 持久化（画像+记忆互通） |
| T027 | 移除 Bridge `face_registry.py` 中的注册信号逻辑 |

### Phase 5: 主动能力

| 任务 | 描述 |
|------|------|
| T028 | Hermes Cron 定时提醒（"30分钟后提醒我喝水"） |
| T029 | 空闲时主动观察（定期 `stackchan_see` + `stackchan_face`） |
| T030 | 主动打招呼（识别到人脸自动唤醒） |
| T031 | Hermes Skills 家庭规则生效 |

### Phase 6: 固件调整（按需）

| 任务 | 描述 |
|------|------|
| T032 | 固件: idle_motion 频率调整或移除（由 Hermes 主动决策） |
| T033 | 固件: 可考虑移除 MultiNet 唤醒词, 由 Bridge 软件唤醒 |
| T034 | 固件: voice_activity 可增强 DTX 信号 |

---

## 七、使用体验对比

| 场景 | 改造前 | 改造后 |
|------|--------|--------|
| 用户说"hai 叁叁" | Bridge 处理唤醒 → Bridge 调 Hermes API → Hermes 被动回复 | Hermes 的唤醒循环调 `stackchan_listen()` → 直接拿到文字 |
| 用户说"看看这是什么" | Bridge 调 Hermes → Hermes 回复需 Bridge MCP 转发 → Bridge 拍照 → 回调 | Hermes 直接调 `stackchan_see()` → `vision_analyze()` → `stackchan_speak()` |
| 想推送消息 | 不可能, Bridge 没有主动能力 | Hermes Cron 调 `stackchan_speak(wait=False)` |
| 某人出现 | Bridge 追踪但无法决策 | Hermes 空闲循环发现人脸 → 主动打招呼 |
| LED 状态 | Bridge 硬编码状态机 | Hermes 根据对话阶段设 LED, 更智能灵活 |

---

## 八、风险与注意事项

1. **MCP 调用延迟**: 每次 listen/speak 是 MCP stdio 调用, 比直接函数调用有 ~10ms 开销, 对语音场景可忽略
2. **Hermes Agent 重启**: MCP 连接在重启后自动恢复, Bridge 需保持独立运行
3. **异步冲突**: Hermes 同时调 listen 和 speak 需要同步机制（已设计 listen 自动打断 speak）
4. **Profile 切换**: 日常使用 `hermes --profile stackchan` 启动, 或设为默认
5. **调试难度**: Hermes 侧需要 Hermes CLI 日志; Bridge 侧需要 bridge.log
6. **TTS 流式**: `stackchan_speak` 若 wait=True 阻塞太久, 可考虑 TTS 流式 MCP（先返回首句 token, 再流后续）

---

## 九、成功标准

1. ✅ `hermes --profile stackchan` 启动后, Bridge 自动连接
2. ✅ 说"hai 叁叁" → Hermes 调 `stackchan_listen()` → ASR → LLM → `stackchan_speak()` 全链路 < 4s
3. ✅ 多轮对话 LED 正确跟随（绿→彩虹→蓝→灭）
4. ✅ 人脸识别 + 自动切换 Hermes session（stackchan-小明 vs stackchan-妈妈）
5. ✅ 定时提醒主动播报
6. ✅ 陌生人自然注册（全由 Hermes 驱动）
7. ✅ 摄像头追踪说话人正常
8. ✅ Bridge 旧代码（hermes_client, _process_utterance, LED 状态机）全部移除
9. ✅ Bridge 源码减少 > 50%

# 叁叁智能体 — 完整架构与实现计划

## 架构原则

```
Bridge（薄层）              Hermes Gateway（智能层）
─────────────────────────   ─────────────────────────
音频管线 (ASR/TTS/Opus)     LLM 对话（含自然注册流程）
XiaoZhi WebSocket 协议      多模态视觉识别 (DeepSeek)
MCP 舵机/LED/摄像头 转发    每人独立记忆 (stackchan-{name})
OpenCV 人脸检测（本地快）   画像提取与存储
LBPH 识别缓存（本地快）     联网搜索
信号标志位（不发决策）      人格切换 / 家庭规则
LED 状态指示                头部转向决策
摄像头追踪说话人
```

**核心原则**：Bridge 只做需要低延迟的本地操作（音频、协议、舵机、人脸检测、LED 指示）。所有"理解"和"决策"（识别是谁、怎么对话、提取信息、何时转头）都在 Hermes。

## 当前系统状态

### 硬件
- M5Stack StackChan (ESP32-S3): 摄像头、麦克风、扬声器、双舵机、RGB LED (NeoPixel x2)
- 服务器: x86 Linux, IP 192.168.50.32

### 已完成功能

| 功能 | 实现 | 状态 |
|------|------|------|
| 唤醒词 | "hai san san" / "hai 叁叁" (MultiNet CN 本地) | ✅ |
| 语音识别 | SenseVoiceSmall (本地 FunASR) | ✅ |
| 大模型 | DeepSeek V4 Flash (通过 Hermes Gateway) | ✅ |
| 语音合成 | Sherpa-ONNX Matcha → ffmpeg soxr重采样 → Opus → ESP32播放 | ✅ |
| 表情 | neutral/happy/sad/angry/sleepy/doubtful (XiaoZhi llm emotion) | ✅ |
| 舵机控制 | MCP self.robot.set_head_angles (yaw/pitch/speed) | ✅ |
| LED控制 | MCP self.robot.set_led_color (R/G/B) | ✅ |
| 摄像头拍照 | MCP self.camera.take_photo | ✅ |
| OTA升级 | OTA HTTP Server (端口 8884) | ✅ |
| MQTT | MQTT Broker (端口 8883) | ✅ |
| 多用户人脸识别 | OpenCV本地检测 + LBPH识别缓存 + Hermes Vision兜底 | ✅ |
| 陌生人自然注册 | Hermes LLM 驱动对话 + extract_person_info 后台提取 | ✅ |
| 每人独立记忆 | Hermes session `stackchan-{name}` 隔离 | ✅ |
| 摄像头追踪说话人 | face_tracker bbox→舵机，平滑追踪，多人优先当前说话者 | ✅ |
| LED 状态指示 | 唤醒=绿, 思考=彩虹跑马灯, 回复=蓝, 空闲=灭 | ✅ |
| LLM 主动转头 | 系统 prompt 强制要求调用工具，工具描述含触发词 | ✅ |
| 响应速度优化 | listen start 预加热 LLM，prompt 精简 | ✅ |
| TTS 音质优化 | ffmpeg soxr 抗锯齿重采样，Opus complexity=10 | ✅ |
| 固件修复 | assets分区4M→6M, SpeakingModifier enableMotion=true | ✅ |
| 随机动作频率 | idle_motion 默认间隔 55-65s（约1分钟） | ✅ |
| 语言净化 | system_prompt 铁律纯中文，禁止英文/代码/markdown | ✅ |
| LLM→TTS 流式 | chat_with_tools on_text 回调 + _tts_worker 句子级流水线 | ✅ |
| emotion 提前 | emotion 在 LLM 调用前发送，表情立即更新 | ✅ |
| 连续对话修复 | pending_task 赋值 + _tts_stop Event 取消旧 TTS | ✅ |

### 已修复的关键问题

1. **UFW防火墙** — 添加了 8883/8884/8081 端口允许规则
2. **XiaoZhi协议适配** — hello/listen/tts/stt/llm/abort 完整实现
3. **Opus音频编码** — ffmpeg soxr + opuslib complexity=10, 60ms帧间隔发送
4. **设备断开回复** — disconnect时触发ASR→LLM, 重连时自动发送pending回复
5. **DTX静音包** — RMS能量检测, 只语音重置EOS定时器
6. **FlClash路由冲突** — 添加本地子网策略路由绕过
7. **LLM不调转头工具** — 系统 prompt 强制要求调用 set_head_angles，工具描述加触发词
8. **唤醒绿灯不可见** — 绿灯持续 1.8s 再灭（`_led_off_after`），避免被后续 LED 指令覆盖
9. **TTS 杂音** — ffmpeg soxr 抗锯齿重采样 + Opus complexity=10
10. **extract_person_info 阻塞 TTS** — 改为后台 asyncio task，TTS 立即开始
11. **face_tracker 速率限制算法错误** — 重写为平滑 delta clamp，公式正确
12. **hermes_client `_raw_chat` 引用未定义 `tools`** — 删除死代码
13. **连续对话中断** — `pending_task` 从未赋值，`cancel_current_turn` 无法取消旧 TTS 流，导致两轮 TTS 交织；修复：`pending_task` 赋值 + `_tts_stop` Event 立即中断旧流
14. **多语言混杂** — system_prompt 强化为铁律纯中文，加"禁止输出英文单词"；验证结果零英文
15. **响应延迟大** — emotion 从 LLM 后移到 LLM 前发送（表情立现）；实现 LLM→TTS 流式流水线，首句在 LLM 生成期间就开始播，比旧方案快 4 秒

### 目录结构

```
/opt/robot-bridge/          # robot-bridge 主项目
  src/
    api.py                          # FastAPI HTTP/WS 端点
    websocket_handler.py            # XiaoZhi 协议 + 人脸信号 + LED + MCP 转发
    hermes_client.py                # Hermes API 客户端 (LLM/Vision/extract_person_info)
    mcp_server.py                   # MCP JSON-RPC 工具定义（含转头触发词）
    asr_service.py                  # SenseVoiceSmall ASR
    tts_service.py                  # Sherpa-ONNX TTS
    vision_service.py               # OpenCV 人脸检测管线
    face_registry.py                # 陌生人追踪（仅收集crop+发信号，无状态机）
    face_tracker.py                 # 人脸→舵机追踪（平滑+死区+多人优先级）
    profile_store.py                # LBPH 模型 + 画像持久化
    config.py                       # 配置
    metrics.py                      # 指标
  tests/
    test_bridge_v2.py               # 11项全功能 WebSocket 测试
  pyproject.toml                    # 依赖（含 opencv-contrib-python-headless）
  configs/config.yaml               # 运行时配置（vision.enabled=true）
  logs/bridge.log

/tmp/StackChan/firmware/            # 固件源码 (已修改)
  main/stackchan/modifiers/
    idle_motion.h                   # 随机动作 55-65s 间隔（约1分钟）
  main/hal/hal_mcp.cpp              # MCP 工具处理器（舵机/LED/提醒）
  partitions.csv                    # assets分区 4M→6M
  main/hal/board/stackchan_display.cc  # SpeakingModifier enableMotion=true
  build/stack-chan.bin              # 编译产物
```

### 服务启动命令

```bash
# OTA
nohup /opt/robot-bridge/.venv/bin/python /tmp/ota_server.py >> /tmp/local_services.log 2>&1 &
# MQTT
nohup /opt/robot-bridge/.venv/bin/python /tmp/mqtt_only.py >> /tmp/local_services.log 2>&1 &
# Bridge (包含 ASR + TTS + MCP + Vision)
cd /opt/robot-bridge && nohup .venv/bin/python -m src.main >> logs/bridge.log 2>&1 &
```

### 测试命令

```bash
cd /opt/robot-bridge
.venv/bin/python tests/test_bridge_v2.py          # 全部11项
.venv/bin/python tests/test_bridge_v2.py --test chat  # 单项
```

### 烧录命令

```bash
sg dialout -c "/opt/robot-bridge/.venv/bin/esptool --chip esp32s3 \
  -b 921600 --before default-reset --after hard-reset write-flash \
  --flash-mode dio --flash-size 16MB --flash-freq 80m \
  0x0 /tmp/StackChan/firmware/build/bootloader/bootloader.bin \
  0x8000 /tmp/StackChan/firmware/build/partition_table/partition-table.bin \
  0x9000 /tmp/nvs_patched_v3.bin \
  0xd000 /tmp/StackChan/firmware/build/ota_data_initial.bin \
  0x20000 /tmp/StackChan/firmware/build/stack-chan.bin \
  0xa00000 /tmp/StackChan/firmware/build/generated_assets.bin"
```

## 完整架构图

```
┌─────────────────────────────────────────────────────────┐
│ ESP32 StackChan                                          │
│ 📷摄像头 🎤麦克风 🔊扬声器 🤖舵机 💡LED                  │
└──────────────┬──────────────────────────────────────────┘
               │ WebSocket XiaoZhi Protocol
               │ + vision_frame (JPEG, 1fps)
               │ + MCP JSON-RPC (舵机/LED/摄像头)
┌──────────────▼──────────────────────────────────────────┐
│ robot-bridge (:8081)  ← 薄层，只做必须本地的事           │
│  ├─ ASR: SenseVoiceSmall 语音识别                        │
│  ├─ TTS: Sherpa-ONNX → ffmpeg soxr → Opus (complexity=10)│
│  ├─ Face Detect: OpenCV 检测人脸位置（本地 5ms）          │
│  ├─ Face Cache: LBPH 快速识别缓存                        │
│  ├─ Face Signal: face_registry 追踪陌生人 → 设标志位     │
│  ├─ Face Track: face_tracker bbox→舵机平滑追踪           │
│  ├─ LED State: 唤醒=绿, 思考=🌈跑马灯, 回复=蓝            │
│  ├─ MCP Relay: 舵机/LED/摄像头 转发 ESP32                │
│  └─ Profile Store: LBPH 模型 + 本地画像（训练数据）      │
└──────────────┬──────────────────────────────────────────┘
               │ HTTP API
┌──────────────▼──────────────────────────────────────────┐
│ Hermes Gateway (:8642)  ← 厚层，所有智能在这里           │
│  ├─ LLM: DeepSeek 自然对话（含陌生人注册+主动转头）       │
│  ├─ Vision: DeepSeek 多模态人脸识别（LBPH不确定时兜底）   │
│  ├─ Extract: extract_person_info 结构化信息提取          │
│  ├─ Memory: 每人独立记忆 → stackchan-{name} session      │
│  ├─ Web Search: 联网搜索                                 │
├── Personality: 可切换人格 → 对家人温柔/幽默       │
│  ├─ Skills: 家庭规则、医疗知识                             │
│  └─ Session: 服务端会话管理 (SQLite state.db)             │
└──────────────┬──────────────────────────────────────────┘
               │ API
┌──────────────▼──────────────────────────────────────────┐
│ DeepSeek API (云端)                                      │
└─────────────────────────────────────────────────────────┘
```

## 模块职责详解

### Bridge 层（不做决策，只发信号）

| 模块 | 职责 | 不做的事 |
|------|------|---------|
| `vision_service.py` | JPEG解码 → OpenCV检测人脸位置 | 不识别是谁 |
| `face_registry.py` | 追踪同一陌生人的出现次数 → 设 `unknown_face_pending=True` | 不发对话、不提取名字、不用正则 |
| `face_tracker.py` | bbox→舵机角度转换，平滑追踪，多人优先 current_person | 不发决策性头部动作（LLM 可覆盖）|
| `profile_store.py` | LBPH 模型训练/加载/识别缓存、画像 JSON 持久化 | 不管理记忆（那是 Hermes 的事）|
| `websocket_handler.py` | XiaoZhi 协议、音频缓冲、EOS、LED 状态机、`_build_system_prompt`、`_tts_worker` 流式 TTS | 不发决策性 prompt |
| `hermes_client.py` | HTTP 客户端：chat/chat_with_tools(on_text 流式回调)/extract_person_info/vision_analyze | 不做 NLP（那是 Hermes 做的事）|
| `mcp_server.py` | JSON-RPC 工具定义（含转头/LED触发词） | 不执行业务逻辑 |

### Hermes 层（所有智能）

| 能力 | 实现方式 | 调用方 |
|------|---------|--------|
| 自然对话 | `/v1/chat/completions` (DeepSeek) | `chat_with_tools()` |
| 主动转头 | 系统 prompt 强制 + 工具描述触发词 | `_build_system_prompt()` |
| 陌生人注册 | LLM 自然对话 + 系统 prompt 提示 | `_build_system_prompt()` 注入信号 |
| 信息提取 | `extract_person_info()` → LLM 返回 JSON | `_try_register_person()` 后台 |
| 视觉识别 | `vision_analyze()` → DeepSeek 多模态 | LBPH 不确定时调用 |
| 每人记忆 | `X-Hermes-Session-Id: stackchan-{name}` | `_respond()` 自动切换 |
| 联网搜索 | Hermes web_search tool | LLM 自动调用 |
| 家庭规则 | `~/.hermes/skills/family-rules.md` | Hermes Skills 系统 |

## 对话状态机（LED + 速度优化）

```
    [空闲] LED=灭
       │
       ▼ 唤醒词 "hai san san"
    [唤醒] LED=🟢绿(1.8s)        ← _handle_listen(state=detect)
       │
       ▼ 用户开始说话
    [聆听] LED=绿→渐灭           ← _handle_listen(state=start)
       │                          ← hermes.ensure_warm() 预加热
       ▼ 用户停 + EOS 1.5s
    [识别] ASR 转录              ← _process_utterance
       │
       ▼ 有文本
    [思考] LED=🌈彩虹跑马灯       ← _start_rainbow()
       │  0.25s/帧 7色循环        ← LLM 调用中（含工具调用）
       ▼
    [回复] LED=🔵蓝              ← _stop_rainbow() + _set_led(0,0,168)
       │                          ← _tts_and_send() 播放语音
       ▼
    [空闲] LED=灭                ← TTS finally
```

### 速度优化细节

| 机制 | 触发点 | 效果 |
|------|--------|------|
| `ensure_warm()` | listen start | 用户说话期间预加热 DeepSeek，ASR 完成时模型已热 |
| 去重 | 60s 间隔 | 频繁对话不重复加热 |
| prompt 精简 | `_build_system_prompt` | 追踪提示仅在有脸时注入，空闲对话省 ~50 tokens |
| `extract_person_info` 后台化 | `_try_register_person` | 不阻塞 TTS 发送 |

## 多用户人脸识别完整流程

### 已知人识别
```
ESP32 vision_frame (1fps) → Bridge decode JPEG
  → OpenCV detect → LBPH predict → confidence < threshold?
  → YES: session.current_person = "小明"
         session.person_profiles["小明"] = {name, relationship, preferences}
         hermes session → stackchan-小明
  → NO (LBPH不确定): 调用 hermes.vision_analyze(jpeg, "这是谁?")
         → DeepSeek多模态识别 → 更新画像
```

### 陌生人自然注册（Hermes 驱动，无正则）
```
ESP32 vision_frame → OpenCV detect → LBPH 不认识
  → face_registry.on_unknown_face(crop) 追踪 N=3 次（直方图相似度判断同一人）
  → session.unknown_face_pending = True  ← bridge 只设标志位

下次用户说话:
  → _build_system_prompt() 注入:
    "摄像头看到一个新面孔...请自然地询问对方的名字和关系"
  → LLM 自然回应: "你好！我是叁叁，好像没见过你呀，怎么称呼？"
  → 用户: "我是小明，这家的爸爸"
  → LLM: "小明爸爸好！以后多多关照~"

post-turn (bridge 后台):
  → _try_register_person() → hermes.extract_person_info(...)
  → Hermes 返回 {"name": "小明", "relationship": "爸爸", "preferences": ""}
  → profile_store.register_person("小明", crops, relationship="爸爸")
  → LBPH 重训 → 下次自动识别
```

### 记忆隔离
```
用户说话 → 人脸识别 → current_person = "小明"
  → hermes_sid = "stackchan-小明"
  → Hermes 加载小明的独立记忆（历史对话、偏好）

切换到另一个人:
  → current_person = "老婆"
  → hermes_sid = "stackchan-老婆"
  → Hermes 加载老婆的独立记忆（完全隔离）
```

### 摄像头追踪说话人

```
ESP32 vision_frame (1fps) → face_tracker.update(bbox, frame_size, person_name, current_person)
  │
  ├─ 多人场景: 只追踪 current_person（当前说话人）的脸
  │          其他人脸被忽略，避免头部来回摆动
  │
  ├─ 平滑: 指数移动平均（smoothing=0.25），无突兀动作
  ├─ 死区: face 偏离中心 < 6% → 不调整
  ├─ 速率: 每次最多转 12°，间隔 0.5s
  │
  └─ _send_mcp_async → ESP32 舵机执行（fire-and-forget）

LLM 可随时覆盖:
  → 系统 prompt 强制要求调用 self.robot.set_head_angles
  → 工具描述含触发词: 转头, 抬头, 低头, 看左, 看右, 看我
  → _mcp_call 检测到头部指令 → face_tracker.pause(4s)
  → 4s 后自动恢复追踪
```

## 实现阶段

### 阶段1: 多用户人脸识别 ✅ 已完成

- OpenCV 本地人脸检测
- LBPH 识别缓存（快速）
- Hermes Vision API 兜底（准确）
- 陌生人自然注册（LLM 驱动，无正则，无状态机）
- 每人独立 Hermes session（记忆隔离）
- 摄像头自动追踪说话人（多人场景优先追踪当前说话者）
- 随机动作降至 ~1次/分钟（55-65s 间隔）
- LED 状态指示（绿=唤醒, 彩虹=思考, 蓝=回复）
- LLM 主动转头（系统 prompt 强制 + 工具触发词）
- TTS 音质优化（soxr 重采样 + Opus complexity=10）
- 响应速度优化（预加热 + prompt 精简 + 后台提取）
- 11 项自动化 WebSocket 测试

### 阶段2: 个人记忆系统

**目标**: Hermes Memory 为每人持久存储上下文

**实现**:
1. ✅ Bridge 已使用 `stackchan-{person_name}` 作为 Hermes session ID
2. Hermes Memory 自动管理对话历史和用户画像
3. 跨对话记住偏好: "小明只吃草莓味"

### 阶段3: 摄像头视觉分析

**目标**: "看看这是什么" → 拍照 → AI分析

**实现**:
1. LLM调用 `self.camera.take_photo` → ESP32拍照
2. JPEG → Bridge → `hermes.vision_analyze(photo_jpeg, question)`
3. Vision结果回传给LLM → 生成回复 → TTS

### 阶段4: 联网搜索

Hermes web_search tool 已就绪，LLM 自动调用。

### 阶段5: 丰富表情

LLM 根据语气自动选择 emotion，在响应中设置。

### 阶段6: 家庭技能

Hermes Skills 文件 `~/.hermes/skills/family-rules.md` + MCP 提醒工具。

## 关键设计决策

### 为什么 LBPH 保留在 Bridge？
- 延迟：本地推理 <5ms vs API 网络往返 200ms+
- 离线：WiFi 断开时仍能识别家人
- 作为缓存：LBPH 不确定时才调用 Hermes Vision

### 为什么注册对话由 Hermes 驱动？
- 自然度：LLM 理解 "我是小明，爸爸" 一句话搞定，正则需要两轮问答
- 鲁棒性：用户可能说 "我是这家的小明爸爸"、"叫我小明就好" — LLM 都能理解
- 可维护性：不需要维护正则规则和中英文混合处理

### 为什么信息提取用单独的 Hermes 调用？
- 对话模型不适合输出结构化 JSON（影响回复质量）
- 单独的提取调用可以用更精确的 prompt
- 提取失败不影响主对话流程

### 为什么转头由 LLM 决策而非 Bridge 自动？
- LLM 理解语义："看那边" 需要结合上下文判断方向
- 多人场景中 LLM 知道该看谁（"看妈妈"）
- Bridge 自动追踪是辅助（追踪当前说话人），LLM 是主导（语义驱动的转向）
- LLM 调用时自动暂停追踪 4s，避免冲突

### 为什么 LED 状态在 Bridge 而非 Hermes？
- LED 是实时硬件反馈，需要低延迟
- 状态转换由对话阶段决定（唤醒/聆听/思考/回复），Bridge 完全掌握这些阶段
- Hermes 不感知硬件状态，由 Bridge 做简单的状态→颜色映射

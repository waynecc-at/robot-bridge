# Feature Specification: 叁叁语音机器人 v2 — Hermes-Owns Architecture

**Feature Branch**: `main` (refactor)

**Created**: 2026-05-16

**Status**: Proposal | Constitution v2.0 Ratified

**Input**: Phase 1 (Bridge-as-MCP-controller) complete, re-architecture to Hermes-Owns

## User Scenarios & Testing

### User Story 1 — 唤醒并对话 (Priority: P1)

家庭成员走近 StackChan, 说"叁叁"或"sansan", 机器人转向说话人,
听用户说话后用自然中文语音回复。

**唤醒规则**:
- **空闲状态**: 说"叁叁" → 唤醒 → 调 listen() 聆听
- **已唤醒/正在对话**: 唤醒词被忽略, 不打断当前对话

**Why this priority**: 核心价值——语音对话。

**Independent Test**:
```
Given Hermes Agent 以 stackchan profile 运行, Bridge MCP 在线, 机器人空闲
When 用户说"叁叁"
Then Hermes 调 stackchan_listen() 聆听
And Hermes 调 stackchan_led(green) 绿灯

Given 机器人已在聆听/对话中
When 用户再说"叁叁"
Then 唤醒词被忽略, 不会重新唤醒, 不会打断当前对话

Given ASR 返回文字
When Hermes 收到文字
Then Hermes 调 stackchan_led(rainbow) 彩虹跑马灯
And Hermes 调 LLM (DeepSeek) 生成回复
And Hermes 调 stackchan_emote() 设置表情

Given LLM 返回文本
When Hermes 开始 TTS
Then Hermes 调 stackchan_led(blue) 蓝灯
And Hermes 调 stackchan_speak() 播放语音
And Hermes 调 stackchan_idle() TTS 完成后进入待机
```

### User Story 2 — 认识家人 (Priority: P1)

Hermes 在空闲循环中周期性调 `stackchan_see()` + `stackchan_face()`,
检测到未知面孔时主动说话问候并自然询问身份。

**Why this priority**: 记忆系统和个性化交互的前提。由 Hermes 主动驱动,
不再依赖 Bridge 的信号机制。

**Independent Test**:
```
Given 摄像头拍到未知面孔
When Hermes 空闲循环检测到新面孔
Then Hermes 主动调 stackchan_speak("你好,我好像没见过你,请问你是谁呀?")
When 用户回答"我是小明,这家的爸爸"
Then Hermes 提取 name=小明, relationship=爸爸
And Hermes 调 stackchan_face_register("小明", "爸爸") 训练 LBPH
And Hermes Memory 记录人物画像（统一 session + 人物标签）
And 下次自动识别
```

### User Story 3 — 区分家人记忆 (Priority: P2)

小明和妈妈分别对话, Hermes 使用统一对话 session,
每轮消息带说话人标签, 同时通过 fact_store 记账人偏好。

**Why this priority**: 统一 session 使对话自然连贯（妈妈问"小明乖不乖"能回答）,
人物画像通过 Memory 独立持久化（偏好不因 session 压缩丢失）。

**Independent Test**:
```
Given 摄像头识别为小明, 人脸已验证
When 小明说"叁叁给我讲个故事"
Then Hermes 在统一 session 中记录 "[小明] 请求讲恐龙故事"
And Hermes 从 Memory 查小明偏好 → 调取恐龙主题
And 回复"小明想听暴龙的故事吗?"

Given 摄像头识别切换为妈妈
When 妈妈接着说话
Then Hermes 在统一 session 中记录 "[妈妈] 加入对话"
And 回复中可自然引用小明刚才的话（同一 session）
And Hermes 从 Memory 查询妈妈偏好（不同于小明）
```

### User Story 4 — 主动转头看人 (Priority: P2)

Hermes 头部追踪逻辑: 当识别人脸时, 调 `stackchan_track_target()` 自动追踪;
当 LLM 需要语义转头时, 调 `stackchan_look()` 覆盖自动追踪。

**Why this priority**: 自然的社交互动体验。追踪逻辑下放 Bridge,
但 Hermes 可随时覆盖, 语义优先。

**Independent Test**:
```
Given 摄像头识别到人脸在画面左侧
When 该人开始说话
Then Hermes 调 stackchan_look() 让舵机转向该人脸

Given Hermes 想在回复时转头示意
When LLM 生成含转头指令的回复
Then Hermes 调 stackchan_look(yaw=30, pitch=0) 覆盖追踪
And 自动追踪暂停 4s 后恢复
```

### User Story 5 — 拍照分析 (Priority: P3)

用户说"看看这是什么", Hermes 直接调 `stackchan_see()` 拍照,
然后调 `vision_analyze()` 分析, 再 `stackchan_speak()` 回复。

**Why this priority**: 展示了 Hermes 直接编排多个工具的能力,
完全不需要 Bridge 中间层干预。

**Independent Test**:
```
Given 对话中用户说"看看这是什么"
When Hermes LLM 决定调用 vision_analyze
Then Hermes 调 stackchan_see() → 拿到 JPEG
And Hermes 调 vision_analyze(jpeg) → 拿到分析结果
And Hermes 调 stackchan_speak() → 语音回复分析内容
```

### User Story 6 — 定时提醒 (Priority: P3)

用户说"30分钟后提醒我", Hermes 创建 cronjob, 到时间主动播报。

**Why this priority**: 展示了 Hermes Agent 的主动能力, 这是 v1 做不到的。

**Independent Test**:
```
Given 对话中用户设置提醒
When 用户说"1分钟后提醒我喝水"
Then Hermes 调 cronjob(action='create', schedule='1m')
When 1 分钟后 cron 触发
Then Hermes 调 stackchan_speak("小明,该喝水啦!")
```

### Edge Cases

- MCP 连接断开: Hermes 检测到工具调用失败 → 打印日志 → 重试连接
- 多人同时说话: ASR 可能识别失败 → Hermes 可再调 listen()
- 陌生人沉默走过: Hermes 空闲循环检测到瞬时面孔 → 不触发注册
- TTS 中断: 新 listen() 调用时自动打断旧 speak()
- ESP32 断连: Bridge 检测到 → MCP 工具返回错误 → Hermes 可重试或提示
- LLM 超时: Hermes 设 tool timeout → LLM 调用超时 → 返回友好提示
- **唤醒防误触**: 已唤醒/对话中, 唤醒词 "叁叁" 不响应, 防止打断对话
- **唤醒词连续触发**: 用户在对话中自然说出"叁叁"（如"叁叁刚才说的..."）不触发重新唤醒

## Requirements

### Functional Requirements

- **FR-001**: Hermes MUST run as the primary agent loop (not passive API consumer)
- **FR-002**: Bridge MUST expose all hardware capabilities as MCP tools
- **FR-003**: MCP tools MUST be discoverable by Hermes Agent at startup
- **FR-004**: Hermes MUST own LED state decisions (not hardcoded in Bridge)
- **FR-005**: Hermes MUST own face registration flow (LLM-driven, not signal-driven)
- **FR-006**: Hermes MUST own conversation orchestration (listen → think → speak loop)
- **FR-007**: Bridge MUST NOT contain conversational state or decision logic
- **FR-008**: Hermes MUST use a unified conversation session with per-turn speaker tagging
- **FR-009**: Hermes MUST extract and persist per-person key facts (preferences, relationships) to fact_store post-turn
- **FR-010**: Hermes MUST be able to proactively initiate speech (cron, observations)
- **FR-011**: Hermes MUST run periodic idle loop (face detection, status check)
- **FR-012**: MCP tools MUST complete within <5s for voice-interactive operations
- **FR-013**: listen() MUST interrupt in-progress speak() for natural conversation
- **FR-014**: Bridge MCP server MUST auto-reconnect if connection drops

### Key Entities

- **Hermes Profile**: stackchan-specific config (MCP, model, SOUL, Skills)
- **MCP Tool**: Bridge-exposed capability (listen, speak, see, look, led, etc.)
- **Hermes Session**: unified family conversation session, per-message tagged with speaker identity
- **PersonProfile**: name, face_id, relationship, preferences, LBPH model
- **CronJob**: scheduled proactive actions (reminders, weather, observations)

## Success Criteria

- **SC-001**: Hermes → stackchan_listen() → ASR → LLM → stackchan_speak() < 4s
- **SC-002**: Bridge source code reduced by >30% (decision logic fully removed, protocol+audio layer preserved)
- **SC-003**: hermes_client.py completely removed (no direct Hermes API calls from Bridge)
- **SC-004**: LED states correctly follow conversation: green (listen) → rainbow (think) → blue (speak) → off (idle)
- **SC-005**: Person registration fully driven by Hermes LLM (no Bridge signal states)
- **SC-006**: Unified session keeps coherent context across speaker switches (Mom can ask about Xiao Ming)
- **SC-007**: Per-person facts (preferences, relationships) persisted to Hermes Memory, survive context compression
- **SC-008**: All existing test_bridge_v2.py tests pass (updated for MCP interface)
- **SC-009**: `hermes --profile stackchan` starts both Hermes agent + Bridge MCP

## Assumptions

- MCP Python SDK is available (`pip install mcp`)
- Hermes Agent v0.13.0+ supports native MCP client (confirmed)
- StackChan ESP32 stays on same local network (192.168.50.x)
- DeepSeek API latency acceptable (<200ms RTT)
- Bridge can run as persistent background process (systemd user service)
- Single ESP32 device per instance (no multi-device routing yet)

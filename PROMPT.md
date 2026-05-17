# 接手提示词

复制下面这段给新会话：

```
这是 M5Stack StackChan 的 robot-bridge 语音机器人项目。

## 项目文件

请先读完以下文件了解全貌：
1. .specify/memory/constitution.md  — 5条核心原则
2. .specify/memory/spec.md          — 6个用户故事 + 验收场景
3. .specify/memory/plan.md          — 技术上下文 + 6阶段规划
4. .specify/memory/tasks.md         — 62个任务（48已完成, 14待做）
5. ARCHITECTURE.md                  — 完整架构图 + 设计决策

## 当前状态

Phase 1 多用户人脸识别已完成，Phase 2-6 规划中。

已完成功能（21项）：
- XiaoZhi 协议语音对话、ASR(SenseVoiceSmall)、TTS(Sherpa-ONNX→Opus)
- 多用户人脸识别(OpenCV+LBPH+陌生人LLM自然注册)
- 摄像头追踪说话人(face_tracker→舵机)
- LED状态指示(唤醒=绿, 思考=彩虹跑马灯, 回复=蓝)
- LLM主动转头(系统prompt强制+工具触发词)
- LLM→TTS流式(on_text回调, 句子级流水线)
- 每人独立记忆(stackchan-{name} Hermes session)
- 11项WebSocket自动化测试

关键修复（15项）：
- 连续对话TTS交织、多语言混杂、蓝灯卡住、响应延迟
- face_tracker算法错误、extract_person_info阻塞TTS
- idle_motion频率4-8s→55-65s(固件已烧录)

## 技术栈

Bridge(Python): FastAPI :8081, WebSocket XiaoZhi协议, httpx→Hermes :8642
Firmware(C++): ESP32-S3, ESP-IDF v5.5.4, 摄像头/舵机/LED
本地模型: SenseVoiceSmall(ASR), Sherpa-ONNX Matcha(TTS), OpenCV(人脸)
云端: DeepSeek V4 Flash via Hermes Gateway

## 当前已知问题

1. 唤醒词响应率低 — ESP32 MultiNet固件问题
2. 绿灯只有一边亮 — StackChan硬件NeoPixel
3. 回复慢 — DeepSeek API延迟, Bridge侧已最大化优化

## 服务管理

启动: cd /home/wayne/robot-bridge && nohup .venv/bin/python -m src.main >> logs/bridge.log 2>&1 &
测试: .venv/bin/python tests/test_bridge_v2.py
固件编译: bash /tmp/build_stackchan.sh
固件烧录: sg dialout -c "esptool ..."
```

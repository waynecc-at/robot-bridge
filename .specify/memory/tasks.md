# Tasks: 叁叁语音机器人 v2 — Hermes-Owns 重构

**Input**: plan.md + spec.md (v2 architecture)

**Status**: Phase 0-3 Complete ✅ | Phase 4-6 Planned

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US6)

---

## Phase 0: Research & Prep ✅

- [x] T-001 Analyze current Bridge architecture — 7 modules too thick
- [x] T-002 [P] Design MCP tool interface (11 tools, naming prefix mcp_stackchan_)
- [x] T-003 [P] Validate Hermes native MCP client support (v0.13.0, yes)
- [x] T-004 [P] Design Hermes profile "stackchan" config
- [x] T-005 [P] Update constitution to v2.0 (Hermes-Owns principle)
- [x] T-006 [P] Write spec.md v2
- [x] T-007 [P] Write plan.md + tasks.md

---

## Phase 1: Bridge MCP Server Foundation ✅

**Goal**: Create Bridge MCP server with all 11 tools, end-to-end testable from Hermes.

- [x] T001 Install `mcp` Python SDK into Bridge venv
- [x] T002 [P] Create `src/mcp_server.py` — MCP stdio server skeleton with tool discovery
- [x] T003 [US1] Implement `mcp_stackchan_listen` — audio buffering, DTX EOS, ASR
- [x] T004 [US1] Implement `mcp_stackchan_speak` — TTS, Opus streaming to ESP32
- [~] T005 [US1] `mcp_stackchan_wakeword` — wake word handled in Bridge bottom-half (listen state=detect, websocket_handler:190-195), not exposed as MCP tool
- [x] T006 [US4] Implement `mcp_stackchan_see` + `mcp_stackchan_face` — camera + detection
- [x] T007 [P] Implement remaining MCP tools: look, led, emote, status, idle, track_target, face_register

---

## Phase 2: Hermes Agent Loop ✅

**Goal**: SOUL.md + profile + skills + time awareness.

- [x] T008 Create `~/.hermes/profiles/stackchan/config.yaml` with MCP server config
- [x] T009 [P] Write `~/.hermes/profiles/stackchan/SOUL.md` — agent personality (v2: voice robot)
- [x] T010 [P] Write `~/.hermes/skills/stackchan-family.md` — family rules
- [x] T011 [US1] SOUL.md defines wake → listen → LLM → speak → idle behavior
- [x] T012 [US3] Unified session with per-message speaker tagging in SOUL.md
- [x] T012b [US3] Post-turn fact extraction instructions in SOUL.md
- [x] T013 [US1] LED state guidance in SOUL.md (blue=speaking)
- [x] T014 [US1] Auto-interrupt on new audio (Bridge barge-in feature)
- [x] T015 [P] Time awareness in SOUL.md
- [x] T016 [P] Agent Driver injects current time into user messages

**Note**: Phase 2 scope changed from "Hermes drives MCP loop" to "Agent Driver drives Gateway API loop". SOUL.md still defines personality and rules. Agent Driver is a temporary bridge until Hermes natively supports auto-loop.

---

## Phase 3: Strip Bridge ✅

**Goal**: Remove all decision-making code from Bridge.

- [x] T017 [P] Delete `src/hermes_client.py`
- [x] T018 [P] Remove `_process_utterance` from websocket_handler
- [x] T019 [P] Remove `_build_system_prompt`
- [x] T020 [P] Remove `_respond`
- [x] T021 [P] Remove LED state machine (keep low-level LED helpers)
- [x] T022 [P] Strip websocket_handler to protocol relay + audio buffer + ASR + MCP relay
- [x] T023 (new) Create `src/agent_driver.py` — ASR hook → LLM (Gateway API) → TTS
- [x] T024 (new) Add ASR hook callback to websocket_handler (event-driven, no polling)

**Bridge source reduced >30%**: 863 → 627 lines, zero decision logic ✅

---

## Phase 3.5: Integration Fixes & Audio Quality ✅ (2026-05-17)

**Goal**: Fix integration gaps found in audit, improve TTS audio quality, LED states, and response latency.

### Code Quality (Audit fixes)

- [x] T045 Fix `websocket_handler.py:85` stale `HermesClient` reference → NameError risk
- [x] T046 Consolidate duplicate `RobotSession` class (api.py + websocket_handler.py → dataclass)
- [x] T047 Delete dead modules: `hermes_cli_bridge.py`, `hermes_agent_loop.py`
- [x] T048 Remove `.bak` files from `src/` (hermes_client.py.bak, websocket_handler.py.bak, api.py.bak)
- [x] T049 Remove unused `STACKCHAN_TOOLS` from mcp_server.py
- [x] T050 Fix SKILL.md MCP tool names: `mcp_stackchan_stackchan_*` → `stackchan_*`
- [x] T051 Fix misleading comment referencing deleted `_build_system_prompt`
- [x] T052 Update SC-002: >50% → >30% (constitution.md, spec.md, tasks.md)

### Agent Driver Hook

- [x] T053 Add `agent_driver.set_ws_handler()` in api.py websocket_endpoint — ASR text wasn't triggering LLM because hook was never registered

### LED State Machine

- [x] T054 Green LED on listen start (`_handle_listen` state=start → `_set_led(0,168,0)`)
- [x] T055 Rainbow LED during LLM thinking (agent_driver starts rainbow before LLM call)
- [x] T056 Stop rainbow before blue LED in `_tts_and_send` (prevents rainbow overwriting blue)
- [x] T057 Rainbow stopped on LLM error/timeout (guard in agent_driver exception path)

### TTS Pipeline Optimization

- [x] T058 Replace scipy FFT resample with ffmpeg soxr — FFT brick-wall filter introduced 10.5% HF noise vs soxr's 7.2%
- [x] T059 Feed float32 PCM directly to ffmpeg (`-f f32le`) — skip WAV creation and pre-quantization
- [x] T060 Single Opus encoder reused across sentences (was: per-sentence encoder → state reset artifacts)
- [x] T061 Opus profile: `'audio'` → `'voip'` (speech-optimized, filters irrelevant HF)
- [x] T062 Opus bitrate: 32000 → 48000 bps (50% more bits, fewer compression artifacts)
- [x] T063 Remove `asyncio.sleep(0.06)` between Opus frames — artificial pacing caused ESP32 buffer underruns (70-80ms actual interval vs 60ms frame → pops/clicks)
- [x] T064 Drop partial final frame instead of zero-padding (eliminates end-of-sentence click)

### Latency Reduction

- [x] T065 EOS silence timeout: 2.5s → 1.5s (faster response after user stops speaking)
- [x] T066 LLM max_tokens: 500 → 200 (shorter replies → faster first token + less TTS)

### Success Criteria Update

- **SC-002 revised**: Bridge source >30% reduction (was >50%) — protocol+audio layer is Bridge's core value
- **SC-010** (new): TTS audio glitch-free (no buffer underruns, no frame-boundary artifacts)
- **SC-011** (new): LED states follow: off(idle) → green(listen) → rainbow(think) → blue(speak) → green(re-listen)

---

## Phase 4: Face Registration Upgrade

**Goal**: Face registration fully driven by Agent Driver via Hermes LLM.

- [ ] T025 (was T021) [US2] Implement face check in Agent Driver idle loop
- [ ] T026 (was T022) [US2] SOUL.md instructions for LLM-driven face registration
- [ ] T027 (was T023) [US2] Wire `stackchan_face_register` tool when LLM decides to register
- [ ] T028 (was T024) [US2] Store registration output → Hermes Memory via session_id

---

## Phase 5: Proactive Capabilities

**Goal**: Hermes proactively uses StackChan (not just reactive dialog). Requires Agent Driver improvements.

- [ ] T029 (was T025) Cron-based reminders (Hermes cronjob tool)
- [ ] T030 (was T026) Non-blocking speak for notifications
- [ ] T031 (was T027) Periodic face check when idle (Agent Driver)
- [ ] T032 (was T028) LLM decides to proactively greet
- [ ] T033 (was T029) Family skill rules injected into conversation
- [ ] T034 (was T030) Weather push scenario
- [ ] T035 (was T031) Time-of-day voice modulation
- [ ] T036 (was T032) Breathing LED during idle
- [ ] T037 (was T033) Time-aware greeting variations

---

## Phase 6: Firmware Tuning

**Goal**: Firmware optimized for Hermes-driven architecture.

- [ ] T038 (was T034) Reduce/remove idle_motion in firmware
- [ ] T039 (was T035) Evaluate software wake word detection (Bridge-side)
- [ ] T040 (was T036) Optimize vision_frame rate

---

## Phase 7 (Future): Function Calling for MCP Tools

**Goal**: Agent Driver forwards LLM tool calls (look, see, register) to Bridge MCP.

- [ ] T041 Parse tool_calls from Gateway API streaming response
- [ ] T042 Implement tool call dispatch: stackchan_look → Bridge MCP relay
- [ ] T043 Implement tool call result injection back to LLM
- [ ] T044 Test: "转头看右边" triggers stackchan_look + continues conversation

---

## Summary

| Phase | Tasks | Status | LOC Change |
|-------|-------|--------|-----------|
| 0. Research | 7 | ✅ | — |
| 1. Bridge MCP | 7 | ✅ | +1200 |
| 2. Hermes Loop | 10 | ✅ | +400 |
| 3. Strip Bridge | 8 | ✅ | -800 (net) |
| 3.5. Integration & Audio | 22 | ✅ | -200 (audit cleanup) |
| 4. Face Register | 4 | ⬜ | +200 |
| 5. Proactive | 9 | ⬜ | +250 |
| 6. Firmware | 3 | ⬜ | Minimal |
| 7. Function Calling | 4 | ⬜ | +300 |
| **Total** | **74** | **54✅ 20⏳** | **Bridge: -600 net** |

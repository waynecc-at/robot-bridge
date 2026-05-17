<!--
Sync Impact Report
==================
Version change: 1.0.0 → 2.0.0 (principle redefinition: Bridge-thin redefined)
Modified principles: I (Bridge-Thin → Hermes-Owns), V (new Progressive Delivery semantics)
Added sections: none
Removed sections: none
Templates requiring updates:
  ✅ plan-template.md — verified, no changes needed
  ✅ spec-template.md — verified, no changes needed
  ✅ tasks-template.md — verified, no changes needed
Follow-up TODOs: Phase 1 re-init with new architecture
-->
# 叁叁 (StackChan) — Project Constitution v2.0

## Core Principles

### I. Hermes-Owns, Bridge-Operates (NON-NEGOTIABLE)

Hermes Agent is the sole orchestrator and decision-maker. Bridge is a stateless
MCP server that executes hardware operations on Hermes' behalf.

**Hermes owns** (all intelligence):
- Conversation orchestration (wake → listen → think → speak → idle loop)
- System prompt and personality (SOUL.md + Skills)
- LED state decisions (what color at what phase)
- Face registration flow (LLM-driven natural conversation)
- **Unified conversation session + per-person Memory tagging** (every message tagged with speaker identity, key facts persisted to Memory)
- When to speak, when to listen, when to look
- Proactive actions (reminders, observations, notifications)

**Bridge operates** (thin execution layer):
- XiaoZhi WebSocket protocol encoding/decoding
- Opus ↔ PCM audio codec
- ASR inference (SenseVoiceSmall)
- TTS inference (Sherpa-ONNX Matcha → ffmpeg → Opus)
- OpenCV face detection on ESP32 vision frames
- LBPH face recognition cache (fast local inference)
- LBPH training for new person registration
- MCP tool interface exposing above capabilities as hermetic functions

**Rationale**: Bridge-as-MCP-server is the cleanest separation of concerns.
Hermes gets full agency via tool-calling; Bridge is a pure capability server
with no decision logic. No conversational state lives in Bridge.

### II. Chinese-Only Voice Interaction

Every LLM response MUST be pure Chinese. English words, code snippets, markdown,
emoji, and kaomoji are FORBIDDEN in TTS output.

- System prompt MUST explicitly state "铁律：纯中文"
- Tool descriptions MUST include Chinese trigger words
- TTS model MUST be Chinese-optimized (Matcha Chinese Baker)

**Rationale**: The device targets Chinese-speaking family users. Mixed-language
output degrades TTS quality and user experience.

### III. Local-First Latency

Performance-critical operations MUST run locally on the server (x86 Linux),
not depend on cloud round-trips.

- ASR: SenseVoiceSmall, local inference
- TTS: Sherpa-ONNX Matcha, local inference
- Face detection: OpenCV Haar cascade, <5ms per frame
- Face recognition cache: LBPH, <5ms per prediction
- Bridge MCP server: local stdio transport, <1ms IPC overhead

Cloud-dependent operations (acceptable latency):
- LLM: DeepSeek via Hermes Gateway (streaming)
- Vision recognition fallback: DeepSeek multimodal via vision_analyze
- Web search: Hermes web_search tool

**Rationale**: Voice interaction requires sub-second perceived latency. Local
inference for the audio pipeline is mandatory. MCP over local stdio adds
negligible overhead (~10μs per call).

### IV. Privacy-First

Camera data MUST NOT leave the local network in raw form.

- Face crops sent to Hermes Vision API only when LBPH uncertain
- Full camera frames processed locally; only face crops transmitted
- Person profiles stored locally (JSON + LBPH model)
- Hermes Memory stores conversation context, not raw images
- No cloud upload of camera feed under any circumstances

**Rationale**: Family home deployment; camera privacy is non-negotiable.

### V. Progressive Delivery

Every refactoring phase MUST be independently testable and reversible.

- Phase N does not block Phase N+1 (parallel development where possible)
- Each phase adds measurable capability (see success criteria per phase)
- Old Bridge code paths and new MCP paths can coexist during migration
- Test suite covers both old and new paths during transition
- After full migration, old Bridge code is deleted (not commented out)
- Bridge source code reduction >30% is a success metric

**Rationale**: Refactoring a working real-time hardware project requires
careful staging. Parallel coexistence of old and new paths enables continuous
testing without service interruption.

## Security & Privacy

- Hermes API key not needed in Bridge (removed: Bridge no longer calls Hermes API directly)
- MCP stdio transport only listens on localhost (no network exposure)
- ESP32 WebSocket connections restricted to local network
- No persistent cloud storage of audio or video data
- OTA firmware updates served over local HTTP (port 8884)

## Development Workflow

- Bridge source: `src/` (Python) — MCP server + protocol layer only
- Firmware: `/tmp/StackChan/firmware/` (C++)
- Runtime config: `configs/config.yaml`
- Hermes Profile: `~/.hermes/profiles/stackchan/config.yaml` (MCP server config)
- Hermes SOUL: `~/.hermes/profiles/stackchan/SOUL.md` (Agent personality)
- Tests: `tests/test_bridge_v2.py` — updated for MCP-based integration tests
- Run: `hermes --profile stackchan` starts both Hermes + Bridge MCP
- Standalone Bridge test: `python -m src.mcp_server` (stdio, for debugging)

## Governance

This constitution supersedes all other project practices. Amendments require:

1. Documented rationale in commit message
2. Version bump per semantic versioning
3. Sync impact report updated in constitution header
4. All dependent templates (plan, spec, tasks) checked for consistency

**Compliance**: Every PR and code review MUST verify alignment with these principles.
Violations require explicit justification or constitution amendment.

**Version**: 2.0.0 | **Ratified**: 2026-05-16 | **Last Amended**: 2026-05-16

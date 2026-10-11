# Connecting a StackChan to the Bridge

This guide walks you from a freshly cloned repo to a StackChan device that is
actually talking to the bridge. It answers issue
[#1](https://github.com/waynecc-at/robot-bridge/issues/1) — _"How do I actually
connect this to the chan?"_

> **Every value in this guide was read from this repository at commit
> `ed4cff7`.** Where the wiring depends on hardware this project does not ship
> (the flashable firmware), that is called out explicitly instead of guessed.

## 0. The link at a glance

```
StackChan (ESP32-S3)                robot-bridge                Hermes Agent
  XiaoZhi WS client  ──ws://─▶  :8081 /ws/robot   ──ASR──▶  :8644 /webhooks/stackchan
        ▲   (Opus audio + JSON)        │  (11 MCP tools, ASR/TTS)
        └──────────────────────────────┘  :8081 /internal/*  ◀── MCP calls
```

The device is the **WebSocket client**. The bridge is the **server**. So the
single most important fact is the URL the device dials out to:

```
ws://<BRIDGE-IP>:8081/ws/robot
```

- Port `8081` and path `/ws/robot` are the bridge defaults
  (`src/config.py:11-12`, `configs/config.example.yaml:10-11`, confirmed by
  loading the config — see the verification note at the end).
- `/ws/robot` is registered as the ESP32 endpoint in `src/api.py:419`.

Everything else in this guide exists to make that one connection work.

## 1. Get the bridge running first

```bash
git clone https://github.com/waynecc-at/robot-bridge.git
cd robot-bridge
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then create the runtime config from the template (the real one is git-ignored —
see `.gitignore:19`):

```bash
cp configs/config.example.yaml configs/config.yaml
```

`configs/config.yaml` is where `server.port` lives if you want a port other than
`8081`. The template ships (`configs/config.example.yaml:8-13`):

```yaml
server:
  host: "0.0.0.0"
  port: 8081
  websocket_path: "/ws/robot"
  ping_interval: 15
  ping_timeout: 10
```

> **Note on Python versions:** `README.md` says Python 3.12+, while
> `pyproject.toml:5` declares `requires-python = ">=3.10"`. Prefer 3.12 if you
> can; if your distro only has 3.10/3.11 the dependency metadata still allows it.

Start the bridge:

```bash
python -m src.main
```

You should see (from `src/main.py:66-67`):

```
[Server] Starting on 0.0.0.0:8081
[Server] WebSocket endpoint: ws://0.0.0.0:8081/ws/robot
```

Sanity check from another terminal:

```bash
curl http://<BRIDGE-IP>:8081/health
# {"status":"healthy","service":"robot-bridge","version":"0.1.0"}
```

`/health` is defined at `src/api.py:84-91`.

## 2. Config run Hermes (the webhook + MCP side)

The bridge does not run the LLM. On every finished utterance it POSTs the
recognised text to the Hermes webhook (`src/websocket_handler.py:458-472`):

```
POST http://127.0.0.1:8644/webhooks/stackchan
Header: X-Webhook-Signature: INSECURE_NO_AUTH
Body:   {"text": "...", "session_id": "stackchan-<id>", "person": ""}
```

So Hermes must be listening on webhook port `8644` and exposing the `stackchan`
MCP server. Add to `~/.hermes/config.yaml` (`README.md:52-72`):

```yaml
mcp_servers:
  stackchan:
    command: /opt/robot-bridge/.venv/bin/python
    args: ["-m", "src.mcp_server"]
    env:
      PYTHONPATH: /opt/robot-bridge
    timeout: 60

platforms:
  api_server:
    enabled: true
    host: 127.0.0.1
    port: 8642
  webhook:
    enabled: true
    extra:
      host: 127.0.0.1
      port: 8644
```

Create the webhook subscription (`README.md:77-81`):

```bash
hermes webhook subscribe stackchan \
  --prompt "你是叁叁，StackChan语音机器人。用户说：{text}。简短回复，纯中文，口语化。用 stackchan_speak 说话。" \
  --deliver log \
  --secret "INSECURE_NO_AUTH"
```

The `--secret` must match the hard-coded `X-Webhook-Signature: INSECURE_NO_AUTH`
header the bridge sends (`src/websocket_handler.py:471`).

Optionally set the API key used for server-side session continuity
(`src/config.py:20`, `start.sh:17-20`):

```bash
export HERMES_API_KEY=robot-bridge-key
```

`8642` is the Hermes **Gateway API** port (`src/config.py:19`), distinct from the
webhook port `8644` and from the bridge's own `8081`.

## 3. Point the device at the bridge

### 3a. The `/ota` endpoint

The bridge exposes an OTA/config endpoint that a device can fetch on boot to
learn the WebSocket URL (`src/api.py:109-126`). It responds to `GET` or `POST`:

```json
{
  "websocket": { "url": "ws://192.168.50.32:8081/ws/robot", "version": 1 },
  "server_time": { "timestamp": 1730000000000, "timezone_offset": 480 }
}
```

> ⚠️ **The URL in that response is hard-coded to the maintainer's own LAN address
> (`192.168.50.32`).** You must replace it with your bridge host's real LAN IP.
> This is literally the line to edit — `src/api.py:119`:
>
> ```python
> "url": "ws://192.168.50.32:8081/ws/robot",
> ```
>
> The bridge does not read this from `configs/config.yaml`; the string is a
> constant in the handler.

Whether your device uses `/ota` at all depends on its firmware (see §4).

### 3b. What the device must do on the WebSocket

Regardless of how the URL is supplied, once connected the device speaks the
XiaoZhi protocol the bridge implements (`src/websocket_handler.py:1-16`):

1. Send a `Device-Id` header when opening the socket (`src/api.py:435`). The
   bridge derives a stable `session_id` from it so Hermes context survives
   reconnects:

   ```python
   device_id = websocket.headers.get("device-id", "") or str(uuid.uuid4())[:12]
   ```

2. Send the hello frame (`src/websocket_handler.py:158-178`,
   `tests/test_bridge_v2.py:109-114`):

   ```json
   {"type":"hello","version":1,"transport":"websocket",
    "audio_params":{"format":"opus","sample_rate":16000,"channels":1,"frame_duration":60}}
   ```

   The bridge replies with its own `hello` carrying `session_id` and the same
   Opus parameters.

3. Then stream raw Opus audio in **binary** WS frames, and listen for
   `stt` / `llm` / `tts` JSON control frames and Opus audio coming back.

There is also a second WebSocket, `/stackChan/ws` (`src/api.py:479`), used for
the binary camera/JPEG stream (Avatar protocol). It is optional; voice works
without it.

## 4. The firmware half — read this carefully

**This repository does not ship a complete, flashable firmware.** The
`firmware/` directory contains only camera/vision *integration examples* to drop
into an existing StackChan project:

```
firmware/CameraDriver.h
firmware/FaceDetector.h
firmware/FaceTracker.h
firmware/WebSocketSender.h
firmware/main_vision_integration.ino   # "Drop this into your existing StackChan.ino"
```

The actual device firmware this project was built against lives **outside** the
repo. `ARCHITECTURE.md:96-103` documents it as a modified
[xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)-style tree at
`/tmp/StackChan/firmware/`, and the only flash recipe in the repo is for those
pre-built binaries (`ARCHITECTURE.md:124-136`):

```bash
sg dialout -c "/opt/robot-bridge/.venv/bin/esptool --chip esp32s3 \
  -b 921600 --before default-reset --after hard-reset write-flash \
  --flash-mode dio --flash-size 16MB --flash-freq 80m \
  0x0   /tmp/StackChan/firmware/build/bootloader/bootloader.bin \
  0x8000 /tmp/StackChan/firmware/build/partition_table/partition-table.bin \
  0x9000 /tmp/nvs_patched_v3.bin \
  0xd000 /tmp/StackChan/firmware/build/ota_data_initial.bin \
  0x20000 /tmp/StackChan/firmware/build/stack-chan.bin \
  0xa00000 /tmp/StackChan/firmware/build/generated_assets.bin"
```

And `PROMPT.md:53` refers to a build script `bash /tmp/build_stackchan.sh`
which is **not in this repo either**.

**Practical path if you are starting from scratch:**

1. Build/flash a XiaoZhi-compatible StackChan firmware (the upstream
   `xiaozhi-esp32` project is the reference implementation the bridge's protocol
   doc cites — `src/websocket_handler.py:3`).
2. Configure that firmware's WebSocket/Ota URL to point at
   `ws://<BRIDGE-IP>:8081/ws/robot` (§3).
3. Optionally fold in the `firmware/` vision headers for camera + face tracking.

> **This hardware half is UNVERIFIED by the author of this doc** — it needs a
> physical StackChan and the out-of-tree firmware tree, which are not in this
> repo. If you maintain the firmware, please confirm (or correct) the exact
> upstream commit, build steps, and `menuconfig`/`app_config` fields a user
> should set. See the PR discussion.

## 5. Verify the bridge without a physical device

The repo's own suite claims to need no hardware —
`tests/test_bridge_v2.py:3-5`:

> _"Tests each feature without requiring the physical ESP32. Simulates
> WebSocket messages and checks server responses."_

Run it against a live bridge (`ARCHITECTURE.md:118-122`):

```bash
.venv/bin/python tests/test_bridge_v2.py          # all tests
.venv/bin/python tests/test_bridge_v2.py --test chat
```

It connects to `http://127.0.0.1:8081` and `ws://127.0.0.1:8081/ws/robot`
(`tests/test_bridge_v2.py:24-25`) and drives a full handshake, wake-word LED,
vision frame, and LED-state flow. If these pass, your bridge-side wiring
(port/path/protocol) is correct and any remaining problem is on the device side.

## 6. Troubleshooting checklist

| Symptom | Check |
|---|---|
| Device never connects | Is the URL `ws://<BRIDGE-IP>:8081/ws/robot`? Wrong IP or `8081` is the usual cause. |
| `/health` works locally but not from the device | Firewall on `8081` (ARCHITECTURE.md:57 mentions allowing `8081`). |
| Connects, no reply | Hermes webhook on `8644` up? `X-Webhook-Signature` matches the subscription `--secret`? |
| Reply is silent | Opus/TTS path — check bridge log `logs/bridge.log`. |
| OTA returns the wrong IP | You forgot to edit `src/api.py:119`. |

## Verification note (for reviewers)

- The port/path defaults in §1 were confirmed by **executing** the real config
  loader, not by reading alone: importing `src.config.config` and printing
  `config.server` yields `host=0.0.0.0`, `port=8081`, `websocket_path=/ws/robot`.
- The full `tests/test_bridge_v2.py` suite was **not run** here: it pulls heavy
  native deps (FunASR, torch, sherpa-onnx, OpenCV, opuslib — `pyproject.toml`)
  that could not be installed in the doc-authoring environment. All endpoint,
  port and header values above are therefore quoted directly from source with
  file:line references so they can be checked by eye.

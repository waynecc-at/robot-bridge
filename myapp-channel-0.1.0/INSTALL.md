# myapp-channel Installation Guide

`myapp-channel` is an OpenClaw custom channel plugin that uses an outbound WebSocket bridge.

It is designed for this flow:

1. Your external service exposes a WebSocket endpoint.
2. OpenClaw loads `myapp-channel`.
3. `myapp-channel` connects out to your service.
4. Your service sends `reply.request`.
5. OpenClaw returns `reply.response`.

## Package Contents

- `index.ts`
- `openclaw.plugin.json`
- `README.md`
- `INSTALL.md`

## Install

Copy the `myapp-channel` folder into the target OpenClaw extensions directory:

```bash
mkdir -p ~/.openclaw/extensions
cp -R myapp-channel ~/.openclaw/extensions/
```

After copying, your target machine should have:

```text
~/.openclaw/extensions/myapp-channel/index.ts
~/.openclaw/extensions/myapp-channel/openclaw.plugin.json
~/.openclaw/extensions/myapp-channel/README.md
```

## OpenClaw Configuration

Add this block to the target OpenClaw config file:

```json
{
  "channels": {
    "myapp-channel": {
      "enabled": true,
      "bridgeUrl": "ws://YOUR_BRIDGE_HOST:8080/openclaw-bridge",
      "bridgeToken": "replace-with-random-token",
      "botName": "OpenClaw",
      "timeoutMs": 120000,
      "reconnectMs": 3000
    }
  }
}
```

### Config Fields

- `enabled`: whether the channel starts.
- `bridgeUrl`: the WebSocket URL that OpenClaw should connect to.
- `bridgeToken`: shared token appended to the bridge URL as query parameter `token`.
- `botName`: logical target name inside OpenClaw.
- `timeoutMs`: max time to wait for OpenClaw reply generation.
- `reconnectMs`: reconnect interval after disconnect.

## External Bridge Service Requirements

Your external service must expose a WebSocket endpoint such as:

```text
ws://YOUR_BRIDGE_HOST:8080/openclaw-bridge
```

It must accept the token as a query parameter:

```text
ws://YOUR_BRIDGE_HOST:8080/openclaw-bridge?token=replace-with-random-token
```

## Bridge Protocol

### Request from external service to OpenClaw

```json
{
  "type": "reply.request",
  "requestId": "uuid",
  "text": "你好",
  "from": "user-or-device-id",
  "senderName": "display-name",
  "conversationId": "conversation-id"
}
```

### Success response from OpenClaw

```json
{
  "type": "reply.response",
  "requestId": "uuid",
  "ok": true,
  "sessionKey": "myapp-channel:user-or-device-id:conversation-id",
  "from": "user-or-device-id",
  "conversationId": "conversation-id",
  "reply": "OpenClaw generated text"
}
```

### Error response from OpenClaw

```json
{
  "type": "reply.response",
  "requestId": "uuid",
  "ok": false,
  "sessionKey": "myapp-channel:user-or-device-id:conversation-id",
  "from": "user-or-device-id",
  "conversationId": "conversation-id",
  "error": "error message"
}
```

## Startup Steps

1. Deploy your bridge service and confirm the WebSocket endpoint is reachable from the OpenClaw host or container.
2. Copy the plugin into `~/.openclaw/extensions/myapp-channel`.
3. Add the `channels.myapp-channel` config.
4. Restart the OpenClaw gateway.
5. Check logs for:

```text
[myapp-channel] connecting bridge to ...
[myapp-channel] bridge connected
```

## Verification

After startup, test by sending a `reply.request` over the bridge.

Expected OpenClaw log flow:

```text
[myapp-channel] reply.request ...
[gateway] [myapp-channel] dispatching reply ...
[myapp-channel] reply.response ... ok=true
```

## Troubleshooting

### Bridge does not connect

- Verify `bridgeUrl` is reachable from the OpenClaw runtime environment.
- Verify the token matches on both sides.
- Verify the target service is really speaking WebSocket, not plain HTTP.

### Request reaches OpenClaw but no reply is returned

- Check OpenClaw logs for `dispatcher finished queuedFinal=false`.
- Check your OpenClaw agent configuration and routing.
- Check whether your agent or message policy is suppressing replies.

### Plugin changes do not take effect

- Restart OpenClaw gateway after editing plugin files.

## Recommended Deployment Layout

```text
~/.openclaw/
  openclaw.json
  extensions/
    myapp-channel/
      index.ts
      openclaw.plugin.json
      README.md
```

## Version

- Package version: `0.1.0`
- Channel id: `myapp-channel`

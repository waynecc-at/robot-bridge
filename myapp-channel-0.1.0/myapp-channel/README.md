# myapp-channel

`myapp-channel` now runs in bridge mode.

It no longer expects your app to call the OpenClaw gateway over HTTP. Instead:

1. `xiaozhi-server` exposes a WebSocket bridge endpoint.
2. The `myapp-channel` plugin inside OpenClaw connects outbound to that endpoint.
3. `xiaozhi-server` sends `reply.request` messages over the bridge.
4. OpenClaw sends `reply.response` messages back on the same long-lived connection.

## OpenClaw config

```json
{
  "channels": {
    "myapp-channel": {
      "enabled": true,
      "bridgeUrl": "ws://YOUR_BRIDGE_HOST:8080/openclaw-bridge",
      "bridgeToken": "your-bridge-token",
      "botName": "OpenClaw",
      "timeoutMs": 120000,
      "reconnectMs": 3000
    }
  }
}
```

## xiaozhi-server config

```env
OPENCLAW_BRIDGE_PATH=/openclaw-bridge
OPENCLAW_BRIDGE_TOKEN=your-bridge-token
OPENCLAW_BRIDGE_REQUEST_TIMEOUT_SECONDS=120
```

## Bridge request

`xiaozhi-server` sends:

```json
{
  "type": "reply.request",
  "requestId": "uuid",
  "text": "你好",
  "from": "device-id",
  "senderName": "client-id",
  "conversationId": "session-id"
}
```

## Bridge response

OpenClaw replies:

```json
{
  "type": "reply.response",
  "requestId": "uuid",
  "ok": true,
  "sessionKey": "myapp-channel:device-id:session-id",
  "from": "device-id",
  "conversationId": "session-id",
  "reply": "..."
}
```

Error response:

```json
{
  "type": "reply.response",
  "requestId": "uuid",
  "ok": false,
  "sessionKey": "myapp-channel:device-id:session-id",
  "from": "device-id",
  "conversationId": "session-id",
  "error": "..."
}
```

## Notes

- The OpenClaw side initiates the WebSocket connection to your external bridge service.
- The bridge token must match on both sides.
- After changing plugin files or channel config, restart the OpenClaw gateway.

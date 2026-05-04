import {
  DEFAULT_ACCOUNT_ID,
  emptyPluginConfigSchema,
  type ChannelPlugin,
  type OpenClawConfig,
  type OpenClawPluginApi,
} from "openclaw/plugin-sdk";

const CHANNEL_ID = "myapp-channel";
const DEFAULT_RECONNECT_MS = 3_000;
const BRIDGE_PING_INTERVAL_MS = 10_000;
const BRIDGE_IDLE_TIMEOUT_MS = 30_000;

type MyappChannelConfig = {
  enabled?: boolean;
  bridgeUrl?: string;
  bridgeToken?: string;
  botName?: string;
  timeoutMs?: number;
  reconnectMs?: number;
};

type ResolvedMyappAccount = {
  accountId: string;
  enabled: boolean;
  configured: boolean;
  bridgeUrl: string;
  bridgeToken: string;
  botName: string;
  timeoutMs: number;
  reconnectMs: number;
};

type BridgeRequest = {
  type: "reply.request";
  requestId: string;
  text?: string;
  from?: string;
  senderName?: string;
  conversationId?: string;
  sessionKey?: string;
};

type BridgeResponse = {
  type: "reply.response";
  requestId: string;
  ok: boolean;
  sessionKey: string;
  from: string;
  conversationId: string | null;
  reply?: string;
  error?: string;
};

const channelConfigSchema = {
  schema: {
    type: "object",
    additionalProperties: true,
    properties: {
      enabled: { type: "boolean" },
      bridgeUrl: { type: "string" },
      bridgeToken: { type: "string" },
      botName: { type: "string" },
      timeoutMs: { type: "integer", minimum: 1, maximum: 300000 },
      reconnectMs: { type: "integer", minimum: 250, maximum: 60000 },
    },
  },
};

let pluginApiRef: OpenClawPluginApi | null = null;

function getPluginApi(): OpenClawPluginApi {
  if (!pluginApiRef) {
    throw new Error("myapp-channel plugin API is not initialized");
  }
  return pluginApiRef;
}

function resolveChannelConfig(cfg: OpenClawConfig): MyappChannelConfig {
  return ((cfg.channels as Record<string, unknown> | undefined)?.[CHANNEL_ID] ??
    {}) as MyappChannelConfig;
}

function resolveAccount(cfg: OpenClawConfig): ResolvedMyappAccount {
  const channelCfg = resolveChannelConfig(cfg);
  const timeoutMs =
    typeof channelCfg.timeoutMs === "number" &&
    Number.isFinite(channelCfg.timeoutMs) &&
    channelCfg.timeoutMs > 0
      ? Math.min(Math.floor(channelCfg.timeoutMs), 300_000)
      : 120_000;
  const reconnectMs =
    typeof channelCfg.reconnectMs === "number" &&
    Number.isFinite(channelCfg.reconnectMs) &&
    channelCfg.reconnectMs >= 250
      ? Math.min(Math.floor(channelCfg.reconnectMs), 60_000)
      : DEFAULT_RECONNECT_MS;
  const bridgeUrl = String(channelCfg.bridgeUrl ?? "").trim();
  const bridgeToken = String(channelCfg.bridgeToken ?? "").trim();

  return {
    accountId: DEFAULT_ACCOUNT_ID,
    enabled: channelCfg.enabled !== false,
    configured: Boolean(bridgeUrl),
    bridgeUrl,
    bridgeToken,
    botName: String(channelCfg.botName ?? "OpenClaw").trim() || "OpenClaw",
    timeoutMs,
    reconnectMs,
  };
}

function resolveSessionKey(payload: {
  sessionKey?: string;
  conversationId?: string;
  from?: string;
}): string {
  const explicitSessionKey = String(payload.sessionKey ?? "").trim();
  if (explicitSessionKey) {
    return explicitSessionKey;
  }
  const from = String(payload.from ?? "").trim() || "unknown";
  const conversationId = String(payload.conversationId ?? "").trim();
  if (conversationId) {
    return `${CHANNEL_ID}:${from}:${conversationId}`;
  }
  return `${CHANNEL_ID}:${from}`;
}

function resolveRoutePeerId(from: string, conversationId: string | null): string {
  return conversationId ? `${from}:${conversationId}` : from;
}

function buildBridgeUrl(account: ResolvedMyappAccount): string {
  const url = new URL(account.bridgeUrl);
  if (account.bridgeToken) {
    url.searchParams.set("token", account.bridgeToken);
  }
  return url.toString();
}

function waitForAbortSignal(signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    signal.addEventListener("abort", () => resolve(), { once: true });
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function dataToText(data: unknown): Promise<string> {
  if (typeof data === "string") {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return Buffer.from(data).toString("utf8");
  }
  if (ArrayBuffer.isView(data)) {
    return Buffer.from(data.buffer, data.byteOffset, data.byteLength).toString("utf8");
  }
  if (data && typeof (data as Blob).arrayBuffer === "function") {
    return Buffer.from(await (data as Blob).arrayBuffer()).toString("utf8");
  }
  return String(data ?? "");
}

async function runAgentReply(params: {
  text: string;
  from: string;
  senderName?: string;
  conversationId?: string;
  sessionKey?: string;
  account: ResolvedMyappAccount;
  cfg: OpenClawConfig;
}): Promise<{ reply: string; sessionKey: string; conversationId: string | null; from: string }> {
  const { account, cfg } = params;
  const runtime = getPluginApi().runtime;
  const logger = getPluginApi().logger;
  const from = String(params.from ?? "").trim() || "unknown";
  const conversationId = String(params.conversationId ?? "").trim() || null;
  const route = runtime.channel.routing.resolveAgentRoute({
    cfg,
    channel: CHANNEL_ID,
    accountId: account.accountId,
    peer: {
      kind: "direct",
      id: resolveRoutePeerId(from, conversationId),
    },
  });
  const sessionKey =
    String(params.sessionKey ?? "").trim() || route.sessionKey || resolveSessionKey({
      sessionKey: params.sessionKey,
      conversationId: params.conversationId,
      from,
    });
  const replyParts: string[] = [];
  const ctxPayload = runtime.channel.reply.finalizeInboundContext({
    Body: params.text,
    BodyForAgent: params.text,
    RawBody: params.text,
    CommandBody: params.text,
    From: from,
    To: account.botName,
    SessionKey: sessionKey,
    AccountId: route.accountId,
    ChatType: "direct",
    ConversationLabel: from,
    SenderName: String(params.senderName ?? "").trim() || from,
    SenderId: from,
    CommandAuthorized: true,
    Provider: CHANNEL_ID,
    Surface: CHANNEL_ID,
    OriginatingChannel: CHANNEL_ID as never,
    OriginatingTo: from,
  });
  const storePath = runtime.channel.session.resolveStorePath(cfg.session?.store, {
    agentId: route.agentId,
  });

  await runtime.channel.session.recordInboundSession({
    storePath,
    sessionKey: ctxPayload.SessionKey ?? sessionKey,
    ctx: ctxPayload,
    onRecordError: (err: unknown) => {
      logger.warn(
        `[${CHANNEL_ID}] failed to record inbound session: ${err instanceof Error ? err.message : String(err)}`,
      );
    },
  });
  logger.info(
    `[${CHANNEL_ID}] dispatching reply from=${from} sessionKey=${sessionKey} routeAgent=${route.agentId} conversationId=${conversationId ?? "-"}`,
  );

  const abortController = new AbortController();
  const timeoutHandle = setTimeout(() => {
    abortController.abort(new Error(`Timed out after ${account.timeoutMs}ms`));
  }, account.timeoutMs);

  const dispatchPromise = runtime.channel.reply.dispatchReplyWithBufferedBlockDispatcher({
    ctx: ctxPayload,
    cfg,
    dispatcherOptions: {
      deliver: async (outbound: { text?: string; body?: string }, info?: { kind?: string }) => {
        const chunk = String(outbound.text ?? outbound.body ?? "").trim();
        if (chunk) {
          replyParts.push(chunk);
          logger.info(
            `[${CHANNEL_ID}] deliver kind=${info?.kind ?? "unknown"} chunkLen=${chunk.length}`,
          );
        }
      },
      onError: (error, info) => {
        const message = error instanceof Error ? error.message : String(error);
        logger.error(
          `[${CHANNEL_ID}] dispatcher error kind=${info.kind} message=${message}`,
        );
      },
    },
    replyOptions: {
      abortSignal: abortController.signal,
      disableBlockStreaming: true,
      timeoutOverrideSeconds: Math.max(1, Math.ceil(account.timeoutMs / 1000)),
      onModelSelected: (ctx) => {
        logger.info(
          `[${CHANNEL_ID}] model selected provider=${ctx.provider} model=${ctx.model} think=${ctx.thinkLevel ?? "-"}`,
        );
      },
    },
  });

  let dispatchResult: { queuedFinal?: boolean } | undefined;
  try {
    dispatchResult = await dispatchPromise;
  } finally {
    clearTimeout(timeoutHandle);
  }

  logger.info(
    `[${CHANNEL_ID}] dispatcher finished queuedFinal=${String((dispatchResult as { queuedFinal?: boolean })?.queuedFinal ?? false)} replyParts=${replyParts.length}`,
  );

  return {
    reply: replyParts.join("\n\n").trim(),
    sessionKey,
    conversationId,
    from,
  };
}

async function sendBridgeJson(ws: WebSocket, payload: unknown): Promise<void> {
  ws.send(JSON.stringify(payload));
}

async function handleBridgeRequest(params: {
  raw: string;
  ws: WebSocket;
  account: ResolvedMyappAccount;
  cfg: OpenClawConfig;
  logger: OpenClawPluginApi["logger"];
  log?: { info?: (message: string) => void; warn?: (message: string) => void; error?: (message: string) => void };
}): Promise<void> {
  const parsed = JSON.parse(params.raw) as { type?: string };
  if (parsed.type === "bridge.hello" || parsed.type === "bridge.pong") {
    return;
  }
  if (parsed.type === "bridge.ping") {
    await sendBridgeJson(params.ws, { type: "bridge.pong", channel: CHANNEL_ID });
    return;
  }
  if (parsed.type !== "reply.request") {
    return;
  }

  const request = parsed as BridgeRequest;
  const text = String(request.text ?? "").trim();
  const from = String(request.from ?? "").trim();
  if (!request.requestId || !text || !from) {
    return;
  }

  params.log?.info?.(
    `[${CHANNEL_ID}] reply.request requestId=${request.requestId} from=${from} conversationId=${String(
      request.conversationId ?? "",
    ).trim() || "-"}`,
  );

  let response: BridgeResponse;
  try {
    const result = await runAgentReply({
      text,
      from,
      senderName: request.senderName,
      conversationId: request.conversationId,
      sessionKey: request.sessionKey,
      account: params.account,
      cfg: params.cfg,
    });
    response = {
      type: "reply.response",
      requestId: request.requestId,
      ok: true,
      sessionKey: result.sessionKey,
      from: result.from,
      conversationId: result.conversationId,
      reply: result.reply,
    };
    params.log?.info?.(
      `[${CHANNEL_ID}] reply.response requestId=${request.requestId} ok=true replyLen=${response.reply?.length ?? 0}`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    params.logger.error(`[${CHANNEL_ID}] bridge request failed`, {
      requestId: request.requestId,
      from,
      error: message,
    });
    response = {
      type: "reply.response",
      requestId: request.requestId,
      ok: false,
      sessionKey: resolveSessionKey({
        sessionKey: request.sessionKey,
        conversationId: request.conversationId,
        from,
      }),
      from,
      conversationId: String(request.conversationId ?? "").trim() || null,
      error: message,
    };
    params.log?.error?.(`[${CHANNEL_ID}] reply.response requestId=${request.requestId} ok=false error=${message}`);
  }

  if (params.ws.readyState === WebSocket.OPEN) {
    await sendBridgeJson(params.ws, response);
  }
}

async function connectBridge(params: {
  account: ResolvedMyappAccount;
  cfg: OpenClawConfig;
  logger: OpenClawPluginApi["logger"];
  log?: { info?: (message: string) => void; warn?: (message: string) => void; error?: (message: string) => void };
  setStatus: (patch: Record<string, unknown>) => void;
  abortSignal: AbortSignal;
}): Promise<void> {
  const bridgeUrl = buildBridgeUrl(params.account);
  params.log?.info?.(`[${CHANNEL_ID}] connecting bridge to ${bridgeUrl}`);
  await new Promise<void>((resolve) => {
    let lastError: string | null = null;
    let lastActivityAt = Date.now();
    let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
    const ws = new WebSocket(bridgeUrl);

    const markActivity = () => {
      lastActivityAt = Date.now();
    };

    const stopHeartbeat = () => {
      if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    };

    const onAbort = () => {
      stopHeartbeat();
      try {
        ws.close(1000, "abort");
      } catch {
        resolve();
      }
    };

    params.abortSignal.addEventListener("abort", onAbort, { once: true });

    ws.addEventListener("open", () => {
      markActivity();
      params.log?.info?.(`[${CHANNEL_ID}] bridge connected`);
      params.setStatus({
        running: true,
        lastStartAt: Date.now(),
        lastError: null,
      });
      void sendBridgeJson(ws, {
        type: "bridge.hello",
        role: "openclaw",
        channel: CHANNEL_ID,
      });
      heartbeatTimer = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN) {
          return;
        }
        if (Date.now() - lastActivityAt > BRIDGE_IDLE_TIMEOUT_MS) {
          lastError = lastError ?? `bridge idle for ${BRIDGE_IDLE_TIMEOUT_MS}ms`;
          params.log?.warn?.(`[${CHANNEL_ID}] bridge idle timeout; closing socket for reconnect`);
          stopHeartbeat();
          try {
            ws.close(4000, "idle timeout");
          } catch {
            // Ignore close errors; the close handler will drive reconnect.
          }
          return;
        }
        try {
          ws.send(JSON.stringify({ type: "bridge.ping", channel: CHANNEL_ID, ts: Date.now() }));
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          params.log?.error?.(`[${CHANNEL_ID}] bridge heartbeat send failed: ${lastError}`);
          stopHeartbeat();
          try {
            ws.close(1011, "heartbeat send failed");
          } catch {
            // Ignore close errors; the close handler will drive reconnect.
          }
        }
      }, BRIDGE_PING_INTERVAL_MS);
    });

    ws.addEventListener("message", (event) => {
      void (async () => {
        try {
          markActivity();
          const raw = await dataToText(event.data);
          await handleBridgeRequest({
            raw,
            ws,
            account: params.account,
            cfg: params.cfg,
            logger: params.logger,
            log: params.log,
          });
        } catch (error) {
          lastError = error instanceof Error ? error.message : String(error);
          params.log?.error?.(`[${CHANNEL_ID}] bridge message error: ${lastError}`);
          params.logger.error(`[${CHANNEL_ID}] bridge message error`, { error: lastError });
        }
      })();
    });

    ws.addEventListener("error", () => {
      lastError = lastError ?? "websocket error";
      params.log?.error?.(`[${CHANNEL_ID}] bridge websocket error`);
    });

    ws.addEventListener("close", (event) => {
      params.abortSignal.removeEventListener("abort", onAbort);
      stopHeartbeat();
      const reason = event.reason ? ` reason=${event.reason}` : "";
      params.log?.warn?.(
        `[${CHANNEL_ID}] bridge closed code=${event.code}${reason}${lastError ? ` error=${lastError}` : ""}`,
      );
      params.setStatus({
        running: false,
        lastStopAt: Date.now(),
        ...(lastError ? { lastError } : {}),
      });
      resolve();
    });
  });
}

const myappChannelPlugin: ChannelPlugin<ResolvedMyappAccount> = {
  id: CHANNEL_ID,
  meta: {
    id: CHANNEL_ID,
    label: "MyApp Channel",
    selectionLabel: "MyApp Channel (Bridge)",
    detailLabel: "MyApp Channel (Bridge)",
    docsPath: "/tools/plugin",
    docsLabel: "plugin",
    blurb: "Outbound WebSocket bridge from Gateway to xiaozhi-server.",
    order: 200,
  },
  capabilities: {
    chatTypes: ["direct"],
    media: false,
    threads: false,
    reactions: false,
    nativeCommands: false,
    blockStreaming: false,
  },
  reload: { configPrefixes: [`channels.${CHANNEL_ID}`] },
  configSchema: channelConfigSchema,
  config: {
    listAccountIds: () => [DEFAULT_ACCOUNT_ID],
    resolveAccount: (cfg) => resolveAccount(cfg),
    defaultAccountId: () => DEFAULT_ACCOUNT_ID,
    isEnabled: (account) => account.enabled,
    isConfigured: (account) => account.configured,
    describeAccount: (account) => ({
      accountId: account.accountId,
      enabled: account.enabled,
      configured: account.configured,
      bridgeUrl: account.bridgeUrl || "[missing]",
      timeoutMs: account.timeoutMs,
      reconnectMs: account.reconnectMs,
      bridgeToken: account.bridgeToken ? "[set]" : "[empty]",
    }),
  },
  security: {
    resolveDmPolicy: () => ({
      policy: "open",
      allowFrom: [],
      policyPath: `channels.${CHANNEL_ID}.bridgeUrl`,
      allowFromPath: `channels.${CHANNEL_ID}.bridgeUrl`,
      approveHint: "Bridge requests are authorized by the WebSocket bridge token.",
      normalizeEntry: (raw) => raw.trim(),
    }),
  },
  directory: {
    self: async () => null,
    listPeers: async () => [],
    listGroups: async () => [],
  },
  status: {
    defaultRuntime: {
      accountId: DEFAULT_ACCOUNT_ID,
      running: false,
      lastStartAt: null,
      lastStopAt: null,
      lastError: null,
    },
    buildAccountSnapshot: ({ account, runtime }) => ({
      accountId: account.accountId,
      enabled: account.enabled,
      configured: account.configured,
      bridgeUrl: account.bridgeUrl || "[missing]",
      timeoutMs: account.timeoutMs,
      reconnectMs: account.reconnectMs,
      running: runtime?.running ?? false,
      lastStartAt: runtime?.lastStartAt ?? null,
      lastStopAt: runtime?.lastStopAt ?? null,
      lastError: runtime?.lastError ?? null,
    }),
  },
  gateway: {
    startAccount: async (ctx) => {
      const logger = getPluginApi().logger;
      const account = resolveAccount(ctx.cfg);
      if (!account.enabled) {
        ctx.log?.info?.("myapp-channel is disabled, skipping bridge startup");
        return { stop: () => {} };
      }
      if (!account.configured) {
        ctx.log?.warn?.("myapp-channel bridgeUrl is missing, skipping bridge startup");
        return { stop: () => {} };
      }

      while (!ctx.abortSignal.aborted) {
        await connectBridge({
          account,
          cfg: ctx.cfg,
          logger,
          log: ctx.log,
          abortSignal: ctx.abortSignal,
          setStatus: (patch) => ctx.setStatus({ accountId: ctx.accountId, ...patch }),
        });
        if (ctx.abortSignal.aborted) {
          break;
        }
        ctx.log?.info?.(`[${CHANNEL_ID}] reconnecting in ${account.reconnectMs}ms`);
        await sleep(account.reconnectMs);
      }
    },
  },
  messaging: {
    normalizeTarget: (target) => {
      const trimmed = target.trim();
      return trimmed || undefined;
    },
    targetResolver: {
      looksLikeId: (value) => Boolean(value.trim()),
      hint: "<from>",
    },
  },
};

const plugin = {
  id: CHANNEL_ID,
  name: "MyApp Channel",
  description: "Outbound WebSocket bridge channel for xiaozhi-server integration.",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    pluginApiRef = api;
    api.registerChannel({ plugin: myappChannelPlugin });
  },
};

export default plugin;

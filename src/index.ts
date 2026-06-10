import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

const PORT = Number(process.env.PI_MLX_MODELS_PORT ?? 11434);
const HOST = process.env.PI_MLX_MODELS_HOST ?? "127.0.0.1";
const BASE_URL = process.env.PI_MLX_MODELS_BASE_URL ?? `http://${HOST}:${PORT}/v1`;
const DEFAULT_MODEL = process.env.PI_MLX_MODELS_DEFAULT_MODEL?.trim() || "";

const PROVIDER_ID = "pi-mlx-models";
const DATA_DIR = join(homedir(), ".pi", "agent", "pi-mlx-models");
const VENV_DIR = join(DATA_DIR, "venv");
const VENV_PYTHON = join(VENV_DIR, "bin", "python3");
const VENV_HF = join(VENV_DIR, "bin", "hf");
const VENV_HUGGINGFACE_CLI = join(VENV_DIR, "bin", "huggingface-cli");
const HF_HOME = process.env.HF_HOME?.trim() || join(homedir(), ".cache", "huggingface");

type CachedModel = {
  key: string;
  modelId: string;
  size?: string;
  lastAccessed?: string;
};

type HfCacheEntry = {
  repo_id?: unknown;
  repo_type?: unknown;
  id?: unknown;
  size?: unknown;
  last_accessed?: unknown;
};

let serverProc: ChildProcess | null = null;
let currentModel = DEFAULT_MODEL;
let spinnerTimer: NodeJS.Timeout | null = null;
let spinnerIndex = 0;
let statusCtx: any | null = null;
let statusLabel = "";
let statusProgress: number | undefined;
let statusStartedAt: number | null = null;
let statusShowElapsed = false;

function keyForModel(modelId: string): string {
  return modelId
    .replace(/^mlx-community\//, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function hfCacheEnv() {
  return { ...process.env, HF_HOME, TRANSFORMERS_CACHE: HF_HOME, HF_HUB_DISABLE_TELEMETRY: "1" };
}

function hfCliCandidates(): string[] {
  return [
    process.env.PI_MLX_MODELS_HF_CLI,
    existsSync(VENV_HF) ? VENV_HF : undefined,
    "hf",
    existsSync(VENV_HUGGINGFACE_CLI) ? VENV_HUGGINGFACE_CLI : undefined,
    "huggingface-cli",
  ].filter((candidate): candidate is string => Boolean(candidate));
}

function parseHfCacheModels(stdout: string): CachedModel[] {
  const parsed = JSON.parse(stdout) as unknown;
  const entries = Array.isArray(parsed) ? (parsed as HfCacheEntry[]) : [];
  const seen = new Set<string>();
  return entries
    .filter((entry) => entry.repo_type === "model")
    .map((entry) => ({
      modelId: typeof entry.repo_id === "string" ? entry.repo_id : typeof entry.id === "string" ? entry.id.replace(/^model\//, "") : "",
      size: typeof entry.size === "string" ? entry.size : undefined,
      lastAccessed: typeof entry.last_accessed === "string" ? entry.last_accessed : undefined,
    }))
    .filter(
      (entry) =>
        entry.modelId &&
        (entry.modelId.startsWith("mlx-community/") || /mlx/i.test(entry.modelId)) &&
        !seen.has(entry.modelId) &&
        seen.add(entry.modelId),
    )
    .map((entry) => ({ key: keyForModel(entry.modelId), ...entry }))
    .sort((a, b) => a.modelId.localeCompare(b.modelId));
}

function listCachedModels(): CachedModel[] {
  for (const command of hfCliCandidates()) {
    const res = spawnSync(command, ["cache", "ls", "--format", "json"], {
      env: hfCacheEnv(),
      encoding: "utf8",
    });
    if (res.status !== 0 || !res.stdout.trim()) continue;
    try {
      return parseHfCacheModels(res.stdout);
    } catch {
      continue;
    }
  }
  return [];
}

function resolveModel(input?: string): { modelId: string; cached?: CachedModel } {
  const normalized = (input || "").trim();
  const cachedModels = listCachedModels();
  if (!normalized) {
    const cached = cachedModels.find((m) => m.modelId === DEFAULT_MODEL || m.key === DEFAULT_MODEL) ?? cachedModels[0];
    return { modelId: cached?.modelId ?? DEFAULT_MODEL, cached };
  }

  if (/^\d+$/.test(normalized)) {
    const idx = Number(normalized) - 1;
    if (idx >= 0 && idx < cachedModels.length) {
      const cached = cachedModels[idx];
      return { modelId: cached.modelId, cached };
    }
  }

  const cached = cachedModels.find((m) => m.key === normalized || m.modelId === normalized);
  if (cached) return { modelId: cached.modelId, cached };
  return { modelId: normalized };
}

async function pickCachedModel(ctx: any): Promise<CachedModel | undefined> {
  const cachedModels = listCachedModels();
  if (!cachedModels.length) {
    ctx.ui.notify(
      `No cached MLX models found in ${HF_HOME}. Start with /mlx-start <hf-model-id> to download one first.`,
      "warning",
    );
    return undefined;
  }

  const options = cachedModels.map((m, i) => {
    const details = [m.size, m.lastAccessed ? `used ${m.lastAccessed}` : undefined].filter(Boolean).join(", ");
    return `${i + 1}. ${m.modelId}${details ? ` — ${details}` : ""}`;
  });
  const selected = await ctx.ui.select("Select cached MLX model", options);
  if (!selected) return undefined;
  const idx = options.indexOf(selected);
  if (idx < 0) return undefined;
  return cachedModels[idx];
}

async function startModelFromInput(
  pi: ExtensionAPI,
  ctx: any,
  args: string | undefined,
  controls: {
    startSpinner: (ctx: any, baseText: string, progress?: number, showElapsed?: boolean) => void;
    stopSpinner: () => void;
  },
) {
  const selected = resolveModel(args);
  const model = selected.modelId;
  if (!model) {
    ctx.ui.notify("No MLX model selected. Use /mlx-start <hf-model-id> or cache a model with Hugging Face first.", "error");
    return;
  }
  currentModel = model;
  await registerProvider(pi, { includeFallback: false });

  const progress = makeProgressController(ctx, "mlx-progress", `MLX start progress (${model})`, [
    "Ensure runtime is installed",
    "Start MLX server process",
    "Wait for server health endpoint",
    "Download/load model",
    "Warm up first inference",
    "Register provider models",
  ]);

  try {
    progress.activateStep(0, "Checking runtime");
    await ensureSetup();
    progress.doneStep(0, "Runtime ready");

    if (serverProc && !serverProc.killed) {
      serverProc.kill("SIGTERM");
      serverProc = null;
      await new Promise((r) => setTimeout(r, 800));
    }

    ctx.ui.notify(
      `Starting MLX server with ${model}${selected.cached ? ` (cached: ${selected.cached.key})` : ""}...`,
      "info",
    );
    controls.startSpinner(ctx, `Start MLX server process (${model})`);
    progress.activateStep(1, "Launching mlx_lm.server");

    serverProc = spawn(VENV_PYTHON, ["-m", "mlx_lm.server", "--model", model, "--host", HOST, "--port", String(PORT)], {
      env: hfCacheEnv(),
      stdio: ["ignore", "pipe", "pipe"],
    });

    progress.doneStep(1, "Server process started");
    progress.activateStep(2, "Probing /v1/models");
    controls.startSpinner(ctx, "Wait for server health endpoint");

    serverProc.stderr?.on("data", (d) => {
      const text = d.toString();
      const fetchMatch = text.match(/Fetching\s+(\d+)\s+files?:\s+(\d+)%.*?(\d+)\/(\d+)/);
      if (fetchMatch) {
        const pct = Number(fetchMatch[2]) / 100;
        const done = fetchMatch[3];
        const total = fetchMatch[4];
        progress.setStep(3, { state: "active", detail: `Downloading model files (${done}/${total})`, progress: pct });
        controls.startSpinner(ctx, `Download/load model (${done}/${total} files)`, pct);
        return;
      }
      if (text.includes("Fetching") || text.includes("Downloading")) {
        progress.setStep(3, { state: "active", detail: "Downloading model files..." });
        controls.startSpinner(ctx, "Download/load model");
        return;
      }
      if (text.toLowerCase().includes("starting") || text.includes("httpd")) {
        controls.startSpinner(ctx, `Start MLX server process (${model})`);
      }
    });

    serverProc.on("exit", () => {
      controls.stopSpinner();
      serverProc = null;
      ctx.ui.setStatus(PROVIDER_ID, "mlx: stopped");
    });

    await waitForServer();
    progress.doneStep(2, "Server responded on /v1/models");

    progress.activateStep(3, "Loading model (first run may download GBs)");
    controls.startSpinner(ctx, "Download/load model");

    progress.activateStep(4, "Running first lightweight completion");
    controls.startSpinner(ctx, "Warm up first inference", undefined, true);
    await waitForInferenceReady(model);
    progress.doneStep(4, "Model is inference-ready");
    progress.doneStep(3, "Model loaded");

    progress.activateStep(5, "Refreshing provider model list");
    await registerProvider(pi, { includeFallback: true });
    progress.doneStep(5, "Provider ready");

    controls.stopSpinner();
    ctx.ui.setStatus(PROVIDER_ID, `mlx: running (${model})`);
    ctx.ui.setWidget("mlx-progress", undefined);
     ctx.ui.setWidget("mlx-model-picker", undefined);
    ctx.ui.notify("MLX server is ready for prompts. Use /model and pick pi-mlx-models/...", "info");
  } catch (e) {
    controls.stopSpinner();
    progress.errorStep(4, e instanceof Error ? e.message : String(e));
    ctx.ui.notify(`mlx-start failed: ${e instanceof Error ? e.message : String(e)}`, "error");
  }
}

type StepState = {
  label: string;
  state: "pending" | "active" | "done" | "error";
  detail?: string;
  progress?: number;
};

function makeProgressController(ctx: any, id: string, title: string, labels: string[]) {
  const steps: StepState[] = labels.map((label) => ({ label, state: "pending" }));

  const setStep = (index: number, patch: Partial<StepState>) => {
    if (index < 0 || index >= steps.length) return;
    steps[index] = { ...steps[index], ...patch };
    render();
  };

  const activateStep = (index: number, detail?: string) => {
    steps.forEach((s, i) => {
      if (i < index && s.state !== "done") s.state = "done";
      if (i > index && s.state === "active") s.state = "pending";
    });
    setStep(index, { state: "active", detail });
  };

  const doneStep = (index: number, detail?: string) => {
    setStep(index, { state: "done", detail, progress: 1 });
  };

  const errorStep = (index: number, detail?: string) => {
    setStep(index, { state: "error", detail });
  };

  const bar = (progress?: number) => {
    if (progress == null || Number.isNaN(progress)) return "";
    const pct = Math.max(0, Math.min(1, progress));
    const width = 18;
    const filled = Math.round(pct * width);
    return `[${"█".repeat(filled)}${"░".repeat(width - filled)}] ${Math.round(pct * 100)}%`;
  };

  const render = () => {
    const activeIndex = steps.findIndex((s) => s.state === "active");
    const errorIndex = steps.findIndex((s) => s.state === "error");
    const doneIndexes = steps
      .map((s, i) => ({ s, i }))
      .filter(({ s }) => s.state === "done")
      .map(({ i }) => i);
    const lastDoneIndex = doneIndexes.length ? doneIndexes[doneIndexes.length - 1] : -1;

    const idx = errorIndex >= 0 ? errorIndex : activeIndex >= 0 ? activeIndex : lastDoneIndex;
    const current = idx >= 0 ? steps[idx] : null;

    const lines: string[] = [title, ""];
    if (current) {
      const icon = current.state === "done" ? "✓" : current.state === "active" ? "●" : current.state === "error" ? "✗" : "○";
      lines.push(`${icon} ${current.label}`);
      if (current.detail) lines.push(`   ${current.detail}`);
      const b = bar(current.progress ?? (current.state === "done" ? 1 : undefined));
      if (b) lines.push(`   ${b}`);
    }

    ctx.ui.setWidget(id, lines, { placement: "belowEditor" });
  };

  render();
  return { setStep, activateStep, doneStep, errorStep };
}

function findPython(): string | null {
  const candidates = [
    "/opt/homebrew/bin/python3.13",
    "/opt/homebrew/bin/python3.12",
    "/opt/homebrew/bin/python3.11",
    "/opt/homebrew/bin/python3.10",
    "/opt/homebrew/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
  ];
  for (const c of candidates) {
    try {
      const s = spawnSync(c, ["--version"], { stdio: ["ignore", "pipe", "pipe"], timeout: 3000 });
      if (s.status !== 0) continue;
      const out = `${s.stdout?.toString() || ""}${s.stderr?.toString() || ""}`;
      const m = out.match(/Python\s+3\.(\d+)/);
      if (!m) continue;
      const minor = Number(m[1]);
      if (minor >= 10 && minor <= 13) return c;
    } catch {
      // ignore
    }
  }
  return null;
}

function run(cmd: string, args: string[], env?: Record<string, string>, onLine?: (line: string) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...env },
    });
    let err = "";
    const handle = (chunk: Buffer) => {
      const text = chunk.toString();
      if (!onLine) return;
      for (const line of text.split(/\r?\n/)) {
        const t = line.trim();
        if (t) onLine(t);
      }
    };
    p.stdout?.on("data", handle);
    p.stderr?.on("data", (d) => {
      err += d.toString();
      handle(d);
    });
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(err.slice(-800) || `${cmd} failed with exit ${code}`))));
    p.on("error", reject);
  });
}

async function ensureSetup(progress?: ReturnType<typeof makeProgressController>) {
  const py = findPython();
  if (!py) throw new Error("Python 3.10–3.13 not found. Install: brew install python@3.13");

  progress?.activateStep(0, `Using ${py}`);
  progress?.doneStep(0, "Compatible Python found");

  progress?.activateStep(1, "Checking virtual environment");
  if (!existsSync(VENV_PYTHON)) {
    await run(py, ["-m", "venv", VENV_DIR], undefined, (line) => progress?.setStep(1, { detail: line }));
  }
  progress?.doneStep(1, "Virtual environment ready");

  progress?.activateStep(2, "Upgrading pip");
  await run(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "pip", "--index-url", "https://pypi.org/simple/"], undefined, (line) => {
    const m = line.match(/(\d+)%/);
    progress?.setStep(2, { detail: line.slice(0, 120), progress: m ? Number(m[1]) / 100 : undefined });
  });
  progress?.doneStep(2, "pip upgraded");

  progress?.activateStep(3, "Installing mlx-lm");
  await run(VENV_PYTHON, ["-m", "pip", "install", "--upgrade", "mlx-lm>=0.24.0", "--index-url", "https://pypi.org/simple/"], undefined, (line) => {
    const m = line.match(/(\d+)%/);
    progress?.setStep(3, { detail: line.slice(0, 120), progress: m ? Number(m[1]) / 100 : undefined });
  });
  progress?.doneStep(3, "mlx-lm installed");
}

async function waitForServer(timeoutMs = 600000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(`${BASE_URL}/models`);
      if (res.ok) return;
    } catch {
      // retry
    }
    await new Promise((r) => setTimeout(r, 1200));
  }
  throw new Error("Local MLX server did not become healthy in time");
}

async function waitForInferenceReady(model: string, timeoutMs = 1800000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 120000);
      const res = await fetch(`${BASE_URL}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "Reply with: ok" }],
          max_tokens: 2,
          stream: false,
          temperature: 0,
        }),
        signal: controller.signal,
      });
      clearTimeout(timer);
      if (res.ok) return;
    } catch {
      // still loading/downloading
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  throw new Error("Model did not become inference-ready in time");
}

async function discoverModelIds(): Promise<string[]> {
  try {
    const res = await fetch(`${BASE_URL}/models`);
    if (!res.ok) return [];
    const body = (await res.json()) as { data?: Array<{ id: string }> };
    return (body.data ?? []).map((m) => m.id).filter(Boolean);
  } catch {
    return [];
  }
}

function asProviderModels(ids: string[]) {
  return ids.map((id) => ({
    id,
    name: `${id} (Local)` ,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 8192,
  }));
}

async function registerProvider(pi: ExtensionAPI, options?: { includeFallback?: boolean }) {
  const discovered = await discoverModelIds();
  const includeFallback = options?.includeFallback ?? false;
  const modelIds = discovered.length ? discovered : includeFallback && currentModel ? [currentModel] : [];

  pi.registerProvider(PROVIDER_ID, {
    name: "PI MLX Models",
    baseUrl: BASE_URL,
    apiKey: "DUMMY",
    api: "openai-completions",
    models: asProviderModels(modelIds),
  });
}

export default async function (pi: ExtensionAPI) {
  await registerProvider(pi, { includeFallback: false });

  function renderStatusBar() {
    if (!statusCtx) return;
    const width = 16;
    const elapsed = statusShowElapsed && statusStartedAt ? ` (${Math.floor((Date.now() - statusStartedAt) / 1000)}s)` : "";

    if (statusProgress != null && !Number.isNaN(statusProgress)) {
      const pct = Math.max(0, Math.min(1, statusProgress));
      const filled = Math.round(pct * width);
      const bar = `[${"█".repeat(filled)}${"░".repeat(width - filled)}] ${Math.round(pct * 100)}%`;
      statusCtx.ui.setStatus(PROVIDER_ID, `${bar} ${statusLabel}${elapsed}`);
      return;
    }

    const pos = spinnerIndex % width;
    const chars = Array.from({ length: width }, (_, i) => (i === pos ? "█" : "░"));
    statusCtx.ui.setStatus(PROVIDER_ID, `[${chars.join("")}] ${statusLabel}${elapsed}`);
  }

  function stopSpinner() {
    if (spinnerTimer) {
      clearInterval(spinnerTimer);
      spinnerTimer = null;
    }
    statusCtx = null;
    statusLabel = "";
    statusProgress = undefined;
    statusStartedAt = null;
    statusShowElapsed = false;
  }

  function startSpinner(ctx: any, baseText: string, progress?: number, showElapsed = false) {
    const labelChanged = statusLabel !== baseText;
    const elapsedModeChanged = statusShowElapsed !== showElapsed;
    statusCtx = ctx;
    statusLabel = baseText;
    statusProgress = progress;
    statusShowElapsed = showElapsed;
    if (labelChanged || elapsedModeChanged || statusStartedAt == null) statusStartedAt = Date.now();
    if (!spinnerTimer) {
      spinnerTimer = setInterval(() => {
        spinnerIndex = (spinnerIndex + 1) % 16;
        renderStatusBar();
      }, 120);
    }
    renderStatusBar();
  }

  pi.registerCommand("mlx-init", {
    description: "Initialize local MLX runtime (python venv + mlx-lm)",
    handler: async (_args, ctx) => {
      ctx.ui.setWidget("mlx-model-picker", undefined);
      const progress = makeProgressController(ctx, "mlx-progress", "MLX init progress", [
        "Find compatible Python",
        "Create virtual environment",
        "Upgrade pip",
        "Install mlx-lm",
      ]);
      try {
        ctx.ui.notify("Installing local MLX runtime...", "info");
        await ensureSetup(progress);
        ctx.ui.notify("MLX runtime installed.", "info");
        ctx.ui.setWidget("mlx-progress", undefined);
      } catch (e) {
        progress.errorStep(3, e instanceof Error ? e.message : String(e));
        ctx.ui.notify(`mlx-init failed: ${e instanceof Error ? e.message : String(e)}`, "error");
      }
    },
  });

  pi.registerCommand("mlx-start", {
    description: "Start local MLX server. Usage: /mlx-start [cache-number|cache-key|hf-model-id]",
    handler: async (args, ctx) => {
      const normalizedArgs = (args || "").trim();
      if (!normalizedArgs) {
        await ensureSetup();
        const cachedModel = await pickCachedModel(ctx);
        if (!cachedModel) {
          ctx.ui.notify("Model selection cancelled.", "info");
          return;
        }
        await startModelFromInput(pi, ctx, cachedModel.key, { startSpinner, stopSpinner });
        return;
      }

      ctx.ui.setWidget("mlx-model-picker", undefined);
      await startModelFromInput(pi, ctx, normalizedArgs, { startSpinner, stopSpinner });
    },
  });

  pi.registerCommand("mlx-stop", {
    description: "Stop local MLX server",
    handler: async (_args, ctx) => {
      stopSpinner();
      if (serverProc && !serverProc.killed) {
        serverProc.kill("SIGTERM");
        serverProc = null;
      }
      ctx.ui.setStatus(PROVIDER_ID, "mlx: stopped");
      ctx.ui.setWidget("mlx-progress", undefined);
      ctx.ui.setWidget("mlx-model-picker", undefined);
      await registerProvider(pi, { includeFallback: false });
      ctx.ui.notify("MLX server stopped.", "info");
    },
  });

  pi.on("session_shutdown", async () => {
    stopSpinner();
    if (serverProc && !serverProc.killed) {
      serverProc.kill("SIGTERM");
      serverProc = null;
    }
    await registerProvider(pi, { includeFallback: false });
  });
}

// Settings, from the container's env (where the platform and the policy's
// registered variables put them) overlaid by the campaign's `policy.params`
// (manifest), so one image serves campaigns that want different models,
// effort or budgets.

import type { ThinkingLevel } from "@earendil-works/pi-agent-core";

export interface LlmSettings {
  baseUrl: string;
  apiKey: string;
  model: string;
  /** Whether the model reasons (sends reasoning_content / thinking). */
  reasoning: boolean;
  thinking: ThinkingLevel;
  contextWindow: number;
  maxOutputTokens: number;
  /** pi-ai OpenAI-compatibility flags for this endpoint (see the README). */
  compat: Record<string, unknown>;
}

export interface Settings {
  apiUrl: string;
  apiKey: string;
  sessionId: number;
  llm: LlmSettings;
  /** Stop asking the model once this many tokens were spent (0 = no limit). */
  maxTokensPerNight: number;
  /** At most this many trials at once (0 = as many as cards and ports allow). */
  maxParallel: number;
  /** Re-bench a result this much better than the best so far before trusting it. */
  anomalyThreshold: number;
  /** Re-measure this many of the best configs near the end, so finalists rest on repeats. */
  confirmTopK: number;
  /** Give up on an unreachable model after this many minutes and finalize. */
  outageMinutes: number;
  /** How often to poll launches and benchmarks. */
  pollMs: number;
  /** Nudges without a new proposal before the harness treats the search as done. */
  nudgeLimit: number;
}

const THINKING: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

function num(value: unknown, fallback: number): number {
  const n = typeof value === "string" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
}

function str(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function bool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["1", "true", "yes", "on"].includes(value.toLowerCase());
  return fallback;
}

function json(value: unknown): Record<string, unknown> {
  if (value && typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>;
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") return parsed;
    } catch {
      throw new Error(`LLM_COMPAT is not a JSON object: ${value}`);
    }
  }
  return {};
}

/** The env the platform gives every policy container. */
export function platformEnv(env: NodeJS.ProcessEnv): { apiUrl: string; apiKey: string; sessionId: number } {
  const apiUrl = (env.AUTOTUNE_API_URL ?? "").replace(/\/+$/, "");
  const apiKey = env.AUTOTUNE_API_KEY ?? "";
  const sessionId = Number(env.AUTOTUNE_SESSION_ID ?? "");
  const missing = [
    ["AUTOTUNE_API_URL", apiUrl],
    ["AUTOTUNE_API_KEY", apiKey],
    ["AUTOTUNE_SESSION_ID", Number.isFinite(sessionId) && sessionId > 0 ? "ok" : ""],
  ].filter(([, v]) => !v).map(([k]) => k);
  if (missing.length) throw new Error(`missing policy env: ${missing.join(", ")}`);
  return { apiUrl, apiKey, sessionId };
}

export function resolveSettings(env: NodeJS.ProcessEnv, params: Record<string, unknown> = {}): Settings {
  const platform = platformEnv(env);
  const thinking = str(params.thinking ?? env.LLM_THINKING, "high") as ThinkingLevel;
  if (!THINKING.includes(thinking)) throw new Error(`thinking must be one of ${THINKING.join(", ")}, not ${thinking}`);
  const llm: LlmSettings = {
    baseUrl: str(params.base_url ?? env.LLM_BASE_URL, ""),
    apiKey: env.LLM_API_KEY ?? "",
    model: str(params.model ?? env.LLM_MODEL, ""),
    reasoning: bool(params.reasoning ?? env.LLM_REASONING, true),
    thinking,
    contextWindow: num(params.context_window ?? env.LLM_CONTEXT_WINDOW, 128000),
    maxOutputTokens: num(params.max_output_tokens ?? env.LLM_MAX_OUTPUT_TOKENS, 16000),
    compat: { ...json(env.LLM_COMPAT), ...json(params.compat) },
  };
  if (!llm.baseUrl) throw new Error("no LLM endpoint: set LLM_BASE_URL (or policy params base_url)");
  if (!llm.model) throw new Error("no LLM model: set LLM_MODEL (or policy params model)");
  return {
    ...platform,
    llm,
    maxTokensPerNight: num(params.max_tokens_per_night ?? env.AGENT_MAX_TOKENS_PER_NIGHT, 0),
    maxParallel: num(params.max_parallel ?? env.AGENT_MAX_PARALLEL, 0),
    anomalyThreshold: num(params.anomaly_threshold, 0.15),
    confirmTopK: num(params.confirm_top_k, 2),
    outageMinutes: num(params.outage_minutes, 20),
    pollMs: num(env.AGENT_POLL_SECONDS, 10) * 1000,
    nudgeLimit: num(params.nudge_limit, 3),
  };
}

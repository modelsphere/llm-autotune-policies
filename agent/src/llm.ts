// The model, as pi-ai sees it: any OpenAI-compatible endpoint registered as a
// custom provider. Plus the one side call the night makes outside the agent's
// conversation — diagnosing a failed launch from its log — so the planner's
// context gets the conclusion, not 200 lines of traceback.

import { createModels, createProvider, type Model, type Models } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import type { LlmSettings } from "./config.js";
import type { Diagnosis, Trial } from "./lab/state.js";
import { configText } from "./lab/space.js";

export function buildModel(llm: LlmSettings): { models: Models; model: Model<"openai-completions"> } {
  const model: Model<"openai-completions"> = {
    id: llm.model,
    name: llm.model,
    api: "openai-completions",
    provider: "autotune-llm",
    baseUrl: llm.baseUrl,
    reasoning: llm.reasoning,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: llm.contextWindow,
    maxTokens: llm.maxOutputTokens,
    // Self-hosted OpenAI-compatible servers (vLLM, SGLang, …) mostly lack the
    // `developer` role and `reasoning_effort`; the endpoint's own settings win.
    compat: { supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStore: false, ...llm.compat },
  } as Model<"openai-completions">;
  const models = createModels();
  models.setProvider(
    createProvider({
      id: "autotune-llm",
      name: "LLM",
      baseUrl: llm.baseUrl,
      auth: { apiKey: { name: "LLM", resolve: async () => ({ auth: llm.apiKey ? { apiKey: llm.apiKey } : {} }) } },
      models: [model],
      api: openAICompletionsApi(),
    }),
  );
  return { models, model };
}

/** A diagnosis from the log alone, for when the model cannot be asked. */
export function heuristicDiagnosis(excerpt: string): Diagnosis {
  const text = excerpt.toLowerCase();
  const flag = /unrecognized arguments?:\s*(--[\w-]+)/i.exec(excerpt)?.[1];
  if (flag) return { cause: `the engine does not know ${flag}`, offending_flag: flag.replace(/^--/, "").replace(/-/g, "_"), source: "heuristic" };
  if (text.includes("out of memory") || text.includes("outofmemory") || /\boom\b/.test(text)) {
    return { cause: "ran out of GPU memory", fix_hint: "lower the memory fraction, batch or graph sizes, or use more cards", source: "heuristic" };
  }
  if (text.includes("timed out") || text.includes("timeout")) {
    return { cause: "did not come up in time", source: "heuristic" };
  }
  const line = excerpt.split("\n").reverse().find((l) => /error|exception/i.test(l));
  return { cause: line ? line.trim().slice(0, 300) : "unknown — see the log", source: "heuristic" };
}

export async function diagnose(
  models: Models,
  model: Model<"openai-completions">,
  prompt: string,
  trial: Trial,
  excerpt: string,
  engine: string,
): Promise<Diagnosis> {
  const fallback = heuristicDiagnosis(excerpt);
  if (!excerpt.trim()) return fallback;
  const response = await models.completeSimple(
    model,
    {
      systemPrompt: prompt,
      messages: [
        {
          role: "user",
          content:
            `Engine: ${engine}\nConfig: ${configText(trial.config)}\n` +
            `Platform failure class: ${trial.failure?.failureClass ?? "?"}\n` +
            `Platform error: ${trial.failure?.error ?? ""}\n\nLog excerpt:\n${excerpt}`,
          timestamp: Date.now(),
        },
      ],
    },
    { reasoning: "low", maxTokens: 2000 },
  );
  if (response.stopReason === "error" || response.stopReason === "aborted") return fallback;
  const text = response.content.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("");
  const match = /\{[\s\S]*\}/.exec(text);
  if (!match) return fallback;
  try {
    const parsed = JSON.parse(match[0]) as Partial<Diagnosis>;
    if (!parsed.cause) return fallback;
    return {
      cause: String(parsed.cause).slice(0, 500),
      offending_flag: parsed.offending_flag ? String(parsed.offending_flag) : undefined,
      fix_hint: parsed.fix_hint ? String(parsed.fix_hint).slice(0, 500) : undefined,
      source: "llm",
    };
  } catch {
    return fallback;
  }
}

/** What a night learned, for other campaigns (PUT /session/digest). */
export interface Digest {
  summary: string;
  lessons: { text: string; scope: string }[];
  constraints: string[];
  open_questions: string[];
}

const SCOPES = new Set(["campaign", "model", "engine", "hardware"]);

/** The night's digest, written by the model from the full lab record. Null
 *  when the model cannot be asked or answers with nothing usable: the
 *  platform then simply has no digest for this night. */
export async function writeDigest(
  models: Models,
  model: Model<"openai-completions">,
  prompt: string,
  record: string,
): Promise<Digest | null> {
  const response = await models.completeSimple(
    model,
    { systemPrompt: prompt, messages: [{ role: "user", content: record, timestamp: Date.now() }] },
    { reasoning: "medium", maxTokens: 8000 },
  );
  if (response.stopReason === "error" || response.stopReason === "aborted") return null;
  const text = response.content.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("");
  const match = /\{[\s\S]*\}/.exec(text);
  if (!match) return null;
  try {
    const raw = JSON.parse(match[0]) as Partial<Digest>;
    if (!raw.summary || typeof raw.summary !== "string") return null;
    const strings = (v: unknown, max: number, len: number) =>
      (Array.isArray(v) ? v : []).filter((x): x is string => typeof x === "string" && x.trim() !== "").slice(0, max).map((x) => x.slice(0, len));
    return {
      summary: raw.summary.slice(0, 20000),
      lessons: (Array.isArray(raw.lessons) ? raw.lessons : [])
        .filter((l) => l && typeof l.text === "string" && l.text.trim() !== "")
        .slice(0, 50)
        .map((l) => ({ text: l.text.slice(0, 2000), scope: SCOPES.has(String(l.scope)) ? String(l.scope) : "campaign" })),
      constraints: strings(raw.constraints, 50, 2000),
      open_questions: strings(raw.open_questions, 20, 2000),
    };
  } catch {
    return null;
  }
}

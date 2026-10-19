#!/usr/bin/env node
// Does the configured LLM endpoint do what the agent needs? One request
// through pi with a tool attached: it must answer with a tool call, and it
// should stream its reasoning. Run it where the policy will run, with the same
// env (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_COMPAT, …):
//
//   docker run --rm -e LLM_BASE_URL=… -e LLM_MODEL=… -e LLM_API_KEY=… <image> node dist/src/check.js
//
// Prints what came back (never the key) and exits 0 when tool calls work.

import { Type } from "typebox";
import { type LlmSettings, resolveSettings } from "./config.js";
import { buildModel } from "./llm.js";

function settingsFromEnv(): LlmSettings {
  // The platform's variables are not needed for a check.
  const env = { AUTOTUNE_API_URL: "http://unused", AUTOTUNE_API_KEY: "unused", AUTOTUNE_SESSION_ID: "1", ...process.env };
  return resolveSettings(env).llm;
}

const llm = settingsFromEnv();
const { models, model } = buildModel(llm);
const started = Date.now();
const response = await models.completeSimple(
  model,
  {
    systemPrompt: "You tune LLM serving engines. Act only through tools.",
    messages: [
      {
        role: "user",
        content: "Propose one sglang config that raises mem_fraction_static from 0.85 to 0.9. Use the propose tool.",
        timestamp: Date.now(),
      },
    ],
    tools: [
      {
        name: "propose",
        description: "Queue engine configs to benchmark",
        parameters: Type.Object({
          configs: Type.Array(Type.Object({ engine_args: Type.Record(Type.String(), Type.Any()) })),
          hypothesis: Type.String(),
        }),
      },
    ],
  },
  { reasoning: llm.reasoning ? llm.thinking === "off" ? undefined : (llm.thinking as never) : undefined, maxTokens: Math.min(llm.maxOutputTokens, 8000) },
);
const seconds = ((Date.now() - started) / 1000).toFixed(1);
const thinking = response.content.filter((b) => b.type === "thinking").map((b) => (b as { thinking: string }).thinking).join("");
const text = response.content.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("");
const call = response.content.find((b) => b.type === "toolCall") as { name: string; arguments: unknown } | undefined;
console.log(JSON.stringify({
  endpoint: llm.baseUrl,
  model: llm.model,
  seconds: Number(seconds),
  stop_reason: response.stopReason,
  error: response.errorMessage ?? null,
  reasoning_chars: thinking.length,
  reasoning_head: thinking.slice(0, 200),
  text_head: text.slice(0, 200),
  tool_call: call ? { name: call.name, arguments: call.arguments } : null,
  usage: response.usage,
}, null, 2));
process.exitCode = call && call.name === "propose" ? 0 : 1;

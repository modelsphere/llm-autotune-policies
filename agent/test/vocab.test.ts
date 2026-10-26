// The engine vocabulary (contract 1.5): what the agent is told about the
// switches an image reads, and the declared values it has not tried.

import assert from "node:assert/strict";
import { test } from "node:test";
import { PolicyClient } from "../src/contract/client.js";
import { untriedDeclared } from "../src/context/render.js";
import { LabState } from "../src/lab/state.js";
import { configKeyOf, vocabLine } from "../src/lab/vocab.js";
import { buildTools, type ToolContext } from "../src/tools.js";
import { FakePlatform } from "./fake-platform.js";

test("an entry reads as the key a config sets, with its facts and recommendation", () => {
  const env = { kind: "env" as const, name: "SGLANG_ENABLE_OVERLAP_PLAN_STREAM", type: "bool", default: false, help: "Plan the next step on a separate stream. Needs a speculative algorithm.", help_source: "docs", level: "high", when: "speculative decoding on" };
  assert.equal(configKeyOf(env), "env.SGLANG_ENABLE_OVERLAP_PLAN_STREAM");
  assert.equal(vocabLine(env), "- `env.SGLANG_ENABLE_OVERLAP_PLAN_STREAM` (bool, default false) [high; when: speculative decoding on] — Plan the next step on a separate stream.");
  assert.match(vocabLine(env, true), /Needs a speculative algorithm\.$/);
  const flag = { kind: "flag" as const, name: "attention_backend", type: "str", choices: ["fa3", "triton"], help: "x = y", help_source: "source", level: "unknown" };
  assert.equal(vocabLine(flag), '- `attention_backend` (str; one of "fa3", "triton") — x = y (from the engine\'s code)');
});

test("declared values no trial has tried are listed, failed trials count as tried", () => {
  const lab = new LabState();
  lab.addTrial({ purpose: "explore", config: { tp: 2, attention_backend: "fa3" }, cards: 2 });
  const failed = lab.addTrial({ purpose: "explore", config: { tp: 1 }, cards: 1 });
  failed.status = "failed";
  const space = { grid: { tp: [1, 2, 4], attention_backend: ["fa3", "flashinfer"] }, tied: [{ speculative_num_steps: [2, 3] }] };
  assert.deepEqual(untriedDeclared(space, lab.trials), [
    "attention_backend: flashinfer",
    "speculative_num_steps: 2, 3",
    "tp: 4",
  ]);
  assert.deepEqual(untriedDeclared({}, lab.trials), []);
});

test("engine_vocab looks switches up on the platform", async () => {
  const platform = new FakePlatform({});
  await platform.start();
  try {
    const client = new PolicyClient({ apiUrl: platform.url, apiKey: "atk_test", retries: 0 });
    const manifest = await client.getSession();
    const ctx = {
      client, lab: new LabState(), runner: {}, manifest, direction: "max", sessionId: 1,
      trialsLeft: () => null, allowEarlyDone: false, onDoneRefused: () => {}, onIntroduced: () => {}, onProposed: () => {}, log: () => {},
    } as unknown as ToolContext;
    const tool = buildTools(ctx).find((t) => t.name === "engine_vocab");
    assert.ok(tool, "offered when the platform has the vocabulary service");
    const out = await tool.execute("t1", { query: "spec" });
    const body = (out.content[0] as { text: string }).text;
    assert.match(body, /1 match/);
    assert.match(body, /`env\.SGLANG_ENABLE_OVERLAP_PLAN_STREAM`.*\[high; when: speculative decoding on\]/);

    const check = buildTools(ctx).find((t) => t.name === "check")!;
    const checked = (await check.execute("t2", { engine_args: { tp: 2, "env.SGLANG_ENABLE_OVERLAP_PLAN_STREAM": true } })).content[0] as { text: string };
    assert.match(checked.text, /env: SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1/);
    const unknown = (await check.execute("t3", { engine_args: { "env.SGLANG_NOPE": 1 } })).content[0] as { text: string };
    assert.match(unknown.text, /problem \[unknown_env\]/);

    const without = buildTools({ ...ctx, manifest: { ...manifest, services: { ...manifest.services, vocabulary: false } } } as ToolContext);
    assert.ok(!without.some((t) => t.name === "engine_vocab"), "not offered by an older platform");
  } finally {
    await platform.stop();
  }
});

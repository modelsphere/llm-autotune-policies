import assert from "node:assert/strict";
import { test } from "node:test";

import type { AgentMessage } from "@earendil-works/pi-agent-core";

import { withoutOldReasoning } from "../src/driver.js";
import { type ConfigStats, conservative, ranking, relativeSd } from "../src/lab/stats.js";

function group(key: string, values: number[]): ConfigStats {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  return { key, config: { k: key }, trialIds: [values.length], values, feasible: true, mean, min: Math.min(...values), max: Math.max(...values) };
}

test("when the noise is wide, one lucky measurement does not outrank a config measured twice", () => {
  // Campaign 28's noise: the same config measured 697.4k and 679.7k (2.6%
  // apart). A single draw a little above a twice-measured config is then not
  // evidence that it is better.
  const production = group("production", [679_700, 697_400]);
  const steady = group("steady", [700_000, 699_000]);
  const lucky = group("lucky", [702_000]);
  const order = ranking([lucky, steady, production], "max").map((g) => g.key);
  assert.deepEqual(order.slice(0, 2), ["steady", "lucky"]);
});

test("when the noise is narrow, a clearly higher single measurement still leads", () => {
  const production = group("production", [690_000, 690_500]);
  const steady = group("steady", [694_000, 692_100]);
  const better = group("better", [702_000]);
  assert.equal(ranking([steady, better, production], "max")[0].key, "better");
});

test("the pooled noise comes from every repeat; the prior until there is one", () => {
  assert.equal(relativeSd([group("a", [100])], 0.02), 0.02);
  const sd = relativeSd([group("a", [99, 101]), group("b", [200])]);
  assert.ok(Math.abs(sd - Math.sqrt(2 * 0.01 ** 2)) < 1e-9);
  // Lower is better: the conservative value moves up.
  assert.ok(conservative(group("c", [100]), 0.05, "min") > 100);
});

test("old turns lose their reasoning; the latest keeps it", () => {
  const turn = (text: string): AgentMessage =>
    ({ role: "assistant", content: [{ type: "thinking", thinking: `why ${text}` }, { type: "text", text }] }) as unknown as AgentMessage;
  const out = withoutOldReasoning([turn("one"), { role: "user", content: "event", timestamp: 0 } as AgentMessage, turn("two")]);
  const blocks = (m: AgentMessage) => (m as unknown as { content: { type: string }[] }).content.map((b) => b.type);
  assert.deepEqual(blocks(out[0]), ["text"]);
  assert.deepEqual(blocks(out[2]), ["thinking", "text"]);
});

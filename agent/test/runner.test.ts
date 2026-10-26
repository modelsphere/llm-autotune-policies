// The runner and a platform that has no room yet.

import assert from "node:assert/strict";
import { test } from "node:test";
import { NoCapacity, type PolicyClient } from "../src/contract/client.js";
import { LabState } from "../src/lab/state.js";
import { TrialRunner } from "../src/runner.js";

test("a trial the platform has no room for waits before it asks again", async () => {
  // Campaign 1 logged "t61 started" hundreds of times 10 ms apart: the requeued
  // trial was restarted by the finishing attempt itself, not after the wait.
  let launches = 0;
  const client = {
    launch: async () => {
      launches += 1;
      throw new NoCapacity(503, "the cards are still being released");
    },
  } as unknown as PolicyClient;
  const lab = new LabState();
  const runner = new TrialRunner(
    client, lab, { gpu_indices: [0, 1], ports: [28200] },
    { pollMs: 100, direction: "max", anomalyThreshold: 0.25, suite: "screen" },
    { onTrialDone: () => {}, diagnose: async () => ({}), log: () => {} } as never,
  );
  lab.addTrial({ purpose: "remeasure", config: { tp: 2 }, cards: 2 });
  runner.pump();
  await new Promise((r) => setTimeout(r, 350));
  await runner.stopAll();
  assert.ok(launches >= 2 && launches <= 5, `asked ${launches} times in 350 ms with a 100 ms wait`);
  assert.equal(lab.trials[0].status, "cancelled");
});

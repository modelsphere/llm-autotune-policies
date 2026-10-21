import assert from "node:assert/strict";
import { test } from "node:test";

import type { Manifest } from "../src/contract/types.js";
import { catalogueOf, deltas, headlineKeys, scenariosOf, unhealthyPoints } from "../src/lab/measures.js";
import { CATALOGUE, replayMeasures } from "./fake-platform.js";

const catalogue = catalogueOf({ measures: CATALOGUE } as unknown as Manifest);
const replay = (config: Record<string, unknown>, value: number) =>
  scenariosOf({ id: 1, status: "succeeded", measures: replayMeasures(config, value) } as never);

test("the ledger shows a module's headline measures, not every bucket", () => {
  const [s] = replay({}, 1000);
  const keys = headlineKeys(s.operating, catalogue);
  assert.ok(keys.includes("output_tpm_per_gpu") && keys.includes("ttft_ms@p99") && keys.includes("success_rate"));
  assert.ok(!keys.includes("ttft_ms@p90[32k_64k]"), "a bucket is for inspect_trial");
  assert.ok(!keys.includes("error_rate"), "a clean lower-is-better rate stays quiet");
  assert.ok(!keys.includes("requests_completed"), "counts are for inspect_trial");
});

test("failed requests are read from the health measures of a replay", () => {
  assert.deepEqual(unhealthyPoints(replay({}, 1000)), []);
  const [line] = unhealthyPoints(replay({ tp_size: 8 }, 1000));
  assert.match(line, /^replay operating point: success_rate 0\.900, error_rate 0\.100/);
});

test("a sweep level that served nothing is caught too", () => {
  const sweep = [{
    scenario: "sweep", module: "sweep", kind: "sweep", operatingLevel: "c4", operatingRule: "",
    operating: { success_rate: 1 }, points: { c4: { success_rate: 1 }, c64: { requests_completed: 0 } },
  }];
  assert.deepEqual(unhealthyPoints(sweep), ["sweep c64: 0 requests completed"]);
});

test("a comparison marks each change better or worse by the measure's direction", () => {
  const [a] = replay({}, 1000);
  const [b] = replay({}, 1100);
  b.operating["ttft_ms@p99"] = 990;
  const out = deltas(a.operating, b.operating, catalogue);
  assert.ok(out.includes("output_tpm_per_gpu +10.0% better"));
  assert.ok(out.includes("ttft_ms@p99 +10.0% worse"));
  assert.ok(!out.some((d) => d.startsWith("concurrency")), "an unchanged measure is not news");
});

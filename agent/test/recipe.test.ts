import assert from "node:assert/strict";
import { test } from "node:test";

import { recipeLines } from "../src/context/render.js";

test("a pd recipe says its roles have their own flags and shows its chart values", () => {
  const lines = recipeLines({
    engine: "sglang",
    chart: { name: "sglang", version: "0.9.26" },
    values: {
      extraArgs: ["--tp-size=8"],
      image: { repository: "lmsysorg/sglang", tag: "v0.5.20" },
      pd: { enabled: true, prefill: { gpus: 8 }, decode: { gpus: 8 } },
      shm: { sizeLimit: "32Gi" },
    },
  });
  assert.match(lines[0], /sglang chart \(0\.9\.26\), pd/);
  assert.match(lines[0], /prefill\.<flag>/);
  assert.match(lines[1], /"shm":\{"sizeLimit":"32Gi"\}/);
  assert.doesNotMatch(lines[1], /extraArgs|lmsysorg/);
});

test("no recipe, no lines", () => {
  assert.deepEqual(recipeLines(undefined), []);
  assert.deepEqual(recipeLines({ values: {} }), []);
});

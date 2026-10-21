// What the agent reads when a config leaves the declared space: a note, never
// something that reads as a refusal (a model took "outside the declared space"
// for one and stopped exploring with most of the night left).

import assert from "node:assert/strict";
import { test } from "node:test";
import { deviationText } from "../src/tools.js";

test("deviations read as allowed notes", () => {
  for (const kind of ["unknown", "off_grid", "out_of_range", "base_override"]) {
    const line = deviationText({ param: "page_size", kind, value: 32 });
    assert.match(line, /^page_size=32 /);
    assert.match(line, /it is allowed$/);
    assert.doesNotMatch(line, /refus|reject|forbidden/i);
  }
});

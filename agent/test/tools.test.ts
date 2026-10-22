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

import { buildTools, type ToolContext } from "../src/tools.js";
import { LabState } from "../src/lab/state.js";
import { cardsNow, timeLines } from "../src/context/render.js";

function doneTool(trialsLeft: number | null, allowEarlyDone: boolean) {
  const lab = new LabState();
  const refused: string[] = [];
  const ctx = {
    client: {}, lab, runner: {}, manifest: { services: {}, objective: {} }, direction: "max", sessionId: 1,
    trialsLeft: () => trialsLeft, allowEarlyDone, onDoneRefused: (r: string) => refused.push(r), onIntroduced: () => {}, onProposed: () => {}, log: () => {},
  } as unknown as ToolContext;
  const tool = buildTools(ctx).find((t) => t.name === "done")!;
  return { lab, refused, run: () => tool.execute("t1", { reason: "nothing left" }) };
}

test("done is refused while time for two or more trials remains", async () => {
  const early = doneTool(20, false);
  await assert.rejects(early.run(), /refused: about 20 more trial/);
  assert.ok(!early.lab.done);
  assert.deepEqual(early.refused, ["nothing left"], "the night learns the agent asked to end");
  const late = doneTool(1, false);
  await late.run();
  assert.ok(late.lab.done);
  const allowed = doneTool(20, true); // the campaign lets the agent stop early
  await allowed.run();
  assert.ok(allowed.lab.done);
});

test("the record states the whole search budget and the validation reserve after it", () => {
  const lines = timeLines(
    { started_at: "2026-10-21T11:00:00Z", search_deadline: "2026-10-21T13:00:00Z", hard_deadline: "2026-10-21T13:45:00Z" },
    30,
    Date.parse("2026-10-21T12:30:00Z"),
  ).join("\n");
  assert.match(lines, /120 min in all/);
  assert.match(lines, /90 min used, 30 min left \(25%\)/);
  assert.match(lines, /45 min reserved for the platform to validate the finalists/);
});

test("the record counts cards, so trials of different widths can share the session", () => {
  assert.equal(cardsNow(7, 1, [{ id: 3, cards: 2 }, { id: 4, cards: 4 }]), "6 of 7 card(s) in use by 2 trial(s) (#3: 2, #4: 4), 1 free");
  assert.equal(cardsNow(8, 8, []), "0 of 8 card(s) in use by 0 trial(s), 8 free");
});

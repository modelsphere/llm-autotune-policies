// The conversation may be trimmed; the system prompt and the lab record never are.

import assert from "node:assert/strict";
import { test } from "node:test";
import { createModels, fauxProvider } from "@earendil-works/pi-ai";
import { AgentDriver, estimateTokens } from "../src/driver.js";

test("trimming drops old turns, keeps the system prompt, and always ends on the fresh record", () => {
  const faux = fauxProvider({ models: [{ id: "m", contextWindow: 20000, maxTokens: 2000 }] });
  const models = createModels();
  models.setProvider(faux.provider);
  let rendered = 0;
  const driver = new AgentDriver({
    models, model: faux.getModel() as never, systemPrompt: "SYSTEM", tools: [], thinking: "off",
    contextWindow: 20000, maxOutputTokens: 2000, outageMinutes: 1,
    record: () => `<lab_record>${++rendered}</lab_record>`, log: () => {},
  });
  const big = "x".repeat(8000); // ~2000 tokens each
  const messages = [
    { role: "system", content: "SYSTEM", timestamp: 0 },
    ...Array.from({ length: 12 }, (_, i) => [
      { role: "user", content: `event ${i} ${big}`, timestamp: i },
      { role: "assistant", content: [{ type: "text", text: `reply ${i}` }], timestamp: i },
    ]).flat(),
  ];
  const out = (driver as unknown as { contextFor(m: unknown[]): { role: string; content: unknown }[] }).contextFor(messages);
  assert.equal(out[0].role, "system");
  assert.equal(out.at(-1)!.content, `<lab_record>1</lab_record>`);
  assert.ok(estimateTokens(out) < 20000 - 2000, "fits the window");
  assert.ok(out.length < messages.length, "old turns were dropped");
  assert.equal(out[1].role, "user", "the kept conversation starts on a user message");
  assert.ok(JSON.stringify(out).includes("event 11"), "the newest turns are kept");
});

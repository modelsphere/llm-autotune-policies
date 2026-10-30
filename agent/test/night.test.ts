// A whole night, offline: the fake platform over real HTTP, and a scripted
// model (pi-ai's faux provider) standing in for the LLM. If this passes, the
// harness honours the contract and the lab's books stay right.

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  type AssistantMessage,
  createModels,
  fauxAssistantMessage,
  fauxProvider,
  fauxText,
  fauxToolCall,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import { CONTRACT_VERSION } from "../src/contract/types.js";
import { runNight } from "../src/night.js";
import { FakePlatform, score } from "./fake-platform.js";

function textOf(message: TranscriptContext["messages"][number] | undefined): string {
  if (!message) return "";
  const content = (message as { content: unknown }).content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((c: { text?: string }) => c.text ?? "").join("");
  return "";
}

/** A scripted model: decides from the lab record what a sensible agent would do. */
function scriptedModel(opts: { stallMs?: number } = {}) {
  const faux = fauxProvider({ models: [{ id: "scripted", reasoning: true, contextWindow: 200000, maxTokens: 8000 }] });
  const models = createModels();
  models.setProvider(faux.provider);
  let phase = 0;
  let calls = 0;
  const seen: string[] = [];
  const respond = async (context: TranscriptContext): Promise<AssistantMessage> => {
    const messages = context.messages;
    const system = textOf(messages[0]);
    if (system.startsWith("You write the digest")) {
      return fauxAssistantMessage(fauxText('{"summary": "chunked_prefill_size 8192 with mem_fraction_static 0.9 beat production.", "lessons": [{"text": "0.95 OOMs at load", "scope": "model"}, {"text": "vague", "scope": "galaxy"}], "constraints": ["mem_fraction_static 0.95 OOMs"], "open_questions": ["does 16384 help too?"]}'));
    }
    if (system.startsWith("You diagnose")) {
      return fauxAssistantMessage(fauxText('{"cause": "out of memory while capturing CUDA graphs", "offending_flag": "mem_fraction_static", "fix_hint": "lower it below 0.93"}'));
    }
    calls += 1;
    if (opts.stallMs && calls === 1) await new Promise((r) => setTimeout(r, opts.stallMs));
    const record = textOf(messages.at(-1));
    seen.push(record);
    const previous = messages.at(-2);
    if (previous?.role === "toolResult") return fauxAssistantMessage(fauxText("Waiting for results."));
    if (phase === 0) {
      phase = 1;
      return fauxAssistantMessage(
        [
          fauxToolCall("open_hypothesis", { statement: "more KV cache (higher mem_fraction_static) raises throughput" }, { id: "c1" }),
          fauxToolCall(
            "propose",
            {
              hypothesis_id: "H1",
              prediction: { direction: "up", magnitude_pct: 15 },
              configs: [
                { engine_args: { mem_fraction_static: 0.9 }, rationale: "more KV cache" },
                { engine_args: { mem_fraction_static: 0.9, quantization: "fp8" }, rationale: "cheat" },
                { engine_args: { mem_fraction_static: 0.95 }, rationale: "even more" },
              ],
            },
            { id: "c2" },
          ),
        ],
        { stopReason: "toolUse" },
      );
    }
    const measured = (re: RegExp) => record.split("\n").some((line) => re.test(line) && line.includes("measured"));
    if (phase === 1 && measured(/mem_fraction_static=0\.9\b/) && /FAILED at launch/.test(record)) {
      phase = 2;
      return fauxAssistantMessage(
        [
          fauxToolCall("add_constraint", { text: "mem_fraction_static 0.95 OOMs at load", evidence_trials: [3] }, { id: "c3" }),
          fauxToolCall("close_hypothesis", { id: "H1", verdict: "supported", evidence: "0.9 beat the anchor" }, { id: "c4" }),
          fauxToolCall("propose", { configs: [{ engine_args: { mem_fraction_static: 0.9, chunked_prefill_size: 8192 } }], prediction: { direction: "up" } }, { id: "c5" }),
        ],
        { stopReason: "toolUse" },
      );
    }
    if (phase === 2 && measured(/chunked_prefill_size=8192/)) {
      phase = 3;
      return fauxAssistantMessage(
        [
          fauxToolCall("record_lesson", { text: "chunked_prefill_size 8192 helps on this workload", scope: "campaign" }, { id: "c6" }),
          fauxToolCall("done", { reason: "the space around the best is covered" }, { id: "c7" }),
        ],
        { stopReason: "toolUse" },
      );
    }
    return fauxAssistantMessage(fauxText("Waiting for results."));
  };
  faux.setResponses(Array.from({ length: 500 }, () => respond));
  return { models, model: faux.getModel() as never, seen, calls: () => calls };
}

test("a whole night: anchor, hypotheses, refusals, a diagnosed failure, confirmation, finalize, exit", async () => {
  const platform = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } }, production: { mem_fraction_static: 0.85 }, params: { allow_early_done: true } });
  await platform.start();
  const lines: string[] = [];
  try {
    const scripted = scriptedModel();
    const code = await runNight({ env: platform.env, model: scripted, log: (l) => lines.push(l), tickMs: 10 });
    assert.equal(code, 0);
    assert.ok(platform.finalized, "the policy called /session/finalized");

    // The handshake, then a status signal: the search ended by itself, not by the deadline.
    assert.deepEqual(platform.heartbeats[0].capabilities, []);
    assert.equal(platform.heartbeats[0].contract_version, CONTRACT_VERSION);
    assert.ok(platform.heartbeats.some((b) => b.status === "exhausted"));

    // The forbidden config was refused before it cost a launch; the OOM was diagnosed.
    assert.ok(lines.some((l) => l.includes("REJECTED config 2") && l.includes("forbidden")), "refusal reached the agent");
    const launched = [...platform.launches.values()].map((l) => l.config);
    assert.ok(!launched.some((c) => "quantization" in c));
    assert.ok(scripted.seen.some((r) => r.includes("out of memory while capturing CUDA graphs")), "the diagnosis is in the record");

    // Every engine was released.
    assert.ok([...platform.launches.values()].every((l) => l.released));

    // The best config is the registered contender, backed by a confirmation re-measure.
    const best = platform.contenders[0] as { launch_spec: { engine_args: Record<string, unknown>; environment: Record<string, unknown> }; evidence: { measurements: number }; trial_ids: number[] };
    assert.equal(best.launch_spec.engine_args.chunked_prefill_size, 8192);
    assert.equal(best.launch_spec.environment.image_digest, "sglang@sha256:abc");
    assert.ok(best.evidence.measurements >= 2, "the finalist was re-measured");
    // Its evidence names the platform's own trials, one per measurement.
    assert.equal(best.trial_ids.length, best.evidence.measurements);
    assert.ok(best.trial_ids.every((id) => id > 1000), "platform trial ids, not the agent's numbers");

    // The record reads the replay in measures, with what they mean.
    const record = scripted.seen.at(-1) ?? "";
    assert.ok(record.includes("replay: output_tpm_per_gpu"), "the replay's numbers are in the ledger");
    assert.ok(record.includes("`success_rate` (ratio, higher is better)"), "the record explains its measures");
    assert.ok(record.includes("there are no levels"), "the record says what the screen sends");
    // The image's recommended switches are in the record, high ones described.
    assert.ok(record.includes("## What this engine image accepts"), "the record says what the image reads");
    assert.match(record, /1 flag\(s\) and 2 environment variable\(s\), read from the image \(sglang 0\.5\.19\)/);
    // The model's facts arrive with the vocabulary when the manifest had none.
    assert.match(record, /What the model is \(its config\.json\): .*MoE, 512 experts \(10 active per token\); built-in draft layers \(MTP\/NEXTN\): 1/);
    assert.ok(record.includes("`env.SGLANG_ENABLE_OVERLAP_PLAN_STREAM` (bool, default false) [high; when: speculative decoding on]"));
    assert.ok(record.includes("- `chunked_prefill_size` (int) [mid]"), "mid switches in one line");
    assert.ok(!record.includes("SGLANG_LOG_GC"), "unknown ones stay out of the record");

    // The plan declared the parameter the agent introduced outside the space.
    const lastPlan = platform.plans.at(-1) as { extras: { param: string }[] };
    assert.ok(lastPlan.extras.some((e) => e.param === "chunked_prefill_size"));

    // What tonight learned is filed for other campaigns, scopes kept honest.
    const digest = platform.digest as { summary: string; lessons: { scope: string }[]; open_questions: string[] };
    assert.ok(digest, "a digest was filed");
    assert.match(digest.summary, /8192/);
    assert.deepEqual(digest.lessons.map((l) => l.scope), ["model", "campaign"]);
    assert.deepEqual(digest.open_questions, ["does 16384 help too?"]);

    // What tonight learned is saved for the next night.
    const state = platform.state as { format: string; lessons: { text: string }[]; constraints: { text: string }[]; nights: { best: { objective: number } }[] };
    assert.equal(state.format, "llm-autotune-agent/1");
    assert.ok(state.lessons.some((l) => l.text.includes("8192")));
    assert.ok(state.constraints.some((c) => c.text.includes("OOM")));
    assert.equal(state.nights[0].best.objective, score({ mem_fraction_static: 0.9, chunked_prefill_size: 8192 }));
  } finally {
    await platform.stop();
  }
});

test("the next night reads what the last one learned", async () => {
  const first = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } }, params: { allow_early_done: true } });
  await first.start();
  try {
    await runNight({ env: first.env, model: scriptedModel(), log: () => {}, tickMs: 10 });
  } finally {
    await first.stop();
  }
  const second = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } }, state: first.state, sessionId: 8, params: { allow_early_done: true } });
  await second.start();
  try {
    const scripted = scriptedModel();
    await runNight({ env: second.env, model: scripted, log: () => {}, tickMs: 10 });
    const record = scripted.seen[0];
    assert.ok(record.includes("## Earlier nights of this campaign"));
    assert.ok(record.includes("mem_fraction_static 0.95 OOMs at load"));
    assert.ok(record.includes("earlier night"));
  } finally {
    await second.stop();
  }
});

test("the heartbeat keeps beating while the model thinks", async () => {
  const platform = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } }, heartbeatIntervalS: 0.05, params: { allow_early_done: true } });
  await platform.start();
  try {
    const scripted = scriptedModel({ stallMs: 600 });
    const beatsBefore = platform.heartbeats.length;
    const night = runNight({ env: platform.env, model: scripted, log: () => {}, tickMs: 10 });
    await new Promise((r) => setTimeout(r, 500));
    assert.equal(scripted.calls(), 1, "the model is still on its first, stalled call");
    assert.ok(platform.heartbeats.length - beatsBefore >= 5, `only ${platform.heartbeats.length} beats during the stall`);
    assert.equal(await night, 0);
  } finally {
    await platform.stop();
  }
});

test("a policy with no LLM configured says so and exits", async () => {
  const platform = new FakePlatform();
  await platform.start();
  try {
    const env = { ...platform.env, LLM_BASE_URL: "" };
    const code = await runNight({ env, log: () => {} });
    assert.equal(code, 1);
    assert.ok(platform.heartbeats.some((b) => b.status === "error" && b.failure_class === "config"));
    assert.ok(platform.events.some((e) => e.level === "error" && e.message.includes("LLM_BASE_URL")));
  } finally {
    await platform.stop();
  }
});

test("a finalize while the model is mid-call ends the night cleanly", async () => {
  const platform = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } } });
  await platform.start();
  try {
    const night = runNight({ env: platform.env, model: scriptedModel({ stallMs: 3000 }), log: () => {}, tickMs: 10 });
    await new Promise((r) => setTimeout(r, 300));
    platform.command = "finalize"; // the operator's force-stop
    const started = Date.now();
    assert.equal(await night, 0);
    assert.ok(Date.now() - started < 2500, "did not wait for the stalled model call");
    assert.ok(platform.finalized);
    assert.ok([...platform.launches.values()].every((l) => l.released));
  } finally {
    await platform.stop();
  }
});

test("an agent that goes quiet does not leave the cards idle or end the search", async () => {
  const platform = new FakePlatform({ space: { grid: { mem_fraction_static: [0.85, 0.9] } }, params: { keep_busy_minutes: 0.003 } });
  await platform.start();
  const lines: string[] = [];
  try {
    // Proposes once, then only waits — as if it believed the search converged.
    const faux = fauxProvider({ models: [{ id: "quiet", reasoning: true, contextWindow: 200000, maxTokens: 8000 }] });
    const models = createModels();
    models.setProvider(faux.provider);
    let first = true;
    const events: string[] = [];
    const respond = async (context: TranscriptContext): Promise<AssistantMessage> => {
      for (const m of context.messages) if (textOf(m).startsWith("[lab event")) events.push(textOf(m));
      if (!first) return fauxAssistantMessage(fauxText("Converged; waiting."));
      first = false;
      return fauxAssistantMessage(
        [fauxToolCall("propose", { configs: [{ engine_args: { mem_fraction_static: 0.9 } }], prediction: { direction: "up" } }, { id: "q1" })],
        { stopReason: "toolUse" },
      );
    };
    faux.setResponses(Array.from({ length: 2000 }, () => respond));
    const night = runNight({ env: platform.env, model: { models, model: faux.getModel() as never }, log: (l) => lines.push(l), tickMs: 10 });
    await new Promise((r) => setTimeout(r, 2000));
    platform.command = "finalize";
    assert.equal(await night, 0);

    assert.ok(lines.some((l) => l.includes("harness queued")), "the harness filled the idle cards");
    assert.ok(platform.events.some((e) => e.message.includes("the harness queued re-measures")));
    assert.ok(events.some((e) => e.includes("Widen the search") && e.includes("mem_fraction_static (0.9)")), "the agent was steered to widen, told what it tried");
    assert.ok(!platform.heartbeats.some((b) => b.status === "exhausted"), "the search did not end itself");
    assert.ok(platform.launches.size > 3, `only ${platform.launches.size} launches`);
  } finally {
    await platform.stop();
  }
});

test("production built by the platform from the recipe: launched by its config, shown whole", async () => {
  // 1.7: the platform sends every flag production runs and the config that
  // launches it — {} when it is the recipe as written. The anchor sends {},
  // never the flags back, and the record shows what the anchor runs.
  const platform = new FakePlatform({
    space: { grid: { mem_fraction_static: [0.85, 0.9] } },
    productionManifest: {
      engine_args: { tp: 1, mem_fraction_static: 0.83, chunked_prefill_size: 8192 },
      env: { PYTORCH_ALLOC_CONF: "expandable_segments:True" },
      image: "lmsysorg/sglang:v0.5.22",
      config: {},
      source: "recipe",
      launch_command: "",
    },
    params: { allow_early_done: true },
  });
  await platform.start();
  try {
    const scripted = scriptedModel();
    const code = await runNight({ env: platform.env, model: scripted, log: () => {}, tickMs: 10 });
    assert.equal(code, 0);
    const [first] = [...platform.launches.values()];
    assert.deepEqual(first.config, {}, "the anchor is production's config, not its flags");
    const record = scripted.seen.find((r) => r.includes("## Production (the anchor)")) ?? "";
    assert.ok(record.includes("Production config (the campaign's recipe as written): chunked_prefill_size=8192 mem_fraction_static=0.83 tp=1"), record.slice(0, 400));
    assert.ok(record.includes("Production env: PYTORCH_ALLOC_CONF=expandable_segments:True"));
    assert.ok(record.includes("Production image: lmsysorg/sglang:v0.5.22"));
  } finally {
    await platform.stop();
  }
});

// The model's only verbs. No shell, no files, no code: every action is an
// experiment-level request the harness can check, schedule and account for.
// Proposals pass through the platform's own dry run (POST /configs/check) and
// the lab's dedup before they can cost a launch; a rejection comes straight
// back to the model with the reason, inside the same turn.

import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { type Static, type TSchema, Type } from "typebox";
import type { PolicyClient } from "./contract/client.js";
import type { Config, Manifest } from "./contract/types.js";
import { configKey, type LabState, now, type Prediction, type Trial } from "./lab/state.js";
import { configText, diffText } from "./lab/space.js";
import { byConfig, deltaText, type Direction, noiseBand, trialObjective } from "./lab/stats.js";
import { catalogueOf, deltas } from "./lab/measures.js";
import type { TrialRunner } from "./runner.js";
import { type VocabEntry, vocabLine } from "./lab/vocab.js";

export interface ToolContext {
  client: PolicyClient;
  lab: LabState;
  runner: TrialRunner;
  manifest: Manifest;
  direction: Direction;
  sessionId: number;
  /** Estimated trials that still fit before the search deadline. */
  trialsLeft(): number | null;
  /** The campaign lets the agent end the search before its deadline. */
  allowEarlyDone: boolean;
  /** done was refused for the time left; the reason the agent gave. */
  onDoneRefused(reason: string): void;
  /** A new parameter outside the declared space was introduced. */
  onIntroduced(param: string, reason: string): void;
  onProposed(trials: Trial[]): void;
  log(line: string): void;
}

/** An AgentTool whose execute() receives its validated, typed arguments. */
function tool<S extends TSchema>(def: {
  name: string;
  label: string;
  description: string;
  parameters: S;
  execute(params: Static<S>): Promise<AgentToolResult<any>>;
}): AgentTool<any> {
  return {
    name: def.name,
    label: def.label,
    description: def.description,
    parameters: def.parameters,
    execute: (_toolCallId: string, params: unknown) => def.execute(params as Static<S>),
  };
}

/** A deviation from the declared space, worded as what it is: a note, never a refusal. */
export function deviationText(d: { param?: unknown; kind?: unknown; value?: unknown }): string {
  const what = `${String(d.param)}=${JSON.stringify(d.value)}`;
  switch (d.kind) {
    case "unknown": return `${what} is a parameter the declared space does not list; it is allowed`;
    case "off_grid": return `${what} is not one of the declared values; it is allowed`;
    case "out_of_range": return `${what} is outside the declared range; it is allowed`;
    case "base_override": return `${what} changes a value the campaign base fixes; it is allowed`;
    default: return `${what}: ${String(d.kind)}`;
  }
}

function text(body: string, details: Record<string, unknown> = {}): AgentToolResult<Record<string, unknown>> {
  return { content: [{ type: "text", text: body }], details };
}

const PredictionSchema = Type.Object({
  direction: Type.Union([Type.Literal("up"), Type.Literal("down"), Type.Literal("same"), Type.Literal("unknown")], {
    description: "Expected effect on the objective, relative to `versus` (default: the anchor)",
  }),
  magnitude_pct: Type.Optional(Type.Number({ description: "Expected change in percent" })),
  versus: Type.Optional(Type.Integer({ description: "Trial id the prediction is relative to" })),
  note: Type.Optional(Type.String()),
});

function trialOr404(lab: LabState, id: number): Trial {
  const trial = lab.trial(id);
  if (!trial) throw new Error(`no trial #${id}`);
  return trial;
}

function truncateJson(value: unknown, limit: number): string {
  const s = JSON.stringify(value, null, 1);
  return s.length > limit ? `${s.slice(0, limit)}\n… (${s.length - limit} more characters)` : s;
}

export function buildTools(ctx: ToolContext): AgentTool<any>[] {
  const { client, lab, runner } = ctx;
  const catalogue = catalogueOf(ctx.manifest);
  const anchorConfig = () => lab.anchor?.config;

  const propose = tool({
    name: "propose",
    label: "Propose",
    description:
      "Queue configs to launch and benchmark. Each is checked by the platform first (forbidden parameters, command-line round trip, safety) and against tonight's ledger (exact duplicates are refused — use remeasure). Accepted configs start as soon as cards free up. Give a hypothesis and a prediction: they are scored when results land.",
    parameters: Type.Object({
      configs: Type.Array(
        Type.Object({
          engine_args: Type.Record(Type.String(), Type.Any(), {
            description: "The COMPLETE config: flat dict of engine args, canonical snake_case keys, scalar values; an environment variable as an `env.NAME` key. Anything not set takes the campaign base / engine default.",
          }),
          rationale: Type.Optional(Type.String()),
        }),
        { minItems: 1, maxItems: 16 },
      ),
      hypothesis_id: Type.Optional(Type.String()),
      prediction: Type.Optional(PredictionSchema),
    }),
    execute: async (params) => {
      if (lab.done) throw new Error(`the search is over (${lab.done.reason}); nothing more will be launched`);
      const hypothesis = params.hypothesis_id ? lab.hypothesis(params.hypothesis_id) : undefined;
      if (params.hypothesis_id && !hypothesis) throw new Error(`no hypothesis ${params.hypothesis_id}; open it first`);
      const accepted: Trial[] = [];
      const lines: string[] = [];
      for (const [index, entry] of params.configs.entries()) {
        const label = `config ${index + 1}`;
        const check = await client.checkConfig(entry.engine_args ?? {});
        if (!check.ok) {
          lines.push(`REJECTED ${label}: ${(check.problems ?? []).map((p) => p.message).join("; ")}`);
          continue;
        }
        const canonical = check.config ?? {};
        const key = configKey(canonical);
        const previous = lab.seen(key);
        if (previous) {
          lines.push(`REJECTED ${label}: identical to #${previous.id} (${previous.status}); use remeasure to measure it again`);
          continue;
        }
        if (!runner.fits(check.cards ?? 0)) {
          lines.push(`REJECTED ${label}: needs ${check.cards} card(s); this session holds ${runner.totalCards}`);
          continue;
        }
        const trial = lab.addTrial({
          purpose: "explore",
          config: canonical,
          cards: check.cards ?? 0,
          hypothesisId: hypothesis?.id,
          prediction: params.prediction,
          rationale: entry.rationale,
        });
        hypothesis?.trialIds.push(trial.id);
        accepted.push(trial);
        for (const d of check.deviations ?? []) {
          if (d.kind === "unknown" && typeof d.param === "string") {
            ctx.onIntroduced(d.param, entry.rationale || hypothesis?.statement || "introduced by the agent");
          }
        }
        const notes = (check.deviations ?? []).map(deviationText);
        const warn = (check.warnings?.length ? ` (warning: ${check.warnings.join("; ")})` : "") + (notes.length ? ` (note: ${notes.join("; ")})` : "");
        lines.push(`QUEUED #${trial.id} (${trial.cards} card(s)): ${diffText(canonical, anchorConfig())}${warn}`);
      }
      runner.pump();
      if (accepted.length) ctx.onProposed(accepted);
      const queued = lab.trials.filter((t) => t.status === "queued").length;
      lines.push(`Now: ${runner.inFlight} running, ${queued} queued, ${runner.freeCardCount} of ${runner.totalCards} card(s) free.`);
      return text(lines.join("\n"), { accepted: accepted.map((t) => t.id) });
    },
  });

  const check = tool({
    name: "check",
    label: "Check",
    description: "Dry-run a config: the platform's canonical form, the engine arguments it renders to, cards used, and anything that would be refused. Free; launches nothing.",
    parameters: Type.Object({ engine_args: Type.Record(Type.String(), Type.Any()) }),
    execute: async (params) => {
      const result = await client.checkConfig(params.engine_args ?? {});
      const previous = result.ok ? lab.seen(configKey(result.config ?? {})) : undefined;
      const lines = [
        `ok: ${result.ok}${previous ? ` — but identical to #${previous.id} (${previous.status})` : ""}`,
        `canonical: ${configText(result.config ?? {})}`,
        `vs anchor: ${diffText(result.config ?? {}, anchorConfig())}`,
        `cards: ${result.cards}`,
      ];
      const shape = result.shape as { topology?: string; pods?: { role: string; gpus: number; count: number }[] } | undefined;
      if (shape?.pods?.length) lines.push(`pods: ${shape.topology} — ${shape.pods.map((p) => `${p.count}× ${p.role} (${p.gpus} card(s))`).join(", ")}`);
      const commands = Object.entries(result.commands ?? {});
      if (commands.length) for (const [role, argv] of commands) lines.push(`command (${role}): ${argv.join(" ")}`);
      else lines.push(`command: ${(result.command ?? []).join(" ")}`);
      const env = Object.entries(result.env ?? {});
      if (env.length) lines.push(`env: ${env.map(([k, v]) => `${k}=${v}`).join(" ")}`);
      for (const p of result.problems ?? []) lines.push(`problem [${p.kind}]: ${p.message}`);
      for (const w of result.warnings ?? []) lines.push(`warning: ${w}`);
      if (result.ok) lines.push("refusals: none — this config may be proposed");
      for (const d of result.deviations ?? []) lines.push(`note (allowed, not a refusal): ${deviationText(d)}`);
      return text(lines.join("\n"));
    },
  });

  const engineVocab = tool({
    name: "engine_vocab",
    label: "Engine vocabulary",
    description:
      "Look up the flags and environment variables this engine image reads: type, default, choices, what it does (the engine's own words), and the platform's recommendation (high, mid, low, unknown). Search by a substring of the name or description. Use it before setting a switch you have not used tonight; an env var the image does not read is refused.",
    parameters: Type.Object({
      query: Type.Optional(Type.String({ description: "Substring of the name or description, e.g. \"spec\" or \"allreduce\"" })),
      kind: Type.Optional(Type.Union([Type.Literal("flag"), Type.Literal("env")])),
      level: Type.Optional(Type.Union([Type.Literal("high"), Type.Literal("mid"), Type.Literal("low"), Type.Literal("unknown")], {
        description: "Lowest recommendation to include (mid = high and mid)",
      })),
    }),
    execute: async (params) => {
      const vocab = await client.engineVocabulary({ ...params, limit: 40 });
      if (vocab.status === "none") return text(`The platform has not read this image's vocabulary yet.${vocab.notes?.length ? ` ${vocab.notes.join("; ")}` : ""}`);
      const lines = [`${vocab.engine} ${vocab.engine_version || ""}: ${vocab.total} match(es)${(vocab.total ?? 0) > (vocab.entries?.length ?? 0) ? `, showing ${vocab.entries?.length}` : ""}`];
      for (const e of (vocab.entries ?? []) as VocabEntry[]) lines.push(vocabLine(e, true));
      if (vocab.notes?.length) lines.push(`notes: ${vocab.notes.join("; ")}`);
      return text(lines.join("\n"));
    },
  });

  const inspectTrial = tool({
    name: "inspect_trial",
    label: "Inspect trial",
    description: "Everything about one trial: every measure of every benchmark module (and each sweep level), the failure and its log excerpt, the engine's own resolved arguments (when it reports them), its environment.",
    parameters: Type.Object({ id: Type.Integer() }),
    execute: async (params) => {
      const t = trialOr404(lab, params.id);
      const { serverInfo, ...rest } = t;
      return text(
        truncateJson(rest, 16000) +
          (serverInfo ? `\n\nEngine-reported server info:\n${truncateJson(serverInfo, 8000)}` : ""),
      );
    },
  });

  const readLog = tool({
    name: "read_log",
    label: "Read log",
    description: "A trial's engine log: tail, or lines matching a case-insensitive regex with context. Startup errors, OOM traces and the engine's own throughput/batch lines are here.",
    parameters: Type.Object({
      trial_id: Type.Integer(),
      grep: Type.Optional(Type.String()),
      tail: Type.Optional(Type.Integer({ minimum: 1, maximum: 2000 })),
      context: Type.Optional(Type.Integer({ minimum: 0, maximum: 20 })),
    }),
    execute: async (params) => {
      const t = trialOr404(lab, params.trial_id);
      if (!t.launchId) throw new Error(`trial #${t.id} never launched`);
      const log = await client.launchLog(t.launchId, { grep: params.grep, tail: params.tail ?? 200, context: params.context });
      return text(log || "(empty)");
    },
  });

  const compare = tool({
    name: "compare",
    label: "Compare",
    description: "Two trials side by side: config difference, objective delta with the noise band, and every measure's change per benchmark module (and per sweep level), marked better or worse.",
    parameters: Type.Object({ a: Type.Integer(), b: Type.Integer() }),
    execute: async (params) => {
      const a = trialOr404(lab, params.a);
      const b = trialOr404(lab, params.b);
      const noise = noiseBand(byConfig(lab.trials));
      const va = trialObjective(a);
      const vb = trialObjective(b);
      const lines = [
        `#${b.id} vs #${a.id}: ${diffText(b.config, a.config)}`,
        `objective: #${a.id} ${va ?? "—"} → #${b.id} ${vb ?? "—"}${va !== null && vb !== null ? ` (${deltaText(vb, va, noise, ctx.direction)})` : ""}`,
      ];
      const sa = a.measurements.at(-1)?.scenarios ?? [];
      const sb = b.measurements.at(-1)?.scenarios ?? [];
      for (const s of sb) {
        const other = sa.find((x) => x.scenario === s.scenario);
        if (!other) continue;
        const at = s.operatingLevel || other.operatingLevel ? ` (#${a.id} at ${other.operatingLevel ?? "—"}, #${b.id} at ${s.operatingLevel ?? "—"})` : "";
        lines.push(`${s.scenario} operating point${at}: ${deltas(other.operating, s.operating, catalogue).join(", ") || "no change"}`);
        for (const [point, values] of Object.entries(s.points)) {
          if (!other.points[point]) continue;
          const d = deltas(other.points[point], values, catalogue);
          if (d.length) lines.push(`${s.scenario} ${point}: ${d.join(", ")}`);
        }
      }
      return text(lines.join("\n"));
    },
  });

  const remeasure = tool({
    name: "remeasure",
    label: "Remeasure",
    description: "Launch and benchmark an already-measured config again (a fresh engine), to tell a real difference from noise.",
    parameters: Type.Object({ trial_id: Type.Integer(), reason: Type.String() }),
    execute: async (params) => {
      if (lab.done) throw new Error("the search is over");
      const of = trialOr404(lab, params.trial_id);
      const trial = lab.addTrial({ purpose: "remeasure", config: of.config, cards: of.cards, remeasureOf: of.id, rationale: params.reason });
      runner.pump();
      ctx.onProposed([trial]);
      return text(`QUEUED #${trial.id}: remeasure of #${of.id}`);
    },
  });

  const cancelTrial = tool({
    name: "cancel_trial",
    label: "Cancel trial",
    description: "Cancel a queued or running trial whose answer you no longer need, to free its cards sooner.",
    parameters: Type.Object({ id: Type.Integer(), reason: Type.String() }),
    execute: async (params) => {
      const t = trialOr404(lab, params.id);
      if (!["queued", "launching", "serving", "benching"].includes(t.status)) throw new Error(`#${t.id} is already ${t.status}`);
      runner.cancel(t, params.reason);
      return text(`cancel requested for #${t.id}`);
    },
  });

  const openHypothesis = tool({
    name: "open_hypothesis",
    label: "Open hypothesis",
    description: "Record a hypothesis before testing it. Returns its id for propose(hypothesis_id=…).",
    parameters: Type.Object({ statement: Type.String() }),
    execute: async (params) => {
      const h = lab.openHypothesis(params.statement);
      return text(`opened ${h.id}`);
    },
  });

  const closeHypothesis = tool({
    name: "close_hypothesis",
    label: "Close hypothesis",
    description: "Settle a hypothesis with its evidence. Settled hypotheses are carried to later nights.",
    parameters: Type.Object({
      id: Type.String(),
      verdict: Type.Union([Type.Literal("supported"), Type.Literal("refuted"), Type.Literal("inconclusive")]),
      evidence: Type.String(),
    }),
    execute: async (params) => {
      const h = lab.hypothesis(params.id);
      if (!h) throw new Error(`no hypothesis ${params.id}`);
      h.status = params.verdict;
      h.evidence = params.evidence;
      h.closedAt = now();
      return text(`${h.id} closed as ${h.status}`);
    },
  });

  const addConstraint = tool({
    name: "add_constraint",
    label: "Add constraint",
    description: "Record something that must not be tried again and why (e.g. 'tp=1 with mem_fraction_static>=0.92 OOMs at load'), with the trials that show it. Carried to later nights.",
    parameters: Type.Object({ text: Type.String(), evidence_trials: Type.Array(Type.Integer()) }),
    execute: async (params) => {
      lab.constraints.push({ text: params.text, evidence: params.evidence_trials, at: now(), night: ctx.sessionId });
      return text("constraint recorded");
    },
  });

  const recordLesson = tool({
    name: "record_lesson",
    label: "Record lesson",
    description: "Write down something learned that a later night (or another campaign on this model, engine or hardware) should know. Lessons survive context trimming and nights.",
    parameters: Type.Object({
      text: Type.String(),
      scope: Type.Union([Type.Literal("campaign"), Type.Literal("model"), Type.Literal("engine"), Type.Literal("hardware")]),
    }),
    execute: async (params) => {
      lab.lessons.push({ text: params.text, scope: params.scope, at: now(), night: ctx.sessionId });
      return text("lesson recorded");
    },
  });

  const done = tool({
    name: "done",
    label: "Done",
    description: "End the search. Refused while time for 2 or more trials remains before the search deadline: that GPU time is the search's, and validation has its own reserve after it.",
    parameters: Type.Object({ reason: Type.String() }),
    execute: async (params) => {
      const left = ctx.trialsLeft();
      if (!ctx.allowEarlyDone && left !== null && left >= 2) {
        ctx.onDoneRefused(params.reason);
        throw new Error(
          `refused: about ${left} more trial(s) fit before the search deadline, and the finalists' validation has its own time after it. Spend them on what makes the verdict trustworthy: re-measure the finalists, ablate the leading stack one change at a time to find which parts carry the gain, test combinations you have not measured, or try a lever you have not touched. The cards are not left idle: when you propose nothing, the harness re-measures the finalists in the free slots and asks again.`,
        );
      }
      lab.done = { reason: params.reason, at: now() };
      return { ...text("Acknowledged. Running trials finish, the finalists are confirmed, and the search ends."), terminate: true };
    },
  });

  const vocab = ctx.manifest.services?.vocabulary ? [engineVocab] : [];
  return [propose, check, ...vocab, inspectTrial, readLog, compare, remeasure, cancelTrial, openHypothesis, closeHypothesis, addConstraint, recordLesson, done];
}

// The lab record: rebuilt from the lab's books before every model request and
// sent as the newest message, so the model always reasons from the complete,
// current truth — every trial, every diagnosed failure, every constraint and
// lesson — however long the night has been. It is never compacted: GPU hours
// are what is scarce here, not tokens. Only the conversation around it is.

import type { Manifest } from "../contract/types.js";
import type { Hypothesis, LabState, Trial } from "../lab/state.js";
import { configText, diffText } from "../lab/space.js";
import { type VocabEntry, vocabLine } from "../lab/vocab.js";
import {
  type Catalogue,
  catalogueOf,
  fmt,
  glossary,
  headlineKeys,
  LEVEL_KEYS,
  unhealthyPoints,
  valuesText,
} from "../lab/measures.js";
import {
  byConfig,
  type ConfigStats,
  deltaText,
  type Direction,
  noiseBand,
  ranking,
  trialFeasible,
  trialObjective,
  anchorValue as anchorMean,
} from "../lab/stats.js";

export interface RecordContext {
  manifest: Manifest;
  lab: LabState;
  direction: Direction;
  targetMetric: string;
  forbidden: string[];
  totalCards: number;
  freeCards: number;
  running: { id: number; cards: number }[];
  inFlight: number;
  sessionId: number;
  nowMs?: number;
  tokensUsed?: number;
  tokenLimit?: number;
  /** The image's high and mid recommendations, read once at the start (1.5). */
  levers?: VocabEntry[];
  /** "ok", "partial", "none", or undefined when the platform has no vocabulary service. */
  vocabularyStatus?: string;
  /** How many flags and env vars the image reads. */
  vocabularyCounts?: { flags: number; env: number; engineVersion: string };
  /** The model's facts from its config.json (manifest, or a later vocabulary read). */
  modelFacts?: Record<string, unknown>;
}

/** "MoE, 512 experts (10 active); built-in draft layers: 1; …" — the model's
 *  facts in the words a tuner uses. Empty when none were read. */
export function factsText(facts: Record<string, unknown> | undefined): string {
  const f = facts ?? {};
  if (!Object.keys(f).length) return "";
  const parts: string[] = [];
  if (Array.isArray(f.architectures)) parts.push(String(f.architectures.join("/")));
  if (f.experts) parts.push(`MoE, ${f.experts} experts${f.experts_per_token ? ` (${f.experts_per_token} active per token)` : ""}${f.shared_experts ? `, ${f.shared_experts} shared` : ""}`);
  else parts.push("dense");
  parts.push(`built-in draft layers (MTP/NEXTN): ${f.draft_layers ?? "none"}`);
  if (f.layer_types && typeof f.layer_types === "object") parts.push(`layers by kind: ${JSON.stringify(f.layer_types)}`);
  if (f.kv_lora_rank) parts.push(`MLA attention (kv_lora_rank ${f.kv_lora_rank})`);
  for (const [key, label] of [["layers", "layers"], ["kv_heads", "KV heads"], ["head_dim", "head dim"], ["max_context", "max context"], ["sliding_window", "sliding window"]] as const) {
    if (f[key] !== undefined) parts.push(`${label} ${f[key]}`);
  }
  if (f.dtype) parts.push(`dtype ${f.dtype}`);
  if (f.quantization) parts.push(`quantized: ${JSON.stringify(f.quantization)}`);
  if (f.multimodal) parts.push("multimodal");
  return parts.join("; ");
}

type DeclaredSpace = { grid?: Record<string, unknown[]>; tied?: Record<string, unknown[]>[] };

/** Declared grid/tied values that no trial tonight has set (failed ones count
 *  as tried): "tp: 4; attention_backend: fa3". */
/** The recipe every launch starts from (1.6): its chart, how many pods it
 *  makes, and the chart values beyond flags and env, which the config's
 *  `values.<path>` keys address. Empty when the platform sends none. */
export function recipeLines(recipe: Record<string, unknown> | undefined): string[] {
  const values = (recipe?.values ?? {}) as Record<string, unknown>;
  if (!Object.keys(values).length) return [];
  const chart = (recipe?.chart ?? {}) as { name?: string; version?: string };
  const pd = (values.pd ?? {}) as Record<string, unknown>;
  const lws = (values.lws ?? {}) as Record<string, unknown>;
  const topology = pd.enabled
    ? "pd: separate prefill and decode pods, each role with its own flags (`prefill.<flag>`, `decode.<flag>`)"
    : lws.enabled
      ? `lws: one engine over a group of ${lws.size ?? "?"} pods`
      : "one pod";
  const out = [`- Deployed as the ${chart.name ?? "engine"} chart${chart.version ? ` (${chart.version})` : ""}, ${topology}.`];
  const rest = Object.fromEntries(
    Object.entries(values).filter(([k]) => !["extraArgs", "env", "image", "model"].includes(k)),
  );
  if (Object.keys(rest).length) out.push(`- Its other chart values (set one as \`values.<dotted.path>\`): ${JSON.stringify(rest)}`);
  return out;
}

export function untriedDeclared(space: unknown, trials: Trial[]): string[] {
  const s = (space ?? {}) as DeclaredSpace;
  const values = new Map<string, unknown[]>();
  for (const [k, vs] of Object.entries(s.grid ?? {})) values.set(k, [...(values.get(k) ?? []), ...(vs ?? [])]);
  for (const group of s.tied ?? []) for (const [k, vs] of Object.entries(group)) values.set(k, [...(values.get(k) ?? []), ...(vs ?? [])]);
  const show = (v: unknown) => (typeof v === "string" ? v : JSON.stringify(v));
  const out: string[] = [];
  for (const [key, vs] of [...values.entries()].sort(([a], [b]) => a.localeCompare(b))) {
    const tried = new Set(trials.filter((t) => key in t.config).map((t) => show(t.config[key])));
    const left = [...new Set(vs.map(show))].filter((v) => !tried.has(v));
    if (left.length) out.push(`${key}: ${left.join(", ")}`);
  }
  return out;
}

function minutesBetween(a?: string, b?: string): number | null {
  if (!a || !b) return null;
  return (Date.parse(b) - Date.parse(a)) / 60000;
}

/** Measured minutes per trial (launch to release), or the menu's estimate. */
export function minutesPerTrial(lab: LabState, manifest: Manifest): number {
  const done = lab.trials
    .filter((t) => t.status === "measured")
    .map((t) => minutesBetween(t.startedAt, t.finishedAt))
    .filter((m): m is number => m !== null && m > 0);
  // No real trial takes under a minute (a model load alone is longer); the
  // floor keeps an instant fake or a clock skew from promising a huge budget.
  if (done.length) return Math.max(1, done.reduce((a, b) => a + b, 0) / done.length);
  const screen = manifest.services?.benchmarks?.screen?.approx_minutes ?? 15;
  return screen + 10; // + model load
}

export function budget(ctx: RecordContext): { minutesLeft: number | null; perTrial: number; trialsLeft: number | null; parallel: number } {
  const deadline = ctx.manifest.time.search_deadline ? Date.parse(ctx.manifest.time.search_deadline) : null;
  const nowMs = ctx.nowMs ?? Date.now();
  const minutesLeft = deadline === null ? null : Math.max(0, (deadline - nowMs) / 60000);
  const perTrial = minutesPerTrial(ctx.lab, ctx.manifest);
  const typicalCards = Math.max(1, Math.round(median(ctx.lab.trials.map((t) => t.cards)) ?? 1));
  const ports = ctx.manifest.hardware.ports?.length ?? 1;
  const parallel = Math.max(1, Math.min(ports, ctx.totalCards ? Math.floor(ctx.totalCards / typicalCards) : ports));
  const trialsLeft = minutesLeft === null ? null : Math.floor(minutesLeft / perTrial) * parallel;
  return { minutesLeft, perTrial, trialsLeft, parallel };
}

/** Cards in use and free, and what each running trial holds. */
export function cardsNow(total: number, free: number, running: { id: number; cards: number }[]): string {
  if (!total) return `${running.length} trial(s) running`;
  const held = running.length ? ` (${running.map((r) => `#${r.id}: ${r.cards}`).join(", ")})` : "";
  return `${total - free} of ${total} card(s) in use by ${running.length} trial(s)${held}, ${free} free`;
}

/** The night's clock: how long the search has in all, how much is left, and
 *  the validation reserve after it that the platform spends, not the agent. */
export function timeLines(
  time: { started_at?: string | null; search_deadline?: string | null; hard_deadline?: string | null },
  minutesLeft: number | null,
  nowMs: number,
): string[] {
  const at = (s?: string | null) => (s ? Date.parse(s) : null);
  const start = at(time.started_at), search = at(time.search_deadline), hard = at(time.hard_deadline);
  if (search === null) return ["- Search ends: no deadline"];
  const lines: string[] = [];
  if (start !== null) {
    const total = (search - start) / 60000;
    const used = Math.max(0, (nowMs - start) / 60000);
    lines.push(`- Search budget: ${fmt(total, 0)} min in all (${time.started_at} to ${time.search_deadline}); ${fmt(used, 0)} min used, ${fmt(minutesLeft ?? 0, 0)} min left (${fmt(total > 0 ? (100 * (minutesLeft ?? 0)) / total : 0, 0)}%)`);
  } else {
    lines.push(`- Search ends ${time.search_deadline} — ${fmt(minutesLeft ?? 0, 0)} min left`);
  }
  if (hard !== null && hard > search) {
    lines.push(`- After the search: ${fmt((hard - search) / 60000, 0)} min reserved for the platform to validate the finalists (until ${time.hard_deadline}). It is not yours to spend and needs nothing from you; \`done\` is refused while time for 2+ trials of the search remains.`);
  }
  return lines;
}

function median(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : null;
}

/** The latest measurement's numbers, module by module: the operating point's
 *  headline measures, then each other level of a sweep. */
function resultLines(t: Trial, catalogue: Catalogue): string[] {
  const m = t.measurements.at(-1);
  if (!m) return [];
  const out: string[] = [];
  for (const s of m.scenarios) {
    const at = s.operatingLevel ? ` at ${s.operatingLevel}` : "";
    out.push(`${s.scenario}${at}: ${valuesText(s.operating, headlineKeys(s.operating, catalogue))}`);
    const levels = Object.entries(s.points)
      .filter(([name]) => /^c\d+$/.test(name))
      .sort(([a], [b]) => Number(a.slice(1)) - Number(b.slice(1)));
    if (levels.length) out.push(`${s.scenario} levels: ${levels.map(([name, v]) => `${name}: ${valuesText(v, LEVEL_KEYS)}`).join(" | ")}`);
  }
  return out;
}

/** Where requests did not succeed in the latest measurement. */
export function unhealthyLevels(t: Trial): string[] {
  const m = t.measurements.at(-1);
  return m ? unhealthyPoints(m.scenarios) : [];
}

function trialRow(t: Trial, anchorConfig: Trial["config"] | undefined, anchorValue: number | null, noise: number | null, direction: Direction, catalogue: Catalogue): string {
  const value = trialObjective(t);
  const delta = value !== null && anchorValue !== null && t.purpose !== "anchor" ? deltaText(value, anchorValue, noise, direction) : "";
  const parts = [
    `#${t.id}`,
    t.purpose + (t.remeasureOf ? ` of #${t.remeasureOf}` : ""),
    t.status,
    t.purpose === "anchor" ? configText(t.config) : diffText(t.config, anchorConfig),
  ];
  if (t.status === "measured") {
    parts.push(`objective ${fmt(value)}${t.measurements.length > 1 ? ` (mean of ${t.measurements.length}: ${t.measurements.map((m) => fmt(m.objective)).join(", ")})` : ""}`);
    if (delta) parts.push(`vs anchor ${delta}`);
    if (!trialFeasible(t)) parts.push(`INFEASIBLE: ${t.measurements.flatMap((m) => m.breaches).join("; ") || "redline crossed"}`);
    const unhealthy = unhealthyLevels(t);
    if (unhealthy.length) parts.push(`REQUESTS FAILED at ${unhealthy.join("; ")}`);
    parts.push(...resultLines(t, catalogue));
  }
  if (t.engineWarnings?.length) parts.push(`engine warnings: ${t.engineWarnings.slice(0, 5).join(" / ")}${t.engineWarnings.length > 5 ? ` (+${t.engineWarnings.length - 5} more, inspect_trial)` : ""}`);
  if (t.failure) {
    const d = t.failure.diagnosis;
    parts.push(
      `FAILED at ${t.failure.stage} (${t.failure.failureClass})` +
        (d ? `: ${d.cause}${d.offending_flag ? ` [flag: ${d.offending_flag}]` : ""}${d.fix_hint ? ` — hint: ${d.fix_hint}` : ""}` : t.failure.error ? `: ${t.failure.error.slice(0, 200)}` : ""),
    );
  }
  if (t.hypothesisId) parts.push(`tests ${t.hypothesisId}`);
  if (t.prediction && t.status === "measured" && value !== null) parts.push(predictionCheck(t, value, anchorValue));
  for (const note of t.notes) parts.push(`note: ${note}`);
  return `- ${parts.join(" · ")}`;
}

function predictionCheck(t: Trial, value: number, anchorValue: number | null): string {
  const p = t.prediction!;
  if (anchorValue === null || anchorValue === 0 || p.direction === "unknown") return `predicted ${p.direction}`;
  const pct = ((value - anchorValue) / Math.abs(anchorValue)) * 100;
  const actual = Math.abs(pct) < 1 ? "same" : pct > 0 ? "up" : "down";
  const hit = actual === p.direction ? "prediction held" : "PREDICTION MISSED";
  return `predicted ${p.direction}${p.magnitude_pct !== undefined ? ` ~${p.magnitude_pct}%` : ""}, got ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% — ${hit}`;
}

function hypothesisLine(h: Hypothesis): string {
  const trials = h.trialIds.length ? ` (trials ${h.trialIds.map((i) => `#${i}`).join(", ")})` : "";
  return `- ${h.id} [${h.status}] ${h.statement}${trials}${h.evidence ? ` — ${h.evidence}` : ""}`;
}

function contenderLines(top: ConfigStats[], anchorConfig: Trial["config"] | undefined): string[] {
  return top.map(
    (g, i) => `${i + 1}. ${diffText(g.config, anchorConfig)} — mean ${fmt(g.mean)} over ${g.values.length} measurement(s), trials ${g.trialIds.map((x) => `#${x}`).join(", ")}`,
  );
}

export function renderRecord(ctx: RecordContext): string {
  const { manifest, lab, direction } = ctx;
  const anchor = lab.anchor;
  const anchorValue = anchor ? anchorMean(lab.trials) : null;
  const stats = byConfig(lab.trials);
  const noise = noiseBand(stats);
  const top = ranking(stats, direction).slice(0, manifest.contenders?.max ?? 2);
  const b = budget(ctx);
  const hw = manifest.hardware;
  const screen = manifest.services?.benchmarks?.screen;
  const catalogue = catalogueOf(manifest);
  const out: string[] = [];

  out.push("<lab_record>");
  out.push("This record is rebuilt by the harness before every message from its own books. It is the source of truth; when it disagrees with anything earlier in the conversation, the record is right.");

  out.push("\n## Tonight");
  out.push(`- Model: ${manifest.model.served_model_name} on ${manifest.model.engine}, image ${manifest.model.image}`);
  const facts = factsText(ctx.modelFacts ?? (manifest.model as { facts?: Record<string, unknown> }).facts);
  out.push(facts ? `- What the model is (its config.json): ${facts}` : "- The model's config has not been read yet.");
  out.push(`- Hardware: ${hw.machine} — ${hw.gpu_type || "unknown GPU"}, ${ctx.totalCards} card(s)${hw.pool ? ` (a pool of ${hw.node_count} node(s), at most ${hw.node_gpu_max || "?"} cards per pod)` : ""}, ${hw.ports?.length ?? 0} port(s)`);
  out.push(...recipeLines((manifest.model as { recipe?: Record<string, unknown> }).recipe));
  out.push(...timeLines(manifest.time, b.minutesLeft, ctx.nowMs ?? Date.now()));
  out.push(`- One trial takes ~${fmt(b.perTrial, 0)} min (launch to release); up to ${b.parallel} at once → room for about ${b.trialsLeft ?? "?"} more trial(s)`);
  out.push(`- Now: ${cardsNow(ctx.totalCards, ctx.freeCards, ctx.running)}, ${lab.trials.filter((t) => t.status === "queued").length} queued. Trials may use different card counts side by side; a queued one starts when enough cards are free.`);
  if (ctx.tokenLimit) out.push(`- Model tokens used: ${ctx.tokensUsed ?? 0} of ${ctx.tokenLimit}`);

  out.push("\n## Objective");
  out.push(`- ${direction === "max" ? "Maximize" : "Minimize"} \`${ctx.targetMetric}\` on the screen benchmark${screen ? ` (${screen.slug}, ~${screen.approx_minutes} min each)` : ""}, subject to its redlines. A config that crosses a redline is infeasible however fast it is.`);
  const redlines = manifest.objective?.screen?.redlines ?? [];
  if (redlines.length) out.push(`- Redlines: ${JSON.stringify(redlines)}`);
  out.push(`- The verdict is the platform's own full-replay measurement of the finalists after the search. Screen numbers choose the finalists.`);

  out.push("\n## What the screen benchmark measures");
  const workload = (screen?.workload ?? []) as { scenario?: string; family?: string; params?: Record<string, unknown>; reads_as?: string }[];
  if (workload.length) {
    for (const w of workload) out.push(`- \`${w.scenario}\` (${w.family}): ${w.reads_as ?? "no description"}${w.params && Object.keys(w.params).length ? ` Params: ${JSON.stringify(w.params)}` : ""}`);
  } else {
    out.push("- The platform did not say what the screen sends; read it from the results.");
  }
  out.push("- Results are in the platform's measures: `name@stat[qualifier]` (`ttft_ms@p99`, `ttft_ms@p90[32k_64k]`, a prompt-length bucket). A measure means the same in every module. A missing one was not reported, not zero. Each ledger row shows the headline measures; `inspect_trial` has every one.");
  const shown = new Set<string>();
  for (const t of lab.trials) for (const s of t.measurements.at(-1)?.scenarios ?? []) {
    for (const k of headlineKeys(s.operating, catalogue)) shown.add(k);
    if (Object.keys(s.points).some((p) => /^c\d+$/.test(p))) for (const k of LEVEL_KEYS) shown.add(k);
  }
  const target = ctx.targetMetric.replace(/^[^/]*\//, "").replace(/:.*$/, "");
  if (catalogue.size) {
    const lines = glossary([target, ...shown], catalogue);
    if (lines.length) out.push(...lines);
  }

  out.push("\n## Rules the platform enforces");
  out.push(`- Forbidden parameters (keep the campaign's value; never change): ${ctx.forbidden.length ? ctx.forbidden.join(", ") : "none"}`);
  out.push("- Every value must be a string, number or boolean that survives the engine command line and back (no lists/objects; JSON flags as one JSON string). Never set placement (model_path, host, port, …).");
  if (manifest.services?.env_keys) out.push("- An environment variable is set as an `env.NAME` key (`\"env.SGLANG_FOO\": true`). A name the image does not read is refused: look it up with `engine_vocab` first.");
  if (manifest.services?.recipes) out.push("- Besides flags and `env.NAME`, a key may set a pd role's flag (`prefill.mem_fraction_static`, `decode.max_running_requests`) or a chart value (`values.shm.sizeLimit`); `null` removes one. Scheduling, routing, scaling, names, the image and how many pods each role runs are the platform's, and refused. `check` shows the pods and each role's command the platform builds.");
  out.push("- Use `check` before proposing anything unusual; it costs nothing.");

  if (ctx.vocabularyStatus !== undefined) {
    out.push("\n## What this engine image accepts");
    const levers = ctx.levers ?? [];
    const counts = ctx.vocabularyCounts;
    if (ctx.vocabularyStatus === "none") {
      out.push("- The platform has not read this image's flags and environment variables yet; rely on the flags you know.");
    } else {
      out.push(`- ${counts ? `${counts.flags} flag(s) and ${counts.env} environment variable(s)` : "Its flags and environment variables"}, read from the image${counts?.engineVersion ? ` (${manifest.model.engine} ${counts.engineVersion})` : ""}: types, defaults, allowed values and the engine's own description of each. Look up any you have not used tonight with \`engine_vocab\` before proposing it; a flag the image does not accept, or an env var it does not read, is refused.`);
    }
    if (levers.length) {
      out.push("Recommended by the platform for this engine image and hardware (`high` first). Try each `high` one early, alone against the anchor, where its `when` fits; `mid` ones when the evidence points at them. They are ideas, not results: measure them. Everything else the image reads: `engine_vocab`.");
      for (const e of levers.filter((x) => x.level === "high")) out.push(vocabLine(e));
      for (const e of levers.filter((x) => x.level === "mid")) out.push(vocabLine({ ...e, help: "" }));
    }
  }

  out.push("\n## Production (the anchor)");
  const production = (manifest.production as { engine_args?: Record<string, unknown> } | undefined)?.engine_args;
  out.push(`- Production config: ${production && Object.keys(production).length ? configText(production) : "(none recorded; the anchor is the campaign's base config)"}`);
  out.push(`- Anchor measurement: ${anchor ? (anchorValue !== null ? fmt(anchorValue) : anchor.status) : "not started"}`);
  const declared = manifest.search_space ?? {};
  if (Object.keys(declared).length) {
    out.push(`- The campaign's declared space (a hint, not a limit): ${JSON.stringify(declared)}`);
    const untried = untriedDeclared(declared, lab.trials);
    if (untried.length) out.push(`- Declared values no trial has tried yet: ${untried.join("; ")}`);
  }

  out.push(`\n## Trials tonight (${lab.trials.length})`);
  if (noise !== null) out.push(`Measurement noise so far: ±${(noise * 100).toFixed(1)}% (half the spread of repeated measurements of the same config).`);
  else out.push("Measurement noise: unknown until a config has been measured twice.");
  if (!lab.trials.length) out.push("- none yet");
  for (const t of lab.trials) out.push(trialRow(t, anchor?.config, anchorValue, noise, direction, catalogue));

  out.push("\n## Current finalists (chosen by the harness: best feasible mean)");
  out.push(top.length ? contenderLines(top, anchor?.config).join("\n") : "- none yet");

  out.push("\n## Known constraints (from failures)");
  out.push(lab.constraints.length ? lab.constraints.map((c) => `- ${c.text}${c.evidence.length ? ` (evidence: ${c.evidence.map((e) => `#${e}`).join(", ")})` : ""}${c.night && c.night !== ctx.sessionId ? " [earlier night]" : ""}`).join("\n") : "- none recorded");

  out.push("\n## Hypotheses");
  out.push(lab.hypotheses.length ? lab.hypotheses.map(hypothesisLine).join("\n") : "- none yet");

  out.push("\n## Lessons");
  const lessons = lab.lessons;
  out.push(lessons.length ? lessons.map((l) => `- [${l.scope}${l.night && l.night !== ctx.sessionId ? ", earlier night" : ""}] ${l.text}`).join("\n") : "- none yet");

  if (lab.priorNights.length) {
    out.push("\n## Earlier nights of this campaign");
    for (const n of lab.priorNights) {
      out.push(`- Night of ${n.startedAt} (session ${n.sessionId}, ended: ${n.endedReason || "?"}): ${n.trials.length} trial(s); anchor ${fmt(n.anchor)}; best ${n.best ? `${fmt(n.best.objective)} with ${configText(n.best.config)}` : "none"}`);
      for (const t of n.trials) {
        out.push(`  - ${configText(t.config)} → ${t.failure ? `FAILED (${t.failure})` : `${fmt(t.objective)}${t.feasible ? "" : " INFEASIBLE"}`}`);
      }
    }
  }
  out.push("</lab_record>");
  return out.join("\n");
}

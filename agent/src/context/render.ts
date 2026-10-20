// The lab record: rebuilt from the lab's books before every model request and
// sent as the newest message, so the model always reasons from the complete,
// current truth — every trial, every diagnosed failure, every constraint and
// lesson — however long the night has been. It is never compacted: GPU hours
// are what is scarce here, not tokens. Only the conversation around it is.

import type { Manifest } from "../contract/types.js";
import type { Hypothesis, LabState, Trial } from "../lab/state.js";
import { configText, diffText } from "../lab/space.js";
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
  freeSlots: number;
  inFlight: number;
  sessionId: number;
  nowMs?: number;
  tokensUsed?: number;
  tokenLimit?: number;
}

function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e5) return value.toExponential(3);
  if (abs >= 100) return value.toFixed(0);
  return value.toFixed(digits);
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

function median(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)] : null;
}

function levelLine(t: Trial): string {
  const m = t.measurements.at(-1);
  if (!m || !m.levels.length) return "";
  // uptime and error_rate are the direct signals that a level served at all;
  // a latency of 0 or a tiny throughput is only a symptom of what they say.
  const keys = ["output_tps", "output_tps_mean", "ttft_p99_ms", "itl_p99_ms", "uptime", "measured_requests", "error_rate", "meets_slo"];
  return m.levels
    .map((l) => `c${l.concurrency}: ${keys.filter((k) => k in l.metrics).map((k) => `${k.replace("_ms", "")} ${fmt(l.metrics[k])}`).join(", ")}`)
    .join(" | ");
}

/** Levels where requests did not succeed: uptime below 1 or any errors. */
export function unhealthyLevels(t: Trial): string[] {
  const m = t.measurements.at(-1);
  if (!m) return [];
  return m.levels
    .filter((l) =>
      (l.metrics.uptime !== undefined && l.metrics.uptime < 1) ||
      (l.metrics.error_rate ?? 0) > 0 ||
      l.metrics.measured_requests === 0)
    .map((l) => `c${l.concurrency}${l.metrics.uptime !== undefined ? ` uptime ${fmt(l.metrics.uptime)}` : ""}${l.metrics.measured_requests === 0 ? " 0 requests completed" : ""}${l.metrics.error_rate ? ` errors ${fmt(l.metrics.error_rate)}` : ""}`);
}

function trialRow(t: Trial, anchorConfig: Trial["config"] | undefined, anchorValue: number | null, noise: number | null, direction: Direction): string {
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
    if (unhealthy.length) parts.push(`REQUESTS FAILED at ${unhealthy.join(", ")}`);
    const levels = levelLine(t);
    if (levels) parts.push(levels);
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
  const out: string[] = [];

  out.push("<lab_record>");
  out.push("This record is rebuilt by the harness before every message from its own books. It is the source of truth; when it disagrees with anything earlier in the conversation, the record is right.");

  out.push("\n## Tonight");
  out.push(`- Model: ${manifest.model.served_model_name} on ${manifest.model.engine}, image ${manifest.model.image}`);
  out.push(`- Hardware: ${hw.machine} — ${hw.gpu_type || "unknown GPU"}, ${ctx.totalCards} card(s)${hw.pool ? ` (a pool of ${hw.node_count} node(s), at most ${hw.node_gpu_max || "?"} cards per launch)` : ""}, ${hw.ports?.length ?? 0} port(s)`);
  out.push(`- Search ends ${manifest.time.search_deadline ?? "(no deadline)"}${b.minutesLeft !== null ? ` — ${fmt(b.minutesLeft, 0)} min left` : ""}`);
  out.push(`- One trial takes ~${fmt(b.perTrial, 0)} min (launch to release); up to ${b.parallel} at once → room for about ${b.trialsLeft ?? "?"} more trial(s)`);
  out.push(`- Now: ${ctx.inFlight} trial(s) running, ${lab.trials.filter((t) => t.status === "queued").length} queued, ${ctx.freeSlots} slot(s) free`);
  if (ctx.tokenLimit) out.push(`- Model tokens used: ${ctx.tokensUsed ?? 0} of ${ctx.tokenLimit}`);

  out.push("\n## Objective");
  out.push(`- ${direction === "max" ? "Maximize" : "Minimize"} \`${ctx.targetMetric}\` on the screen benchmark${screen ? ` (${screen.slug}, ~${screen.approx_minutes} min each)` : ""}, subject to its redlines. A config that crosses a redline is infeasible however fast it is.`);
  const redlines = manifest.objective?.screen?.redlines ?? [];
  if (redlines.length) out.push(`- Redlines: ${JSON.stringify(redlines)}`);
  out.push(`- The verdict is the platform's own full-replay measurement of the finalists after the search. Screen numbers choose the finalists.`);

  out.push("\n## Rules the platform enforces");
  out.push(`- Forbidden parameters (keep the campaign's value; never change): ${ctx.forbidden.length ? ctx.forbidden.join(", ") : "none"}`);
  out.push("- Every value must be a string, number or boolean that survives the engine command line and back (no lists/objects; JSON flags as one JSON string). Never set placement (model_path, host, port, …).");
  out.push("- Use `check` before proposing anything unusual; it costs nothing.");

  out.push("\n## Production (the anchor)");
  const production = (manifest.production as { engine_args?: Record<string, unknown> } | undefined)?.engine_args;
  out.push(`- Production config: ${production && Object.keys(production).length ? configText(production) : "(none recorded; the anchor is the campaign's base config)"}`);
  out.push(`- Anchor measurement: ${anchor ? (anchorValue !== null ? fmt(anchorValue) : anchor.status) : "not started"}`);
  const declared = manifest.search_space ?? {};
  if (Object.keys(declared).length) out.push(`- The campaign's declared space (a hint, not a limit): ${JSON.stringify(declared)}`);

  out.push(`\n## Trials tonight (${lab.trials.length})`);
  if (noise !== null) out.push(`Measurement noise so far: ±${(noise * 100).toFixed(1)}% (half the spread of repeated measurements of the same config).`);
  else out.push("Measurement noise: unknown until a config has been measured twice.");
  if (!lab.trials.length) out.push("- none yet");
  for (const t of lab.trials) out.push(trialRow(t, anchor?.config, anchorValue, noise, direction));

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

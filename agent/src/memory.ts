// What one night leaves the next: the lessons, the constraints learned from
// failures, closed hypotheses, and each night's ledger in brief. Stored with
// PUT /state (per campaign, so night 2 reads what night 1 wrote) and written
// after every trial, so a crash loses nothing that was learned.

import type { Manifest } from "./contract/types.js";
import type { Constraint, Hypothesis, LabState, Lesson, PriorNight } from "./lab/state.js";
import { byConfig, type Direction, ranking, trialFeasible, trialObjective } from "./lab/stats.js";

export const STATE_FORMAT = "llm-autotune-agent/1";
const MAX_NIGHTS = 12;
const MAX_TRIALS_PER_NIGHT = 150;
const MAX_NOTES = 200;

export interface Memory {
  format: string;
  nights: PriorNight[];
  lessons: Lesson[];
  constraints: Constraint[];
  hypotheses: Hypothesis[];
}

export function emptyMemory(): Memory {
  return { format: STATE_FORMAT, nights: [], lessons: [], constraints: [], hypotheses: [] };
}

/** Our own blob, or an empty memory for anything else (night one, or a blob
 *  another policy wrote for this campaign). */
export function loadMemory(blob: unknown): Memory {
  if (!blob || typeof blob !== "object" || (blob as Memory).format !== STATE_FORMAT) return emptyMemory();
  const m = blob as Memory;
  return {
    format: STATE_FORMAT,
    nights: Array.isArray(m.nights) ? m.nights : [],
    lessons: Array.isArray(m.lessons) ? m.lessons : [],
    constraints: Array.isArray(m.constraints) ? m.constraints : [],
    hypotheses: Array.isArray(m.hypotheses) ? m.hypotheses : [],
  };
}

export function applyMemory(lab: LabState, memory: Memory): void {
  lab.priorNights = memory.nights;
  lab.lessons = [...memory.lessons];
  lab.constraints = [...memory.constraints];
  // Closed hypotheses stay as knowledge; open ones from a dead night do not.
  lab.hypotheses = memory.hypotheses.filter((h) => h.status !== "open").map((h) => ({ ...h, id: `prior-${h.id}` }));
}

export function tonight(lab: LabState, manifest: Manifest, endedReason: string, direction: Direction): PriorNight {
  const top = ranking(byConfig(lab.trials), direction)[0];
  const anchor = lab.anchor ? trialObjective(lab.anchor) : null;
  const measured = lab.trials.filter((t) => t.status === "measured" || t.status === "failed");
  return {
    sessionId: manifest.session_id,
    startedAt: manifest.time.started_at ?? new Date().toISOString(),
    endedReason,
    best: top ? { config: top.config, objective: top.mean, n: top.values.length } : null,
    anchor,
    trials: measured.slice(-MAX_TRIALS_PER_NIGHT).map((t) => ({
      config: t.config,
      objective: trialObjective(t),
      feasible: trialFeasible(t),
      failure: t.failure
        ? `${t.failure.failureClass}${t.failure.diagnosis ? `: ${t.failure.diagnosis.cause}` : ""}`
        : undefined,
      purpose: t.purpose,
    })),
    summary: lab.done?.reason ?? "",
  };
}

export function snapshot(lab: LabState, manifest: Manifest, endedReason: string, direction: Direction): Memory {
  const night = tonight(lab, manifest, endedReason, direction);
  const nights = [...lab.priorNights.filter((n) => n.sessionId !== night.sessionId), night].slice(-MAX_NIGHTS);
  return {
    format: STATE_FORMAT,
    nights,
    lessons: lab.lessons.slice(-MAX_NOTES),
    constraints: lab.constraints.slice(-MAX_NOTES),
    hypotheses: lab.hypotheses
      .filter((h) => h.status !== "open")
      .map((h) => ({ ...h, id: h.id.replace(/^prior-/, "") }))
      .slice(-MAX_NOTES),
  };
}

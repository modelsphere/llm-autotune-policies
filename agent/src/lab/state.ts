// The lab's books: every trial of the night, the hypotheses, the constraints
// learned from failures, the lessons. The single source of truth the context
// is rebuilt from before every model request, and what is carried to the next
// night (memory.ts). The model never edits it directly — only through tools.

import type { Config } from "../contract/types.js";

export type TrialStatus =
  | "queued"
  | "launching"
  | "serving"
  | "benching"
  | "measured"
  | "failed"
  | "cancelled";

export type Purpose = "anchor" | "explore" | "remeasure" | "confirm";

export interface Prediction {
  direction: "up" | "down" | "same" | "unknown";
  /** Expected change of the objective vs the reference, in percent. */
  magnitude_pct?: number;
  /** The trial the prediction is relative to; default: the anchor. */
  versus?: number;
  note?: string;
}

/** One concurrency level of a sweep, reduced to the metrics worth reading. */
export interface Level {
  concurrency: number;
  metrics: Record<string, number>;
}

export interface Measurement {
  benchId: number;
  objective: number | null;
  feasible: boolean;
  breaches: string[];
  /** Headline numbers of the benchmark (throughput per card, TTFT, …). */
  headline: Record<string, number>;
  levels: Level[];
  bestLevel: Record<string, number>;
  at: string;
}

export interface Diagnosis {
  cause: string;
  offending_flag?: string;
  fix_hint?: string;
  source: "llm" | "heuristic";
}

export interface Failure {
  stage: "launch" | "bench" | "refused" | "deadline";
  failureClass: string;
  error: string;
  logExcerpt?: string;
  diagnosis?: Diagnosis;
}

export interface Trial {
  id: number;
  purpose: Purpose;
  /** Canonical config, as the platform echoed it from /configs/check. */
  config: Config;
  key: string;
  cards: number;
  hypothesisId?: string;
  prediction?: Prediction;
  rationale?: string;
  remeasureOf?: number;
  status: TrialStatus;
  launchId?: number;
  benchIds: number[];
  gpuIndices: number[];
  port?: number;
  queuedAt: string;
  startedAt?: string;
  servingAt?: string;
  finishedAt?: string;
  measurements: Measurement[];
  failure?: Failure;
  /** The engine's own account of its resolved arguments, when it offers one. */
  serverInfo?: Record<string, unknown>;
  environment?: Record<string, unknown>;
  /** Distinct warning lines the engine logged while serving this trial. */
  engineWarnings?: string[];
  /** Harness notes: "anomaly: re-benched before trusting", … */
  notes: string[];
}

export interface Hypothesis {
  id: string;
  statement: string;
  status: "open" | "supported" | "refuted" | "inconclusive";
  evidence?: string;
  trialIds: number[];
  openedAt: string;
  closedAt?: string;
}

export interface Constraint {
  text: string;
  evidence: number[];
  at: string;
  night?: number;
}

export interface Lesson {
  text: string;
  scope: "campaign" | "model" | "engine" | "hardware";
  at: string;
  night?: number;
}

export interface PriorNight {
  sessionId: number;
  startedAt: string;
  endedReason: string;
  best: { config: Config; objective: number; n: number } | null;
  anchor: number | null;
  trials: {
    config: Config;
    objective: number | null;
    feasible: boolean;
    failure?: string;
    purpose: Purpose;
  }[];
  summary: string;
}

/** A stable identity for a canonical config: sorted keys, JSON values. */
export function configKey(config: Config): string {
  return JSON.stringify(Object.keys(config).sort().map((k) => [k, config[k]]));
}

export function now(): string {
  return new Date().toISOString();
}

export class LabState {
  trials: Trial[] = [];
  hypotheses: Hypothesis[] = [];
  constraints: Constraint[] = [];
  lessons: Lesson[] = [];
  priorNights: PriorNight[] = [];
  /** Parameters introduced outside the declared space, with the first reason. */
  introduced = new Map<string, string>();
  done: { reason: string; at: string } | null = null;
  private nextId = 1;
  private nextHypothesis = 1;

  addTrial(fields: Omit<Trial, "id" | "status" | "benchIds" | "gpuIndices" | "queuedAt" | "measurements" | "notes" | "key">): Trial {
    const trial: Trial = {
      ...fields,
      id: this.nextId++,
      key: configKey(fields.config),
      status: "queued",
      benchIds: [],
      gpuIndices: [],
      queuedAt: now(),
      measurements: [],
      notes: [],
    };
    this.trials.push(trial);
    return trial;
  }

  trial(id: number): Trial | undefined {
    return this.trials.find((t) => t.id === id);
  }

  get anchor(): Trial | undefined {
    return this.trials.find((t) => t.purpose === "anchor" && t.status === "measured")
      ?? this.trials.find((t) => t.purpose === "anchor");
  }

  /** Trials still to run or running. */
  get pending(): Trial[] {
    return this.trials.filter((t) => ["queued", "launching", "serving", "benching"].includes(t.status));
  }

  /** Was this exact config already tried or queued tonight (excluding remeasures)? */
  seen(key: string): Trial | undefined {
    return this.trials.find((t) => t.key === key && t.status !== "cancelled");
  }

  openHypothesis(statement: string): Hypothesis {
    const h: Hypothesis = {
      id: `H${this.nextHypothesis++}`,
      statement,
      status: "open",
      trialIds: [],
      openedAt: now(),
    };
    this.hypotheses.push(h);
    return h;
  }

  hypothesis(id: string): Hypothesis | undefined {
    return this.hypotheses.find((h) => h.id === id);
  }
}

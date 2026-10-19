// What a finished benchmark becomes in the ledger: the platform's objective
// and feasibility, plus the per-concurrency numbers worth reasoning about.

import type { BenchmarkOut, ResultsPart } from "../contract/types.js";
import { type Level, type Measurement, now } from "./state.js";

/** Metric names worth keeping per level; anything with "card_norm" in it too. */
const KEEP = [
  "output_tps", "total_tps", "input_tps", "request_output_tps", "requests_per_second",
  "ttft_p50_ms", "ttft_p90_ms", "ttft_p95_ms", "ttft_p99_ms",
  "itl_p50_ms", "itl_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
  "e2e_latency_p50_ms", "e2e_latency_p99_ms",
  "uptime", "reported_concurrency", "error_rate",
];

function numeric(metrics: Record<string, unknown> | undefined): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [key, value] of Object.entries(metrics ?? {})) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const short = key.includes(".") ? key.slice(key.lastIndexOf(".") + 1) : key;
    if (KEEP.includes(short) || short.includes("card_norm")) out[short] = value;
  }
  return out;
}

export function measurementOf(bench: BenchmarkOut, results: ResultsPart | null): Measurement {
  const summary = bench.summary;
  const levels: Level[] = [];
  let bestLevel: Record<string, number> = {};
  for (const scenario of results?.scenarios ?? []) {
    for (const level of scenario.levels ?? []) {
      levels.push({ concurrency: level.concurrency, metrics: numeric(level.metrics) });
    }
    if (scenario.best_level && Object.keys(scenario.best_level).length) {
      bestLevel = numeric(scenario.best_level as Record<string, unknown>);
      const c = (scenario.best_level as Record<string, unknown>).concurrency;
      if (typeof c === "number") bestLevel.concurrency = c;
    }
  }
  const headline = numeric({ ...(bench.metrics ?? {}), ...((results?.headline as Record<string, unknown>) ?? {}) });
  return {
    benchId: bench.id,
    objective: summary?.objective_value ?? null,
    feasible: Boolean(summary?.feasible),
    breaches: summary?.breaches ?? [],
    headline,
    levels,
    bestLevel,
    at: now(),
  };
}

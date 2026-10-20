// Measurement trust, in code rather than in the model's head: the same config
// measured more than once gives the noise; every delta the model sees carries
// it; contenders are picked here from repeated measurements, never by the model.

import type { Config } from "../contract/types.js";
import type { Trial } from "./state.js";

export type Direction = "max" | "min";

/** All measurements of one config across trials (a trial and its remeasures). */
export interface ConfigStats {
  key: string;
  config: Config;
  trialIds: number[];
  values: number[];
  feasible: boolean;
  mean: number;
  min: number;
  max: number;
}

export function trialObjective(trial: Trial): number | null {
  const values = trial.measurements.map((m) => m.objective).filter((v): v is number => v !== null);
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Production's value tonight: the mean over every measurement of the
 *  anchor's config (the anchor and its repeats), not the first one alone. */
export function anchorValue(trials: Trial[]): number | null {
  const anchor = trials.find((t) => t.purpose === "anchor");
  if (!anchor) return null;
  const group = byConfig(trials).find((g) => g.key === anchor.key);
  return group ? group.mean : trialObjective(anchor);
}

export function trialFeasible(trial: Trial): boolean {
  return trial.measurements.length > 0 && trial.measurements.every((m) => m.feasible);
}

/** Group measured trials by config. */
export function byConfig(trials: Trial[]): ConfigStats[] {
  const groups = new Map<string, ConfigStats>();
  for (const t of trials) {
    if (t.status !== "measured") continue;
    for (const m of t.measurements) {
      if (m.objective === null) continue;
      let g = groups.get(t.key);
      if (!g) {
        g = { key: t.key, config: t.config, trialIds: [], values: [], feasible: true, mean: 0, min: Infinity, max: -Infinity };
        groups.set(t.key, g);
      }
      if (!g.trialIds.includes(t.id)) g.trialIds.push(t.id);
      g.values.push(m.objective);
      g.feasible &&= m.feasible;
    }
  }
  for (const g of groups.values()) {
    g.mean = g.values.reduce((a, b) => a + b, 0) / g.values.length;
    g.min = Math.min(...g.values);
    g.max = Math.max(...g.values);
  }
  return [...groups.values()];
}

/** Relative noise (half the spread over the mean) of repeated measurements,
 *  the median over configs measured more than once; null until there is one. */
export function noiseBand(stats: ConfigStats[]): number | null {
  const spreads = stats
    .filter((g) => g.values.length >= 2 && g.mean !== 0)
    .map((g) => (g.max - g.min) / 2 / Math.abs(g.mean))
    .sort((a, b) => a - b);
  if (!spreads.length) return null;
  return spreads[Math.floor(spreads.length / 2)];
}

function better(a: number, b: number, direction: Direction): boolean {
  return direction === "max" ? a > b : a < b;
}

/** Relative standard deviation of one measurement, pooled over every config
 *  measured more than once (production first of all: the night measures it
 *  twice); `prior` until there is a repeat. */
export function relativeSd(stats: ConfigStats[], prior = 0.01): number {
  let sum = 0;
  let dof = 0;
  for (const g of stats) {
    if (g.values.length < 2 || g.mean === 0) continue;
    for (const v of g.values) sum += ((v - g.mean) / g.mean) ** 2;
    dof += g.values.length - 1;
  }
  return dof ? Math.sqrt(sum / dof) : prior;
}

/** A config's value as it can be trusted: its mean, moved one standard error
 *  toward worse. One lucky measurement no longer outranks a config measured
 *  twice at nearly the same mean; measuring a config again can only make its
 *  score more honest, never pull a good config below a lucky one. */
export function conservative(g: ConfigStats, sd: number, direction: Direction): number {
  const se = (sd * Math.abs(g.mean)) / Math.sqrt(g.values.length);
  return direction === "max" ? g.mean - se : g.mean + se;
}

/** Feasible configs, best first by their conservative value. Ties go to
 *  more measurements, then the earlier trial. */
export function ranking(stats: ConfigStats[], direction: Direction, prior = 0.01): ConfigStats[] {
  const sd = relativeSd(stats, prior);
  const score = new Map(stats.map((g) => [g.key, conservative(g, sd, direction)]));
  return stats
    .filter((g) => g.feasible)
    .sort((a, b) => {
      const sa = score.get(a.key)!;
      const sb = score.get(b.key)!;
      if (sa !== sb) return better(sa, sb, direction) ? -1 : 1;
      if (a.values.length !== b.values.length) return b.values.length - a.values.length;
      return Math.min(...a.trialIds) - Math.min(...b.trialIds);
    });
}

/** "+3.1% (within noise ±4%)" style text for a delta of `value` vs `reference`. */
export function deltaText(value: number, reference: number, noise: number | null, direction: Direction): string {
  if (reference === 0) return "n/a";
  const pct = ((value - reference) / Math.abs(reference)) * 100;
  const sign = pct >= 0 ? "+" : "";
  let text = `${sign}${pct.toFixed(1)}%`;
  if (noise !== null) {
    const band = noise * 100;
    const significant = Math.abs(pct) > 2 * band;
    text += significant ? "" : ` (within noise ±${band.toFixed(1)}%)`;
  }
  if (direction === "min") text += " (lower is better)";
  return text;
}

/** Would this result be suspiciously good — worth re-benching before trusting? */
export function isAnomalous(value: number, bestSoFar: number | null, threshold: number, direction: Direction): boolean {
  if (bestSoFar === null || bestSoFar === 0) return false;
  const gain = direction === "max" ? (value - bestSoFar) / Math.abs(bestSoFar) : (bestSoFar - value) / Math.abs(bestSoFar);
  return gain > threshold;
}

export function bestValue(stats: ConfigStats[], direction: Direction): number | null {
  const top = ranking(stats, direction)[0];
  return top ? top.mean : null;
}

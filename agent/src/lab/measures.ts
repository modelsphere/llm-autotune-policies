// Benchmark results in the platform's measures (its docs/api/measures.md):
// one vocabulary whichever benchmark module reported a number, with each
// measure's unit, direction and meaning in the manifest's catalogue. The
// harness reads nothing else, so a replay, a sweep and a module it has never
// seen are rendered and compared the same way.
//
// A key is `<measure>[@<stat>][[<qualifier>]]`: `output_tpm_per_gpu`,
// `ttft_ms@p99`, `ttft_ms@p90[32k_64k]`. A sweep's other levels are points
// (`c16`, `peak`) beside its operating point.

import type { BenchmarkOut, Manifest } from "../contract/types.js";

/** One module of a benchmark, as measures. */
export interface ScenarioResult {
  scenario: string;
  module: string;
  kind: string;
  /** The sweep level the operating point is (`c16`); null for anything else. */
  operatingLevel: string | null;
  operatingRule: string;
  operating: Record<string, number>;
  points: Record<string, Record<string, number>>;
}

export interface MeasureInfo {
  name: string;
  label: string;
  unit: string;
  better: "higher" | "lower" | null;
  group: string;
  meaning: string;
}

export type Catalogue = Map<string, MeasureInfo>;

export function catalogueOf(manifest: Manifest): Catalogue {
  const out: Catalogue = new Map();
  for (const raw of (manifest.measures ?? []) as Record<string, unknown>[]) {
    const name = String(raw.name ?? "");
    if (!name) continue;
    const better = raw.better === "higher" || raw.better === "lower" ? raw.better : null;
    out.set(name, {
      name,
      label: String(raw.label ?? name),
      unit: String(raw.unit ?? ""),
      better,
      group: String(raw.group ?? ""),
      meaning: String(raw.meaning ?? ""),
    });
  }
  return out;
}

export function scenariosOf(bench: BenchmarkOut): ScenarioResult[] {
  return (bench.measures ?? []).map((s) => ({
    scenario: s.scenario,
    module: s.module,
    kind: s.kind,
    operatingLevel: s.operating_level ?? null,
    operatingRule: s.operating_rule ?? "",
    operating: { ...(s.operating ?? {}) },
    points: Object.fromEntries(Object.entries(s.points ?? {}).map(([k, v]) => [k, { ...v }])),
  }));
}

/** `ttft_ms@p90[32k_64k]` → measure `ttft_ms`, stat `p90`, qualifier `32k_64k`. */
export function parseKey(key: string): { measure: string; stat: string; qualifier: string } {
  const match = /^([a-z0-9_]+)(?:@([a-z0-9]+))?(?:\[([^\]]+)\])?$/.exec(key);
  if (!match) return { measure: key, stat: "", qualifier: "" };
  return { measure: match[1], stat: match[2] ?? "", qualifier: match[3] ?? "" };
}

export function betterOf(key: string, catalogue: Catalogue): "higher" | "lower" | null {
  return catalogue.get(parseKey(key).measure)?.better ?? null;
}

/** The keys worth a line in the ledger, picked by the catalogue's groups so
 *  that any module's results get the same treatment: throughput per GPU,
 *  per-request speed (mean and the slow tail), latency (median and tail),
 *  every health rate that is not clean, the workload, quality and score.
 *  Everything else is one `inspect_trial` away. */
export function headlineKeys(operating: Record<string, number>, catalogue: Catalogue): string[] {
  const keys: string[] = [];
  for (const key of Object.keys(operating)) {
    const { measure, stat, qualifier } = parseKey(key);
    const info = catalogue.get(measure);
    if (!info || qualifier) continue;
    const value = operating[key];
    switch (info.group) {
      case "throughput":
        if (measure.endsWith("_per_gpu") || measure === "request_rate") keys.push(key);
        break;
      case "per request":
        if (stat === "mean" || stat === "p10") keys.push(key);
        break;
      case "latency":
        if (stat === "p50" || stat === "p99") keys.push(key);
        break;
      case "health":
        // Counts are for inspect_trial; a rate shows when it says something.
        if (info.better === "higher" || (info.better === "lower" && value > 0)) keys.push(key);
        break;
      case "workload":
        if (measure !== "window_s") keys.push(key);
        break;
      default:
        keys.push(key);
    }
  }
  return keys;
}

/** What a sweep's other levels are read by in the ledger. */
export const LEVEL_KEYS = ["output_tpm_per_gpu", "request_output_tps@mean", "ttft_ms@p99", "itl_ms@p99", "success_rate", "meets_slo"];

/** Where requests did not succeed, read from the health measures: a success
 *  rate below 1, any errors, or nothing completed. Empty when every point
 *  served. A missing measure is not read as a failure. */
export function unhealthyPoints(scenarios: ScenarioResult[]): string[] {
  const out: string[] = [];
  for (const s of scenarios) {
    const points: [string, Record<string, number>][] = [[s.operatingLevel ?? "operating point", s.operating]];
    for (const [name, values] of Object.entries(s.points)) if (/^c\d+$/.test(name) && name !== s.operatingLevel) points.push([name, values]);
    for (const [name, m] of points) {
      const signs: string[] = [];
      if (m.success_rate !== undefined && m.success_rate < 1) signs.push(`success_rate ${fmt(m.success_rate, 3)}`);
      if ((m.error_rate ?? 0) > 0) signs.push(`error_rate ${fmt(m.error_rate, 3)}`);
      if (m.requests_completed === 0) signs.push("0 requests completed");
      if (signs.length) out.push(`${s.scenario} ${name}: ${signs.join(", ")}`);
    }
  }
  return out;
}

export function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1e5) return value.toExponential(3);
  if (Number.isInteger(value)) return String(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs > 0 && abs < 0.01) return value.toPrecision(2);
  return value.toFixed(digits);
}

export function valuesText(values: Record<string, number>, keys: string[]): string {
  return keys.filter((k) => k in values).map((k) => `${k} ${fmt(values[k], Math.abs(values[k]) < 10 ? 3 : 1)}`).join(", ");
}

/** `b` against `a`, key by key, each marked better or worse by the
 *  catalogue's direction. Only keys both report; a zero base is skipped. */
export function deltas(a: Record<string, number>, b: Record<string, number>, catalogue: Catalogue): string[] {
  const out: string[] = [];
  for (const key of Object.keys(b)) {
    if (!(key in a) || a[key] === 0) continue;
    const pct = ((b[key] - a[key]) / Math.abs(a[key])) * 100;
    if (Math.abs(pct) < 0.05) continue;
    const better = betterOf(key, catalogue);
    const verdict = better === null ? "" : (pct > 0) === (better === "higher") ? " better" : " worse";
    out.push(`${key} ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%${verdict}`);
  }
  return out;
}

/** The catalogue entries behind a set of keys, one line each. */
export function glossary(keys: Iterable<string>, catalogue: Catalogue): string[] {
  const seen = new Set<string>();
  for (const key of keys) seen.add(parseKey(key).measure);
  return [...seen]
    .map((name) => catalogue.get(name))
    .filter((m): m is MeasureInfo => m !== undefined)
    .map((m) => `- \`${m.name}\` (${m.unit || "ratio"}${m.better ? `, ${m.better} is better` : ""}): ${m.meaning}`);
}

// Small helpers over configs and the declared search space — the same
// vocabulary the platform uses (docs/api/policy-contract.md, "Config format").

import type { Config } from "../contract/types.js";

type Space = {
  base?: Config;
  grid?: Record<string, unknown[]>;
  tied?: Record<string, unknown[]>[];
  range?: Record<string, unknown>;
  conditions?: Record<string, unknown>;
};

/** Parameters the declared space varies (grid, tied groups, ranges). */
export function sweptKeys(space: Record<string, unknown> | undefined): string[] {
  const s = (space ?? {}) as Space;
  const keys = new Set<string>(Object.keys(s.grid ?? {}));
  for (const group of s.tied ?? []) for (const k of Object.keys(group)) keys.add(k);
  for (const k of Object.keys(s.range ?? {})) keys.add(k);
  return [...keys].sort();
}

/** Every parameter the declared space mentions at all. */
export function declaredKeys(space: Record<string, unknown> | undefined): Set<string> {
  const s = (space ?? {}) as Space;
  const keys = new Set(sweptKeys(space));
  for (const k of Object.keys(s.base ?? {})) keys.add(k);
  for (const k of Object.keys(s.conditions ?? {})) keys.add(k);
  return keys;
}

function show(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** "tp=2 chunked_prefill_size=8192 −enable_mixed_chunk" — how `config`
 *  differs from `reference`; "(same as production)" when it does not. */
export function diffText(config: Config, reference: Config | undefined): string {
  const ref = reference ?? {};
  const parts: string[] = [];
  for (const key of Object.keys(config).sort()) {
    if (!(key in ref) || show(ref[key]) !== show(config[key])) parts.push(`${key}=${show(config[key])}`);
  }
  for (const key of Object.keys(ref).sort()) {
    if (!(key in config)) parts.push(`−${key}`);
  }
  return parts.length ? parts.join(" ") : "(same as production)";
}

/** "kv_cache_dtype (fp8_e4m3, fp8_e5m2); chunked_prefill_size (8192, 24576)" —
 *  every parameter the configs set away from `reference`, with the values tried. */
export function variedText(configs: Config[], reference: Config | undefined, maxValues = 6): string {
  const ref = reference ?? {};
  const varied = new Map<string, Set<string>>();
  for (const config of configs) {
    for (const key of Object.keys(config)) {
      if (key in ref && show(ref[key]) === show(config[key])) continue;
      if (!varied.has(key)) varied.set(key, new Set());
      varied.get(key)!.add(show(config[key]));
    }
  }
  if (!varied.size) return "nothing yet";
  return [...varied.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, values]) => {
      const shown = [...values].slice(0, maxValues);
      return `${key} (${shown.join(", ")}${values.size > shown.length ? `, +${values.size - shown.length} more` : ""})`;
    })
    .join("; ");
}

export function configText(config: Config): string {
  const keys = Object.keys(config).sort();
  return keys.length ? keys.map((k) => `${k}=${show(config[k])}`).join(" ") : "(engine defaults)";
}

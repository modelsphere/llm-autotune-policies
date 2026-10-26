// The engine image's vocabulary as the platform serves it (GET
// /engine/vocabulary, contract 1.5): every flag and environment variable the
// image reads, with the engine's own description and the platform's
// recommendation level. One line per entry, in the words a config uses.

export type VocabEntry = {
  kind: "flag" | "env";
  name: string;
  type?: string;
  default?: unknown;
  choices?: unknown[];
  /** The engine accepts names beyond `choices` too (a plugin registry). */
  choices_open?: boolean;
  help?: string;
  help_source?: string;
  level?: string;
  when?: string;
  /** The platform's hint of where to search: values worth a trial, or an interval. */
  suggested?: { values?: unknown[]; range?: { min: number; max: number; step?: number } } | null;
};

/** "try 0.75–0.93 (step 0.02)" / "try fa3, flashinfer"; "" without a hint. */
export function suggestionText(e: VocabEntry): string {
  const s = e.suggested;
  if (s?.range) return `try ${s.range.min}–${s.range.max}${s.range.step ? ` (step ${s.range.step})` : ""}`;
  if (s?.values?.length) return `try ${s.values.map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(", ")}`;
  return "";
}

/** The key a config sets it with: a flag's snake_case name, `env.NAME`. */
export function configKeyOf(e: VocabEntry): string {
  return e.kind === "env" ? `env.${e.name}` : e.name;
}

/** "`env.X` (bool, default false) [high; when: …] — what it does". `full`
 *  keeps the whole description; otherwise its first sentence. */
export function vocabLine(e: VocabEntry, full = false): string {
  const facts = [e.type, e.default !== undefined && e.default !== null ? `default ${JSON.stringify(e.default)}` : ""]
    .filter(Boolean)
    .join(", ");
  const open = e.choices_open ? " or a registered plugin" : "";
  const n = e.choices?.length ?? 0;
  const choices = !n
    ? ""
    : full || n <= 6
      ? `; one of ${e.choices!.map((c) => JSON.stringify(c)).join(", ")}${open}`
      : `; one of ${n} values${open} (engine_vocab lists them)`;
  const tag = [e.level && e.level !== "unknown" ? e.level : "", e.when ? `when: ${e.when}` : "", suggestionText(e)].filter(Boolean).join("; ");
  const help = (e.help ?? "").trim();
  const said = full ? help : help.split(/(?<=\.)\s/)[0];
  const source = e.help_source === "source" ? " (from the engine's code)" : "";
  return `- \`${configKeyOf(e)}\`${facts || choices ? ` (${facts}${choices})` : ""}${tag ? ` [${tag}]` : ""}${said ? ` — ${said}${source}` : ""}`;
}

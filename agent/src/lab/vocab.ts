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
  /** For a string env var with no declared choices: the engine code that reads
   *  it, its own check of the value first. The only list of values there is. */
  reads?: { where: string; code: string }[];
};

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
  const tag = [e.level && e.level !== "unknown" ? e.level : "", e.when ? `when: ${e.when}` : ""].filter(Boolean).join("; ");
  const help = (e.help ?? "").trim();
  const said = full ? help : help.split(/(?<=\.)\s/)[0];
  const source = e.help_source === "source" ? " (from the engine's code)" : "";
  const line = `- \`${configKeyOf(e)}\`${facts || choices ? ` (${facts}${choices})` : ""}${tag ? ` [${tag}]` : ""}${said ? ` — ${said}${source}` : ""}`;
  if (!full || !e.reads?.length) return line;
  const reads = e.reads.map((r) => `    read at ${r.where}:\n${r.code.split("\n").map((l) => `      ${l}`).join("\n")}`);
  return [line, ...reads].join("\n");
}

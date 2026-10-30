// One night, start to exit — the contract's lifecycle around the agent.
//
//   manifest → handshake → plan → memory → anchor queued → agent orients →
//   [agent proposes ⇄ runner measures ⇄ lab events] → (done | deadline) →
//   confirm the finalists → finalize → wait for exit
//
// The harness, not the model, owns: the clock and the heartbeat, the anchor,
// re-measuring the finalists, which configs are registered as contenders, the
// plan's declared extras, and what is saved for the next night.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { platformEnv, resolveSettings, type Settings } from "./config.js";
import { PhaseError, PolicyClient, PolicyError } from "./contract/client.js";
import { Heartbeat } from "./contract/heartbeat.js";
import type { Config, ContenderIn, Manifest, PlanExtra } from "./contract/types.js";
import { budget, type RecordContext, renderRecord } from "./context/render.js";
import type { VocabEntry } from "./lab/vocab.js";
import { AgentDriver } from "./driver.js";
import { buildModel, diagnose, writeDigest } from "./llm.js";
import { applyMemory, loadMemory, snapshot } from "./memory.js";
import { LabState, type Trial } from "./lab/state.js";
import { diffText, sweptKeys, variedText } from "./lab/space.js";
import { byConfig, deltaText, type Direction, noiseBand, ranking, trialObjective, anchorValue as anchorMean } from "./lab/stats.js";
import { TrialRunner } from "./runner.js";
import { buildTools } from "./tools.js";
import { VERSION } from "./version.js";

export interface NightOptions {
  env?: NodeJS.ProcessEnv;
  /** Replace the model (tests: a scripted provider). */
  model?: ReturnType<typeof buildModel>;
  log?(line: string): void;
  /** Main-loop tick. */
  tickMs?: number;
}

const PROMPTS = fileURLToPath(new URL("../../prompts/", import.meta.url));

async function prompt(name: string): Promise<string> {
  return readFile(`${PROMPTS}${name}`, "utf8");
}

function stamp(line: string): void {
  console.log(`${new Date().toISOString()} ${line}`);
}

export async function runNight(opts: NightOptions = {}): Promise<number> {
  const env = opts.env ?? process.env;
  const log = opts.log ?? stamp;
  const platform = platformEnv(env);
  const client = new PolicyClient({ apiUrl: platform.apiUrl, apiKey: platform.apiKey });
  const manifest = await client.getSession();
  const hb = new Heartbeat(client, (manifest.time.heartbeat_interval_s || 30) * 1000, VERSION, log);
  await hb.handshake();

  let settings: Settings;
  try {
    settings = resolveSettings(env, (manifest.policy?.params ?? {}) as Record<string, unknown>);
  } catch (err) {
    // A misconfigured policy says so at once instead of idling to the watchdog.
    log(`cannot start: ${(err as Error).message}`);
    await client.event("error", `agent policy cannot start: ${(err as Error).message}`).catch(() => undefined);
    hb.current.status = "error";
    hb.current.failureClass = "config";
    hb.current.message = (err as Error).message;
    await hb.beat().catch(() => undefined);
    return 1;
  }
  return new Night(client, manifest, hb, settings, log, opts).run();
}

class Night {
  private readonly lab = new LabState();
  private readonly direction: Direction;
  private readonly targetMetric: string;
  private readonly forbidden: string[];
  private readonly runner: TrialRunner;
  private driver!: AgentDriver;
  private endedReason = "";
  private confirmed = false;
  private nudges = 0;
  private lastNudge = 0;
  /** Since when cards have sat free with nothing queued (null while busy). */
  private idleSince: number | null = null;
  private announced = false;
  private readonly tickMs: number;

  constructor(
    private readonly client: PolicyClient,
    private readonly manifest: Manifest,
    private readonly hb: Heartbeat,
    private readonly settings: Settings,
    private readonly log: (line: string) => void,
    private readonly opts: NightOptions,
  ) {
    const screen = manifest.objective?.screen;
    this.direction = (screen?.direction ?? "max").toLowerCase() === "min" ? "min" : "max";
    this.targetMetric = screen?.target_metric ?? "objective";
    this.forbidden = ((manifest.space_policy as { forbidden?: string[] } | undefined)?.forbidden ?? []).map(String);
    this.tickMs = opts.tickMs ?? 1000;
    this.runner = new TrialRunner(
      client,
      this.lab,
      manifest.hardware,
      {
        pollMs: settings.pollMs,
        direction: this.direction,
        anomalyThreshold: settings.anomalyThreshold,
        suite: "screen",
      },
      {
        onTrialDone: (t) => this.trialDone(t),
        onTrialStarted: (t) => log(`t${t.id} started (${t.purpose})`),
        diagnose: (t, excerpt) => this.diagnoseFailure(t, excerpt),
        log,
      },
    );
  }

  private model!: ReturnType<typeof buildModel>;
  private diagnosePrompt = "";
  private digestPrompt = "";

  async run(): Promise<number> {
    const { client, manifest, hb, settings, lab, log } = this;
    this.model = this.opts.model ?? buildModel(settings.llm);
    this.diagnosePrompt = await prompt("diagnose.md");
    this.digestPrompt = await prompt("digest.md");
    const system = (await prompt("system.md"))
      .replaceAll("{{engine}}", manifest.model.engine)
      .replaceAll("{{metric}}", this.targetMetric)
      .replaceAll("{{direction}}", this.direction === "max" ? "maximize" : "minimize");

    applyMemory(lab, loadMemory(await client.getState().catch(() => null)));
    await this.putPlan();
    hb.current.phase = "searching";
    hb.current.message = "starting: measuring the production config first";
    hb.start();

    await this.queueAnchor();
    await this.readVocabulary();

    const tools = buildTools({
      client,
      lab,
      runner: this.runner,
      manifest,
      direction: this.direction,
      sessionId: manifest.session_id,
      trialsLeft: () => budget(this.recordContext()).trialsLeft,
      allowEarlyDone: settings.allowEarlyDone,
      onIntroduced: (param, reason) => {
        if (lab.introduced.has(param)) return;
        lab.introduced.set(param, reason);
        this.putPlan().catch((err) => log(`plan update failed: ${(err as Error).message}`));
      },
      onDoneRefused: (reason) => {
        log(`done refused (time for more trials remains): ${reason}`);
      },
      onProposed: (trials) => {
        this.nudges = 0;
        const hypotheses = [...new Set(trials.map((t) => t.hypothesisId).filter(Boolean))];
        client
          .event("info", `agent queued ${trials.map((t) => `#${t.id}`).join(", ")}${hypotheses.length ? ` to test ${hypotheses.join(", ")}` : ""}: ${trials.map((t) => diffText(t.config, lab.anchor?.config)).join(" | ")}`)
          .catch(() => undefined);
      },
      log,
    });

    this.driver = new AgentDriver({
      models: this.model.models,
      model: this.model.model,
      systemPrompt: system,
      tools,
      thinking: settings.llm.thinking,
      contextWindow: settings.llm.contextWindow,
      maxOutputTokens: settings.llm.maxOutputTokens,
      outageMinutes: settings.outageMinutes,
      requestTimeoutMs: settings.llm.timeoutMs,
      requestRetries: settings.llm.maxRetries,
      record: () => renderRecord(this.recordContext()),
      log,
    });
    this.runner.pump();
    this.driver.start(
      "The night begins. The production config is already queued as the anchor (#1), and again (#2) for the measurement noise. Read the lab record, open your first hypotheses, and propose enough configs to fill every free slot now. You will be woken whenever a trial finishes or a slot frees up.",
    );

    const command = await this.searchLoop();
    if (command === "exit" || command === "abort") {
      await this.teardown(command);
      return 0;
    }
    await this.finalize();
    await this.awaitExit();
    return 0;
  }

  // -- the search ---------------------------------------------------------------

  private async searchLoop(): Promise<string> {
    const { hb, lab, runner } = this;
    for (;;) {
      const command = hb.command;
      if (command === "exit" || command === "abort" || command === "finalize") {
        this.endedReason ||= command === "finalize" ? "deadline or platform finalize" : command;
        return command;
      }
      if (command === "serve") this.log("the platform asked us to serve a contender; this policy never registers a foreign image, so ignoring");
      if (this.driver.outage && !lab.done) this.end("the model was unreachable");
      if (this.settings.maxTokensPerNight && this.driver.tokens >= this.settings.maxTokensPerNight && !lab.done) {
        this.end(`token limit of ${this.settings.maxTokensPerNight} reached`);
      }
      if (lab.done && !this.announced) {
        this.announced = true;
        this.log(`search ending: ${lab.done.reason}`);
        this.client.event("info", `agent search ending: ${lab.done.reason}`).catch(() => undefined);
      }
      this.maybeConfirm();
      this.maybeNudge();
      this.maybeKeepBusy();
      this.updateStatus();
      if (lab.done && runner.idle) {
        this.endedReason ||= lab.done.reason;
        if (hb.current.status !== "exhausted") {
          hb.current.status = "exhausted";
          hb.current.message = `search done: ${lab.done.reason}`;
          await hb.beat().catch(() => undefined);
          continue;
        }
      }
      await new Promise((r) => setTimeout(r, this.tickMs));
    }
  }

  private end(reason: string): void {
    this.lab.done ??= { reason, at: new Date().toISOString() };
  }

  /** Near the end, measure each finalist that rests on one measurement again,
   *  so the configs the platform validates are backed by repeats. */
  private maybeConfirm(): void {
    if (this.confirmed || this.settings.confirmTopK <= 0) return;
    const b = budget(this.recordContext());
    const endgame = Boolean(this.lab.done) || (b.minutesLeft !== null && b.minutesLeft < b.perTrial * 2);
    if (!endgame) return;
    if (this.lab.done && !this.runner.idle) return; // let the agent's last trials land first
    this.confirmed = true;
    if (b.minutesLeft !== null && b.minutesLeft < b.perTrial) return; // no time left to confirm anything
    const top = ranking(byConfig(this.lab.trials), this.direction).slice(0, this.settings.confirmTopK);
    const queued: number[] = [];
    for (const g of top) {
      if (g.values.length >= 2) continue;
      if (this.lab.trials.some((t) => t.key === g.key && t.status !== "measured" && t.status !== "failed" && t.status !== "cancelled")) continue;
      const of = this.lab.trial(g.trialIds[0])!;
      const trial = this.lab.addTrial({ purpose: "confirm", config: of.config, cards: of.cards, remeasureOf: of.id, rationale: "harness: confirm a finalist" });
      queued.push(trial.id);
    }
    if (queued.length) {
      this.log(`confirming finalists: ${queued.map((i) => `#${i}`).join(", ")}`);
      this.driver.notify(`The harness queued ${queued.map((i) => `#${i}`).join(", ")} to re-measure the current finalists before the search ends.`);
      this.runner.pump();
    }
  }

  /** The card width of what the agent is running now: the narrowest of its
   *  last few proposals that launched, else the anchor's. */
  private nudgeWidth(): number {
    const recent = this.lab.trials
      .filter((t) => t.purpose !== "anchor" && t.cards > 0 && t.status !== "failed")
      .slice(-5)
      .map((t) => t.cards);
    if (recent.length) return Math.min(...recent);
    return this.lab.anchor?.cards || 1;
  }

  /** The model went quiet with cards free and nothing queued: remind it, and
   *  after enough fruitless reminders, call the search done. */
  private maybeNudge(): void {
    const { lab, runner, driver } = this;
    if (lab.done || driver.busy) return;
    // A slot is room for the configs the agent is running, not one spare card:
    // seven cards hold three tp=2 trials and leave one that fits none of them,
    // and nudging about that card ended a night with nine hours left.
    const width = this.nudgeWidth();
    if (lab.trials.some((t) => t.status === "queued") || runner.freeSlots(width) === 0) return;
    const b = budget(this.recordContext());
    if (b.minutesLeft !== null && b.minutesLeft < b.perTrial) return;
    if (Date.now() - this.lastNudge < 30000) return;
    if (this.nudges >= this.settings.nudgeLimit) {
      // Ending early is the agent's call only where the campaign allows it;
      // otherwise keep reminding it, less often, until the deadline (and the
      // harness keeps the cards busy meanwhile, see maybeKeepBusy).
      if (this.settings.allowEarlyDone || (b.trialsLeft !== null && b.trialsLeft < 2)) {
        this.end("the agent stopped proposing");
        return;
      }
      if (Date.now() - this.lastNudge < 5 * 60000) return;
    }
    this.nudges += 1;
    this.lastNudge = Date.now();
    driver.notify(
      `${runner.freeCardCount} of ${runner.totalCards} card(s) are free and nothing is queued, with room for about ${b.trialsLeft ?? "?"} more trial(s). Idle cards are wasted GPU time: propose the next experiments (the declared space is only a hint — any engine parameter that is not forbidden may be tried).`,
    );
  }

  /** Cards the agent leaves idle are the harness's to fill. The platform ends
   *  a session that runs nothing for a while, and an agent that believes it
   *  has converged proposes nothing: together they ended a 60-hour search
   *  after 11. So past a few idle minutes, re-measure the closest-ranked
   *  configs (fewest measurements first) in the free slots, which sharpens
   *  the verdict, and steer the agent to widen the search. */
  private maybeKeepBusy(): void {
    const { lab, runner } = this;
    const width = this.nudgeWidth();
    const idle = !lab.done && this.settings.keepBusyMinutes > 0
      && !lab.trials.some((t) => t.status === "queued") && runner.freeSlots(width) > 0;
    if (!idle) {
      this.idleSince = null;
      return;
    }
    const b = budget(this.recordContext());
    if (b.minutesLeft !== null && b.minutesLeft < b.perTrial) return;
    this.idleSince ??= Date.now();
    const idleMinutes = (Date.now() - this.idleSince) / 60000;
    if (idleMinutes < this.settings.keepBusyMinutes) return;
    this.idleSince = null;

    const busy = new Set(lab.trials.filter((t) => !["measured", "failed", "cancelled"].includes(t.status)).map((t) => t.key));
    const picks = ranking(byConfig(lab.trials), this.direction)
      .slice(0, 6)
      .filter((g) => g.feasible && !busy.has(g.key))
      .map((g, rank) => ({ g, rank }))
      .sort((a, c) => a.g.values.length - c.g.values.length || a.rank - c.rank)
      .slice(0, runner.freeSlots(width));
    const queued: Trial[] = [];
    for (const { g } of picks) {
      const of = lab.trial(g.trialIds[0])!;
      queued.push(lab.addTrial({ purpose: "remeasure", config: of.config, cards: of.cards, remeasureOf: of.id, rationale: "harness: the cards sat idle; re-measure a close finalist" }));
    }
    if (queued.length) runner.pump();
    const ids = queued.map((t) => `#${t.id}`).join(", ");
    this.log(`cards idle ${idleMinutes.toFixed(1)} min: ${queued.length ? `harness queued ${ids}` : "nothing to re-measure"}`);
    if (queued.length) this.client.event("info", `agent idle with cards free; the harness queued re-measures ${ids}`).catch(() => undefined);
    const hours = b.minutesLeft === null ? "" : ` with about ${(b.minutesLeft / 60).toFixed(1)} h of search left`;
    const tried = variedText(lab.trials.filter((t) => t.purpose !== "anchor").map((t) => t.config), lab.anchor?.config);
    this.driver.notify(
      `The cards sat idle for ${idleMinutes.toFixed(0)} min${hours}${queued.length ? `, so the harness queued ${ids} (re-measures of close finalists) to keep them working` : ""}. ` +
        "A plateau around one stack is not the end of the search: the time left is the search's, and validation has its own. " +
        `What tonight has set away from production so far: ${tried}. ` +
        "Widen the search: levers you have not touched (the engine's other scheduler, memory, attention, CUDA-graph, overlap/compile, parallelism and speculative-decoding options — `check` any you have not used), larger moves away from the current stack, values beyond the range tried, combinations of levers measured only alone, and ablations of the leading stack. Propose the next experiments now.",
    );
  }

  private async trialDone(trial: Trial): Promise<void> {
    // The platform reads the image's vocabulary in the background when the
    // session starts; it may land after our first look.
    const facts = this.modelFacts ?? (this.manifest.model as { facts?: object }).facts ?? {};
    const stale = this.vocabularyStatus === "none" || !Object.keys(facts).length;
    if (stale && Date.now() - this.vocabularyReadAt > 5 * 60_000) await this.readVocabulary();
    await this.refreshContenders();
    await this.saveState("in progress");
    this.driver?.notify(this.describe(trial));
  }

  private describe(trial: Trial): string {
    const anchor = this.lab.anchor;
    const anchorValue = anchor ? anchorMean(this.lab.trials) : null;
    const value = trialObjective(trial);
    const noise = noiseBand(byConfig(this.lab.trials));
    let line = `Trial #${trial.id} (${trial.purpose}) ${trial.status}`;
    if (trial.status === "measured" && value !== null) {
      line += `: objective ${value.toFixed(1)}`;
      if (anchorValue !== null && trial.purpose !== "anchor") line += ` (${deltaText(value, anchorValue, noise, this.direction)} vs anchor)`;
      if (trial.measurements.some((m) => !m.feasible)) line += " — INFEASIBLE (redline crossed)";
    }
    if (trial.failure) {
      const d = trial.failure.diagnosis;
      line += `: failed at ${trial.failure.stage} (${trial.failure.failureClass})${d ? ` — ${d.cause}${d.offending_flag ? ` [flag ${d.offending_flag}]` : ""}` : ""}`;
    }
    return `${line}. ${this.runner.freeCardCount} of ${this.runner.totalCards} card(s) free. The lab record has the details.`;
  }

  private async diagnoseFailure(trial: Trial, excerpt: string) {
    return diagnose(this.model.models, this.model.model, this.diagnosePrompt, trial, excerpt, this.manifest.model.engine);
  }

  // -- the contract's obligations --------------------------------------------------

  /** Say what we tune: the declared space's axes, plus every parameter the
   *  agent has introduced so far, each with the reason it was introduced. */
  private async putPlan(): Promise<void> {
    const extras: PlanExtra[] = [...this.lab.introduced.entries()].map(([param, reason]) => ({ param, reason: reason.slice(0, 500) }));
    await this.client.putPlan(
      sweptKeys(this.manifest.search_space),
      extras,
      `LLM agent (${this.settings.llm.model}): proposes any engine parameter that is not forbidden; parameters outside the declared space are listed as extras as they are introduced.`,
    );
  }

  private async queueAnchor(): Promise<void> {
    // 1.7: the platform names the config that launches production (`{}` when
    // it is the campaign's recipe as written); before, the recorded args were it.
    const production = (this.manifest.production ?? {}) as { config?: Config; engine_args?: Config };
    const first = (production.config ?? production.engine_args ?? {}) as Config;
    for (const candidate of Object.keys(first).length ? [first, {}] : [{}]) {
      const check = await this.client.checkConfig(candidate);
      if (check.ok) {
        // Twice, side by side when the budget allows: the two give the noise
        // every later delta is judged against, before anything else is.
        const anchor = this.lab.addTrial({ purpose: "anchor", config: check.config ?? {}, cards: check.cards ?? 0, rationale: "the production config, measured tonight as the reference" });
        this.lab.addTrial({ purpose: "remeasure", remeasureOf: anchor.id, config: anchor.config, cards: anchor.cards, rationale: "production again: the first estimate of the measurement noise" });
        return;
      }
      this.log(`anchor ${Object.keys(candidate).length ? "production config" : "base config"} refused: ${(check.problems ?? []).map((p) => p.message).join("; ")}`);
    }
  }

  /** The best feasible configs, as the platform's contenders. Re-registered
   *  after every trial: what is registered when we die is what gets validated. */
  private contendersQueue: Promise<void> = Promise.resolve();

  /** One update at a time: two trials finishing together must not send two
   *  overlapping lists (the later one waits and sends the newer ranking). */
  private refreshContenders(): Promise<void> {
    this.contendersQueue = this.contendersQueue.then(() => this.sendContenders());
    return this.contendersQueue;
  }

  private async sendContenders(): Promise<void> {
    const top = ranking(byConfig(this.lab.trials), this.direction).slice(0, this.manifest.contenders?.max ?? 2);
    if (!top.length) return;
    // The platform's evidence is its own trial records, one per benchmark it
    // measured; the lab's trial numbers mean nothing to it.
    const platformTrial = new Map<number, number>();
    try {
      for (const t of await this.client.listTrials()) if (t.run_id !== null && t.run_id !== undefined) platformTrial.set(t.run_id, t.id);
    } catch (err) {
      if (!(err instanceof PhaseError)) this.log(`trials not read; contenders go without their evidence ids: ${(err as Error).message}`);
    }
    const contenders: ContenderIn[] = top.map((g, index) => {
      const withEnv = g.trialIds.map((id) => this.lab.trial(id)).find((t) => t?.environment);
      return {
        rank: index + 1,
        launch_spec: {
          engine_args: g.config,
          image: this.manifest.model.image,
          env: {},
          volumes: {},
          environment: withEnv?.environment ?? {},
        },
        evidence: { screen_objective: g.mean, measurements: g.values.length, values: g.values, agent_trials: g.trialIds },
        trial_ids: g.trialIds
          .flatMap((id) => this.lab.trial(id)?.benchIds ?? [])
          .map((bench) => platformTrial.get(bench))
          .filter((id): id is number => id !== undefined),
      };
    });
    try {
      const echo = await this.client.putContenders(contenders);
      for (const r of echo.rejected ?? []) this.log(`contender refused: ${JSON.stringify(r)}`);
    } catch (err) {
      if (!(err instanceof PhaseError)) this.log(`contenders not updated: ${(err as Error).message}`);
    }
  }

  /** The night's findings for other campaigns, when the platform takes them.
   *  Best effort: a night without a digest is only a night others cannot
   *  learn from. */
  private async fileDigest(): Promise<void> {
    if (!this.manifest.services?.history || !this.lab.trials.some((t) => t.status === "measured")) return;
    this.hb.current.message = "writing the night's digest";
    try {
      const record = renderRecord(this.recordContext());
      let outcome = await writeDigest(this.model.models, this.model.model, this.digestPrompt, record, this.settings.llm.maxOutputTokens);
      if (!outcome.digest) {
        this.log(`digest: first try unusable (${outcome.why}); retrying with less reasoning`);
        outcome = await writeDigest(this.model.models, this.model.model, this.digestPrompt, record, this.settings.llm.maxOutputTokens, "low");
      }
      const digest = outcome.digest;
      if (!digest) {
        this.log(`digest: the model gave nothing usable (${outcome.why}); none filed`);
        return;
      }
      await this.client.putDigest(digest);
      this.log(`digest filed: ${digest.lessons.length} lesson(s), ${digest.constraints.length} constraint(s), ${digest.open_questions.length} open question(s)`);
    } catch (err) {
      this.log(`digest not filed: ${(err as Error).message}`);
    }
  }

  private async saveState(reason: string): Promise<void> {
    try {
      await this.client.putState(snapshot(this.lab, this.manifest, reason, this.direction));
    } catch (err) {
      this.log(`state not saved: ${(err as Error).message}`);
    }
  }

  private updateStatus(): void {
    const measured = this.lab.trials.filter((t) => t.status === "measured");
    const top = ranking(byConfig(this.lab.trials), this.direction)[0];
    const anchorValue = anchorMean(this.lab.trials);
    const gain = top && anchorValue ? ` (${deltaText(top.mean, anchorValue, null, this.direction)} vs production)` : "";
    this.hb.current.message = `${measured.length} measured, ${this.runner.inFlight} running; best ${top ? top.mean.toFixed(1) : "—"}${gain}`;
    const start = this.manifest.time.started_at ? Date.parse(this.manifest.time.started_at) : null;
    const end = this.manifest.time.search_deadline ? Date.parse(this.manifest.time.search_deadline) : null;
    if (start && end && end > start) this.hb.current.progress = (Date.now() - start) / (end - start);
  }

  private levers: VocabEntry[] = [];
  private vocabularyStatus: string | undefined;
  private vocabularyReadAt = 0;
  private vocabularyCounts: RecordContext["vocabularyCounts"];
  private modelFacts: Record<string, unknown> | undefined;

  /** The image's recommended switches, once: the record lists them every turn. */
  private async readVocabulary(): Promise<void> {
    if (!this.manifest.services?.vocabulary) return;
    this.vocabularyReadAt = Date.now();
    try {
      const vocab = await this.client.engineVocabulary({ level: "mid", limit: 60 });
      this.vocabularyStatus = vocab.status;
      this.levers = (vocab.entries ?? []) as VocabEntry[];
      if (vocab.model_facts && Object.keys(vocab.model_facts).length) this.modelFacts = vocab.model_facts;
      if (vocab.status !== "none") {
        const [flags, env] = await Promise.all([
          this.client.engineVocabulary({ kind: "flag", limit: 1 }),
          this.client.engineVocabulary({ kind: "env", limit: 1 }),
        ]);
        this.vocabularyCounts = { flags: flags.total ?? 0, env: env.total ?? 0, engineVersion: vocab.engine_version ?? "" };
      }
      this.log(`vocabulary: ${vocab.status}, ${this.levers.length} recommended switch(es)${this.modelFacts ? ", model facts read" : ""}`);
    } catch (err) {
      this.vocabularyStatus = "none";
      this.log(`vocabulary read failed: ${(err as Error).message}`);
    }
  }

  recordContext(): RecordContext {
    const cards = this.lab.trials.map((t) => t.cards).sort((a, b) => a - b);
    const typical = cards.length ? cards[Math.floor(cards.length / 2)] : 1;
    return {
      manifest: this.manifest,
      lab: this.lab,
      direction: this.direction,
      targetMetric: this.targetMetric,
      forbidden: this.forbidden,
      totalCards: this.runner.totalCards,
      freeCards: this.runner.freeCardCount,
      running: this.runner.runningCards(),
      inFlight: this.runner.inFlight,
      sessionId: this.manifest.session_id,
      tokensUsed: this.driver?.tokens,
      tokenLimit: this.settings.maxTokensPerNight || undefined,
      levers: this.levers,
      vocabularyStatus: this.vocabularyStatus,
      vocabularyCounts: this.vocabularyCounts,
      modelFacts: this.modelFacts,
    };
  }

  // -- the end -------------------------------------------------------------------------

  private async finalize(): Promise<void> {
    const { hb, client, log } = this;
    hb.current.phase = "finalizing";
    hb.current.message = "finalizing: releasing engines";
    await this.driver.stop();
    await this.runner.stopAll();
    await this.refreshContenders();
    await this.saveState(this.endedReason);
    try {
      await client.finalized();
    } catch (err) {
      if (!(err instanceof PolicyError)) throw err;
    }
    // After finalized: the platform validates the finalists meanwhile, so the
    // model call costs no GPU time.
    await this.fileDigest();
    hb.current.phase = "validating";
    hb.current.message = "finalized; the platform is measuring the finalists";
    log(`finalized after ${this.lab.trials.length} trial(s); model tokens used: ${this.driver.tokens}`);
  }

  private async awaitExit(): Promise<void> {
    while (this.hb.command !== "exit" && this.hb.command !== "abort") {
      await new Promise((r) => setTimeout(r, this.tickMs));
    }
    this.hb.stop();
  }

  private async teardown(command: string): Promise<void> {
    this.log(`platform says ${command}: releasing engines and exiting`);
    await this.driver.stop();
    await this.runner.stopAll();
    await this.saveState(command);
    this.hb.stop();
  }
}

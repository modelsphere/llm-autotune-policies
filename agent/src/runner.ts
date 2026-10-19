// Runs trials on the platform: as many at once as the session's cards and
// ports allow, each on its own disjoint cards. A trial is: delegated launch →
// wait for it to serve → record what it runs on → screen benchmark → full
// results → release. The model decides WHAT to run; this decides nothing but
// placement and order, and keeps the books honest:
//   - a suspiciously good result is benchmarked again on the same engine
//     before anyone treats it as real (AutoTuneBench's anomaly breaker);
//   - a failed launch arrives with its log excerpt and a diagnosis.

import { randomBytes } from "node:crypto";
import { ConfigRejected, NoCapacity, PhaseError, type PolicyClient, PolicyError, SessionOver } from "./contract/client.js";
import type { LaunchOut } from "./contract/types.js";
import { measurementOf } from "./lab/results.js";
import { type Diagnosis, type LabState, now, type Trial, type TrialStatus } from "./lab/state.js";
import { bestValue, byConfig, type Direction, isAnomalous } from "./lab/stats.js";

export interface RunnerOptions {
  pollMs: number;
  direction: Direction;
  /** Re-bench a result this much better (fraction) than the best so far. */
  anomalyThreshold: number;
  maxParallel?: number;
  suite: string;
}

export interface RunnerHooks {
  onTrialDone(trial: Trial): void | Promise<void>;
  onTrialStarted?(trial: Trial): void;
  diagnose(trial: Trial, logExcerpt: string): Promise<Diagnosis>;
  log(line: string): void;
}

const LAUNCH_DONE = new Set(["serving", "failed", "released"]);
const BENCH_DONE = new Set(["succeeded", "failed", "killed"]);
const ERROR_PATTERN = "error|exception|traceback|out of memory|oom|unrecognized|invalid|killed|fatal|assert";

export class TrialRunner {
  private freeCards: number[];
  private freePorts: number[];
  private readonly hasCards: boolean;
  private readonly running = new Map<number, Promise<void>>();
  private stopped = false;
  private waitTimer: NodeJS.Timeout | null = null;
  /** Idempotency keys are scoped to this process: a restarted container must
   *  not be handed the runs its predecessor asked for under the same key. */
  private readonly runId = randomBytes(4).toString("hex");

  constructor(
    private readonly client: PolicyClient,
    private readonly lab: LabState,
    hardware: { gpu_indices?: number[]; ports?: number[] },
    private readonly opts: RunnerOptions,
    private readonly hooks: RunnerHooks,
  ) {
    this.freeCards = [...(hardware.gpu_indices ?? [])];
    this.freePorts = [...(hardware.ports ?? [])];
    this.hasCards = this.freeCards.length > 0;
  }

  /** Cards the session holds (0 on a machine without cards). */
  get totalCards(): number {
    return this.hasCards ? this.freeCards.length + this.busyCards() : 0;
  }

  private busyCards(): number {
    return this.lab.trials.filter((t) => this.running.has(t.id)).reduce((n, t) => n + t.gpuIndices.length, 0);
  }

  get inFlight(): number {
    return this.running.size;
  }

  get idle(): boolean {
    return this.running.size === 0 && !this.lab.trials.some((t) => t.status === "queued");
  }

  /** How many more trials of `cards` cards could start right now. */
  freeSlots(cards = 1): number {
    const byPorts = this.freePorts.length;
    const byCards = this.hasCards ? Math.floor(this.freeCards.length / Math.max(1, cards)) : byPorts;
    const byCap = this.opts.maxParallel ? Math.max(0, this.opts.maxParallel - this.running.size) : Infinity;
    return Math.max(0, Math.min(byPorts, byCards, byCap));
  }

  fits(cards: number): boolean {
    return !this.hasCards || cards <= this.totalCards;
  }

  /** Start every queued trial that fits, in order, letting a smaller one
   *  start ahead of a wider one that has to wait for cards. */
  pump(): void {
    if (this.stopped) return;
    for (const trial of this.lab.trials.filter((t) => t.status === "queued")) {
      if (this.opts.maxParallel && this.running.size >= this.opts.maxParallel) break;
      if (!this.freePorts.length) break;
      const cards = this.hasCards ? trial.cards : 0;
      if (this.hasCards && this.freeCards.length < cards) continue;
      const gpus = this.hasCards ? this.freeCards.splice(0, cards) : [];
      const port = this.freePorts.shift();
      trial.gpuIndices = gpus;
      trial.port = port;
      trial.status = "launching";
      trial.startedAt = now();
      const run = this.runTrial(trial).finally(() => {
        this.running.delete(trial.id);
        this.freeCards.push(...trial.gpuIndices);
        this.freeCards.sort((a, b) => a - b);
        if (trial.port !== undefined) this.freePorts.push(trial.port);
        this.pump();
      });
      this.running.set(trial.id, run);
      this.hooks.onTrialStarted?.(trial);
    }
  }

  /** Cancel a queued trial, or stop a running one (its engine is released). */
  cancel(trial: Trial, reason: string): void {
    if (trial.status === "queued") {
      trial.status = "cancelled";
      trial.notes.push(`cancelled before it started: ${reason}`);
      trial.finishedAt = now();
      return;
    }
    trial.notes.push(`cancel requested: ${reason}`);
    (trial as Trial & { cancelRequested?: boolean }).cancelRequested = true;
  }

  /** Stop everything: no new trials, release every engine, cancel benchmarks. */
  async stopAll(): Promise<void> {
    this.stopped = true;
    if (this.waitTimer) clearTimeout(this.waitTimer);
    for (const trial of this.lab.trials) {
      if (trial.status === "queued") {
        trial.status = "cancelled";
        trial.notes.push("cancelled: the search ended before it started");
      }
    }
    await Promise.allSettled([...this.running.values()]);
  }

  private shouldStop(trial: Trial): boolean {
    return this.stopped || Boolean((trial as Trial & { cancelRequested?: boolean }).cancelRequested);
  }

  private sleep(): Promise<void> {
    return new Promise((r) => setTimeout(r, this.opts.pollMs));
  }

  private async runTrial(trial: Trial): Promise<void> {
    let launch: LaunchOut | null = null;
    let requeued = false;
    try {
      try {
        launch = await this.client.launch(trial.config, trial.gpuIndices, trial.port, `${this.runId}-t${trial.id}-launch`);
      } catch (err) {
        if (err instanceof NoCapacity) {
          // A just-released engine is still tearing down: try again shortly.
          trial.status = "queued";
          trial.startedAt = undefined;
          requeued = true;
          this.waitTimer = setTimeout(() => this.pump(), this.opts.pollMs);
          return;
        }
        if (err instanceof ConfigRejected || err instanceof PhaseError) {
          this.fail(trial, err instanceof PhaseError ? "deadline" : "refused", "rejected", err.message);
          return;
        }
        throw err;
      }
      trial.launchId = launch.id;
      this.hooks.log(`t${trial.id}: launch ${launch.id} on cards [${trial.gpuIndices.join(",")}] port ${trial.port}`);

      while (!LAUNCH_DONE.has(launch.status)) {
        if (this.shouldStop(trial)) return this.cancelled(trial);
        await this.sleep();
        launch = await this.client.getLaunch(launch.id);
      }
      if (launch.status !== "serving") {
        const excerpt = await this.logExcerpt(launch.id);
        const failure = launch.failure_class || "launch_failed";
        this.fail(trial, "launch", failure, launch.error ?? "", excerpt);
        try {
          trial.failure!.diagnosis = await this.hooks.diagnose(trial, excerpt || launch.error || "");
        } catch (err) {
          this.hooks.log(`t${trial.id}: diagnosis failed: ${(err as Error).message}`);
        }
        return;
      }

      trial.status = "serving";
      trial.servingAt = now();
      await this.recordWhatRuns(trial, launch);
      if (this.shouldStop(trial)) return this.cancelled(trial);

      await this.bench(trial);
      const first = trial.measurements.at(-1);
      if (first && first.objective !== null && first.feasible && !this.shouldStop(trial)) {
        const others = byConfig(this.lab.trials.filter((t) => t.id !== trial.id));
        const best = bestValue(others, this.opts.direction);
        if (isAnomalous(first.objective, best, this.opts.anomalyThreshold, this.opts.direction)) {
          trial.notes.push(
            `anomaly breaker: ${(this.opts.anomalyThreshold * 100).toFixed(0)}%+ better than the best so far, benchmarked again on the same engine before trusting it`,
          );
          await this.bench(trial);
        }
      }
      const settled = trial.status as TrialStatus;
      if (settled !== "failed" && settled !== "cancelled") {
        trial.status = trial.measurements.length ? "measured" : "failed";
      }
    } catch (err) {
      if (err instanceof SessionOver) {
        this.stopped = true;
        return this.cancelled(trial);
      }
      const message = err instanceof PolicyError || err instanceof Error ? err.message : String(err);
      this.fail(trial, "launch", "harness_error", message);
    } finally {
      if (launch && launch.id) {
        try {
          await this.client.release(launch.id);
        } catch {
          // already gone, or the session is over
        }
      }
      if (!requeued) {
        trial.finishedAt = trial.finishedAt ?? now();
        try {
          await this.hooks.onTrialDone(trial);
        } catch (err) {
          this.hooks.log(`onTrialDone failed: ${(err as Error).message}`);
        }
      }
    }
  }

  private async bench(trial: Trial): Promise<void> {
    let bench;
    try {
      bench = await this.client.benchLaunch(trial.launchId!, this.opts.suite, `${this.runId}-t${trial.id}-b${trial.benchIds.length + 1}`);
    } catch (err) {
      if (err instanceof PhaseError) {
        this.fail(trial, "deadline", "refused", err.message);
        return;
      }
      throw err;
    }
    trial.status = "benching";
    trial.benchIds.push(bench.id);
    while (!BENCH_DONE.has(bench.status)) {
      if (this.shouldStop(trial)) {
        try {
          await this.client.cancelBenchmark(bench.id);
        } catch {
          // best effort
        }
        this.cancelled(trial);
        return;
      }
      await this.sleep();
      bench = await this.client.getBenchmark(bench.id);
    }
    if (bench.status !== "succeeded") {
      this.fail(trial, "bench", bench.failure_class || bench.status, bench.error ?? "");
      return;
    }
    let results = null;
    try {
      results = await this.client.benchmarkResults(bench.id);
    } catch (err) {
      this.hooks.log(`t${trial.id}: full results unavailable: ${(err as Error).message}`);
    }
    trial.measurements.push(measurementOf(bench, results));
  }

  private async recordWhatRuns(trial: Trial, launch: LaunchOut): Promise<void> {
    try {
      const env = await this.client.launchEnvironment(launch.id);
      trial.environment = Object.fromEntries(
        Object.entries(env).filter(([k, v]) => v !== null && v !== "" && !["kind", "schema_version", "run_id", "snapshot"].includes(k)),
      );
    } catch {
      // older platform, or not recorded yet
    }
    if (launch.endpoint_url) trial.serverInfo = await serverInfo(launch.endpoint_url);
  }

  private async logExcerpt(launchId: number): Promise<string> {
    try {
      const matches = await this.client.launchLog(launchId, { grep: ERROR_PATTERN, context: 3, tail: 120 });
      const tail = await this.client.launchLog(launchId, { tail: 60 });
      return `--- lines matching errors ---\n${matches}\n--- last lines ---\n${tail}`.slice(-12000);
    } catch {
      return "";
    }
  }

  private fail(trial: Trial, stage: NonNullable<Trial["failure"]>["stage"], failureClass: string, error: string, logExcerpt?: string): void {
    trial.status = "failed";
    trial.failure = { stage, failureClass, error: error.slice(0, 2000), logExcerpt };
    trial.finishedAt = now();
  }

  private cancelled(trial: Trial): void {
    if (trial.status !== "failed") trial.status = trial.measurements.length ? "measured" : "cancelled";
    trial.finishedAt = now();
  }
}

/** The engine's own account of what it resolved its arguments to — sglang's
 *  /get_server_info carries every server arg with its effective value. Best
 *  effort: an engine without such an endpoint just has none. */
async function serverInfo(endpoint: string): Promise<Record<string, unknown> | undefined> {
  for (const path of ["/get_server_info", "/server_info"]) {
    try {
      const response = await fetch(endpoint.replace(/\/+$/, "") + path, { signal: AbortSignal.timeout(5000) });
      if (!response.ok) continue;
      const body = (await response.json()) as Record<string, unknown>;
      if (body && typeof body === "object") return body;
    } catch {
      // unreachable from here, or not offered
    }
  }
  return undefined;
}

// HTTP client for the policy session API (docs/api/policy-contract.md in the
// platform repo). One method per endpoint; it does not interpret responses.
//
// The same semantics as the Python SDK's client: every call carries
// X-API-Key; transient failures (network, 500/502/503/504 — a platform
// restarting) are retried with backoff, safe because the POSTs that create
// work carry an Idempotency-Key; a 4xx is a decision and raises a typed error.

import type {
  BenchmarkOut,
  Config,
  ConfigCheckOut,
  ContenderIn,
  ContenderOut,
  ContendersEcho,
  EnvironmentPart,
  HeartbeatOut,
  LaunchOut,
  Manifest,
  PlanExtra,
  ResultsPart,
  TrialOut,
} from "./types.js";

export class PolicyError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly body: unknown = {},
  ) {
    super(`HTTP ${status}: ${message}`);
  }
}
/** 401 or 410: the session is over for us. Exit. */
export class SessionOver extends PolicyError {}
/** 409: the phase forbids it. Follow the heartbeat. */
export class PhaseError extends PolicyError {}
/** 422: refused (safety, forbidden parameter, round trip). Fix the config. */
export class ConfigRejected extends PolicyError {}
/** 429: no free cards or ports in our allocation. Release something or wait. */
export class NoCapacity extends PolicyError {}

const TYPED: Record<number, typeof PolicyError> = {
  401: SessionOver,
  410: SessionOver,
  409: PhaseError,
  422: ConfigRejected,
  429: NoCapacity,
};
const TRANSIENT = new Set([0, 500, 502, 503, 504]);

export interface ClientOptions {
  apiUrl: string;
  apiKey: string;
  timeoutMs?: number;
  retries?: number;
  backoffMs?: number;
  backoffCapMs?: number;
  sleep?: (ms: number) => Promise<void>;
}

export interface HeartbeatBody {
  phase: string;
  message?: string;
  progress?: number;
  status?: string;
  failure_class?: string;
  sdk_version?: string;
  contract_version?: string;
  capabilities?: string[];
}

const BASE_PATH = "/api/policy/v1";

export class PolicyClient {
  private readonly base: string;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(private readonly opts: ClientOptions) {
    this.base = opts.apiUrl.replace(/\/+$/, "") + BASE_PATH;
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    idempotencyKey?: string,
    as: "json" | "text" = "json",
  ): Promise<T> {
    const retries = this.opts.retries ?? 8;
    for (let attempt = 0; ; attempt++) {
      try {
        return await this.attempt<T>(method, path, body, idempotencyKey, as);
      } catch (err) {
        if (err instanceof PolicyError && TRANSIENT.has(err.status) && attempt < retries) {
          const base = this.opts.backoffMs ?? 1000;
          await this.sleep(Math.min(base * 2 ** attempt, this.opts.backoffCapMs ?? 15000));
          continue;
        }
        throw err;
      }
    }
  }

  private async attempt<T>(
    method: string,
    path: string,
    body: unknown,
    idempotencyKey: string | undefined,
    as: "json" | "text",
  ): Promise<T> {
    const headers: Record<string, string> = { "X-API-Key": this.opts.apiKey };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (idempotencyKey) headers["Idempotency-Key"] = idempotencyKey;
    let response: Response;
    try {
      response = await fetch(this.base + path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: AbortSignal.timeout(this.opts.timeoutMs ?? 30000),
      });
    } catch (err) {
      throw new PolicyError(0, `network error: ${(err as Error).message}`);
    }
    const text = await response.text();
    if (!response.ok) {
      let parsed: unknown = {};
      try {
        parsed = text ? JSON.parse(text) : {};
      } catch {
        parsed = { detail: text };
      }
      const detail = (parsed as { detail?: unknown }).detail;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail ?? response.statusText);
      const Kind = TYPED[response.status] ?? PolicyError;
      throw new Kind(response.status, message, parsed);
    }
    if (as === "text") return text as T;
    return (text ? JSON.parse(text) : {}) as T;
  }

  // -- session ----------------------------------------------------------------
  getSession(): Promise<Manifest> {
    return this.request("GET", "/session");
  }
  heartbeat(body: HeartbeatBody): Promise<HeartbeatOut> {
    return this.request("POST", "/session/heartbeat", body);
  }
  putPlan(tuning: string[], extras: PlanExtra[], notes: string): Promise<void> {
    return this.request("PUT", "/session/plan", { tuning, extras, notes });
  }
  finalized(): Promise<void> {
    return this.request("POST", "/session/finalized", {});
  }

  // -- launches -----------------------------------------------------------------
  launch(engineArgs: Config, gpuIndices: number[], port: number | undefined, key: string): Promise<LaunchOut> {
    const body: Record<string, unknown> = { engine_args: engineArgs, gpu_indices: gpuIndices };
    if (port !== undefined) body.port = port;
    return this.request("POST", "/launches", body, key);
  }
  getLaunch(id: number): Promise<LaunchOut> {
    return this.request("GET", `/launches/${id}`);
  }
  release(id: number): Promise<LaunchOut> {
    return this.request("DELETE", `/launches/${id}`);
  }
  launchLog(id: number, opts: { tail?: number; grep?: string; context?: number } = {}): Promise<string> {
    const q = new URLSearchParams();
    if (opts.tail) q.set("tail", String(opts.tail));
    if (opts.grep) q.set("grep", opts.grep);
    if (opts.context) q.set("context", String(opts.context));
    const qs = q.toString();
    return this.request("GET", `/launches/${id}/log${qs ? `?${qs}` : ""}`, undefined, undefined, "text");
  }
  launchEnvironment(id: number): Promise<EnvironmentPart> {
    return this.request("GET", `/launches/${id}/environment`);
  }

  // -- benchmarks ---------------------------------------------------------------
  benchLaunch(launchId: number, suite: string, key: string): Promise<BenchmarkOut> {
    return this.request("POST", `/launches/${launchId}/benchmarks`, { suite }, key);
  }
  getBenchmark(id: number): Promise<BenchmarkOut> {
    return this.request("GET", `/benchmarks/${id}`);
  }
  benchmarkResults(id: number): Promise<ResultsPart> {
    return this.request("GET", `/benchmarks/${id}/results`);
  }
  cancelBenchmark(id: number): Promise<BenchmarkOut> {
    return this.request("DELETE", `/benchmarks/${id}`);
  }

  // -- configs, contenders, state, events -----------------------------------------
  checkConfig(engineArgs: Config): Promise<ConfigCheckOut> {
    return this.request("POST", "/configs/check", { engine_args: engineArgs });
  }
  putContenders(contenders: ContenderIn[]): Promise<ContendersEcho> {
    return this.request("PUT", "/contenders", { contenders });
  }
  /** The trials the platform recorded this session: one per benchmark it
   *  measured (`run_id` is the benchmark's id), plus any the policy reported. */
  listTrials(): Promise<TrialOut[]> {
    return this.request("GET", "/trials");
  }
  /** What tonight learned, for other campaigns (1.4). */
  putDigest(digest: unknown): Promise<void> {
    return this.request("PUT", "/session/digest", digest);
  }
  getContenders(): Promise<ContenderOut[]> {
    return this.request("GET", "/contenders");
  }
  async getState(): Promise<unknown | null> {
    try {
      const text = await this.request<string>("GET", "/state", undefined, undefined, "text");
      return text ? JSON.parse(text) : null;
    } catch (err) {
      if (err instanceof PolicyError && err.status === 404) return null; // first night
      if (err instanceof SyntaxError) return null; // not ours (another policy's blob)
      throw err;
    }
  }
  putState(blob: unknown): Promise<void> {
    return this.request("PUT", "/state", blob);
  }
  event(level: string, message: string): Promise<void> {
    return this.request("POST", "/events", { level, message: message.slice(0, 4000) });
  }
}

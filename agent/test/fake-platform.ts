// An in-memory LLM AutoTune for tests: the policy session API over real HTTP,
// with launches that come up (or fail) on the next poll and benchmarks whose
// score is a deterministic function of the config. No GPUs, no network.

import http from "node:http";
import type { AddressInfo } from "node:net";

type Config = Record<string, unknown>;

export interface FakeOptions {
  space?: Record<string, unknown>;
  production?: Config;
  forbidden?: string[];
  cards?: number;
  ports?: number;
  searchMinutes?: number;
  heartbeatIntervalS?: number;
  params?: Record<string, unknown>;
  state?: unknown;
  sessionId?: number;
}

/** Throughput as a function of the config. mem_fraction_static above 0.93 OOMs. */
export function score(config: Config): number {
  const mem = Number(config.mem_fraction_static ?? 0.85);
  const chunk = Number(config.chunked_prefill_size ?? 4096);
  return 1000 + (mem - 0.85) * 4000 + (chunk === 8192 ? 150 : 0);
}

/** A slice of the platform's measure catalogue, as the manifest carries it. */
export const CATALOGUE = [
  { name: "output_tpm_per_gpu", label: "Output tokens/min per GPU", unit: "tok/min/GPU", better: "higher", group: "throughput", meaning: "Generated tokens per minute, per GPU the config occupies.", stats: [] },
  { name: "request_output_tps", label: "Per-request output speed", unit: "tok/s", better: "higher", group: "per request", meaning: "One request's output tokens per second.", stats: ["mean", "p10", "p50", "p90"] },
  { name: "ttft_ms", label: "Time to first token", unit: "ms", better: "lower", group: "latency", meaning: "Time from sending a request to its first token.", stats: ["mean", "p50", "p90", "p99"] },
  { name: "success_rate", label: "Success rate", unit: "", better: "higher", group: "health", meaning: "Share of requests that succeeded.", stats: [] },
  { name: "error_rate", label: "Error rate", unit: "", better: "lower", group: "health", meaning: "Share of requests that errored.", stats: [] },
  { name: "requests_completed", label: "Requests completed", unit: "requests", better: null, group: "health", meaning: "Requests that completed.", stats: [] },
  { name: "concurrency", label: "Concurrency", unit: "requests", better: null, group: "workload", meaning: "Requests in flight.", stats: [] },
];

/** A config's replay result in measures. tp_size 8 drops a tenth of its requests. */
export function replayMeasures(config: Config, value: number) {
  const success = Number(config.tp_size ?? 1) === 8 ? 0.9 : 1;
  return [{
    scenario: "replay", module: "replay", kind: "replay", operating_level: null,
    operating_rule: "the replay itself, at the concurrency it was run at",
    operating: {
      output_tpm_per_gpu: value * 60, "request_output_tps@mean": value / 16, "request_output_tps@p10": value / 20,
      "ttft_ms@p50": 200, "ttft_ms@p99": 900, "ttft_ms@p90[32k_64k]": 2500, success_rate: success,
      error_rate: 1 - success, requests_completed: 400 * success, concurrency: 16,
    },
    points: {},
    supports: [],
  }];
}

export class FakePlatform {
  readonly heartbeats: Record<string, unknown>[] = [];
  readonly events: { level: string; message: string }[] = [];
  readonly plans: Record<string, unknown>[] = [];
  readonly launches = new Map<number, { config: Config; status: string; polls: number; released: boolean; gpus: number[]; port: number }>();
  readonly benches = new Map<number, { launch: number; polls: number; status: string }>();
  contenders: Record<string, unknown>[] = [];
  state: unknown;
  finalized = false;
  command = "run";
  private nextId = 1;
  private server!: http.Server;
  url = "";

  constructor(private readonly opts: FakeOptions = {}) {
    this.state = opts.state;
  }

  async start(): Promise<void> {
    this.server = http.createServer((req, res) => this.handle(req, res));
    await new Promise<void>((r) => this.server.listen(0, "127.0.0.1", () => r()));
    this.url = `http://127.0.0.1:${(this.server.address() as AddressInfo).port}`;
  }

  async stop(): Promise<void> {
    await new Promise<void>((r) => this.server.close(() => r()));
  }

  get env(): NodeJS.ProcessEnv {
    return {
      AUTOTUNE_API_URL: this.url,
      AUTOTUNE_API_KEY: "atk_test",
      AUTOTUNE_SESSION_ID: String(this.opts.sessionId ?? 7),
      LLM_BASE_URL: "http://unused.invalid/v1",
      LLM_MODEL: "scripted",
      AGENT_POLL_SECONDS: "0.02",
    };
  }

  private canonical(config: Config): Config {
    const base = ((this.opts.space ?? {}) as { base?: Config }).base ?? {};
    return { ...base, ...config };
  }

  private check(config: Config) {
    const problems: { kind: string; param: string; message: string }[] = [];
    const canonical = this.canonical(config);
    for (const [key, value] of Object.entries(canonical)) {
      if (["port", "host", "model_path"].includes(key)) problems.push({ kind: "placement", param: key, message: `${key} is placement` });
      if ((this.opts.forbidden ?? ["quantization*"]).some((p) => new RegExp(`^${p.replace("*", ".*")}$`).test(key))) {
        problems.push({ kind: "forbidden", param: key, message: `${key} is forbidden for this campaign` });
      }
      if (Array.isArray(value) || (value && typeof value === "object")) problems.push({ kind: "not_renderable", param: key, message: `${key} is a list` });
    }
    const space = (this.opts.space ?? {}) as { grid?: Record<string, unknown>; base?: Config };
    const declared = new Set([...Object.keys(space.grid ?? {}), ...Object.keys(space.base ?? {})]);
    const cards = Number(canonical.tp ?? canonical.tp_size ?? 1);
    return {
      ok: problems.length === 0,
      config: canonical,
      cards,
      command: Object.entries(canonical).flatMap(([k, v]) => [`--${k.replaceAll("_", "-")}`, String(v)]),
      problems,
      warnings: [],
      deviations: Object.keys(canonical).filter((k) => !declared.has(k)).map((param) => ({ param, kind: "unknown", value: canonical[param] })),
    };
  }

  private manifest() {
    const now = Date.now();
    const cards = this.opts.cards ?? 4;
    return {
      contract: { version: "1.3" },
      session_id: this.opts.sessionId ?? 7,
      campaign_id: 3,
      policy: { name: "agent", image: "agent:test", version: "0.1.0", params: this.opts.params ?? {} },
      model: { engine: "sglang", image: "sglang:test", model_path: "/model", served_model_name: "m", extra_env: {}, extra_volumes: {} },
      hardware: {
        machine: "gpu-test", gpu_type: "H100", gpu_indices: Array.from({ length: cards }, (_, i) => i),
        ports: Array.from({ length: this.opts.ports ?? 4 }, (_, i) => 28200 + i), gpus_visible_in_container: false, pool: false, node_count: 1, node_gpu_max: cards,
      },
      objective: {
        screen: { target_metric: "perf.output_tps", direction: "max", redlines: [] },
        verify: { target_metric: "replay.output_tps", direction: "max", redlines: [] },
      },
      search_space: this.opts.space ?? {},
      space_policy: { deviations: "allowed_if_declared", forbidden: this.opts.forbidden ?? ["quantization*"] },
      production: { engine_args: this.opts.production ?? {}, launch_command: "" },
      time: {
        started_at: new Date(now).toISOString(),
        search_deadline: new Date(now + (this.opts.searchMinutes ?? 120) * 60000).toISOString(),
        hard_deadline: new Date(now + 180 * 60000).toISOString(),
        heartbeat_interval_s: this.opts.heartbeatIntervalS ?? 0.1,
        heartbeat_timeout_s: 180,
        tick_s: 10,
      },
      contenders: { max: 2, validation_suite: "verify", approx_minutes_each: 30 },
      services: {
        launch: true,
        benchmarks: {
          screen: {
            slug: "screen", approx_minutes: 15, dataset_build_id: "",
            workload: [{ scenario: "replay", module: "replay", family: "replay", params: { concurrency: 16 }, reads_as: "Recorded production requests, replayed at one fixed concurrency: there are no levels." }],
          },
        },
        state: true,
        diagnostics: true,
      },
      prior: { contenders: [] },
      measures: CATALOGUE,
    };
  }

  private launchOut(id: number) {
    const l = this.launches.get(id)!;
    return {
      id, status: l.released ? "released" : l.status, endpoint_url: "", launch_command: "", failure_class: l.status === "failed" ? (l.config.bad_flag ? "bad_config" : "oom") : "",
      error: l.status === "failed" ? "engine exited" : "", config: l.config, deviations: [], waiting_reason: "", waiting_since: null,
    };
  }

  private benchOut(id: number) {
    const b = this.benches.get(id)!;
    const config = this.launches.get(b.launch)!.config;
    const value = score(config);
    return {
      id, status: b.status, suite: "screen", failure_class: "", error: "", config, deviations: [],
      summary: b.status === "succeeded" ? { objective_value: value, feasible: true, constraints: [], breaches: [] } : null,
      metrics: b.status === "succeeded" ? { "perf.output_tps": value } : {},
      measures: b.status === "succeeded" ? replayMeasures(config, value) : [],
    };
  }

  private handle(req: http.IncomingMessage, res: http.ServerResponse): void {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const send = (status: number, body: unknown, text = false) => {
        res.writeHead(status, { "content-type": text ? "text/plain" : "application/json" });
        res.end(text ? String(body) : JSON.stringify(body));
      };
      if (req.headers["x-api-key"] !== "atk_test") return send(401, { detail: "bad token" });
      const url = new URL(req.url ?? "/", "http://x");
      const path = url.pathname.replace("/api/policy/v1", "");
      const body = raw ? JSON.parse(raw) : {};
      const m = (re: RegExp) => re.exec(path);
      let match: RegExpExecArray | null;

      if (req.method === "GET" && path === "/session") return send(200, this.manifest());
      if (req.method === "POST" && path === "/session/heartbeat") {
        this.heartbeats.push(body);
        if (body.status === "exhausted" && this.command === "run") this.command = "finalize";
        if (this.finalized && this.command === "finalize") this.command = "exit";
        return send(200, { command: this.command, search_deadline: null, hard_deadline: null });
      }
      if (req.method === "PUT" && path === "/session/plan") {
        this.plans.push(body);
        return send(204, "");
      }
      if (req.method === "POST" && path === "/session/finalized") {
        this.finalized = true;
        return send(204, "");
      }
      if (req.method === "POST" && path === "/configs/check") return send(200, this.check(body.engine_args ?? {}));
      if (req.method === "POST" && path === "/launches") {
        const check = this.check(body.engine_args ?? {});
        if (!check.ok) return send(422, { detail: check.problems.map((p) => p.message).join("; ") });
        const id = this.nextId++;
        this.launches.set(id, { config: check.config, status: "launching", polls: 0, released: false, gpus: body.gpu_indices, port: body.port });
        return send(201, this.launchOut(id));
      }
      if ((match = m(/^\/launches\/(\d+)$/))) {
        const id = Number(match[1]);
        const l = this.launches.get(id);
        if (!l) return send(404, { detail: "no such run" });
        if (req.method === "DELETE") {
          l.released = true;
          return send(200, this.launchOut(id));
        }
        l.polls += 1;
        if (l.status === "launching" && l.polls >= 1) {
          l.status = Number(l.config.mem_fraction_static ?? 0) > 0.93 || l.config.bad_flag ? "failed" : "serving";
        }
        return send(200, this.launchOut(id));
      }
      if ((match = m(/^\/launches\/(\d+)\/log$/))) {
        const l = this.launches.get(Number(match[1]))!;
        const log = l.config.bad_flag
          ? "loading\nsglang: error: unrecognized arguments: --bad-flag\n"
          : l.status === "failed"
            ? "loading weights\ncapturing cuda graphs\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
            : "loading weights\nDecode batch. #running-req: 60, token usage: 0.71\n";
        return send(200, log, true);
      }
      if ((match = m(/^\/launches\/(\d+)\/environment$/))) {
        return send(200, { kind: "environment", schema_version: 1, run_id: Number(match[1]), machine: { name: "gpu-test" }, engine_version: "0.5-test", image_digest: "sglang@sha256:abc", snapshot: {} });
      }
      if ((match = m(/^\/launches\/(\d+)\/benchmarks$/))) {
        const id = this.nextId++;
        this.benches.set(id, { launch: Number(match[1]), polls: 0, status: "benching" });
        return send(201, this.benchOut(id));
      }
      if ((match = m(/^\/benchmarks\/(\d+)$/))) {
        const b = this.benches.get(Number(match[1]))!;
        b.polls += 1;
        if (b.polls >= 1) b.status = "succeeded";
        return send(200, this.benchOut(Number(match[1])));
      }
      if ((match = m(/^\/benchmarks\/(\d+)\/results$/))) {
        const out = this.benchOut(Number(match[1]));
        const v = out.summary?.objective_value ?? 0;
        return send(200, {
          kind: "results", schema_version: 1, run_id: out.id, status: "measured", objective_value: v,
          headline: { output_tpm_card_norm: v * 60 },
          scenarios: [{ key: "sweep", kind: "sweep", levels: [1, 16, 64].map((c) => ({ concurrency: c, metrics: { output_tps: (v * c) / 64, ttft_p99_ms: 100 + c * 5 } })), best_level: { concurrency: 64, output_tps: v } }],
          metrics: {},
        });
      }
      if (req.method === "GET" && path === "/trials") {
        // One platform trial per benchmark it measured, numbered apart from
        // the agent's own trial numbers.
        return send(200, [...this.benches.entries()].filter(([, b]) => b.status === "succeeded").map(([id], i) => ({
          id: 1000 + id, seq: i + 1, config: {}, config_hash: "", source: "platform", run_id: id,
          reported_metrics: {}, reported_objective_value: null, reported_feasible: null, failure_class: "", duration_seconds: null, notes: "",
        })));
      }
      if (req.method === "PUT" && path === "/contenders") {
        this.contenders = body.contenders;
        return send(200, { accepted: body.contenders, rejected: [] });
      }
      if (req.method === "GET" && path === "/state") {
        return this.state === undefined ? send(404, { detail: "none" }) : send(200, this.state);
      }
      if (req.method === "PUT" && path === "/state") {
        this.state = body;
        return send(204, "");
      }
      if (req.method === "POST" && path === "/events") {
        this.events.push(body);
        return send(204, "");
      }
      return send(404, { detail: `fake has no ${req.method} ${path}` });
    });
  }
}

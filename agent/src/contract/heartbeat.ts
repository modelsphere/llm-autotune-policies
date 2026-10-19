// The heartbeat runs on its own timer, never inside the agent's work: a model
// that thinks for five minutes must not make the platform think we are wedged.
// Its reply is the platform's command channel (run | finalize | serve | exit |
// abort); the latest one is kept here for the night loop to read.

import { type HeartbeatBody, type PolicyClient, SessionOver } from "./client.js";
import { CONTRACT_VERSION, type HeartbeatOut } from "./types.js";

export type Command = "run" | "finalize" | "serve" | "exit" | "abort";

export interface HeartbeatStatus {
  phase: string;
  message: string;
  progress?: number;
  /** "exhausted" | "error" | "" — the v1.1 machine-read signal. */
  status: string;
  failureClass?: string;
}

export class Heartbeat {
  readonly current: HeartbeatStatus = { phase: "starting", message: "", status: "" };
  last: HeartbeatOut | null = null;
  private timer: NodeJS.Timeout | null = null;
  private inFlight: Promise<HeartbeatOut> | null = null;
  private listeners: ((hb: HeartbeatOut) => void)[] = [];

  constructor(
    private readonly client: PolicyClient,
    private readonly intervalMs: number,
    private readonly sdkVersion: string,
    private readonly log: (line: string) => void = () => {},
  ) {}

  get command(): Command {
    return ((this.last?.command as Command) ?? "run");
  }

  onCommand(listener: (hb: HeartbeatOut) => void): void {
    this.listeners.push(listener);
  }

  /** The first beat, with the capability handshake. We declare no
   *  capabilities: this policy never serves an engine itself, and every
   *  contender it names is on the campaign's own image, which the platform
   *  launches for validation without us. */
  handshake(): Promise<HeartbeatOut> {
    return this.beat({
      sdk_version: this.sdkVersion,
      contract_version: CONTRACT_VERSION,
      capabilities: [],
    });
  }

  start(): void {
    if (this.timer) return;
    this.timer = setInterval(() => {
      if (this.inFlight) return; // never stack beats behind a slow platform
      this.beat().catch((err) => this.log(`heartbeat failed: ${(err as Error).message}`));
    }, this.intervalMs);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  /** Beat now (e.g. to deliver a status signal at once). */
  async beat(extra: Partial<HeartbeatBody> = {}): Promise<HeartbeatOut> {
    const body: HeartbeatBody = {
      phase: this.current.phase,
      message: this.current.message.slice(0, 4000),
      status: this.current.status || undefined,
      failure_class: this.current.failureClass || undefined,
      ...extra,
    };
    if (this.current.progress !== undefined) body.progress = Math.max(0, Math.min(1, this.current.progress));
    const run = (async () => {
      try {
        return await this.client.heartbeat(body);
      } catch (err) {
        // A terminal session reads as "exit": the platform is done with us.
        if (err instanceof SessionOver) {
          return { command: "exit", search_deadline: null, hard_deadline: null } as HeartbeatOut;
        }
        throw err;
      }
    })();
    this.inFlight = run;
    try {
      const hb = await run;
      const changed = hb.command !== this.last?.command;
      this.last = hb;
      if (changed) this.log(`platform command: ${hb.command}`);
      for (const listener of this.listeners) listener(hb);
      return hb;
    } finally {
      this.inFlight = null;
    }
  }
}


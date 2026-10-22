// Drives one pi agent through the night.
//
// pi-agent-core runs the model ↔ tool loop. This adds what a night needs:
//   - context: before every request the lab record is rebuilt from the books
//     and sent as the newest message; only the conversation before it is ever
//     trimmed, and the model is asked to write down what it learned first;
//   - events: lab news (a trial finished, a slot freed) reaches a running
//     agent through steer() and an idle one through followUp() + continue();
//   - resilience: a failed model call is retried with pi-ai's own classifier
//     and backoff; a context overflow trims harder; a model unreachable for
//     too long is an outage the night loop answers by finalizing.

import { Agent, type AgentEvent, type AgentMessage, type AgentTool, type ThinkingLevel } from "@earendil-works/pi-agent-core";
import {
  type AssistantMessage,
  isContextOverflow,
  isRetryableAssistantError,
  type Message,
  type Model,
  type Models,
  retryDelayMs,
} from "@earendil-works/pi-ai";

export interface DriverOptions {
  models: Models;
  model: Model<"openai-completions">;
  systemPrompt: string;
  tools: AgentTool<any>[];
  thinking: ThinkingLevel;
  contextWindow: number;
  maxOutputTokens: number;
  outageMinutes: number;
  /** Per-request timeout and in-call retries for the model endpoint. */
  requestTimeoutMs?: number;
  requestRetries?: number;
  /** The lab record, rendered fresh. */
  record(): string;
  log(line: string): void;
  /** Called with every assistant message's token usage. */
  onUsage?(tokens: number): void;
  sleep?(ms: number): Promise<void>;
}

const STANDARD_ROLES = new Set(["system", "user", "assistant", "toolResult"]);

/** Rough token count; only used to decide when to trim the conversation. */
export function estimateTokens(value: unknown): number {
  return Math.ceil(JSON.stringify(value).length / 4);
}

/** The conversation without the reasoning of all but the latest assistant
 *  turn. What a turn concluded is in its text, its tool calls and the lab
 *  record; its thinking is long, and kept for every turn it fills the window
 *  within a few turns and forces the conversation to be trimmed. */
export function withoutOldReasoning(messages: AgentMessage[]): AgentMessage[] {
  const last = messages.map((m) => m.role).lastIndexOf("assistant");
  return messages.map((m, i) => {
    if (m.role !== "assistant" || i === last) return m;
    const content = (m as AssistantMessage).content.filter((block) => block.type !== "thinking");
    return { ...(m as AssistantMessage), content } as AgentMessage;
  });
}

export class AgentDriver {
  readonly agent: Agent;
  tokens = 0;
  outage = false;
  private running: Promise<void> | null = null;
  private stopped = false;
  private firstErrorAt: number | null = null;
  private retries = 0;
  private warnedFull = false;
  /** Lab events since the last "conversation nearly full" warning: with
   *  nothing new there is nothing new to write down, and warning again only
   *  makes the model answer the warning. */
  private newsSinceWarning = 0;
  private trimHarder = false;
  private readonly sleep: (ms: number) => Promise<void>;

  constructor(private readonly opts: DriverOptions) {
    this.sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    this.agent = new Agent({
      initialState: { systemPrompt: opts.systemPrompt, model: opts.model, tools: opts.tools, thinkingLevel: opts.thinking },
      // A throttled endpoint or a dropped connection is retried inside the
      // call first; a call that still fails is retried by afterRun, and only
      // a failure lasting outageMinutes ends the search.
      streamFn: (model, context, options) =>
        opts.models.streamSimple(model, context, {
          ...options,
          timeoutMs: opts.requestTimeoutMs,
          maxRetries: opts.requestRetries,
        }),
      convertToLlm: (messages) => messages.filter((m) => STANDARD_ROLES.has(m.role)) as Message[],
      transformContext: async (messages) => this.contextFor(messages),
      // Tools change the lab's books; one at a time keeps every result
      // consistent with the record the next request is built from.
      toolExecution: "sequential",
      // Several events at once (two trials landing together) arrive as one batch.
      steeringMode: "all",
      followUpMode: "all",
    });
    this.agent.subscribe((event) => this.observe(event));
  }

  get busy(): boolean {
    return this.running !== null;
  }

  /** The night's first message. */
  start(text: string): void {
    this.running = this.loop(() => this.agent.prompt(text));
  }

  /** Lab news. A running agent sees it after its current tool calls; an idle
   *  one is woken with it. */
  notify(text: string): void {
    if (this.stopped) return;
    this.newsSinceWarning += 1;
    const message: AgentMessage = { role: "user", content: `[lab event ${new Date().toISOString()}] ${text}`, timestamp: Date.now() };
    if (this.running) this.agent.steer(message);
    else {
      this.agent.followUp(message);
      this.running = this.loop(() => this.agent.continue());
    }
  }

  async stop(): Promise<void> {
    this.stopped = true;
    this.agent.clearAllQueues();
    this.agent.abort();
    // A provider that ignores the abort must not hold up the finalize: wait a
    // little for the run to wind down, then carry on without it.
    const settle = this.running?.catch(() => undefined) ?? Promise.resolve();
    await Promise.race([settle, new Promise((r) => setTimeout(r, 1500))]);
  }

  async idle(): Promise<void> {
    while (this.running) await this.running.catch(() => undefined);
  }

  private async loop(first: () => Promise<void>): Promise<void> {
    try {
      let next: (() => Promise<void>) | null = first;
      while (next && !this.stopped) {
        await next();
        next = await this.afterRun();
      }
    } catch (err) {
      this.opts.log(`agent run failed: ${(err as Error).message}`);
    } finally {
      this.running = null;
      // News that arrived while the run was winding down must not be stranded.
      if (!this.stopped && !this.outage && this.agent.hasQueuedMessages()) {
        this.running = this.loop(() => this.agent.continue());
      }
    }
  }

  /** Decide what follows a finished run: a retry, the queued news, or nothing. */
  private async afterRun(): Promise<(() => Promise<void>) | null> {
    const last = this.agent.state.messages.at(-1) as AssistantMessage | undefined;
    if (last && last.role === "assistant" && last.stopReason === "error") {
      if (isContextOverflow(last, this.opts.contextWindow)) {
        this.opts.log("context overflow: trimming the conversation harder and retrying");
        this.trimHarder = true;
      } else {
        const nowMs = Date.now();
        this.firstErrorAt ??= nowMs;
        if (nowMs - this.firstErrorAt > this.opts.outageMinutes * 60000) {
          this.outage = true;
          this.opts.log(`the model has failed for ${this.opts.outageMinutes} min: ${last.errorMessage}`);
          return null;
        }
        this.retries += 1;
        const delay = retryDelayMs({ baseDelayMs: 2000, maxAgentDelayMs: 60000 }, Math.min(this.retries, 6));
        this.opts.log(
          `model call failed (${isRetryableAssistantError(last) ? "transient" : "not transient"}): ${last.errorMessage}; retrying in ${Math.round(delay / 1000)}s`,
        );
        await this.sleep(delay);
      }
      // Drop the failed reply; the tail is the user/tool message again.
      this.agent.state.messages = this.agent.state.messages.slice(0, -1);
      return () => this.agent.continue();
    }
    if (last?.role === "assistant" && last.stopReason !== "aborted") {
      this.firstErrorAt = null;
      this.retries = 0;
    }
    if (this.agent.hasQueuedMessages()) return () => this.agent.continue();
    return null;
  }

  /** The request context: system prompt, the conversation (trimmed only when
   *  it must be), and the freshly rendered lab record as the newest message. */
  private contextFor(messages: AgentMessage[]): AgentMessage[] {
    const record: AgentMessage = { role: "user", content: this.opts.record(), timestamp: Date.now() };
    const budget = this.opts.contextWindow - this.opts.maxOutputTokens - estimateTokens(record) - 2000;
    const system = messages.filter((m) => m.role === "system");
    let rolling = withoutOldReasoning(messages.filter((m) => m.role !== "system"));
    const size = estimateTokens(rolling);
    if (!this.warnedFull && this.newsSinceWarning > 0 && size > budget * 0.7) {
      this.warnedFull = true;
      this.newsSinceWarning = 0;
      this.agent.steer({
        role: "user",
        content:
          "[harness] The conversation is nearly full. Older turns will soon be dropped (the lab record is never dropped). Record now, with record_lesson / add_constraint / close_hypothesis, anything you learned that is not yet in the record.",
        timestamp: Date.now(),
      });
    }
    const limit = this.trimHarder ? budget * 0.4 : budget * 0.85;
    if (size > limit) {
      const target = this.trimHarder ? budget * 0.3 : budget * 0.6;
      while (rolling.length > 1 && estimateTokens(rolling) > target) {
        rolling = rolling.slice(1);
        // Never start on a tool result or an assistant turn: cut to a user message.
        const start = rolling.findIndex((m) => m.role === "user");
        rolling = start === -1 ? [] : rolling.slice(start);
      }
      this.trimHarder = false;
      this.warnedFull = false;
      this.opts.log(`trimmed the conversation to ${rolling.length} message(s)`);
    }
    return [...system, ...rolling, record];
  }

  private observe(event: AgentEvent): void {
    if (event.type === "message_end" && event.message.role === "assistant") {
      const message = event.message as AssistantMessage;
      const usage = message.usage?.totalTokens || (message.usage?.input ?? 0) + (message.usage?.output ?? 0);
      if (usage) {
        this.tokens += usage;
        this.opts.onUsage?.(usage);
      }
      for (const block of message.content) {
        if (block.type === "thinking" && block.thinking) this.opts.log(`[thinking] ${block.thinking.slice(0, 2000)}`);
        if (block.type === "text" && block.text) this.opts.log(`[agent] ${block.text.slice(0, 4000)}`);
        if (block.type === "toolCall") this.opts.log(`[tool call] ${block.name} ${JSON.stringify(block.arguments).slice(0, 2000)}`);
      }
      if (message.stopReason === "error") this.opts.log(`[model error] ${message.errorMessage}`);
    }
    if (event.type === "tool_execution_end") {
      const body = event.result?.content?.map((c: { type: string; text?: string }) => (c.type === "text" ? c.text : "")).join("") ?? "";
      this.opts.log(`[tool ${event.isError ? "error" : "result"}] ${event.toolName}: ${body.slice(0, 1500)}`);
    }
  }
}

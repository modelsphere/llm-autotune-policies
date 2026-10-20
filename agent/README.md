# agent — an LLM tunes the engine

A **policy** for [LLM AutoTune](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md)
in which a language model decides what to try. It reads every result, every
failed launch and every lesson from earlier nights, forms hypotheses, and
proposes complete engine configurations — any engine flag the campaign does
not forbid, not only the ones in the declared search space. The platform still
launches, measures and judges every config; the model never scores anything.

It is written in TypeScript on [pi](https://github.com/earendil-works/pi)
(`@earendil-works/pi-agent-core` and `pi-ai`, MIT), and needs platform
contract **1.3** (diagnostics, `POST /configs/check`, forbidden parameters).

## How a night goes

1. The production config is measured first, twice and side by side, as the
   **anchor** every result is compared with; the pair is the first estimate of
   the measurement noise.
2. The model reads the **lab record**, opens hypotheses, and fills every free
   slot with experiments. Trials run in parallel on disjoint cards.
3. Each config passes the platform's dry run before it can cost a launch:
   forbidden parameters, the command-line round trip, safety. A refusal goes
   straight back to the model with the reason.
4. When a trial finishes the model is woken with the result; a failed launch
   arrives with its log excerpt and a diagnosis.
5. A result suspiciously better than the best so far is benchmarked again on
   the same engine before anyone trusts it. Near the end the finalists are
   re-measured, so the configs the platform validates rest on repeats.
6. The best feasible configs are kept registered as contenders after every
   trial, chosen in code by their mean moved one standard error toward worse
   (the noise pooled over every repeat), so a single lucky measurement does
   not outrank a config measured twice at nearly the same level. At the deadline, or when the
   model says it is done, the night finalizes and the platform validates them.
7. Lessons, constraints, settled hypotheses and the night's ledger are saved
   with `PUT /state` for the next night.

## What pi does, and what this adds

pi runs the agent loop: the model ↔ tool cycle, tool-argument validation,
streaming, provider differences, abort, steering and follow-up queues, and the
retry/overflow classifiers. This directory adds the parts that make it a tuning
lab rather than a chat:

| Part | File | What it does |
| --- | --- | --- |
| Contract client and heartbeat | `src/contract/` | The policy API, typed from the platform's OpenAPI; a heartbeat on its own timer, so a model thinking for minutes never looks wedged. |
| Lab state and statistics | `src/lab/` | The trial ledger, hypotheses, constraints, lessons; noise bands, the anomaly check, contender ranking. |
| Lab record | `src/context/render.ts` | Rebuilt before every model request from the books and sent as the newest message: every trial with per-concurrency numbers, every diagnosed failure, constraints, lessons, earlier nights, the time left. Never trimmed. |
| Driver | `src/driver.ts` | Wraps pi's `Agent`: the record via `transformContext`, lab events via `steer`/`followUp`, turn retries, trimming only the conversation (after asking the model to write down what it learned). |
| Tools | `src/tools.ts` | The model's only verbs: `propose`, `check`, `inspect_trial`, `read_log`, `compare`, `remeasure`, `cancel_trial`, `open_hypothesis`, `close_hypothesis`, `add_constraint`, `record_lesson`, `done`. No shell, no files. |
| Runner | `src/runner.ts` | Places trials on cards and ports, runs launch → benchmark → results → release, captures the engine's environment and resolved arguments, diagnoses failures. |
| Night | `src/night.ts` | The lifecycle: plan (with the parameters the model introduced, declared as extras), anchor, nudges when cards sit idle, confirmation, contenders, state, finalize. |
| Prompts | `prompts/` | The system prompt (the search protocol) and the diagnostician. The main tuning surface. |

## Configuration

The LLM is any OpenAI-compatible chat endpoint the policy container can reach
(vLLM, SGLang, LiteLLM, a hosted API). Register the policy with its endpoint in
`env` and its key in `secret_env` (stored encrypted, never shown, redacted from
the launch command):

```bash
curl -X POST https://<autotune>/api/policies -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{
    "name": "agent", "image": "<registry>/llm-autotune-policy-agent:0.1.0", "version": "0.1.0",
    "env": {"LLM_BASE_URL": "https://llm.example.com/v1", "LLM_MODEL": "my-model"},
    "secret_env": {"LLM_API_KEY": "sk-…"}}'
```

| Variable | Default | |
| --- | --- | --- |
| `LLM_BASE_URL` | — (required) | OpenAI-compatible base URL, ending in `/v1`. |
| `LLM_MODEL` | — (required) | Model id at that endpoint. |
| `LLM_API_KEY` | none | Put it in `secret_env`. |
| `LLM_REASONING` | `true` | Whether the model streams reasoning (`reasoning_content`). |
| `LLM_THINKING` | `high` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. |
| `LLM_CONTEXT_WINDOW` | `128000` | Tokens; the conversation is trimmed to fit, the record never. |
| `LLM_MAX_OUTPUT_TOKENS` | `16000` | |
| `LLM_COMPAT` | `{}` | JSON: pi-ai's [OpenAI compatibility flags](https://github.com/earendil-works/pi/tree/main/packages/ai#openai-compatibility-settings) for this endpoint, e.g. `{"thinkingFormat": "qwen-chat-template"}`. The defaults suit self-hosted servers (no `developer` role, no `reasoning_effort`). |
| `AGENT_POLL_SECONDS` | `10` | How often launches and benchmarks are polled. |

How much runs at once is the campaign's **GPU budget** (`policy_settings.cards`)
and the policy's ports: each trial takes the cards its config uses.

A campaign can override these per campaign in its policy settings, under
Advanced → **Policy parameters** (`policy.params` in the manifest):

| Parameter | Default | |
| --- | --- | --- |
| `model`, `base_url`, `thinking`, `reasoning`, `context_window`, `max_output_tokens`, `compat` | from env | As above. |
| `max_tokens_per_night` | `0` (no limit) | Stop asking the model after this many tokens. |
| `anomaly_threshold` | `0.15` | Re-bench a result this much better than the best so far. |
| `confirm_top_k` | `2` | Finalists re-measured near the end. |
| `outage_minutes` | `20` | Finalize if the model stays unreachable this long. |
| `nudge_limit` | `3` | Reminders with idle cards and no proposal before the search is called done. |

The campaign's forbidden-parameter list (Advanced → *Parameters the policy may
not change*) is what keeps the model from "winning" by trading quality away —
lower-precision KV cache, a shorter context, a dropped parser. Keep it on.

## Check the endpoint first

`dist/src/check.js` (also `autotune-agent-check`) sends one request through pi
with a tool attached, using the same `LLM_*` variables, and prints what came
back (never the key): whether the model answered with a tool call, whether it
streamed reasoning, the latency. Exit 0 means tool calls work.

```bash
docker run --rm -e LLM_BASE_URL=… -e LLM_MODEL=… -e LLM_API_KEY=… <image> node dist/src/check.js
```

## Develop and test

```bash
npm ci
npm test          # typecheck + full offline nights: a fake platform, a scripted model
npm run licenses  # THIRD_PARTY_LICENSES; fails on a copyleft dependency
```

`test/night.test.ts` drives whole nights through the real HTTP contract with a
scripted model (pi-ai's faux provider): refusals, a diagnosed OOM, the anomaly
re-bench, confirmation, contenders, state carried to a second night, the
heartbeat during a stalled model call, a finalize mid-call.

When the platform's policy API changes, regenerate the types:
`contract/policy-api.openapi.json` is the API's `/api/policy/v1` paths from the
platform's OpenAPI, and `npm run gen:contract` writes `src/contract/api.d.ts`.

## Building an image

```bash
docker build -t <registry>/llm-autotune-policy-agent:0.1.0 agent/
# behind a mirror:
docker build --build-arg NPM_REGISTRY=https://<npm-mirror>/ --build-arg BASE=<registry>/node:22-slim …
```

Unlike the Python policies this one has dependencies, so the build needs an
npm registry (or a mirror); the running container needs only the platform and
the LLM endpoint.

## Limits

- The model sees the engine's arguments through `check` and, for engines that
  report them, the resolved values of a running launch; there is no list of
  every flag the engine image accepts yet, so a misspelled flag still costs a
  launch (and comes back diagnosed).
- Every config is launched on the campaign's engine image; the agent does not
  change images.

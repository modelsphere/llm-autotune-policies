# random-search — reference policy + template

A complete, working **policy** for [LLM AutoTune](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md):
a container that runs a night of tuning on a machine the platform manages. It
does random search over the campaign's declared space — a baseline every smarter
search should beat — and it doubles as the **fork-and-edit starting point** for a
new policy.

The whole policy is one line (`main.py`):

```python
run_policy(SearchLoop(RandomSearch()))
```

`run_policy` reads the boot env, `SearchLoop` runs the entire contract (heartbeat
on cadence, delegated launches, screen benchmarks, finalize, serve-for-
validation, exit), and `RandomSearch` — the only part you rewrite — decides what
to try.

## Layout

```
autotune_policy/        the SDK (stdlib-only, no deps) — you don't edit this
  client.py             one method per contract endpoint
  loop.py               SearchLoop: the standard night
  strategy.py           the Strategy interface you implement
  space.py              sample the {base,grid,tied,range,conditions} space
  run.py                run_policy()
strategies/
  random_search.py      the reference algorithm  ← copy this to start
main.py                 entrypoint: wires a strategy into the loop
Dockerfile              tiny image (stdlib SDK ⇒ nothing to install)
tests/                  a full fake night, offline
```

`autotune_policy/` is vendored, not installed: it is stdlib-only precisely so a
policy image needs no wheels and no reachable index. The copy of record is
[`autotune_policy/`](../autotune_policy/) at the top of this repository, and CI
checks that every policy's copy still matches it.

## Write your own policy

1. Copy this directory into a repository of your own, keep `autotune_policy/` as it is.
2. Copy `strategies/random_search.py` to `strategies/your_algo.py` and implement
   the five methods of `Strategy`:
   - `setup(manifest)` — read `search_space`, `objective`, `hardware`, `prior`.
   - `plan()` — declare what you'll tune (and any out-of-space `extras`).
   - `next_config()` — the next engine-args dict to measure, or `None` when done
     (returning `None` auto-signals `exhausted`; the platform finalizes you now
     instead of waiting out the clock).
   - `observe(result)` — record the platform's measurement (a failed launch is a
     signal, not an error).
   - `contenders()` — your best-so-far as `build_contender(...)` entries.
   Optionally `status()`, `coverage()`, `load_state()`/`dump_state()` for
   warm-start.
3. Point `main.py` at your strategy.

See [BEST_PRACTICES.md](../BEST_PRACTICES.md) for the dos and don'ts.

You never touch `loop.py` or `client.py` unless you need something exotic
(serving your own engines, many concurrent launches) — subclass `SearchLoop`
then.

## What the SDK guarantees for you

- **Dial-out only**: the platform never connects into your container; commands
  arrive on the heartbeat response.
- **The clock is the platform's**: `finalize` on the heartbeat means stop; the
  loop tears engines down, freezes contenders, and moves to validation.
- **Crash-safe**: whatever `contenders()` last returned is what gets validated
  if you die — the loop re-PUTs it after every measurement. An uncaught
  exception in your strategy becomes an explicit `error` signal, not silent
  death, so the platform validates what you found instead of timing you out.
- **You can't stall the night**: liveness (heartbeat) and productivity (an idle
  watchdog over the delegated work the platform sees) both wind you down — even
  if you forget to signal `exhausted`, or livelock while still beating.
- **Coverage for free**: the loop reports which part of the space you've tried,
  per axis, built from your measured configs — override `coverage()` only to add
  a completion estimate or deliberate skips.
- **Network-resilient**: the client retries *every* call on a transient blip or
  a platform restart (network error / 502-503-504) with bounded backoff; the
  creating POSTs carry idempotency keys, so a retry after a lost response never
  double-launches. A 4xx is a decision (auth, phase, safety, capacity) and is
  never retried.
- **The verdict is the platform's**, on its pinned verify dataset, of what you
  serve — nothing you report ranks a leaderboard.

## Test

```bash
uv run --extra dev pytest      # a full night against an in-memory platform, no network
uv run --extra dev ruff check autotune_policy/ strategies/ tests/
```

CI runs exactly this on every push, and builds and pushes the image on a
version tag. Because the fake night exercises the whole contract, a green check
means a change to `strategies/` didn't break the protocol — change the algorithm
freely.

## Build & run

The image is stdlib-only, so there's nothing to install:

```bash
docker build -t <registry>/autotune-policy-random-search:0.1.0 .
docker push  <registry>/autotune-policy-random-search:0.1.0
```

Register it as a policy on the platform (name + image), then start a campaign
with `policy_id` set. The platform injects `AUTOTUNE_API_URL`,
`AUTOTUNE_API_KEY`, `AUTOTUNE_SESSION_ID` and the policy takes it from there.

> Restricted network: set `--build-arg BASE=<your-registry>/python:3.12-slim` if
> Docker Hub isn't reachable.

## Contract version

Built against policy contract **1.1** (`/api/policy/v1`) — adds the `exhausted`
/ `error` heartbeat signals, self-reported `coverage`, and the productivity
watchdog. Both sides ignore unknown JSON fields, so new manifest fields and
services are non-breaking, and a 1.0 image still runs unchanged.

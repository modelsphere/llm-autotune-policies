# Writing a policy — best practices

A policy is a search algorithm in a container. The [contract](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md)
is the full specification; this is the shorter "how to do it well" companion.
The one rule that makes everything else easy:

> **You own the decisions. The platform owns the clock, the cards, and the
> verdict.** Everything the SDK does is in service of that split — so you write
> *what to try*, never *how to talk to the platform*.

## The only code you write

Copy [`random-search/`](random-search/),
keep the `autotune_policy/` SDK untouched, and implement a `Strategy`
([`strategy.py`](autotune_policy/strategy.py)):

| method | your job |
| --- | --- |
| `setup(manifest)` | Read the night's shape: `search_space`, `objective`, `hardware`, `prior`. |
| `plan()` | Declare what you'll vary (and any out-of-space `extras`, with a reason). |
| `next_config()` | The next engine-args dict to measure — or `None` when you're done. |
| `observe(result)` | Record the platform's measurement. A failed launch is a *signal*, not an error. |
| `contenders()` | Your best-so-far, best first, as `build_contender(...)` entries. |

Optional: `status()` (a UI line + progress), `coverage()` (only to add what the
loop can't infer — see below), `load_state()`/`dump_state()` (warm-start across
nights). That's the whole surface. `loop.py` and `client.py` are not yours to
edit unless you serve your own engines — then subclass `SearchLoop`.

## What the platform enforces, so you don't have to

These are handled *for* you. Knowing they exist is why you can change the search
logic freely and trust the night still ends correctly:

- **Liveness.** The SDK heartbeats on cadence. If your container dies or wedges,
  the platform fails you *toward validation* — your registered contenders are
  still measured. Don't build your own keep-alive.
- **Productivity (v1.1).** If a searching session sits with nothing running and
  makes no launch/benchmark request for `search_idle_timeout_s`, it earns an
  idle strike; enough consecutive strikes finalize you. **You cannot livelock
  the platform.** A long benchmark in flight, or any new request, resets it — so
  legitimate work never trips it, but a stuck or silently-finished policy always
  gets reaped.
- **Done, politely.** When `next_config()` returns `None`, the SDK sends
  `status: "exhausted"` and the platform finalizes you *now* instead of idling
  to the deadline. You get this for free — just return `None`.
- **Crashes, honestly.** An uncaught exception in your strategy becomes a
  `status: "error"` signal and a non-zero exit. The platform validates what you
  already found. Don't swallow your own bugs to look alive.
- **Network resilience, idempotency, the clock, the verdict.** The client
  retries every call through a transient blip or a platform restart, and retries
  can't double-launch (creating POSTs carry idempotency keys); the deadline is
  the platform's; the leaderboard verdict is measured by the platform on its
  pinned dataset, of what you *serve*. Nothing you self-report ranks anything.

## Do

- **Speak canonical config.** Flat dict, snake_case keys, JSON-typed values, in
  the platform's spelling. Use `space.py` to sample the declared space; it dedups
  the way the platform canonicalizes.
- **Let placement alone.** Never put `model_path`, `port`, `host`, `node_rank`,
  … in a config — the platform owns placement and will 422 you. `tp`/`dp`/`pp`
  *are* yours; request exactly `space.cards_for(config)` cards per launch.
- **Declare deviations.** Stepping outside the space is allowed — put it in
  `plan().extras` with a reason and it's a badge, not a rejection. Undeclared
  still works but flags louder.
- **Learn from failures.** OOM, bad-config, and crash results arrive in
  `observe()` with `objective=None` and a `failure_class`. Feed them back — a
  Bayesian policy that ignores infeasible regions wastes the night.
- **Keep `contenders()` complete and current.** Whatever it last returned is
  what gets validated if you die. The launch_spec must be self-sufficient.

## Don't

- Don't reimplement scoring — `result.objective` is the campaign's metric,
  already reduced.
- Don't hold cards you're not using — release a launch as soon as its benchmark
  lands, so the next config (or a co-tenant) can run.
- Don't trust your own coverage over the platform's. The platform derives the
  authoritative "what was searched" from the trial ledger. Your `coverage()` is
  a *supplement* — worth overriding only for a completion `fraction` on an
  infinite space, or `skipped_values` you're deliberately avoiding. For a
  fully-delegated policy the loop auto-reports per-axis coverage; leave it.

## Test and ship

```bash
uv run --extra dev pytest      # a full fake night, offline, milliseconds
uv run --extra dev ruff check autotune_policy/ strategies/ tests/
```

`tests/test_night.py` drives your strategy through a whole lifecycle against an
in-memory platform — no network, no GPUs. If it passes, your policy is
contract-correct. CI runs exactly this on every push and builds and pushes the
image on a version tag.
Because the fake night exercises the entire contract, a green check means a
change to `strategies/` didn't break the protocol — that's the safety net that
lets you iterate on the algorithm without re-reading this document each time.

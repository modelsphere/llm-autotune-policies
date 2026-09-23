# chaos — fault-injection test policy

**Not a tuner.** A policy that deliberately misbehaves, so you can see how the
platform handles a policy gone wrong — end to end, against the mock engine, no
GPUs. One image emits any of the faults below; the campaign picks which with the
`CHAOS_SCENARIO` environment variable.

It vendors the same stdlib SDK (`autotune_policy/`) as the real policies, so a
fault exercises exactly the code path a genuine bug would. The copy of record
for the SDK is [`autotune_policy/`](../autotune_policy/) at the top of this
repository, and CI checks that this one still matches it. The contract it speaks is documented in
[the platform repository](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md).

## The fault menu

| `CHAOS_SCENARIO` | What the policy does | Platform behaviour it exercises |
| --- | --- | --- |
| `happy` *(default)* | A few real trials on the mock engine → one contender → serve → exit | The whole happy path: search → finalize → validate → done |
| `exhaust_immediately` | Proposes nothing at all | `exhausted` signal → finalize with zero contenders |
| `crash_next_config` | Raises in the proposal code after one trial | Uncaught bug → `error` signal + non-zero exit → validate what exists |
| `crash_observe` | Raises while recording a measurement | Same `error` path, mid-trial |
| `crash_setup` | Raises before the first heartbeat | Dies before handshake → **container_died / ready_timeout** (no signal possible) |
| `bad_config` | Proposes a config carrying a placement flag (`port`) | Hard **422** at the launch endpoint → loop records `bad_config`, moves on |
| `deviate` | Proposes an undeclared, out-of-space parameter | **Not** rejected — trial is **badged** as a deviation and run |
| `bad_contender` | PUTs a malformed contender (empty `engine_args`, no image) | Rejected-contender echo; session survives |
| `idle` | Heartbeats forever, never launches, never signals done | **Productivity watchdog** → idle strikes → `policy_idle` finalize |
| `wedge` | Beats once, then goes silent while alive | Heartbeat timeout → **`policy_wedged`** at 2× the timeout |
| `exit_hard` | `os._exit(137)` mid-search | Abrupt **container death** → fail toward validation |
| `flaky_network` | Runs the happy path over a socket that drops ~30% of calls | SDK rides out transient blips; platform tolerates the gaps |
| `serve_fail` | Reaches validation with a contender, then never serves it | Serve budget expires → platform **fallback-launches** the contender itself |
| `hw_constraints` | Launches configs that violate the machine: `tp` > cards, card-count mismatch, cards outside its allocation | Each is a **422** — the safety rejections, never a wrong launch |
| `bench_unserved` | Benches a launch it already released, an unknown suite, a bogus launch id | **409** (not serving) / **422** (suite) / **404** (no such launch) |
| `gpu_packing` | Fills the machine with `tp=2` engines on distinct card pairs, then one more | All accepted (**packing works**), the last refused **429** for capacity |

Tunable via env (all optional): `CHAOS_BEATS`, `CHAOS_INTERVAL_S` (idle),
`CHAOS_WEDGE_S` (wedge), `CHAOS_FLAKE_PCT`, `CHAOS_SEED` (flaky_network).

## Run it against the platform

1. **Build & push** to a registry the GPU machines can pull from. The image is
   stdlib-only, so there is nothing to install:
   ```bash
   docker build -t <registry>/autotune-policy-chaos:0.1.0 .
   docker push <registry>/autotune-policy-chaos:0.1.0
   ```
2. **Register** it as a policy (name + image), with `env` carrying the scenario —
   e.g. `{"CHAOS_SCENARIO": "idle"}`. Register it once per fault you want on tap,
   or set the env when you start the campaign.
3. **Start a campaign** with `policy_id` set, pointed at the **mock engine +
   mock LLMBench** machine (the GPU-free E2E path). The platform injects
   `AUTOTUNE_API_URL` / `AUTOTUNE_API_KEY` / `AUTOTUNE_SESSION_ID`; the policy
   dials out and misbehaves on cue.
4. **Watch** the session: `search_end_reason`, the failure class, the events
   stream, and whether contenders still got validated tell you the platform
   handled it as designed.

> The `idle` and `wedge` faults are slow by nature — they wait out the idle
> windows / the heartbeat timeout. Shrink the campaign's `search_idle_timeout_s`
> / `heartbeat_timeout_s` (and `CHAOS_INTERVAL_S`) to see them fire in minutes.

## Test

```bash
uv run --extra dev pytest      # every fault, driven against an in-memory platform
uv run --extra dev ruff check faults.py main.py tests/
```

`tests/test_faults.py` asserts each scenario still emits the failure it claims —
so the menu can't silently rot. `exit_hard` and `flaky_network` are validated by
their presence in the menu rather than executed (one would kill the test process;
the other needs a real socket).

## Contract version

Built against policy contract **1.1** (`/api/policy/v1`).

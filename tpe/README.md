# tpe — model-based search with Optuna

A [policy](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md)
for LLM AutoTune that searches a campaign's space with TPE (Optuna's
Tree-structured Parzen Estimator). Random search draws blind; TPE fits where
the good results are and proposes there, so a night's later trials are
better spent. It also learns where configs fail: an OOM or a crash counts as
an infeasible point, and the search moves away from it.

```python
run_policy(SearchLoop(TPE()))
```

## How it searches

- **Startup:** the first 6 trials are seeded random draws across the space.
  A model fitted on two points knows nothing yet.
- **Then TPE:** it models the space jointly (engine args interact: the memory
  fraction that runs out of memory depends on tp) and reads the platform's
  redline slacks as constraints. A config that crossed a redline counts
  against its region.
- **Failures:**
  - **A config's own fault** (`oom`, `bad_config`, `benchmark_not_passed`,
    `ready_timeout`, `nccl`, `segfault`, `model_load`): replayed as the worst
    value seen with a violated constraint.
  - **Infrastructure** (an ssh timeout, an image pull, a busy card): says
    nothing about the parameters and is not replayed. That config isn't
    proposed again tonight.
- **Running out:** when the model only finds configs already tried, and the
  space has at most 5,000 configs, it measures the ones left. Then it tells
  the platform it is done.

## State and restarts

The observations are the whole state. Before every proposal the policy builds
a fresh Optuna study and replays them into it, which takes milliseconds at
this scale. After every measurement it writes them to the platform. So:

- **A restarted container** reloads the observations, keeps the night's trial
  count, and never proposes a config it already tried.
- **The next night** starts with everything measured before, and a fresh
  trial budget.
- **Proposals are deterministic:** the sampler's seed comes from the space,
  the session and the number of observations. That holds only with the
  same Optuna version, so Optuna is pinned (`>=4.9,<5`).

## Settings

| Variable | Default | |
|---|---|---|
| `TPE_MAX_TRIALS` | 60 | Trials per night at most |
| `TPE_STARTUP_TRIALS` | 6 | Seeded random draws before the model |

Set them in the policy's environment when you register it.

## Building

Unlike `random-search`, this policy has dependencies: Optuna and numpy. The
image installs them from `requirements.txt`, exported with hashes from
`uv.lock`:

```bash
docker build -t <registry>/llm-autotune-policy-tpe:<version> tpe/
# behind a mirror:
docker build --build-arg PIP_INDEX_URL=https://<mirror>/simple \
  -t <registry>/llm-autotune-policy-tpe:<version> tpe/
```

After changing dependencies, run `uv lock` and then
`uv export --frozen --no-dev --no-emit-project -o requirements.txt`.

## Tests

```bash
uv run --extra dev pytest
```

`tests/test_night.py` runs whole nights against an in-memory platform,
including a restart and a second night. `tests/test_tpe.py` covers the space
translation, determinism, exhaustion, and whether it learns: it should move
toward the better region and away from the one that crashes.

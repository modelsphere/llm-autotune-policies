# LLM AutoTune policies

Search policies for [LLM AutoTune](https://github.com/modelsphere/llm-autotune).
A **policy** is a container that decides what to try next. The platform launches
it for a campaign's window, and it proposes configurations over the
[policy contract](https://github.com/modelsphere/llm-autotune/blob/main/docs/api/policy-contract.md) — an HTTP API, so a policy can
be written in any language and versioned on its own.

The platform judges every proposal itself: it validates the configuration,
launches it, benchmarks it and scores it. A policy never reports its own result.
That split is what makes two policies comparable, and it is why adding a smarter
search requires no platform change at all.

| | |
|---|---|
| [`autotune_policy/`](autotune_policy/) | The SDK, and the copy of record — each policy directory vendors it verbatim, and CI checks they still match. Stdlib only. `run_policy(SearchLoop(your_strategy))` satisfies every MUST in the contract — liveness, the productivity watchdog, idempotency — so you write search logic, not protocol. |
| [`random-search/`](random-search/) | The reference policy, and the fork-and-edit template. Random search over the declared space: the baseline any smarter policy should beat. |
| [`tpe/`](tpe/) | TPE (Optuna's Tree-structured Parzen Estimator): proposes where earlier trials paid off and learns which configs fail, so a night's later trials are better spent than random draws. |
| [`heuristic-search/`](heuristic-search/) | Constraint-aware heuristic search with bounded steps, warm starts, decision logs and offline simulation. |
| [`chaos/`](chaos/) | Not a tuner. It emits common policy failures on demand (crash, wedge, idle, bad config, container death) via `CHAOS_SCENARIO`, to prove the platform handles a policy gone wrong. |

Each policy directory is self-contained — its own `pyproject.toml`, Dockerfile,
tests and a vendored copy of the SDK — so it builds and tests on its own and
lifts out into a separate repository unchanged. The top-level `autotune_policy/`
is the copy of record: change it there, run `scripts/sync-sdk.sh`, and commit
the copies with it (CI fails if they drift).

LLM AutoTune checks this repository out under `policies/` as a git submodule.
Changes to a policy or the SDK are pull requests here; the platform only moves
its pointer.

## Writing one

Copy [`random-search/`](random-search/) — it is the template, not just an
example — into a repository of your own:

```bash
cp -r random-search ../my-policy && cd ../my-policy && git init
# edit strategies/ and point main.py at your Strategy
uv run --extra dev pytest        # a full offline night, no platform needed
```

[The reference policy's README](random-search/README.md#write-your-own-policy)
describes the five methods a strategy implements. [`BEST_PRACTICES.md`](BEST_PRACTICES.md) covers what the
platform enforces for you, so you can change the search logic without reasoning
about the protocol.

## Using the published image

`random-search` and `tpe` are published with every release, for amd64 and arm64:

```text
ghcr.io/modelsphere/llm-autotune-policy-random-search:<version>
ghcr.io/modelsphere/llm-autotune-policy-tpe:<version>
```

If your GPU nodes can pull from ghcr.io, register that image and skip the
build. If they can't, copy it into a registry they can reach
(`docker pull`, `docker tag`, `docker push`), or build it as below.

## Building an image

Each policy vendors the SDK, so its own directory is the build context and there
is nothing to install:

```bash
docker build -t <registry>/llm-autotune-policy-random-search:0.1.1 random-search/
docker push <registry>/llm-autotune-policy-random-search:0.1.1
```

Put it where your GPU machines can pull it. Without a registry: on an ssh
machine, `docker save … | ssh <host> docker load`; on kind,
`kind load docker-image …`. Use a version tag rather than `latest`, so
Kubernetes runs the loaded image instead of trying to pull.

Then register the image with the platform's API (an API key from its API Keys
page) and pick it as a campaign's policy in the New campaign form:

```bash
curl -X POST https://<autotune>/api/policies -H "X-API-Key: $KEY" \
  -H 'Content-Type: application/json' -d '{"name": "random-search",
    "image": "ghcr.io/modelsphere/llm-autotune-policy-random-search:0.1.1", "version": "0.1.1",
    "gpus_in_container": false, "needs_model": false}'
```

`gpus_in_container` and `needs_model` are false for a policy that only
delegates launches to the platform, as both policies here do.

## License

Apache-2.0 — see [LICENSE](LICENSE).

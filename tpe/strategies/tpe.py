"""TPE: a model-based search with Optuna's Tree-structured Parzen Estimator.

Random search draws blind. TPE fits two densities over what it has measured
(where the good results are, where the rest are) and proposes where the ratio
is best, so a night's later trials go where the earlier ones paid off. It
also learns where configs fail: an OOM or a crash is fed back as an infeasible
observation, so the search moves away from the cliff instead of probing it
all night.

How it uses Optuna:

- **Replayed, not held.** Before every proposal it builds a fresh in-memory
  study and replays the observations into it. The observations are the whole
  state; they are written to the platform after each one (``dump_state``), so
  a restarted container, or the next night, carries on exactly where the
  search was. Replay costs milliseconds at this scale (tens of trials).
- **Deterministic.** The sampler's seed comes from the space, the session and
  how many observations there are, so the same history proposes the same
  next config. Pin Optuna (``optuna>=4.9,<5``): the sampler's output is
  deterministic per seed *and* version.
- **Constraint-aware.** A measured trial carries the platform's signed redline
  slacks (``summary.constraints``, <= 0 is satisfied), which TPE's
  ``constraints_func`` reads. A config-induced failure (OOM, bad config, a
  benchmark that did not pass, …) is replayed as the worst value seen with a
  violated constraint. An infrastructure failure (ssh timeout, a pull error,
  a busy card) says nothing about the parameters and is not replayed at all.
- **Startup.** The first ``n_startup_trials`` proposals are seeded random
  draws: a model fitted on two points is confidence nobody earned. Six, not
  Optuna's ten, because a night is 15–30 trials.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, create_trial

from autotune_policy.space import canonical_key, swept_keys
from autotune_policy.strategy import Strategy, TrialResult, build_contender
from strategies.optuna_space import decode, distributions, encode, enumerate_space

# Optuna logs every study creation at INFO; here that is once per proposal.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# The platform's failure classes that are a verdict on the parameters: running
# the same config again fails the same way, so they are observations.
CONFIG_FAILURES = frozenset({
    "oom", "bad_config", "benchmark_not_passed", "ready_timeout", "nccl",
    "segfault", "model_load",
})

# Where Optuna's constraint-aware samplers read a trial's constraints.
CONSTRAINTS_KEY = "constraints"

# Slack for "crossed by an unmeasurable amount": a redline whose metric was
# never measured, or a config that crashed before producing any. Finite,
# because infinity breaks the density fit and JSON.
PENALTY_SLACK = 1e6

# Consecutive asks allowed to land on already-tried configs before the space
# counts as exhausted.
MAX_STALE_ASKS = 30

# Spaces up to this many configs are listed to find what is left once the
# model stops finding untried configs.
ENUMERABLE = 5000


def stored_constraints(trial: FrozenTrial) -> list[float]:
    return list(trial.system_attrs.get(CONSTRAINTS_KEY, ()))


class TPE(Strategy):
    def __init__(self, *, max_trials: int = 60, n_startup_trials: int = 6,
                 seed: int | None = None):
        self.max_trials = max_trials
        self.n_startup_trials = n_startup_trials
        self._seed = seed
        self.space: dict = {}
        self.direction = "maximize"
        self.image = ""
        self.max_contenders = 2
        self.session_id: Any = None
        self.base_seed = 0
        # What replay needs, JSON-ready: {"config", "value", "constraints"},
        # value None for a config-induced failure.
        self.observations: list[dict] = []
        self.seen: set[str] = set()
        # This night's measured results, for contenders and status.
        self.results: list[TrialResult] = []
        self._proposed = 0

    # -- lifecycle -------------------------------------------------------------
    def setup(self, manifest: dict) -> None:
        self.space = manifest.get("search_space") or {}
        screen = (manifest.get("objective") or {}).get("screen") or {}
        goal = (screen.get("direction") or "max").lower()
        self.direction = "minimize" if goal.startswith("min") else "maximize"
        self.image = (manifest.get("model") or {}).get("image", "")
        self.max_contenders = (manifest.get("contenders") or {}).get("max") or 2
        self.session_id = manifest.get("session_id")
        self.base_seed = self._seed if self._seed is not None else int(self.session_id or 0)
        # Never re-measure a config a previous night already validated.
        for c in (manifest.get("prior") or {}).get("contenders", []):
            cfg = (c.get("launch_spec") or {}).get("engine_args") or c.get("config")
            if cfg:
                self.seen.add(canonical_key(cfg))

    def plan(self) -> dict:
        return {
            "tuning": swept_keys(self.space),
            "extras": [],
            "notes": (f"TPE (Optuna), up to {self.max_trials} trials, "
                      f"{self.n_startup_trials} seeded random draws first"),
        }

    # -- search ----------------------------------------------------------------
    def next_config(self) -> dict | None:
        if self._proposed >= self.max_trials:
            return None
        with warnings.catch_warnings():
            # Multivariate, grouped, constant-liar TPE is flagged experimental
            # and warns on every construction.
            warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
            config = self._ask()
        if config is not None:
            self.seen.add(canonical_key(config))
            self._proposed += 1
        return config

    def observe(self, result: TrialResult) -> None:
        config = result.config
        self.seen.add(canonical_key(config))
        if result.objective is not None:
            self.results.append(result)
            self.observations.append({
                "config": config,
                "value": float(result.objective),
                "constraints": list(_slacks(result)),
            })
        elif result.failure_class in CONFIG_FAILURES:
            self.observations.append({
                "config": config, "value": None, "constraints": [PENALTY_SLACK],
            })
        # Anything else (an infrastructure failure, a released launch) says
        # nothing about the parameters.

    def contenders(self) -> list[dict]:
        out = []
        for rank, result in enumerate(self._ranked(), start=1):
            environment = result.raw.get("environment") if isinstance(result.raw, dict) else {}
            out.append(build_contender(
                rank=rank,
                engine_args=result.config,
                image=self.image,
                environment=environment or {},
                evidence={"screen_objective": result.objective},
                trial_ids=[result.trial_id] if result.trial_id else [],
            ))
        return out

    def status(self) -> tuple[str, float | None]:
        n = len(self.results)
        progress = min(1.0, self._proposed / max(1, self.max_trials))
        phase = "random startup" if len(self.observations) < self.n_startup_trials else "TPE"
        head = f"trial {self._proposed}/{self.max_trials} ({phase})"
        ranked = self._ranked()
        if ranked:
            return f"{head}, best {ranked[0].objective:.1f}", progress
        return f"{head}, {n} measured, no feasible config yet", progress

    # -- warm-start ------------------------------------------------------------
    def load_state(self, blob: dict) -> None:
        blob = blob or {}
        for obs in blob.get("observations", []):
            if isinstance(obs, dict) and isinstance(obs.get("config"), dict):
                self.observations.append({
                    "config": obs["config"],
                    "value": obs.get("value"),
                    "constraints": list(obs.get("constraints") or []),
                })
        for key in blob.get("seen", []):
            self.seen.add(key)
        # The same session restarting mid-night keeps its trial budget; a new
        # night starts a fresh one.
        if blob.get("session_id") is not None and blob.get("session_id") == self.session_id:
            self._proposed = int(blob.get("proposed") or 0)

    def dump_state(self) -> dict:
        return {
            "session_id": self.session_id,
            "proposed": self._proposed,
            "observations": self.observations,
            "seen": sorted(self.seen),
            "best": [{"config": r.config, "objective": r.objective} for r in self._ranked()],
        }

    # -- internal --------------------------------------------------------------
    def _ask(self) -> dict | None:
        dists = distributions(self.space)
        if not dists:
            # Nothing varies: the base config is the one point.
            config = decode({}, self.space)
            return None if canonical_key(config) in self.seen else config
        study = self._replayed_study(dists)
        stale = 0
        while stale < MAX_STALE_ASKS:
            trial = study.ask(fixed_distributions=dists)
            config = decode(trial.params, self.space)
            if canonical_key(config) not in self.seen:
                return config
            # Not re-drawn by hand: the asked trial stays RUNNING in the
            # study, so the constant liar steers the next ask away from it.
            stale += 1
        # The model keeps landing on tried configs, which happens as a space
        # runs out. If the space is small enough to list, measure what is
        # left rather than end the night with points never tried.
        for config in enumerate_space(self.space, limit=ENUMERABLE):
            if canonical_key(config) not in self.seen:
                return config
        return None

    def _replayed_study(self, dists: dict) -> optuna.Study:
        sampler = TPESampler(
            seed=_seed(self.space, self.base_seed, len(self.observations)),
            n_startup_trials=self.n_startup_trials,
            # Engine args interact (the memory fraction that OOMs depends on
            # tp), so model the space jointly...
            multivariate=True,
            # ...while tolerating parameters that conditions drop from some
            # trials.
            group=True,
            constant_liar=True,
            constraints_func=stored_constraints,
        )
        study = optuna.create_study(direction=self.direction, sampler=sampler)

        # A crashed config has no value, but a COMPLETE trial needs one. The
        # worst measured value is the honest stand-in: in scale, direction-
        # aware, and dominated by the violated constraint anyway.
        measured = [o["value"] for o in self.observations if o["value"] is not None]
        fallback = 0.0
        if measured:
            fallback = min(measured) if self.direction == "maximize" else max(measured)

        # Replay order is part of determinism (the sampler's RNG is consumed
        # per trial), so impose one rather than trusting storage order.
        for obs in sorted(self.observations, key=_replay_order):
            params = encode(obs["config"], self.space)
            if not params:
                continue  # predates a space edit entirely
            study.add_trial(create_trial(
                params=params,
                distributions={k: dists[k] for k in params if k in dists},
                value=fallback if obs["value"] is None else obs["value"],
                system_attrs={CONSTRAINTS_KEY: list(obs["constraints"]) or [-1.0]},
            ))
        return study

    def _ranked(self) -> list[TrialResult]:
        feasible = [r for r in self.results if r.ok]
        feasible.sort(key=lambda r: r.objective, reverse=self.direction == "maximize")
        return feasible[: self.max_contenders]


def _slacks(result: TrialResult) -> tuple[float, ...]:
    """The platform's signed redline slacks for a measured trial, made finite.
    Without them, feasibility is all there is to go on."""
    summary = (result.raw or {}).get("summary") if isinstance(result.raw, dict) else None
    raw = (summary or {}).get("constraints")
    if isinstance(raw, list) and raw:
        out = []
        for slack in raw:
            try:
                value = float(slack)
            except (TypeError, ValueError):
                value = PENALTY_SLACK
            out.append(max(-PENALTY_SLACK, min(PENALTY_SLACK, value)))
        return tuple(out)
    return (-1.0,) if result.feasible else (1.0,)


def _replay_order(obs: dict) -> tuple[str, str]:
    return (canonical_key(obs["config"]), str(obs["value"]))


def _seed(space: dict, base_seed: int, tried: int) -> int:
    """Deterministic per (space, session, number of observations). Eight hex
    digits: numpy seeds cap at 2**32."""
    canon = json.dumps(space or {}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{canon}|{base_seed}|{tried}".encode()).hexdigest()[:8]
    return int(digest, 16)

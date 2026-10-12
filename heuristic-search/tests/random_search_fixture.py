"""Random search — the reference strategy, and the one to replace.

It draws unseen configurations from the campaign's declared space, lets the loop
measure each with the platform's screen benchmark, and keeps the best-feasible
few as contenders. It is intentionally the simplest thing that is genuinely
useful: a baseline every smarter search should beat, and a worked example of the
five methods a `Strategy` implements. To write your own algorithm, copy this
file and change `next_config`/`observe`/`contenders`; the loop and the wire
protocol do not change.
"""

from __future__ import annotations

import random

from autotune_policy.space import canonical_key, sample, swept_keys
from autotune_policy.strategy import Strategy, TrialResult, build_contender


class RandomSearch(Strategy):
    def __init__(self, *, max_trials: int = 60, seed: int | None = None):
        self.max_trials = max_trials
        self._seed = seed
        self._rng = random.Random()
        self.space: dict = {}
        self.direction = "max"
        self.image = ""
        self.max_contenders = 2
        self.seen: set[str] = set()
        self.observations: list[TrialResult] = []
        self._proposed = 0

    # -- lifecycle -------------------------------------------------------------
    def setup(self, manifest: dict) -> None:
        self.space = manifest.get("search_space") or {}
        screen = (manifest.get("objective") or {}).get("screen") or {}
        self.direction = (screen.get("direction") or "max").lower()
        self.image = (manifest.get("model") or {}).get("image", "")
        self.max_contenders = (manifest.get("contenders") or {}).get("max") or 2
        seed = self._seed if self._seed is not None else manifest.get("session_id", 0)
        self._rng = random.Random(seed)
        # Never re-measure a point a previous night already validated.
        for c in (manifest.get("prior") or {}).get("contenders", []):
            cfg = (c.get("launch_spec") or {}).get("engine_args") or c.get("config")
            if cfg:
                self.seen.add(canonical_key(cfg))

    def plan(self) -> dict:
        return {
            "tuning": swept_keys(self.space),
            "extras": [],
            "notes": f"random search, up to {self.max_trials} trials, reproducible from the session seed",
        }

    # -- search ----------------------------------------------------------------
    def next_config(self) -> dict | None:
        if self._proposed >= self.max_trials:
            return None
        for _ in range(200):  # resample past collisions; give up if the space is exhausted
            config = sample(self.space, self._rng)
            key = canonical_key(config)
            if key not in self.seen:
                self.seen.add(key)
                self._proposed += 1
                return config
        return None

    def observe(self, result: TrialResult) -> None:
        self.observations.append(result)

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
        n = len(self.observations)
        progress = min(1.0, n / max(1, self.max_trials))
        ranked = self._ranked()
        if ranked:
            return (f"trial {n}/{self.max_trials}, best {ranked[0].objective:.1f}", progress)
        return (f"trial {n}/{self.max_trials}, no feasible config yet", progress)

    # -- warm-start ------------------------------------------------------------
    def load_state(self, blob: dict) -> None:
        for key in (blob or {}).get("seen", []):
            self.seen.add(key)

    def dump_state(self) -> dict:
        return {
            "seen": sorted(self.seen),
            "best": [{"config": r.config, "objective": r.objective} for r in self._ranked()],
        }

    # -- internal --------------------------------------------------------------
    def _ranked(self) -> list[TrialResult]:
        feasible = [r for r in self.observations if r.ok]
        feasible.sort(key=lambda r: r.objective, reverse=self.direction != "min")
        return feasible[: self.max_contenders]

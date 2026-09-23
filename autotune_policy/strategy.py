"""The one thing a policy author writes.

Fork the template, implement a ``Strategy``, and hand it to
``run_policy(SearchLoop(YourStrategy()))``. The loop does the whole contract
dance — heartbeats, launches, benchmarks, finalize, serve — and calls the four
methods below at the right moments. Everything about *how to tune* lives here;
nothing about *how to talk to the platform* does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TrialResult:
    """One measurement the loop hands back to you.

    ``objective`` is the platform's own ``objective_value`` on the screen suite
    — the campaign's metric, already reduced for you, so you never reimplement
    scoring. A failed launch (OOM, bad config, crash) also arrives here, with
    ``objective=None`` and a ``failure_class``: that is a signal to learn from,
    not an error to swallow.
    """

    config: dict
    objective: float | None
    feasible: bool
    failure_class: str = ""
    trial_id: int | None = None
    error: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.objective is not None and self.feasible and not self.failure_class


class Strategy(ABC):
    """A search algorithm. The SearchLoop owns the clock and the API; you own
    the decisions."""

    @abstractmethod
    def setup(self, manifest: dict) -> None:
        """Receive the night's shape once, before search starts: the manifest's
        ``search_space``, ``objective``, ``hardware``, ``model`` and ``prior``."""

    def plan(self) -> dict:
        """Declare what you will vary — the contract's honesty obligation.

        Return ``{"tuning": [...], "extras": [...], "notes": "..."}``. ``extras``
        are parameters outside the declared space you intend to introduce, each
        ``{"param", "values"|"range", "reason"}``; the platform badges those
        deviations rather than rejecting them. Default: tune nothing declared.
        """
        return {"tuning": [], "extras": [], "notes": ""}

    @abstractmethod
    def next_config(self) -> dict | None:
        """The next engine-args config to measure, or ``None`` when you have
        nothing more to explore (the loop then idles until finalize). A config
        is a flat dict in the platform's canonical spelling — snake_case keys,
        JSON-typed values, and never any placement flag (model_path, port, …)."""

    @abstractmethod
    def observe(self, result: TrialResult) -> None:
        """Record the measurement the loop just took for ``result.config``."""

    @abstractmethod
    def contenders(self) -> list[dict]:
        """Your best-so-far as PUT /contenders entries (see ``build_contender``),
        best first. The loop re-PUTs these after every observation, so whatever
        you return here is what survives a crash and gets validated."""

    def status(self) -> tuple[str, float | None]:
        """A short line and progress (0..1) for the heartbeat and the UI."""
        return ("", None)

    def coverage(self) -> dict | None:
        """Optionally, how much of the space you have covered — in the
        platform's own vocabulary (``{"fraction", "dimensions": [{"param",
        "tried_values", "tried_ranges", "skipped_values"}]}``).

        Return ``None`` (the default) and the loop builds this for you from the
        configs you have measured — you get per-axis coverage for free. Override
        only to add what the loop cannot infer: a completion estimate for an
        infinite/model-based space, or values you are *deliberately* skipping.
        The platform derives its own authoritative coverage from the trial
        ledger regardless; this is the supplement, badged self-reported.
        """
        return None

    def load_state(self, blob: dict) -> None:
        """Warm-start from what a previous night wrote for this campaign (the
        blob may be empty on night one)."""

    def dump_state(self) -> dict:
        """JSON-serialisable state to carry to the next night."""
        return {}


def build_contender(rank: int, engine_args: dict, image: str, *,
                    environment: dict | None = None, env: dict | None = None,
                    volumes: dict | None = None, evidence: dict | None = None,
                    trial_ids: list[int] | None = None) -> dict:
    """Assemble one ``PUT /contenders`` entry.

    ``launch_spec`` must be complete enough that the platform could launch it
    without you — it is also the dead-policy fallback. ``environment``
    (engine/torch/CUDA versions + image digest) is what makes a verdict
    *promotable*; fill it from a launch's reported environment when you have it.
    """
    return {
        "rank": rank,
        "launch_spec": {
            "engine_args": engine_args,
            "image": image,
            "env": env or {},
            "volumes": volumes or {},
            "environment": environment or {},
        },
        "evidence": evidence or {},
        "trial_ids": trial_ids or [],
    }

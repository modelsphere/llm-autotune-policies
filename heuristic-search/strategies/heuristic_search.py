"""Workload-prioritized local search over the declared search space.

All declared multi-value axes may vary. Workload profiles only prioritize axes;
parameters that must stay fixed belong in the search space base or singleton axes.
No telemetry is inferred from an opaque objective value.

Platform interface::

    manifest["objective"] = {
        "screen": {"target_metric": "perf_guidellm_sweep.total_tpm_card_norm",
                   "direction": "maximize",
                   "redlines": [{"metric": "perf_guidellm_sweep.reported_level_meets_slo",
                                 "op": ">=", "value": 1}]},
        "verify": {"target_metric": "perf_guidellm_sweep.total_tpm_card_norm",
                   "direction": "maximize",
                   "redlines": [{"metric": "opencompass.gpqa_diamond",
                                 "op": ">=", "value": 0.6}]},
    }

    # GET /api/policy/v1/benchmarks/{id}; stored intact in TrialResult.raw:
    benchmark = {
        "status": "succeeded",
        "summary": {"objective_value": 1234.5, "feasible": True,
                    "constraints": [-120.0], "breaches": []},
        "metrics": {"perf_guidellm_sweep.reported_level_meets_slo": 1,
                    "perf_guidellm_sweep.ttft_p99": 7.2},
    }

Metric names are literal, flattened dictionary keys; dots are NOT path separators.
SearchLoop passes summary.objective_value and summary.feasible as TrialResult
fields. summary.constraints/breaches are platform diagnostics, not metric values.
Search runs the screen suite only: enforce screen redlines here, retain verify
redlines for context, and leave final verify evaluation to the platform. A missing
verify-only metric in a screen benchmark must not disqualify a screen contender.

"""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal

from autotune_policy.space import canonical_key, prune_inactive
from autotune_policy.strategy import Strategy, TrialResult, build_contender

PROFILES = {
    "interactive": ("max_running_requests", "chunked_prefill_size",
                    "cuda_graph_max_bs_decode", "mamba_full_memory_ratio"),
    "long_prefill": ("chunked_prefill_size", "prefill_max_requests",
                     "mamba_full_memory_ratio", "max_running_requests"),
    "long_decode": ("max_running_requests", "mamba_full_memory_ratio",
                    "cuda_graph_max_bs_decode", "chunked_prefill_size"),
    "agent": ("mamba_full_memory_ratio", "max_running_requests",
              "chunked_prefill_size", "prefill_max_requests"),
    "batch": ("mamba_full_memory_ratio", "max_running_requests",
              "chunked_prefill_size", "prefill_max_requests"),
    "mixed": ("chunked_prefill_size", "prefill_max_requests",
              "schedule_conservativeness", "max_running_requests"),
}
COMMON = ("mem_fraction_static", "schedule_conservativeness")
# Seeds only; they are projected onto declared values, never added as extras.
DEFAULTS = {"chunked_prefill_size": 4096, "mem_fraction_static": 0.85,
            "cuda_graph_max_bs_decode": 32, "schedule_conservativeness": 1.0}


logger = logging.getLogger(__name__)


class HeuristicSearch(Strategy):
    """Keep feasible winners separate from an optionally infeasible search center.

    Local models use measurements with identical other axes. Numeric single
    axes use actual values; tied numeric rows use row coordinates (group effect).
    Categorical axes are explored without linear extrapolation. Predictions only
    prioritize experiments; measured feasibility alone admits contenders.
    """

    state_strategy = "heuristic-search"

    def __init__(self, *, workload: str = "agent", max_trials: int = 24,
                 min_improvement: float = 0.01, baseline: dict | None = None,
                 safety_margin: float = 0.05, max_step: int = 4,
                 pruning_threshold: float = 0.1, decision_log=None):
        if workload not in PROFILES:
            raise ValueError(f"unknown workload {workload!r}; choose {list(PROFILES)}")
        if max_trials < 1 or not math.isfinite(min_improvement) or min_improvement < 0:
            raise ValueError("max_trials must be positive; min_improvement must be finite >= 0")
        self.workload = workload
        self.max_trials = max_trials
        self.min_improvement = min_improvement
        self.baseline = deepcopy(baseline or {})

        if not math.isfinite(safety_margin) or not 0 <= safety_margin < 1:
            raise ValueError("safety_margin must be finite and in [0, 1)")
        if isinstance(max_step, bool) or not isinstance(max_step, int) or max_step < 1:
            raise ValueError("max_step must be a positive integer")
        if not self._finite(pruning_threshold) or pruning_threshold <= 0:
            raise ValueError("pruning_threshold must be finite and positive")
        self.safety_margin = safety_margin
        self.max_step = max_step
        self.pruning_threshold = pruning_threshold
        self.decision_log = decision_log

    def setup(self, manifest: dict) -> None:
        self.session_id = manifest.get("session_id")
        self.space = deepcopy(manifest.get("search_space") or {})
        self.model = deepcopy(manifest.get("model") or {})
        objective = manifest.get("objective") or {}
        self.screen_objective = deepcopy(objective.get("screen") or {})
        self.verify_objective = deepcopy(objective.get("verify") or {})
        direction = self.screen_objective.get("direction", "maximize").lower()
        self.direction = {"maximize": "max", "minimize": "min"}.get(direction, direction)
        if self.direction not in ("min", "max"):
            raise ValueError("objective direction must be maximize/minimize (or max/min)")
        self.screen_redlines = self._redlines(self.screen_objective, "screen")
        self.verify_redlines = self._redlines(self.verify_objective, "verify")
        self.max_contenders = (manifest.get("contenders") or {}).get("max") or 2
        self.context = canonical_key({
            "space": self.space, "model": self.model,
            "hardware": manifest.get("hardware"), "objective": manifest.get("objective"),
            "workload": self.workload, "baseline": self.baseline,
        })
        self.axes = self._axes()
        priority = list(dict.fromkeys((*PROFILES[self.workload], *COMMON)))
        self.tunable = sorted(
            [i for i, rows in enumerate(self.axes)
             if len(rows) > 1],
            key=lambda i: min(priority.index(k) if k in priority else len(priority)
                              for k in self.axes[i][0]),
        )
        preferred = dict(DEFAULTS)
        preferred.update((manifest.get("production") or {}).get("engine_args") or {})
        preferred.update(self.baseline)
        self.anchor = self._project(preferred)
        self.best: TrialResult | None = None
        self.observations: list[TrialResult] = []
        self.seen: set[str] = set()
        self.pending: list[tuple[dict, str]] = [(deepcopy(self.anchor), "baseline")]
        self.proposed = 0
        self.phase = "baseline"
        self.reason = "baseline"
        self.done = False
        self.inflight: dict | None = None

        # The campaign manifest is the single source of truth for SLOs.
        self.constraints = self.screen_redlines
        self.required_metrics = {spec["metric"] for spec in self.constraints}
        # Retain the historical context envelope to accept existing V2 checkpoints.
        self.context = canonical_key({"v1": self.context, "constraints": self.constraints,
                                      "safety_margin": self.safety_margin,
                                      "max_step": self.max_step})
        # Keep default-policy checkpoints compatible: their samples can supply
        # pruning evidence even when written before direction pruning existed.
        if self.pruning_threshold != 0.1:
            self.context = canonical_key({"context": self.context,
                                          "pruning_threshold": self.pruning_threshold})
        # Old checkpoints may have exhausted a smaller, profile-filtered space.
        if any(k not in priority for i in self.tunable for k in self.axes[i][0]):
            self.context = canonical_key({"context": self.context,
                                          "axis_policy": "all-declared-v1"})
        self.center: TrialResult | None = None
        self.samples: list[tuple[dict, TrialResult]] = []
        self.direction_blocks: list[dict] = []
        self.radius = {axis: min(2, self.max_step) for axis in self.tunable}
        self.expected: dict | None = None
        logger.info("Decision criteria: screen_objective=%r direction=%s redlines=%r "
                    "safety_margin=%s min_improvement=%s pruning_threshold=%s",
                    self.screen_objective, self.direction, self.constraints,
                    self.safety_margin, self.min_improvement, self.pruning_threshold)

    @staticmethod
    def _finite(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    @classmethod
    def _redlines(cls, objective: dict, suite: str) -> list[dict]:
        """Read objective.<suite>.redlines: [{metric: str, op: str, value: number}]."""
        redlines = objective.get("redlines", [])
        if not isinstance(redlines, list):
            raise TypeError(f"objective.{suite}.redlines must be a list")
        for spec in redlines:
            if (not isinstance(spec, dict) or not isinstance(spec.get("metric"), str)
                    or not spec["metric"] or spec.get("op") not in ("<", "<=", ">", ">=")
                    or not cls._finite(spec.get("value"))):
                raise ValueError(f"invalid {suite} redline; expected metric, op and finite value")
        # Keep the list: multiple bounds on the same metric must all be checked.
        return deepcopy(redlines)

    def _metrics(self, result: TrialResult) -> dict:
        """Read raw['metrics'][literal_metric_name], never summary or nested paths."""
        metrics = result.raw.get("metrics") or {}
        if not isinstance(metrics, dict):
            return {}
        return {spec["metric"]: float(metrics[spec["metric"]])
                for spec in self.screen_redlines if self._finite(metrics.get(spec["metric"]))}

    def _feasible(self, result: TrialResult) -> bool:
        """Both platform feasibility and all measured screen redlines are required."""
        if not result.ok or not self._finite(result.objective):
            return False
        metrics = self._metrics(result)
        for spec in self.screen_redlines:
            if spec["metric"] not in metrics:
                return False
            value, limit = metrics[spec["metric"]], spec["value"]
            passes = {"<": value < limit, "<=": value <= limit,
                      ">": value > limit, ">=": value >= limit}
            if not passes[spec["op"]]:
                return False
        return True

    def _axes(self) -> list[list[dict]]:
        axes = []
        occupied: set[str] = set()

        def add(rows):
            keys = set(rows[0])
            if occupied & keys:
                raise ValueError("grid, range and tied axes must not overlap")
            occupied.update(keys)
            axes.append(rows)

        for key, values in (self.space.get("grid") or {}).items():
            if not values:
                raise ValueError(f"empty grid: {key}")
            add([{key: v} for v in values])
        for group in self.space.get("tied") or []:
            lengths = {len(v) for v in group.values()}
            if not group or len(lengths) != 1 or 0 in lengths:
                raise ValueError("tied columns must have equal, nonzero lengths")
            add([{k: v[i] for k, v in group.items()} for i in range(next(iter(lengths)))])
        for key, spec in (self.space.get("range") or {}).items():
            lo, hi, step = (float(spec[k]) for k in ("min", "max", "step"))
            if not all(math.isfinite(x) for x in (lo, hi, step)) or step <= 0 or hi < lo:
                raise ValueError(f"invalid range: {key}")
            if (hi - lo) / step > 10000:
                raise ValueError(f"range too large; use a coarser step: {key}")
            # Decimal arithmetic preserves offset minima and scientific-notation
            # steps; rounding to the step's decimal places can leave the grid.
            start, stop, stride = (Decimal(str(spec[k])) for k in ("min", "max", "step"))
            values = [start + i * stride for i in range(int((stop - start) / stride) + 1)]
            add([{key: int(v) if v == v.to_integral_value() else float(v)} for v in values])
        # Numeric axes are searched in numeric order, not declaration order.
        for rows in axes:
            if len(rows[0]) == 1:
                key = next(iter(rows[0]))
                if all(isinstance(r[key], (int, float)) and not isinstance(r[key], bool)
                       for r in rows):
                    rows.sort(key=lambda r: r[key])
        return axes

    def _project(self, preferred: dict) -> dict:
        config = dict(self.space.get("base") or {})
        for rows in self.axes:
            def distance(row):
                score = 0.0
                for key, value in row.items():
                    target = preferred.get(key, value)
                    if isinstance(value, (int, float)) and isinstance(target, (int, float)):
                        score += abs(value - target) / max(abs(target), 1e-9)
                    else:
                        score += 0 if value == target else 1
                return score
            config.update(min(rows, key=distance))
        return config

    def plan(self) -> dict:
        return {
            "tuning": sorted({k for i in self.tunable for k in self.axes[i][0]}),
            "extras": [],
            "notes": (f"{self.workload}: constraint-first local search, measured "
                      f"directions and adaptive steps; <= {self.max_trials} trials; "
                      "predictions do not guarantee SLO compliance or full coverage"),
        }

    def observe(self, result: TrialResult) -> None:
        if self.inflight is None:
            raise RuntimeError("no outstanding proposal")
        previous_best = deepcopy(self.best)
        previous_center_metrics = self._metric_context(self.center)
        full, self.inflight = self.inflight, None
        result = deepcopy(result)
        self.observations.append(result)
        self.seen.add(canonical_key(result.config))
        # Do not fit measurements whose platform-canonical config differs from
        # the proposal: the measured point may no longer be on this local axis.
        matches = result.config == prune_inactive(full, self.space)
        if matches and not result.failure_class:
            self._update_direction_blocks(full, result)
            self.samples.append((deepcopy(full), result))
        improved = matches and self._better_center(result)
        metric_context = self._metric_context(result)
        logger.info("Decision %s observation: inputs=%r previous_center=%r prediction=%r "
                    "proposal_matches=%s center_updated=%s",
                    self.proposed, metric_context, previous_center_metrics,
                    self.expected, matches, improved)
        if metric_context["missing_metrics"] or metric_context["invalid_metrics"]:
            logger.warning("Decision %s: required metrics missing=%r invalid=%r metrics_type=%s; "
                           "expected finite numeric values under literal benchmark.metrics keys",
                           self.proposed, metric_context["missing_metrics"],
                           metric_context["invalid_metrics"], metric_context["metrics_type"])
        if self.expected is not None:
            axis = self.expected["axis"]
            metrics = self._metrics(result)
            errors = [abs(metrics[k] - v) / max(abs(v), 1e-9)
                      for k, v in self.expected["metrics"].items() if k in metrics]
            if self._finite(result.objective) and self.expected["objective"] is not None:
                errors.append(abs(result.objective - self.expected["objective"])
                              / max(abs(self.expected["objective"]), 1e-9))
            reliable = (matches and not result.failure_class
                        and metrics.keys() == self.required_metrics
                        and bool(errors) and max(errors) <= 0.2)
            logger.info("Decision %s prediction check: axis=%s relative_errors=%r reliable=%s",
                        self.proposed, axis, errors, reliable)
            self.radius[axis] = (min(self.max_step, self.radius[axis] + 1)
                                 if reliable and improved else max(1, self.radius[axis] // 2))
        ranked = self._ranked()
        self.best = deepcopy(ranked[0]) if ranked else None
        if improved:
            self.anchor, self.center = deepcopy(full), result
            self.phase = "coarse"
        logger.info("Decision %s outcome: center_updated=%s best_updated=%s center=%r best=%r "
                    "radius=%r", self.proposed, improved, self.best != previous_best,
                    self._metric_context(self.center), self._metric_context(self.best), self.radius)
        # Re-rank even when a trial did not move the center: it supplies direction evidence.
        self.pending = self._neighbors(self.anchor, self.phase != "fine")
        if "oom" in result.failure_class.lower() or "out of memory" in result.error.lower():
            repairs = []
            for candidate, _ in self._adjacent_neighbors(full):
                changed = [k for k in candidate if candidate[k] != full.get(k)]
                if (changed and all(k in ("chunked_prefill_size", "max_running_requests",
                                          "mem_fraction_static") for k in changed)
                        and all(0 < candidate[k] < full[k] for k in changed)):
                    repairs.append((candidate, "OOM repair: " + ",".join(changed)))
            self.pending = repairs + self.pending
        if self.decision_log is not None:
            self.decision_log({
                "event": "observation", "decision_id": self.proposed,
                "config": result.config, "trial_id": result.trial_id,
                "objective": result.objective, "platform_feasible": result.feasible,
                "feasible": self._feasible(result), "metrics": self._metrics(result),
                "raw_metrics": result.raw.get("metrics"),
                "metric_context": metric_context,
                "previous_center_metrics": previous_center_metrics,
                "prediction": deepcopy(self.expected),
                "failure_class": result.failure_class, "error": result.error,
                "proposal_matches": matches, "center_updated": improved,
                "best_updated": self.best != previous_best,
                "center_config": prune_inactive(self.anchor, self.space),
                "radius": {str(axis): radius for axis, radius in self.radius.items()},
            })
        self.expected = None

    def _ranked(self) -> list[TrialResult]:
        results = sorted((r for r in self.observations if self._feasible(r)),
                         key=lambda r: r.objective, reverse=self.direction == "max")
        unique = {}
        for result in results:
            unique.setdefault(canonical_key(result.config), result)
        return list(unique.values())[:self.max_contenders]

    def contenders(self) -> list[dict]:
        return [build_contender(
            rank, r.config, self.model.get("image", ""),
            environment=r.raw.get("environment") or {},
            env=self.model.get("extra_env"), volumes=self.model.get("extra_volumes"),
            evidence={"screen_objective": r.objective, "workload": self.workload},
            trial_ids=[r.trial_id] if r.trial_id is not None else [],
        ) for rank, r in enumerate(self._ranked(), 1)]

    def status(self) -> tuple[str, float | None]:
        message = (f"{self.workload} {self.phase}: {len(self.observations)}/{self.max_trials}; "
                   f"{self.reason}")
        return message, min(1.0, self.proposed / self.max_trials)

    def dump_state(self) -> dict:
        ranked = self._ranked()
        return {
            "strategy": self.state_strategy, "context": self.context, "version": 2,
            "seed": deepcopy(ranked[0].config if ranked else self.anchor),
            "session_id": self.session_id, "min_improvement": self.min_improvement,
            "seen": sorted(self.seen),
            "observations": [asdict(result) for result in self.observations],
            "best": asdict(self.best) if self.best is not None else None,
            "anchor": deepcopy(self.anchor), "pending": deepcopy(self.pending),
            "phase": self.phase, "reason": self.reason, "done": self.done,
            "proposed": self.proposed, "inflight": deepcopy(self.inflight),
            "center": asdict(self.center) if self.center is not None else None,
            "samples": [[deepcopy(config), asdict(result)] for config, result in self.samples],
            "radius": {str(axis): radius for axis, radius in self.radius.items()},
        }

    def load_state(self, blob: dict) -> None:
        # Accept the previous V2 identifier only when reading existing checkpoints.
        if (blob.get("strategy") != self.state_strategy or blob.get("context") != self.context):
            return
        if blob.get("version") != 2 or blob.get("min_improvement") != self.min_improvement:
            return
        self.seen = set(blob["seen"])
        self.observations = [TrialResult(**deepcopy(row)) for row in blob["observations"]]
        self.best = TrialResult(**deepcopy(blob["best"])) if blob["best"] is not None else None
        self.anchor = deepcopy(blob["anchor"])
        self.pending = [(deepcopy(config), reason) for config, reason in blob["pending"]]
        self.phase, self.reason, self.done = blob["phase"], blob["reason"], blob["done"]
        same_session = self.session_id is not None and blob.get("session_id") == self.session_id
        self.proposed = blob["proposed"] if same_session else 0
        self.inflight = None
        if blob["inflight"] is not None:
            # A proposal without a completed observation must remain eligible for retry.
            full = deepcopy(blob["inflight"])
            self.seen.discard(canonical_key(prune_inactive(full, self.space)))
            self.pending.insert(0, (full, blob["reason"]))
            if same_session:
                self.proposed = max(0, self.proposed - 1)

        self.center = (TrialResult(**deepcopy(blob["center"]))
                       if blob["center"] is not None else None)
        self.samples = [(deepcopy(config), TrialResult(**deepcopy(result)))
                        for config, result in blob["samples"]]
        samples, self.samples = self.samples, []
        self.direction_blocks = []
        for config, result in samples:
            self._update_direction_blocks(config, result, emit=False)
            self.samples.append((config, result))
        self.radius = {int(axis): radius for axis, radius in blob["radius"].items()}
        self.expected = None

    def _adjacent_neighbors(self, config: dict) -> list[tuple[dict, str]]:
        proposals = []
        for axis in self.tunable:
            rows = self.axes[axis]
            index = next(i for i, row in enumerate(rows)
                         if all(config.get(k) == v for k, v in row.items()))
            # Workload priority chooses the axis; measurement chooses direction.
            for j in (index - 1, index + 1):
                if 0 <= j < len(rows):
                    candidate = dict(config)
                    candidate.update(rows[j])
                    proposals.append((candidate, f"{self.phase}: {','.join(rows[j])}"))
        return proposals

    def _next_proposal(self) -> dict | None:
        if self.inflight is not None:
            raise RuntimeError("observe the previous proposal before requesting another")
        while not self.done and self.proposed < self.max_trials:
            while self.pending:
                full, reason = self.pending.pop(0)
                config = prune_inactive(full, self.space)
                key = canonical_key(config)
                if key in self.seen:
                    continue
                self.seen.add(key)
                self.inflight = full
                self.reason = reason
                self.proposed += 1
                return deepcopy(config)
            if self.phase == "fine":
                self.done = True
                break
            self.phase = "coarse" if self.phase == "baseline" else "fine"
            self.pending = self._neighbors(self.anchor, self.phase == "coarse")
        return None

    def _metric_context(self, result: TrialResult | None) -> dict | None:
        """Snapshot the measured inputs actually used for ranking and feasibility."""
        if result is None:
            return None
        metrics = self._metrics(result)
        raw = result.raw.get("metrics")
        raw_metrics = raw if isinstance(raw, dict) else {}
        return {
            "trial_id": result.trial_id, "config": result.config,
            "objective": result.objective, "objective_valid": self._finite(result.objective),
            "platform_feasible": result.feasible, "feasible": self._feasible(result),
            "failure_class": result.failure_class, "metrics": metrics,
            "missing_metrics": sorted(self.required_metrics - raw_metrics.keys()),
            "invalid_metrics": {key: raw_metrics[key] for key in sorted(self.required_metrics)
                                if key in raw_metrics and key not in metrics},
            "metrics_type": type(raw).__name__,
            "violation": self._violation(metrics),
        }

    def _violation(self, metrics: dict, *, margin: bool = False) -> tuple[float, float]:
        """Worst relative breach, then total breach; missing telemetry is unknown."""
        breaches = []
        for spec in self.constraints:
            name = spec["metric"]
            if name not in metrics:
                return math.inf, math.inf
            limit = spec["value"]
            scale = max(abs(limit), 1e-9)
            sign = 1 if spec["op"] in ("<", "<=") else -1
            target = limit - sign * scale * self.safety_margin if margin else limit
            delta = sign * (metrics[name] - target) / scale
            strict = spec["op"] in ("<", ">")
            breaches.append(max(0.0, delta, 1e-12 if strict and delta >= 0 else 0.0))
        return max(breaches, default=0.0), sum(breaches)

    def _better_center(self, result: TrialResult) -> bool:
        if result.failure_class:
            return False
        if self.center is None:
            return self._feasible(result) or (
                bool(self.constraints) and math.isfinite(self._violation(self._metrics(result))[0]))
        feasible, old_feasible = self._feasible(result), self._feasible(self.center)
        if feasible != old_feasible:
            return feasible
        if not feasible:
            return self._violation(self._metrics(result)) < self._violation(self._metrics(self.center))
        sign = 1 if self.direction == "max" else -1
        return ((result.objective - self.center.objective) * sign
                > self.min_improvement * max(abs(self.center.objective), 1e-9))

    def _index(self, config: dict, axis: int) -> int:
        return next(i for i, row in enumerate(self.axes[axis])
                    if all(config.get(k) == v for k, v in row.items()))

    def _coordinate(self, axis: int, index: int) -> float | None:
        rows = self.axes[axis]
        if not all(self._finite(v) for row in rows for v in row.values()):
            return None
        return float(next(iter(rows[index].values()))) if len(rows[index]) == 1 else float(index)

    def _predict(self, config: dict, axis: int) -> dict | None:
        """Nearest local secant through the current center, holding other axes fixed."""
        if self.center is None or self.center.failure_class:
            return None
        keys = set(self.axes[axis][0])
        active = prune_inactive(self.anchor, self.space)
        candidate_active = prune_inactive(config, self.space)
        # Conditional activation changes cannot be attributed to this axis alone.
        if set(active) != set(candidate_active) or not keys <= active.keys():
            return None
        current = self._coordinate(axis, self._index(self.anchor, axis))
        target = self._coordinate(axis, self._index(config, axis))
        if current is None or target is None:
            return None
        neighbors = []
        for full, result in self.samples:
            observed = prune_inactive(full, self.space)
            if (result.failure_class or set(observed) != set(active)
                    or any(observed[k] != active[k] for k in active if k not in keys)):
                continue
            x = self._coordinate(axis, self._index(full, axis))
            if x != current:
                neighbors.append((abs(x - current), x, result))
        if not neighbors:
            return None
        _, x, prior = min(neighbors, key=lambda item: item[0])
        ratio = (target - current) / (current - x)
        now_metrics, old_metrics = self._metrics(self.center), self._metrics(prior)
        metrics = {k: value + ratio * (value - old_metrics[k])
                   for k, value in now_metrics.items() if k in old_metrics}
        objective = None
        if self._finite(self.center.objective) and self._finite(prior.objective):
            objective = self.center.objective + ratio * (self.center.objective - prior.objective)
        if not self._finite(objective):
            objective = None
        metrics = {k: v for k, v in metrics.items() if self._finite(v)}
        if metrics.keys() != self.required_metrics or (not self.constraints and objective is None):
            return None
        logger.debug("Prediction inputs: axis=%s candidate=%r center=%r prior=%r ratio=%s "
                     "predicted_metrics=%r predicted_objective=%s",
                     axis, config, self._metric_context(self.center),
                     self._metric_context(prior), ratio, metrics, objective)
        return {"metrics": metrics, "objective": objective}

    def _direction_context(self, config: dict, axis: int) -> str | None:
        """Directions are meaningful only for active, single numeric axes."""
        keys = set(self.axes[axis][0])
        active = prune_inactive(config, self.space)
        if (len(keys) != 1 or not keys <= active.keys()
                or self._coordinate(axis, 0) is None):
            return None
        return canonical_key({"active": sorted(active),
                              "other": {k: v for k, v in active.items() if k not in keys}})

    def _update_direction_blocks(self, full: dict, result: TrialResult, *, emit=True) -> None:
        """Pause points beyond an observed worsening breach, within the same context.

        Compare the nearest previous measurement on either side. Missing telemetry,
        execution failures and changes in conditional activation supply no evidence.
        Samples are replayed through this method on restore, so blocks are derived
        from measurements rather than an independent checkpoint format.
        """
        metrics = self._metrics(result)
        if result.failure_class or metrics.keys() != self.required_metrics:
            return
        for axis in self.tunable:
            context = self._direction_context(full, axis)
            if context is None:
                continue
            value = self._coordinate(axis, self._index(full, axis))
            nearest = {}
            for prior_full, prior in self.samples:
                if (prior.failure_class or self._direction_context(prior_full, axis) != context
                        or self._metrics(prior).keys() != self.required_metrics):
                    continue
                x = self._coordinate(axis, self._index(prior_full, axis))
                if x == value:
                    continue
                direction = 1 if value > x else -1
                if direction not in nearest or abs(value - x) < nearest[direction][0]:
                    nearest[direction] = (abs(value - x), prior)
            for direction, (_, prior) in nearest.items():
                old_metrics = self._metrics(prior)
                for spec in self.constraints:
                    name, limit = spec["metric"], spec["value"]
                    sign = 1 if spec["op"] in ("<", "<=") else -1
                    breach = sign * (metrics[name] - limit)
                    fails = breach > 0 or (breach == 0 and spec["op"] in ("<", ">"))
                    worsening = sign * (metrics[name] - old_metrics[name])
                    scale = max(abs(limit), abs(old_metrics[name]), 1e-9)
                    if not fails or worsening / scale < self.pruning_threshold:
                        continue
                    block = {"axis": axis, "context": context, "direction": direction,
                             "boundary": value, "metric": name, "op": spec["op"],
                             "limit": limit, "previous": old_metrics[name],
                             "measured": metrics[name], "worsening": worsening / scale,
                             "config": deepcopy(result.config),
                             "previous_config": deepcopy(prior.config)}
                    if block in self.direction_blocks:
                        continue
                    self.direction_blocks.append(block)
                    if emit:
                        logger.info("Direction pruned: parameter=%s direction=%s boundary=%s "
                                    "metric=%s previous=%s measured=%s limit=%s",
                                    next(iter(self.axes[axis][0])), direction, value,
                                    name, old_metrics[name], metrics[name], limit)
                        if self.decision_log is not None:
                            self.decision_log({"event": "direction_pruned",
                                               "decision_id": self.proposed,
                                               "threshold": self.pruning_threshold,
                                               **deepcopy(block)})

    def _direction_pruned(self, config: dict) -> bool:
        for block in self.direction_blocks:
            axis = block["axis"]
            if self._direction_context(config, axis) != block["context"]:
                continue
            value = self._coordinate(axis, self._index(config, axis))
            if block["direction"] * (value - block["boundary"]) > 0:
                logger.debug("Candidate pruned: config=%r evidence=%r", config, block)
                return True
        return False

    def _neighbors(self, config: dict, coarse: bool) -> list[tuple[dict, str]]:
        candidates = []
        for axis in self.tunable:
            index = self._index(config, axis)
            # Unknown directions start with one-row probes. Reliable secants can
            # rank larger legal steps, bounded by the per-axis trust radius.
            for offset in range(1, (self.radius[axis] if coarse else 1) + 1):
                for direction in (-1, 1):
                    j = index + direction * offset
                    if not 0 <= j < len(self.axes[axis]):
                        continue
                    candidate = dict(config)
                    candidate.update(self.axes[axis][j])
                    if self._direction_pruned(candidate):
                        continue
                    prediction = self._predict(candidate, axis)
                    if prediction is None and offset > 1:
                        continue
                    # Unknown directions stay available, ahead of predicted violations
                    # when feasible. In repair mode known reductions take priority.
                    if prediction is None:
                        rank = (1, 0, 0, 0, offset)
                    else:
                        violation = self._violation(prediction["metrics"], margin=True)
                        sign = -1 if self.direction == "max" else 1
                        score = (sign * prediction["objective"]
                                 if prediction["objective"] is not None else math.inf)
                        safe = violation == (0, 0)
                        repairing = self.center is not None and not self._feasible(self.center)
                        improves = (repairing and violation < self._violation(
                            self._metrics(self.center), margin=True))
                        rank = (0 if safe or improves else 2, *violation, score, offset)
                    reason = (f"{self.phase}: {','.join(self.axes[axis][j])}; "
                              f"step={direction * offset}; "
                              f"{'local prediction' if prediction else 'exploration'}")
                    candidates.append((rank, candidate, reason))
                    logger.debug("Candidate ranking: config=%r prediction=%r rank=%r reason=%s",
                                 candidate, prediction, rank, reason)
        candidates.sort(key=lambda item: item[0])
        return [(config, reason) for _, config, reason in candidates]

    def next_config(self) -> dict | None:
        # Also cover restored queues and OOM repair candidates queued after observations.
        self.pending = [(config, reason) for config, reason in self.pending
                        if not self._direction_pruned(config)]
        config = self._next_proposal()
        self.expected = None
        if config is not None and self.center is not None:
            changed = [axis for axis in self.tunable
                       if self._index(self.inflight, axis) != self._index(self.anchor, axis)]
            if len(changed) == 1:
                axis = changed[0]
                prediction = self._predict(self.inflight, axis)
                if prediction is not None:
                    self.expected = {"axis": axis, **prediction}
        if config is not None:
            logger.info("Decision %s proposal: reason=%s config=%r center=%r prediction=%r",
                        self.proposed, self.reason, config,
                        self._metric_context(self.center), self.expected)
        if config is not None and self.decision_log is not None:
            center = prune_inactive(self.anchor, self.space)
            changes = [{"param": key, "from": center.get(key), "to": config.get(key),
                        "operation": "remove" if key not in config else
                                     "add" if key not in center else "change"}
                       for key in sorted(center.keys() | config.keys())
                       if (key in center) != (key in config) or center.get(key) != config.get(key)]
            self.decision_log({
                "event": "proposal", "decision_id": self.proposed,
                "strategy": self.state_strategy, "phase": self.phase, "reason": self.reason,
                "center_config": center, "config": config, "changes": changes,
                "prediction": deepcopy(self.expected),
                "center_metrics": self._metric_context(self.center),
                "radius": {str(axis): radius for axis, radius in self.radius.items()},
            })
        return config

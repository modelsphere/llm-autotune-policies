"""Sampling the campaign's declared search space.

The manifest hands you ``search_space`` in the platform's own vocabulary — the
same four kinds a campaign author writes:

    {
      "base":  {"mem_fraction_static": 0.9},        # fixed on every config
      "grid":  {"tp_size": [1, 2, 4]},              # one value, chosen freely
      "tied":  [{"tp_size": [1,2], "mem": [0.9,0.85]}],   # one ROW, columns zipped
      "range": {"chunked_prefill_size": {"min": 2048, "max": 32768, "step": 2048}},
      "conditions": {"speculative_num_steps": {"speculative_algorithm": ["EAGLE"]}}
    }

This module turns that into concrete configs. It is deliberately small and
dependency-free so it can live inside any policy image; it mirrors the
platform's semantics closely enough to dedup locally (the platform is still the
authority — it canonicalises and echoes every config back), but a strategy that
wants to step *outside* the space just builds its own dict and declares the
extra in its plan.
"""

from __future__ import annotations

import json
import random
from typing import Any


def _grid(space: dict) -> dict[str, list]:
    return {k: v for k, v in (space.get("grid") or {}).items() if v}


def _tied(space: dict) -> list[dict[str, list]]:
    return [g for g in (space.get("tied") or []) if isinstance(g, dict) and any(g.values())]


def range_values(spec: dict) -> list:
    """Every value in an interval, inclusive of ``max`` when it lands on a step.

    Floats are rounded to the step's precision so the value we send hashes the
    same as the platform's own expansion of it (0.7 + 3*0.05 is 0.850000…1 in
    binary, which would look like a different config)."""
    lo, hi, step = float(spec["min"]), float(spec["max"]), float(spec["step"])
    is_int = all(float(v).is_integer() for v in (lo, hi, step))
    text = repr(step)
    decimals = 0 if ("e" in text or "." not in text) else len(text.split(".")[1].rstrip("0"))
    out, i = [], 0
    while True:
        v = lo + i * step
        if v > hi + step * 1e-9:
            break
        out.append(round(v) if is_int else round(v, decimals))
        i += 1
        if i > 100_000:  # a typo'd step is a bug, not a 10k-config night
            break
    return out


def _ranges(space: dict) -> dict[str, list]:
    return {k: range_values(v) for k, v in (space.get("range") or {}).items()
            if isinstance(v, dict) and {"min", "max", "step"} <= set(v)}


def _same(a: Any, b: Any) -> bool:
    if a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).strip().lower() == str(b).strip().lower()
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def prune_inactive(config: dict, space: dict) -> dict:
    """Drop parameters whose condition is not met — before hashing, so two
    configs differing only in an ignored flag are one config, as they are on
    the platform."""
    conditions = space.get("conditions") or {}
    if not conditions:
        return config
    active = dict(config)
    for _ in range(len(conditions) + 1):
        removed = False
        for name, clause in conditions.items():
            if name not in active:
                continue
            holds = all(
                key in active and any(_same(active[key], v) for v in vals)
                for key, vals in clause.items()
            )
            if not holds:
                del active[name]
                removed = True
        if not removed:
            break
    return active


def canonical_key(config: dict) -> str:
    """A stable dedup key — the platform hashes the same canonical form."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def sample(space: dict, rng: random.Random) -> dict:
    """One random configuration from the space: base, one value per grid axis,
    one row per tied group, one step per range, conditions pruned."""
    config = dict(space.get("base") or {})
    for key, values in _grid(space).items():
        config[key] = rng.choice(values)
    for group in _tied(space):
        keys = sorted(group)
        rows = min(len(group[k]) for k in keys)
        i = rng.randrange(rows)
        for k in keys:
            config[k] = group[k][i]
    for key, values in _ranges(space).items():
        config[key] = rng.choice(values)
    return prune_inactive(config, space)


def swept_keys(space: dict) -> list[str]:
    """Every parameter the space actually varies — a good default ``plan.tuning``."""
    keys = list(_grid(space))
    for group in _tied(space):
        keys += [k for k in group if k not in keys]
    keys += [k for k in _ranges(space) if k not in keys]
    return sorted(keys)


def _declared_values(space: dict) -> dict[str, list]:
    """Every value each swept parameter is declared to take, by parameter."""
    out: dict[str, list] = {}
    for key, values in _grid(space).items():
        out.setdefault(key, [])
        out[key] += [v for v in values if v not in out[key]]
    for group in _tied(space):
        for key, values in group.items():
            out.setdefault(key, [])
            out[key] += [v for v in values if v not in out[key]]
    for key, values in _ranges(space).items():
        out.setdefault(key, [])
        out[key] += [v for v in values if v not in out[key]]
    return out


def _axis_count(space: dict) -> int:
    """Candidates before conditions prune anything — the product of the axes.
    Used only for a completion estimate, so an over-count under conditions is
    harmless (the platform derives the exact coverage from its own ledger)."""
    total = 1
    for values in _grid(space).values():
        total *= max(1, len(values))
    for group in _tied(space):
        total *= max(1, min(len(v) for v in group.values()))
    for values in _ranges(space).values():
        total *= max(1, len(values))
    return total


def coverage_from_trials(space: dict, configs: list[dict]) -> dict:
    """A self-reported coverage payload, built for you from the configs tried.

    Keyed in the platform's own vocabulary (``swept_keys``) so it lines up with
    the platform's authoritative view. Reports the distinct value reached on
    each swept axis and a rough completion fraction for an enumerable space.
    This is the *supplement* the platform cannot derive on its own — override
    ``Strategy.coverage`` to add deliberate skips or a model-based estimate.
    """
    declared = _declared_values(space)
    seen: dict[str, list] = {k: [] for k in declared}
    for config in configs:
        for key in declared:
            if key in config and not any(_same(config[key], v) for v in seen[key]):
                seen[key].append(config[key])
    dimensions = [{"param": k, "tried_values": seen[k]} for k in sorted(declared)]

    fraction = None
    total = _axis_count(space)
    if 0 < total <= 5000:
        distinct = len({canonical_key(c) for c in configs})
        fraction = round(min(1.0, distinct / total), 4)
    return {"fraction": fraction, "dimensions": dimensions}


def cards_for(config: dict) -> int:
    """How many GPUs an engine with this config occupies: tp × dp × pp.

    The platform allocates a whole machine's cards to the session, but a single
    engine launch must request exactly the cards its parallelism uses — the
    launch endpoint rejects a mismatch. Accepts either the short or `_size`
    spelling of each degree; a missing one is 1."""
    def _deg(*names: str) -> int:
        for n in names:
            if n in config:
                try:
                    return max(1, int(config[n]))
                except (TypeError, ValueError):
                    return 1
        return 1
    return _deg("tp_size", "tp") * _deg("dp_size", "dp") * _deg("pp_size", "pp")

"""The campaign's search space in Optuna's vocabulary, both ways.

The manifest describes the space as ``{base, grid, tied, range, conditions}``;
Optuna wants (param name → distribution) and (param name → value). This module
is the only translation between the two.

- ``distributions()`` maps each independent axis to a distribution. A grid key
  becomes a categorical. A tied group becomes ONE integer index over its rows:
  the group varies together, so modelling its columns separately would propose
  combinations the author ruled out, and the rows are author-ordered (a tp
  ladder), so an integer is closer to the truth than an unordered label. A
  range becomes a numeric distribution: sixteen chunk sizes are an interval to
  a model, not sixteen unrelated categories.

- ``decode()`` turns a sampled point back into engine args: base merged, tied
  row expanded, range values snapped to the declared steps, inactive
  parameters pruned. Snapping matters: Optuna computes ``low + k*step`` in
  float arithmetic and returns 0.7999999999999999 where the space says 0.8,
  which the platform would treat as a different config.

- ``encode()`` turns a measured config back into params, for replay. It is
  deliberately partial: a config that no longer fits the space (the author
  edited it, or a value fell outside a range) keeps the params that still
  match. Raising on old history would strand the search, and a partial
  observation still teaches the model something.

Float ranges are given to Optuna WITHOUT their step. A stepped
FloatDistribution must end exactly ``k`` steps above its low, which is a
float-equality bet lost on inputs as plain as (0.0, 0.9, 0.3). A continuous
interval plus snapping on decode proposes the same configs and cannot lose it.
Integer ranges keep their step: integer arithmetic is exact.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

from autotune_policy.space import prune_inactive, range_values

# A tied group has no parameter name of its own, so its index param is
# synthetic. The prefix keeps it from colliding with a real engine arg, and
# decode() strips it back out.
TIED_PREFIX = "__tied_"


@dataclass(frozen=True)
class RangeAxis:
    name: str
    values: tuple
    step: float
    is_integer: bool


def grid_axes(space: dict) -> dict[str, list]:
    return {k: list(v) for k, v in sorted((space.get("grid") or {}).items()) if v}


def tied_groups(space: dict) -> list[dict[str, list]]:
    return [g for g in (space.get("tied") or []) if isinstance(g, dict) and any(g.values())]


def range_axes(space: dict) -> list[RangeAxis]:
    out = []
    for name, spec in sorted((space.get("range") or {}).items()):
        if not (isinstance(spec, dict) and {"min", "max", "step"} <= set(spec)):
            continue
        values = tuple(range_values(spec))
        if not values:
            continue
        is_integer = all(float(spec[k]).is_integer() for k in ("min", "max", "step"))
        out.append(RangeAxis(name, values, float(spec["step"]), is_integer))
    return out


def _rows(group: dict[str, list]) -> int:
    return min(len(group[k]) for k in sorted(group))


def _same(a: Any, b: Any) -> bool:
    """Equal as the platform compares values: 2 == 2.0, "true" == True."""
    if a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return str(a).strip().lower() == str(b).strip().lower()
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snap(value: Any, axis: RangeAxis) -> Any:
    """The declared value nearest to what Optuna sampled."""
    return min(axis.values, key=lambda v: abs(v - float(value)))


def distributions(space: dict) -> dict[str, BaseDistribution]:
    """Every axis of the space as (param name → Optuna distribution)."""
    out: dict[str, BaseDistribution] = {}
    for key, values in grid_axes(space).items():
        out[key] = CategoricalDistribution(tuple(values))
    for index, group in enumerate(tied_groups(space)):
        rows = _rows(group)
        # A one-row group still has to appear: decode() applies the row's
        # values, and dropping the param would drop them from every proposal.
        if rows == 1:
            out[f"{TIED_PREFIX}{index}"] = CategoricalDistribution((0,))
        else:
            out[f"{TIED_PREFIX}{index}"] = IntDistribution(0, rows - 1)
    for axis in range_axes(space):
        if len(axis.values) == 1:
            out[axis.name] = CategoricalDistribution((axis.values[0],))
        elif axis.is_integer:
            out[axis.name] = IntDistribution(
                int(axis.values[0]), int(axis.values[-1]), step=int(axis.step))
        else:
            out[axis.name] = FloatDistribution(float(axis.values[0]), float(axis.values[-1]))
    return out


def decode(params: dict[str, Any], space: dict) -> dict[str, Any]:
    """A sampled point as the engine args it deploys."""
    merged: dict[str, Any] = dict(space.get("base") or {})
    groups = tied_groups(space)
    axes = {axis.name: axis for axis in range_axes(space)}
    for key in sorted(params):
        value = params[key]
        if key.startswith(TIED_PREFIX):
            index = int(key[len(TIED_PREFIX):])
            if index >= len(groups):
                continue  # a stale param from an edited space
            group = groups[index]
            row = min(int(value), _rows(group) - 1)
            for column in sorted(group):
                merged[column] = group[column][row]
        elif key in axes:
            merged[key] = _snap(value, axes[key])
        else:
            merged[key] = value
    return prune_inactive(merged, space)


def encode(config: dict[str, Any], space: dict) -> dict[str, Any]:
    """A measured config as Optuna params: the parts of it this space still
    declares, spelled the way the SPACE spells them. Optuna checks a param
    against its distribution by value and type, and the config may have
    round-tripped through JSON."""
    params: dict[str, Any] = {}
    for key, values in grid_axes(space).items():
        if key not in config:
            continue
        match = next((v for v in values if _same(config[key], v)), None)
        if match is not None:
            params[key] = match
    for index, group in enumerate(tied_groups(space)):
        columns = sorted(group)
        for row in range(_rows(group)):
            if all(column in config and _same(config[column], group[column][row])
                   for column in columns):
                params[f"{TIED_PREFIX}{index}"] = row
                break
    for axis in range_axes(space):
        raw = _as_float(config.get(axis.name))
        if raw is None:
            continue
        margin = axis.step * 1e-9
        if raw < axis.values[0] - margin or raw > axis.values[-1] + margin:
            continue  # outside the current interval: the space moved
        snapped = _snap(raw, axis)
        params[axis.name] = int(snapped) if axis.is_integer else float(snapped)
    return params


def enumerate_space(space: dict, limit: int = 100_000) -> list[dict[str, Any]]:
    """Every config the space declares, conditions applied, without
    duplicates. For tests and for a small space's exhaustion check."""
    axes: list[list[dict[str, Any]]] = []
    for key, values in grid_axes(space).items():
        axes.append([{key: v} for v in values])
    for group in tied_groups(space):
        columns = sorted(group)
        axes.append([{c: group[c][row] for c in columns} for row in range(_rows(group))])
    for axis in range_axes(space):
        axes.append([{axis.name: v} for v in axis.values])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for combo in itertools.islice(itertools.product(*axes), limit):
        config = dict(space.get("base") or {})
        for part in combo:
            config.update(part)
        config = prune_inactive(config, space)
        key = repr(sorted(config.items()))
        if key not in seen:
            seen.add(key)
            out.append(config)
    return out

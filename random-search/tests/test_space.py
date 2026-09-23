"""The space sampler: samples stay inside the declared space, ranges land on
steps, tied columns move together, and conditions prune before hashing."""

from __future__ import annotations

import random

from autotune_policy.space import (
    canonical_key,
    prune_inactive,
    range_values,
    sample,
    swept_keys,
)

SPACE = {
    "base": {"mem_fraction_static": 0.9},
    "grid": {"tp_size": [1, 2, 4]},
    "tied": [{"a": [1, 2, 3], "b": [10, 20, 30]}],
    "range": {"chunked_prefill_size": {"min": 2048, "max": 8192, "step": 2048}},
    "conditions": {"speculative_num_steps": {"speculative_algorithm": ["EAGLE"]}},
}


def test_range_values_land_on_steps_and_include_max():
    assert range_values({"min": 2048, "max": 8192, "step": 2048}) == [2048, 4096, 6144, 8192]
    assert range_values({"min": 0.7, "max": 0.9, "step": 0.05}) == [0.7, 0.75, 0.8, 0.85, 0.9]


def test_samples_stay_inside_the_declared_space_and_tie_moves_together():
    rng = random.Random(0)
    for _ in range(200):
        c = sample(SPACE, rng)
        assert c["mem_fraction_static"] == 0.9  # base is always present
        assert c["tp_size"] in (1, 2, 4)
        assert c["chunked_prefill_size"] in (2048, 4096, 6144, 8192)
        # tied columns zip: a=1<->b=10, a=2<->b=20, a=3<->b=30
        assert {1: 10, 2: 20, 3: 30}[c["a"]] == c["b"]


def test_conditions_prune_before_hashing():
    active = prune_inactive(
        {"speculative_algorithm": "NONE", "speculative_num_steps": 3}, SPACE
    )
    assert "speculative_num_steps" not in active  # gate not met -> dropped
    kept = prune_inactive(
        {"speculative_algorithm": "EAGLE", "speculative_num_steps": 3}, SPACE
    )
    assert kept["speculative_num_steps"] == 3


def test_canonical_key_is_order_independent():
    assert canonical_key({"a": 1, "b": 2}) == canonical_key({"b": 2, "a": 1})


def test_swept_keys_lists_every_varied_parameter():
    assert swept_keys(SPACE) == ["a", "b", "chunked_prefill_size", "tp_size"]

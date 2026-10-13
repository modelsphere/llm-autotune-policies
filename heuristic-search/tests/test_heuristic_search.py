from copy import deepcopy

import pytest

from autotune_policy import SearchLoop
from autotune_policy.strategy import TrialResult
from strategies.heuristic_search import HeuristicSearch
from tests.test_night import MANIFEST, FakePlatform


def manifest(space, direction="max"):
    result = deepcopy(MANIFEST)
    result["search_space"] = space
    result["objective"]["screen"]["direction"] = direction
    return result


@pytest.mark.parametrize("direction", ["min", "max"])
def test_local_search_improves_and_stops_without_cartesian_sweep(direction):
    strategy = HeuristicSearch(workload="batch", max_trials=30)
    strategy.setup(manifest({"grid": {
        "max_running_requests": [8, 16, 32, 64, 128],
        "chunked_prefill_size": [1024, 2048, 4096, 8192, 16384],
    }}, direction))
    measured = []
    while (config := strategy.next_config()) is not None:
        cost = abs(config["max_running_requests"] - 64)
        cost += abs(config["chunked_prefill_size"] - 8192) / 1024
        objective = cost if direction == "min" else 100 - cost
        measured.append(config)
        strategy.observe(TrialResult(config, objective, True, trial_id=len(measured)))
    best = strategy.contenders()[0]["launch_spec"]["engine_args"]
    assert best == {"max_running_requests": 64, "chunked_prefill_size": 8192}
    assert len(measured) < 25


def test_tied_conditions_and_declared_topology_are_preserved():
    strategy = HeuristicSearch(max_trials=12, baseline={"tp_size": 2})
    strategy.setup(manifest({
        "base": {"speculative_algorithm": "NONE"},
        "grid": {"tp_size": [1, 2], "prefill_max_requests": [1, 2]},
        "tied": [{"chunked_prefill_size": [2048, 4096, 8192],
                  "max_running_requests": [16, 32, 64]}],
        "conditions": {"prefill_max_requests": {"speculative_algorithm": ["DSPARK"]}},
    }))
    assert "tp_size" in strategy.plan()["tuning"]
    topologies = set()
    while (config := strategy.next_config()) is not None:
        topologies.add(config["tp_size"])
        assert "prefill_max_requests" not in config
        assert config["chunked_prefill_size"] == config["max_running_requests"] * 128
        strategy.observe(TrialResult(config, 1.0, True))
    assert topologies == {1, 2}


def test_oom_repairs_lower_neighbor_and_excludes_failure_from_contenders():
    strategy = HeuristicSearch(workload="long_prefill")
    strategy.setup(manifest({"grid": {"chunked_prefill_size": [2048, 4096, 8192]}}))
    first = strategy.next_config()
    strategy.observe(TrialResult(first, None, False, failure_class="prefill_oom"))
    repair = strategy.next_config()
    assert repair["chunked_prefill_size"] < first["chunked_prefill_size"]
    strategy.observe(TrialResult(repair, 10.0, True))
    assert len(strategy.contenders()) == 1


def test_state_restores_best_and_rejects_different_context():
    spec = manifest({"grid": {"max_running_requests": [16, 32, 64]}})
    original = HeuristicSearch()
    original.setup(spec)
    config = original.next_config()
    original.observe(TrialResult(config, 100, True))
    state = original.dump_state()
    restored = HeuristicSearch()
    restored.setup(spec)
    restored.load_state(state)
    assert restored.contenders() == original.contenders()
    assert restored.next_config() != config
    changed = HeuristicSearch(baseline={"max_running_requests": 16})
    changed.setup(spec)
    changed.load_state(state)
    assert changed.next_config()["max_running_requests"] == 16


def test_budget_and_full_platform_lifecycle():
    fake = FakePlatform(MANIFEST, target_trials=999)
    strategy = HeuristicSearch(max_trials=2)
    assert SearchLoop(strategy, sleep=lambda _: None).run(fake) == 0
    assert fake.trials == 2
    assert "exhausted" in fake.heartbeat_statuses
    assert fake.finalized_called and fake.served


@pytest.mark.parametrize("space", [
    {"range": {"max_running_requests": {"min": 1, "max": 10, "step": 0}}},
    {"tied": [{"a": [1], "b": [1, 2]}]},
    {"grid": {"max_running_requests": []}},
])
def test_invalid_space_fails_at_setup(space):
    with pytest.raises(ValueError):
        HeuristicSearch().setup(manifest(space))


def test_nonfinite_and_infeasible_scores_never_become_contenders():
    strategy = HeuristicSearch()
    strategy.setup(manifest({"grid": {"max_running_requests": [16, 32, 64]}}))
    config = strategy.next_config()
    strategy.observe(TrialResult(config, float("nan"), True))
    config = strategy.next_config()
    strategy.observe(TrialResult(config, 999, False))
    assert strategy.contenders() == []


def test_offset_range_values_stay_on_declared_grid():
    strategy = HeuristicSearch()
    strategy.setup(manifest({"range": {
        "mem_fraction_static": {"min": 0.75, "max": 0.95, "step": 0.1},
    }}))
    measured = []
    while (config := strategy.next_config()) is not None:
        measured.append(config["mem_fraction_static"])
        strategy.observe(TrialResult(config, 1.0, True))
    assert set(measured) == {0.75, 0.85, 0.95}


def test_unlisted_tied_column_does_not_lock_group():
    strategy = HeuristicSearch()
    strategy.setup(manifest({"tied": [{
        "max_running_requests": [16, 32], "tp_size": [1, 2],
    }]}))
    assert strategy.plan()["tuning"] == ["max_running_requests", "tp_size"]
    measured = []
    while (config := strategy.next_config()) is not None:
        measured.append(config)
        strategy.observe(TrialResult(config, 1.0, True))
    assert {tuple(sorted(c.items())) for c in measured} == {
        (("max_running_requests", 16), ("tp_size", 1)),
        (("max_running_requests", 32), ("tp_size", 2)),
    }


def test_oom_does_not_treat_disabling_chunking_as_memory_reduction():
    strategy = HeuristicSearch(workload="long_prefill")
    strategy.setup(manifest({"grid": {"chunked_prefill_size": [-1, 4096, 8192]}}))
    config = strategy.next_config()
    strategy.observe(TrialResult(config, None, False, failure_class="oom"))
    assert all(candidate["chunked_prefill_size"] > 0
               for candidate, reason in strategy.pending if reason.startswith("OOM"))


def test_profile_only_prioritizes_axes_and_preserves_unlisted_order():
    strategy = HeuristicSearch()
    strategy.setup(manifest({
        "base": {"fixed": 7},
        "grid": {"custom_first": [0, 1], "singleton": [5],
                 "custom_second": [0, 1], "max_running_requests": [32, 64],
                 "mamba_full_memory_ratio": [0.5, 0.8]},
        "range": {"custom_range": {"min": 0, "max": 1, "step": 1}},
    }))
    measured = []
    while (config := strategy.next_config()) is not None:
        measured.append(config)
        assert config["fixed"] == 7 and config["singleton"] == 5
        strategy.observe(TrialResult(config, 1.0, True))
    baseline = measured[0]
    changes = [[k for k in config if config[k] != baseline[k]] for config in measured[1:]]
    assert changes == [["mamba_full_memory_ratio"], ["max_running_requests"],
                       ["custom_first"], ["custom_second"], ["custom_range"]]


def test_expanded_axes_reject_profile_filtered_checkpoint():
    import json

    spec = manifest({"grid": {"custom": [0, 1]}})
    original = HeuristicSearch()
    original.setup(spec)
    state = original.dump_state()
    state["context"] = json.loads(state["context"])["context"]
    state["done"] = True
    state["pending"] = []
    restored = HeuristicSearch()
    restored.setup(spec)
    restored.load_state(state)
    assert restored.next_config() == {"custom": 0}

from copy import deepcopy

import pytest

from autotune_policy import SearchLoop
from autotune_policy.strategy import TrialResult
from strategies.heuristic_search import HeuristicSearch
from tests.test_heuristic_search import manifest
from tests.test_night import MANIFEST, FakePlatform

CONSTRAINTS = [{"metric": "ttft", "op": "<", "value": 8}]


def strategy(values=(16, 32, 48, 64, 80, 96), baseline=64, **kwargs):
    redlines = kwargs.pop("constraints", [])
    obj = HeuristicSearch(baseline={"max_running_requests": baseline}, **kwargs)
    spec = manifest({"grid": {"max_running_requests": list(values)}})
    spec["objective"]["screen"]["redlines"] = redlines
    obj.setup(spec)
    return obj


def report(obj, config, objective, ttft=None, feasible=True, failure=""):
    obj.observe(TrialResult(config, objective, feasible, failure_class=failure,
                            raw={"metrics": {"ttft": ttft}}))


@pytest.mark.parametrize("raw_metrics,missing,invalid", [
    ({"ttft": 6.0}, [], {}),
    ({"ttft": "6.0"}, [], {"ttft": "6.0"}),
    ({"unrelated": 6.0}, ["ttft"], {}),
    (None, ["ttft"], {}),
])
def test_metric_diagnostics_preserve_raw_and_actual_inputs(caplog, raw_metrics, missing, invalid):
    records = []
    obj = strategy(constraints=CONSTRAINTS, decision_log=records.append)
    with caplog.at_level("INFO"):
        config = obj.next_config()
        obj.observe(TrialResult(config, 100, True, trial_id=17, raw={"metrics": raw_metrics}))
        obj.next_config()
    observation = next(row for row in records if row["event"] == "observation")
    assert observation["raw_metrics"] == raw_metrics
    context = observation["metric_context"]
    assert context["missing_metrics"] == missing
    assert context["invalid_metrics"] == invalid
    assert context["metrics"] == ({} if missing or invalid else {"ttft": 6.0})
    assert context["objective"] == 100
    assert context["trial_id"] == 17
    assert "Decision 1 observation" in caplog.text
    assert "Decision 2 proposal" in caplog.text
    assert ("required metrics missing=" in caplog.text) == bool(missing or invalid)
    if not missing and not invalid:
        assert records[-1]["center_metrics"]["metrics"] == {"ttft": 6.0}


def test_repairs_infeasible_center_then_optimizes_inside_slo():
    obj = strategy(constraints=CONSTRAINTS, max_trials=15)
    measured = []
    while (config := obj.next_config()) is not None:
        n = config["max_running_requests"]
        measured.append(n)
        ttft = 6 + n / 20
        report(obj, config, n, ttft, feasible=ttft < 8)
    assert measured[:2] == [64, 48]
    assert measured[2] == 16  # slope predicts that 32 lacks the requested safety margin
    assert obj.best.config["max_running_requests"] == 32
    assert obj.center.feasible
    assert all(c["launch_spec"]["engine_args"]["max_running_requests"] < 40
               for c in obj.contenders())


@pytest.mark.parametrize("direction", ["min", "max"])
def test_objective_selects_direction_and_expands_step(direction):
    obj = strategy(baseline=48, max_trials=12)
    obj.direction = direction
    sign = 1 if direction == "max" else -1
    first = obj.next_config()
    report(obj, first, sign * 48)
    second = obj.next_config()
    assert second["max_running_requests"] == 32
    report(obj, second, sign * 32)  # lower direction is worse
    third = obj.next_config()
    assert third["max_running_requests"] == 80  # two-row predicted improvement
    report(obj, third, sign * 80)
    assert obj.radius[0] == 3
    assert obj.next_config()["max_running_requests"] == 96


def test_prediction_error_shrinks_step_and_keeps_reverse_direction():
    obj = strategy(baseline=48)
    report(obj, obj.next_config(), 48)
    report(obj, obj.next_config(), 32)
    config = obj.next_config()
    assert config["max_running_requests"] == 80
    report(obj, config, 10)  # the extrapolation was wrong
    assert obj.radius[0] == 1
    assert obj.anchor["max_running_requests"] == 48
    assert obj.next_config()["max_running_requests"] == 64


@pytest.mark.parametrize("ttft", [8, 9, None, float("nan"), float("inf")])
def test_platform_feasible_does_not_override_explicit_slo(ttft):
    obj = strategy(constraints=CONSTRAINTS)
    report(obj, obj.next_config(), 1000, ttft, feasible=True)
    assert obj.contenders() == []
    assert obj.best is None


@pytest.mark.parametrize("op,value,expected", [
    ("<", 8, False), ("<=", 8, True), (">", 8, False), (">=", 8, True),
    (">", 9, True), (">=", 7, False),
])
def test_constraint_operators(op, value, expected):
    constraints = deepcopy(CONSTRAINTS)
    constraints[0]["op"] = op
    obj = strategy(constraints=constraints)
    report(obj, obj.next_config(), 1, value)
    assert bool(obj.contenders()) == expected


def test_missing_objective_can_repair_but_never_win():
    obj = strategy(constraints=CONSTRAINTS)
    report(obj, obj.next_config(), None, 12, feasible=False)
    report(obj, obj.next_config(), None, 10, feasible=False)
    assert obj.anchor["max_running_requests"] == 48
    assert obj.best is None
    assert obj.next_config()["max_running_requests"] == 16


def test_multiple_constraints_and_platform_feasibility():
    constraints = [*CONSTRAINTS, {"metric": "rate", "op": ">=", "value": 100}]
    obj = strategy(constraints=constraints)
    config = obj.next_config()
    obj.observe(TrialResult(config, 1000, True,
                            raw={"metrics": {"ttft": 7, "rate": 90}}))
    assert obj.contenders() == []
    assert obj._violation({"ttft": 10, "rate": 80}) == pytest.approx((0.25, 0.45))
    config = obj.next_config()
    obj.observe(TrialResult(config, 1000, False,
                            raw={"metrics": {"ttft": 7, "rate": 110}}))
    assert obj.contenders() == []


def test_uneven_numeric_values_use_actual_distance():
    obj = strategy(values=(8, 16, 64, 128), baseline=64)
    report(obj, obj.next_config(), 64)
    report(obj, obj.next_config(), 16)
    prediction = obj._predict({"max_running_requests": 128}, 0)
    assert prediction["objective"] == pytest.approx(128)


def test_other_axis_changes_do_not_supply_a_slope():
    obj = HeuristicSearch()
    obj.setup(manifest({"grid": {"max_running_requests": [16, 32, 64],
                                "chunked_prefill_size": [2048, 4096, 8192]}}))
    report(obj, obj.next_config(), 100)
    full = {"max_running_requests": 16, "chunked_prefill_size": 2048}
    obj.samples.append((full, TrialResult(full, 50, True)))
    assert obj._predict({"max_running_requests": 64, "chunked_prefill_size": 4096}, 0) is None


def test_tied_axes_fixed_topology_conditions_and_budget():
    obj = HeuristicSearch(max_trials=5, baseline={"tp_size": 2})
    obj.setup(manifest({"base": {"mode": "off"},
                        "grid": {"tp_size": [1, 2], "prefill_max_requests": [1, 2]},
                        "tied": [{"max_running_requests": [16, 32, 64],
                                  "chunked_prefill_size": [2048, 4096, 8192]}],
                        "conditions": {"prefill_max_requests": {"mode": ["on"]}}}))
    configs = []
    while (config := obj.next_config()) is not None:
        assert config["tp_size"] == 2
        assert "prefill_max_requests" not in config
        assert config["chunked_prefill_size"] == config["max_running_requests"] * 128
        configs.append(config)
        report(obj, config, config["max_running_requests"])
    assert len(configs) <= 5
    assert len({str(c) for c in configs}) == len(configs)


def test_oom_repair_and_inflight_contract():
    obj = strategy()
    config = obj.next_config()
    with pytest.raises(RuntimeError):
        obj.next_config()
    report(obj, config, None, feasible=False, failure="oom")
    assert obj.next_config()["max_running_requests"] == 48


def test_warm_start_restores_and_rejects_changed_constraints():
    obj = strategy(constraints=CONSTRAINTS)
    report(obj, obj.next_config(), 64, 7)
    state = obj.dump_state()
    restored = strategy(constraints=CONSTRAINTS)
    restored.load_state(state)
    assert restored.center == obj.center and restored.best == obj.best
    assert restored.samples == obj.samples
    assert restored.next_config() != state["seed"]
    other = strategy(constraints=[])
    before = deepcopy(other.pending)
    other.load_state(state)
    assert other.pending == before


def test_full_platform_lifecycle_without_explicit_constraints():
    fake = FakePlatform(MANIFEST, target_trials=999)
    obj = HeuristicSearch(max_trials=2)
    assert SearchLoop(obj, sleep=lambda _: None).run(fake) == 0
    assert fake.trials == 2
    assert fake.finalized_called and fake.served


def test_manifest_constraints_and_invalid_definition():
    spec = manifest({"grid": {"max_running_requests": [16, 32]}})
    spec["objective"]["screen"]["redlines"] = CONSTRAINTS
    obj = HeuristicSearch()
    obj.setup(spec)
    assert obj.constraints == CONSTRAINTS
    with pytest.raises(ValueError, match="invalid screen redline"):
        strategy(constraints=[{"value": 8}])


def test_small_gain_updates_winner_without_moving_center():
    obj = strategy(baseline=48, min_improvement=0.1)
    report(obj, obj.next_config(), 100)
    report(obj, obj.next_config(), 101)
    assert obj.best.objective == 101
    assert obj.center.objective == 100
    assert obj.anchor["max_running_requests"] == 48


def test_slo_breach_cannot_replace_feasible_center_even_with_higher_score():
    obj = strategy(constraints=CONSTRAINTS)
    report(obj, obj.next_config(), 100, 7)
    baseline = deepcopy(obj.anchor)
    report(obj, obj.next_config(), 10000, 9)
    assert obj.anchor == baseline
    assert obj.best.objective == 100


def test_changed_platform_config_is_not_used_for_slope_or_center():
    obj = strategy()
    config = obj.next_config()
    config["max_running_requests"] = 123
    report(obj, config, 100)
    assert obj.center is None
    assert obj.samples == []


def test_categorical_axis_has_no_linear_prediction():
    obj = strategy(values=("small", "medium", "large"), baseline="medium")
    report(obj, obj.next_config(), 100)
    report(obj, obj.next_config(), 50)
    assert obj._predict({"max_running_requests": "large"}, 0) is None
    assert obj.next_config()["max_running_requests"] == "large"


def measured_at(obj, value, ttft, *, feasible=False, failure=""):
    """Feed a chosen legal experiment through the proposal/observation contract."""
    config = {**obj.anchor, "max_running_requests": value}
    obj.pending = [(config, "test experiment")]
    proposed = obj.next_config()
    assert proposed == config
    report(obj, proposed, 100, ttft, feasible=feasible, failure=failure)


@pytest.mark.parametrize("direction", [-1, 1])
def test_worsening_breach_prunes_only_beyond_measured_boundary(direction):
    records = []
    obj = strategy(values=(16, 32, 48, 64, 80), baseline=48,
                   constraints=CONSTRAINTS, decision_log=records.append)
    measured_at(obj, 48, 9)
    boundary = 48 + direction * 16
    measured_at(obj, boundary, 12)
    beyond = {"max_running_requests": boundary + direction * 16}
    assert obj._direction_pruned(beyond)
    assert not obj._direction_pruned({"max_running_requests": boundary})
    assert not obj._direction_pruned({"max_running_requests": 48})
    assert not obj._direction_pruned({"max_running_requests": 48 - direction * 16})
    neighbors = obj._neighbors({"max_running_requests": boundary}, coarse=True)
    assert all(config != beyond for config, _ in neighbors)
    # A queued candidate must also be filtered before it consumes trial budget.
    obj.pending.insert(0, (beyond, "previously queued"))
    before = obj.proposed
    assert obj.next_config() != beyond
    assert obj.proposed <= before + 1
    events = [row for row in records if row["event"] == "direction_pruned"]
    assert events[0]["direction"] == direction
    assert events[0]["previous"] == 9 and events[0]["measured"] == 12


def test_large_failed_step_preserves_unmeasured_intermediate_points():
    obj = strategy(baseline=48, constraints=CONSTRAINTS)
    measured_at(obj, 48, 9)
    measured_at(obj, 80, 12)
    assert not obj._direction_pruned({"max_running_requests": 64})
    assert obj._direction_pruned({"max_running_requests": 96})
    obj.pending = [({"max_running_requests": 96}, "farther"),
                   ({"max_running_requests": 64}, "intermediate")]
    assert obj.next_config() == {"max_running_requests": 64}


@pytest.mark.parametrize("old,new,feasible,failure", [
    (12, 9, False, ""),       # Still failing, but improving.
    (9, 9.5, False, ""),      # Below the noise threshold.
    (5, 7, True, ""),         # Worse, but still passes the redline.
    (9, None, False, ""),     # No measured evidence.
    (9, float("nan"), False, ""),
    (None, 12, False, ""),
    (9, 12, False, "oom"),    # Execution failures are not SLO evidence.
])
def test_direction_pruning_requires_significant_measured_breach(old, new, feasible, failure):
    obj = strategy(baseline=48, constraints=CONSTRAINTS)
    measured_at(obj, 48, old)
    measured_at(obj, 64, new, feasible=feasible, failure=failure)
    assert obj.direction_blocks == []


@pytest.mark.parametrize("op,old,new,blocked", [
    (">=", 10, 7, True),
    (">=", 6, 7, False),
    (">=", 10, 8, False),
    (">", 10, 8, True),
    ("<=", 6, 8, False),
    ("<", 6, 8, True),
])
def test_direction_pruning_respects_lower_bounds_and_strictness(op, old, new, blocked):
    obj = strategy(baseline=48, constraints=[{"metric": "ttft", "op": op, "value": 8}])
    measured_at(obj, 48, old)
    measured_at(obj, 64, new)
    assert obj._direction_pruned({"max_running_requests": 80}) == blocked


def test_other_parameter_change_reopens_pruned_direction():
    obj = HeuristicSearch(baseline={"max_running_requests": 48,
                                    "chunked_prefill_size": 8192})
    spec = manifest({"grid": {"max_running_requests": [48, 64, 80],
                              "chunked_prefill_size": [4096, 8192]}})
    spec["objective"]["screen"]["redlines"] = CONSTRAINTS
    obj.setup(spec)
    measured_at(obj, 48, 9)
    measured_at(obj, 64, 12)
    assert obj._direction_pruned({"max_running_requests": 80, "chunked_prefill_size": 8192})
    assert not obj._direction_pruned({"max_running_requests": 80,
                                      "chunked_prefill_size": 4096})


def test_pruning_is_reconstructed_on_restart_without_reemitting_events():
    obj = strategy(baseline=48, constraints=CONSTRAINTS)
    measured_at(obj, 48, 9)
    measured_at(obj, 64, 12)
    records = []
    restored = strategy(baseline=48, constraints=CONSTRAINTS, decision_log=records.append)
    restored.load_state(obj.dump_state())
    assert restored.direction_blocks == obj.direction_blocks
    assert restored._direction_pruned({"max_running_requests": 80})
    assert records == []
    assert restored.next_config() == obj.next_config()


def test_real_log_measurements_block_both_outward_directions():
    obj = strategy(values=(1, 1.87, 2, 4, 8), baseline=2,
                   constraints=[{"metric": "ttft", "op": "<=", "value": 6000}])
    measured_at(obj, 2, 9655.073753744364)
    measured_at(obj, 1.87, 10741.828607022762)
    measured_at(obj, 4, 13852.568387612684)
    assert obj._direction_pruned({"max_running_requests": 1})
    assert obj._direction_pruned({"max_running_requests": 8})
    assert obj.best is None
    assert obj.next_config() is None


@pytest.mark.parametrize("threshold", [0, -1, float("inf"), float("nan"), True])
def test_invalid_pruning_threshold(threshold):
    with pytest.raises(ValueError, match="pruning_threshold"):
        strategy(pruning_threshold=threshold)


def test_pruning_threshold_is_configurable_and_part_of_checkpoint_context():
    obj = strategy(baseline=48, constraints=CONSTRAINTS, pruning_threshold=0.5)
    measured_at(obj, 48, 9)
    measured_at(obj, 64, 12)
    assert not obj.direction_blocks
    other = strategy(baseline=48, constraints=CONSTRAINTS)
    other.load_state(obj.dump_state())
    assert not other.samples


def test_platform_rewritten_measurement_cannot_prune_direction():
    obj = strategy(baseline=48, constraints=CONSTRAINTS)
    measured_at(obj, 48, 9)
    obj.pending = [({"max_running_requests": 64}, "test experiment")]
    config = obj.next_config()
    config["max_running_requests"] = 80
    report(obj, config, 100, 12, feasible=False)
    assert not obj.direction_blocks


def test_categorical_measurements_do_not_prune_by_declaration_order():
    obj = strategy(values=("small", "medium", "large"), baseline="medium",
                   constraints=CONSTRAINTS)
    measured_at(obj, "medium", 9)
    measured_at(obj, "small", 12)
    assert not obj.direction_blocks


def test_tied_axes_and_activation_changes_supply_no_direction_evidence():
    for space, baseline, changed in [
        ({"tied": [{"max_running_requests": [16, 32, 48],
                    "chunked_prefill_size": [2048, 4096, 8192]}]},
         {"max_running_requests": 16, "chunked_prefill_size": 2048},
         {"max_running_requests": 32, "chunked_prefill_size": 4096}),
        ({"grid": {"max_running_requests": [16, 32, 48], "prefill_max_requests": [1]},
          "conditions": {"prefill_max_requests": {"max_running_requests": [32, 48]}}},
         {"max_running_requests": 16},
         {"max_running_requests": 32, "prefill_max_requests": 1}),
    ]:
        obj = HeuristicSearch(baseline=baseline)
        spec = manifest(space)
        spec["objective"]["screen"]["redlines"] = CONSTRAINTS
        obj.setup(spec)
        report(obj, obj.next_config(), 100, 9, feasible=False)
        obj.pending = [(changed, "test experiment")]
        report(obj, obj.next_config(), 100, 12, feasible=False)
        assert not obj.direction_blocks

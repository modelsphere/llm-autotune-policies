"""Exercise the actual manifest and GET benchmark response contract for heuristic search."""
from copy import deepcopy

import pytest

from autotune_policy import SearchLoop
from autotune_policy.strategy import TrialResult
from strategies.heuristic_search import HeuristicSearch
from tests.test_night import MANIFEST, FakePlatform

SLO = "perf_guidellm_sweep.reported_level_meets_slo"
TARGET = "perf_guidellm_sweep.total_tpm_card_norm"
TTFT = "perf_guidellm_sweep.ttft_p99"


def campaign(direction="maximize"):
    spec = deepcopy(MANIFEST)
    spec["objective"] = {
        "screen": {"target_metric": TARGET, "direction": direction,
                   "redlines": [{"metric": SLO, "op": ">=", "value": 1}]},
        "verify": {"target_metric": TARGET, "direction": direction,
                   "redlines": [{"metric": SLO, "op": ">=", "value": 1},
                                {"metric": "opencompass.gpqa_diamond",
                                 "op": ">=", "value": 0.6}]},
    }
    spec["search_space"] = {"grid": {"max_running_requests": [16, 32, 48, 64]}}
    return spec


@pytest.mark.parametrize("direction,normalized", [("maximize", "max"), ("minimize", "min"),
                                                  ("max", "max"), ("min", "min")])
def test_direction_flat_metrics_and_verify_separation(direction, normalized):
    obj = HeuristicSearch()
    obj.setup(campaign(direction))
    assert obj.direction == normalized
    assert len(obj.verify_redlines) == 2
    config = obj.next_config()
    result = TrialResult(config, 1234.5, True, raw={
        "status": "succeeded",
        "summary": {"objective_value": 1234.5, "feasible": True,
                    "constraints": [-120.0], "breaches": []},
        "metrics": {SLO: 1, TARGET: 9999, TTFT: 7.2},
    })
    obj.observe(result)
    assert obj.contenders()[0]["evidence"]["screen_objective"] == 1234.5
    assert obj._metrics(result) == {SLO: 1.0}
    # Missing GPQA in screen is intentional, not a local feasibility failure.
    assert obj.best is not None


@pytest.mark.parametrize("metrics", [{}, {SLO: 0}, {SLO: float("nan")},
                                     {"perf_guidellm_sweep": {"reported_level_meets_slo": 1}}])
def test_missing_breached_or_nested_metrics_cannot_pass(metrics):
    obj = HeuristicSearch()
    obj.setup(campaign())
    config = obj.next_config()
    obj.observe(TrialResult(config, 100, True, raw={
        "metrics": metrics,
        "summary": {"metrics": {SLO: 1}, "constraints": [-120.0], "breaches": []},
    }))
    assert obj.best is None and obj.contenders() == []


def test_multiple_bounds_on_one_metric_and_exact_boundary():
    spec = campaign()
    spec["objective"]["screen"]["redlines"] = [
        {"metric": TTFT, "op": ">=", "value": 2},
        {"metric": TTFT, "op": "<", "value": 8},
    ]
    obj = HeuristicSearch()
    obj.setup(spec)
    for measured, passes in [(1, False), (2, True), (8, False)]:
        result = TrialResult({}, 100, True, raw={"metrics": {TTFT: measured}})
        assert obj._feasible(result) == passes


def test_flat_metrics_guide_repair_and_duplicate_bounds_allow_prediction():
    spec = campaign()
    spec["objective"]["screen"]["redlines"] = [
        {"metric": TTFT, "op": "<", "value": 8},
        {"metric": TTFT, "op": ">=", "value": 0},
    ]
    obj = HeuristicSearch(baseline={"max_running_requests": 64})
    obj.setup(spec)
    for measured in [10, 9]:
        config = obj.next_config()
        obj.observe(TrialResult(config, 100, False, raw={"metrics": {TTFT: measured}}))
    assert obj.anchor["max_running_requests"] == 48
    prediction = obj._predict({"max_running_requests": 16}, 0)
    assert prediction["metrics"][TTFT] == pytest.approx(7)
    assert obj.next_config()["max_running_requests"] == 16


def test_verify_change_invalidates_warm_start():
    obj = HeuristicSearch()
    spec = campaign()
    obj.setup(spec)
    config = obj.next_config()
    obj.observe(TrialResult(config, 100, True, raw={"metrics": {SLO: 1}}))
    state = obj.dump_state()
    spec["objective"]["verify"]["redlines"][1]["value"] = 0.7
    restored = HeuristicSearch()
    restored.setup(spec)
    restored.load_state(state)
    assert restored.pending[0][1] == "baseline"


@pytest.mark.parametrize("redlines", [{}, [{"metric": SLO, "op": ">=", "value": "1"}],
                                      [{"metric": SLO, "op": "bad", "value": 1}]])
def test_invalid_manifest_redlines_rejected(redlines):
    spec = campaign()
    spec["objective"]["screen"]["redlines"] = redlines
    with pytest.raises((TypeError, ValueError), match="redline"):
        HeuristicSearch().setup(spec)


def test_real_response_shape_through_search_loop():
    class ContractPlatform(FakePlatform):
        def bench_launch(self, launch_id, *, suite, idempotency_key):
            assert suite == "screen"
            response = super().bench_launch(launch_id, suite=suite,
                                            idempotency_key=idempotency_key)
            self.benchmarks[response["id"]]["metrics"] = {SLO: 1, TARGET: 9999}
            return response

    obj = HeuristicSearch(max_trials=2)
    platform = ContractPlatform(campaign(), target_trials=999)
    assert SearchLoop(obj, sleep=lambda _: None).run(platform) == 0
    assert platform.trials == 2
    assert obj.contenders()
    assert all(result.raw["metrics"][SLO] == 1 for result in obj.observations)
    assert platform.finalized_called and platform.served

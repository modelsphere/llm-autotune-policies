import json
from pathlib import Path

import pytest

from autotune_policy import SearchLoop
from autotune_policy.dry_run import SLO, TPM, TTFT, DryRunPlatform, MetricSimulator
from main import main
from strategies.heuristic_search import HeuristicSearch


@pytest.fixture
def manifest():
    return json.loads((Path(__file__).parents[1] / "examples/dry_run_manifest.json").read_text())


def test_response_trends_and_saturation(manifest):
    model = MetricSimulator(manifest)
    low = model.benchmark({"max_running_requests": 32})["metrics"]
    mid = model.benchmark({"max_running_requests": 64})["metrics"]
    high = model.benchmark({"max_running_requests": 128})["metrics"]
    assert mid[TTFT] < low[TTFT]
    assert high[TTFT] > mid[TTFT]
    assert mid[TPM] > low[TPM]
    assert mid[SLO] == 1 and low[SLO] == 0
    assert model.is_oom({"max_running_requests": 128, "chunked_prefill_size": 16384})


def test_boundary_and_minimize_objective(manifest):
    screen = manifest["objective"]["screen"]
    screen.update(target_metric=TTFT, direction="minimize",
                  redlines=[{"metric": TTFT, "op": "<", "value": 9}])
    result = MetricSimulator(manifest).benchmark({})
    assert result["summary"]["objective_value"] == 9
    assert result["summary"]["feasible"] is False
    assert result["summary"]["constraints"] == [0]
    screen["redlines"][0]["op"] = "<="
    assert MetricSimulator(manifest).benchmark({})["summary"]["feasible"]


def test_custom_effect_and_seeded_noise(manifest):
    overrides = {"effects": {"max_running_requests": {TTFT: 1.0}}}
    model = MetricSimulator(manifest, model=overrides)
    assert model.benchmark({"max_running_requests": 64})["metrics"][TTFT] == 18
    a = MetricSimulator(manifest, seed=7, noise=0.03)
    b = MetricSimulator(manifest, seed=7, noise=0.03)
    assert [a.benchmark({}) for _ in range(3)] == [b.benchmark({}) for _ in range(3)]


def test_unknown_metrics_fail_explicitly(manifest):
    manifest["objective"]["screen"]["target_metric"] = "unknown.metric"
    with pytest.raises(ValueError, match="no simulation model"):
        MetricSimulator(manifest)


def test_full_search_records_repairs_and_releases_engines(manifest):
    platform = DryRunPlatform(manifest)
    strategy = HeuristicSearch(workload="agent", max_trials=24, safety_margin=0)
    assert SearchLoop(strategy, sleep=lambda _: None).run(platform) == 0
    assert len(platform.trace) == len(strategy.observations) == 24
    assert platform.trace[0]["summary"]["feasible"] is False
    assert any(t.get("failure_class") == "oom" for t in platform.trace)
    assert len(platform.contenders) == 2
    assert len(platform.served) == 2
    assert all(row["status"] == "released" for row in platform.launches.values())
    assert all(r.feasible for r in strategy.observations if r.trial_id in
               {tid for c in platform.contenders for tid in c["trial_ids"]})


def test_no_feasible_contender_still_exits(manifest):
    platform = DryRunPlatform(manifest, model={"oom_pressure": 0.01})
    assert SearchLoop(HeuristicSearch(max_trials=3)).run(platform) == 0
    assert platform.finished and not platform.contenders
    assert len(platform.trace) == 3


def test_cli_offline_report(tmp_path, monkeypatch):
    import urllib.request

    def no_network(*args, **kwargs):
        pytest.fail("dry run must never contact platform")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    output = tmp_path / "report.json"
    assert main(["--dry-run", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["simulated"] and report["exit_code"] == 0
    assert len(report["trials"]) == 24

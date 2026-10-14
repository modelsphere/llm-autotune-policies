import json

from main import main


def test_decisions_are_persisted_and_results_correlated(tmp_path):
    path = tmp_path / "nested" / "decisions.jsonl"
    report_path = tmp_path / "report.json"
    assert main(["--dry-run", "--max-trials", "8", "--output", str(report_path),
                 "--decision-log", str(path)]) == 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    proposals = [row for row in rows if row["event"] == "proposal"]
    results = [row for row in rows if row["event"] == "observation"]
    assert len(proposals) == len(results) == 8
    report = json.loads(report_path.read_text())
    for proposal, result, trial in zip(proposals, results, report["trials"]):
        assert proposal["decision_id"] == result["decision_id"]
        assert proposal["run_id"] == result["run_id"]
        assert proposal["config"] == trial["config"]
        assert proposal["reason"] and proposal["timestamp"]
        assert result["config"] == trial["config"]
        assert isinstance(result["center_updated"], bool)
        for change in proposal["changes"]:
            name = change["param"]
            assert change["from"] == proposal["center_config"].get(name)
            assert change["to"] == proposal["config"].get(name)
    assert any(p["changes"] for p in proposals[1:])
    old = path.read_text()
    assert main(["--dry-run", "--max-trials", "1", "--output", str(report_path),
                 "--decision-log", str(path)]) == 0
    assert path.read_text().startswith(old)
    appended = [json.loads(line) for line in path.read_text()[len(old):].splitlines()]
    assert len(appended) == 2
    assert appended[0]["run_id"] != proposals[0]["run_id"]


def test_unwritable_log_does_not_abort_search(tmp_path, caplog):
    assert main(["--dry-run", "--max-trials", "1", "--output", str(tmp_path / "report.json"),
                 "--decision-log", str(tmp_path)]) == 0
    assert "Cannot write decision log" in caplog.text


def test_nonfinite_metrics_are_valid_json(tmp_path):
    from decision_log import DecisionLog

    path = tmp_path / "decisions.jsonl"
    DecisionLog(path)({"event": "observation", "metrics": {"latency": float("inf")}})
    assert json.loads(path.read_text())["metrics"]["latency"] is None

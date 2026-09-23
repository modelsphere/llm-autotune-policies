"""Each fault, driven against an in-memory platform, asserting the client-visible
behaviour the real platform keys its handling off. No network, no GPUs.

This is a test *of the test policy* — it keeps the fault menu honest, so a
scenario cannot silently stop emitting the failure it claims to.
"""

from __future__ import annotations

import pytest

from autotune_policy.client import ConfigRejected, PolicyError
from faults import _probe, runner_for, scenarios

MANIFEST = {
    "contract": {"version": "1.1"},
    "session_id": 7,
    "campaign_id": 1,
    "model": {"engine": "sglang", "image": "reg/mock:v1", "model_path": "/m",
              "served_model_name": "m", "extra_env": {}, "extra_volumes": {}},
    "hardware": {"machine": "mock-01", "gpu_type": "mock", "gpu_indices": [0, 1],
                 "ports": [28200, 28201], "gpus_visible_in_container": True},
    "objective": {"screen": {"target_metric": "perf.tpm", "direction": "max", "redlines": []}},
    "search_space": {"base": {"mem_fraction_static": 0.9}, "grid": {"tp_size": [1, 2]}},
    "time": {"started_at": None, "search_deadline": None, "hard_deadline": None,
             "heartbeat_interval_s": 30, "heartbeat_timeout_s": 180, "tick_s": 10},
    "contenders": {"max": 2, "validation_suite": "verify", "approx_minutes_each": 60},
    "services": {"launch": True, "benchmarks": {"screen": {"slug": "s", "approx_minutes": 5}},
                 "state": True},
    "prior": {"contenders": []},
}


def _objective(engine_args: dict) -> float:
    return 1000.0 + 100 * int(engine_args.get("tp_size", 0))


class FakePlatform:
    """Just enough of the platform to run a night, plus the two levers the tests
    need: finalize after N trials, or finalize an idle policy after N beats (the
    watchdog, simulated)."""

    def __init__(self, *, target_trials=None, idle_finalize_beats=None):
        self.target_trials = target_trials
        self.idle_finalize_beats = idle_finalize_beats
        self.plan = None
        self.state: dict = {}
        self.events: list = []
        self.launches: dict = {}
        self.attempted_launches: list = []
        self.benchmarks: dict = {}
        self.contender_rows: list = []
        self.bad_contenders: list = []
        self.trials = 0
        self.launch_count = 0
        self.searching_beats = 0
        self.finalized_called = False
        self.served: list = []
        self._serve_issued = False
        self._seq = 0
        self.statuses: list = []
        self.phases: list = []

    def get_session(self):
        return MANIFEST

    def _command(self):
        if self.served:
            return "exit", None, None
        if self.finalized_called or "exhausted" in self.statuses:
            if self.contender_rows and not self.served:
                if not self._serve_issued:
                    self._serve_issued = True
                    return "serve", self.contender_rows[0]["id"], MANIFEST["hardware"]["ports"][0]
                return "exit", None, None  # serve was commanded but never honoured
            return "exit", None, None
        if self.idle_finalize_beats and self.launch_count == 0 \
                and self.searching_beats >= self.idle_finalize_beats:
            return "finalize", None, None
        if self.target_trials is not None and self.trials >= self.target_trials:
            return "finalize", None, None
        return "run", None, None

    def heartbeat(self, *, phase, message="", progress=None, status="",
                  failure_class="", coverage=None, handshake=None):
        self.phases.append(phase)
        self.statuses.append(status)
        if phase == "searching":
            self.searching_beats += 1
        cmd, cid, port = self._command()
        return {"command": cmd, "search_deadline": None, "hard_deadline": None,
                "contender_id": cid, "port": port}

    def put_plan(self, *, tuning, extras, notes):
        self.plan = {"tuning": tuning, "extras": extras, "notes": notes}

    def finalized(self):
        self.finalized_called = True

    def launch(self, *, engine_args, gpu_indices, port, idempotency_key):
        self.attempted_launches.append(dict(engine_args))
        # The platform owns placement: a config carrying one is a hard 422.
        if {"port", "host", "model_path", "node_rank"} & set(engine_args):
            raise ConfigRejected(422, "placement is the platform's")
        self._seq += 1
        lid = self._seq
        self.launch_count += 1
        self.launches[lid] = {"id": lid, "status": "serving", "config": dict(engine_args),
                              "endpoint_url": f"http://mock:{port}", "environment": {}}
        return {"id": lid, "status": "queued"}

    def get_launch(self, launch_id):
        return self.launches[launch_id]

    def release(self, launch_id):
        self.launches[launch_id]["status"] = "released"
        return self.launches[launch_id]

    def bench_launch(self, launch_id, *, suite, idempotency_key):
        self.trials += 1
        self._seq += 1
        bid = self._seq
        cfg = self.launches[launch_id]["config"]
        self.benchmarks[bid] = {
            "id": bid, "status": "succeeded", "suite": suite, "trial_id": 900 + bid,
            "summary": {"objective_value": _objective(cfg), "feasible": True,
                        "constraints": [], "breaches": []},
            "config": cfg, "environment": {},
        }
        return {"id": bid, "status": "queued"}

    def get_benchmark(self, benchmark_id):
        return self.benchmarks[benchmark_id]

    def put_contenders(self, contenders):
        placement = {"port", "host", "model_path", "node_rank"}

        def bad(spec):
            ea = spec.get("engine_args") or {}
            return not ea or not spec.get("image") or bool(set(ea) & placement)

        for c in contenders:
            if bad(c.get("launch_spec") or {}):
                self.bad_contenders.append(c)  # the platform rejects this entry per-entry
        self.contender_rows = [
            {**c, "id": 500 + i} for i, c in enumerate(contenders)
            if not bad(c.get("launch_spec") or {})
        ]
        return {"contenders": self.contender_rows}

    def get_contenders(self):
        return self.contender_rows

    def contender_serving(self, contender_id):
        self.served.append(contender_id)
        return {"id": contender_id}

    def get_state(self):
        return self.state

    def put_state(self, blob):
        self.state = blob

    def event(self, level, message):
        self.events.append((level, message))


def _run(scenario, **fake_kw):
    fake = FakePlatform(**fake_kw)
    rc = runner_for(scenario, sleep=lambda _s: None)(fake)
    return fake, rc


# -- the menu is complete ------------------------------------------------------


def test_the_menu_lists_every_scenario():
    menu = scenarios()
    for expected in ("happy", "crash_next_config", "crash_setup", "exhaust_immediately",
                     "bad_config", "deviate", "bad_contender", "idle", "wedge",
                     "exit_hard", "flaky_network", "serve_fail",
                     "hw_constraints", "bench_unserved", "gpu_packing"):
        assert expected in menu


class _EventClient:
    def __init__(self):
        self.events = []

    def event(self, level, message):
        self.events.append((level, message))


def _raises(exc):
    def fn():
        raise exc
    return fn


def test_a_probe_passes_only_on_the_expected_refusal():
    c = _EventClient()
    # right status -> pass
    assert _probe(c, "x", 422, _raises(ConfigRejected(422, "no"))) is True
    # wrong status -> fail
    assert _probe(c, "x", 422, _raises(PolicyError(429, "busy"))) is False
    # the bad call unexpectedly succeeds -> fail (the guardrail is missing)
    assert _probe(c, "x", 422, lambda: {"ok": 1}) is False
    assert any(lvl == "error" for lvl, _ in c.events)


def test_an_unknown_scenario_is_a_clear_error():
    with pytest.raises(SystemExit):
        runner_for("does_not_exist")


# -- happy path ----------------------------------------------------------------


def test_happy_runs_through_to_a_served_contender():
    fake, rc = _run("happy", target_trials=2)
    assert rc == 0
    assert fake.plan is not None
    assert fake.trials >= 2
    assert fake.finalized_called
    assert fake.served == [fake.contender_rows[0]["id"]]


# -- strategy-level faults -----------------------------------------------------


def test_a_crash_in_the_search_code_becomes_an_error_signal():
    fake, rc = _run("crash_next_config", target_trials=99)
    assert rc == 1
    assert "error" in fake.statuses  # the loop told the platform we gave up
    assert not fake.finalized_called


def test_a_crash_in_observe_also_signals_error():
    fake, rc = _run("crash_observe", target_trials=99)
    assert rc == 1
    assert "error" in fake.statuses


def test_a_crash_in_setup_dies_before_any_heartbeat():
    # setup runs before the first beat, so no error signal is possible — the
    # exception propagates and the process dies (container_died on the platform).
    with pytest.raises(RuntimeError):
        _run("crash_setup")


def test_running_out_immediately_signals_exhausted():
    fake, rc = _run("exhaust_immediately")
    assert rc == 0
    assert "exhausted" in fake.statuses
    assert fake.trials == 0  # never launched anything


def test_a_placement_flag_config_is_rejected_and_survived():
    fake, rc = _run("bad_config")
    assert rc == 0
    assert fake.attempted_launches, "it should have tried to launch the bad config"
    assert "port" in fake.attempted_launches[0]  # the placement flag that gets 422'd
    assert fake.trials == 0  # nothing measured; the loop moved on


def test_a_deviation_config_is_run_not_rejected():
    fake, rc = _run("deviate", target_trials=1)
    assert rc == 0
    assert any("enable_torch_compile" in a for a in fake.attempted_launches)
    assert fake.trials >= 1  # deviations run, they are only badged


def test_a_malformed_contender_is_seen_and_rejected():
    fake, _rc = _run("bad_contender", target_trials=1)
    assert fake.bad_contenders, "the malformed contender should reach the platform"
    assert not fake.contender_rows  # and be rejected, leaving none accepted


# -- below-the-loop faults -----------------------------------------------------


def test_idle_beats_without_ever_launching():
    # The watchdog's target: alive, heartbeating, doing no delegated work.
    fake, rc = _run("idle", idle_finalize_beats=3)
    assert rc == 0
    assert fake.launch_count == 0
    assert fake.searching_beats >= 3
    assert fake.finalized_called  # the (simulated) watchdog wound it down


def test_wedge_beats_then_goes_silent():
    fake, rc = _run("wedge")
    assert rc == 0
    assert fake.launch_count == 0
    assert fake.phases.count("searching") == 1  # one beat, then silence


def test_serve_fail_never_serves_the_contender():
    fake, rc = _run("serve_fail", target_trials=2)
    assert rc == 0
    assert fake.contender_rows  # it produced a contender
    assert fake.served == []    # but never brought it up when told to serve

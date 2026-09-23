"""Drive a whole night against an in-memory platform.

No HTTP, no sleeps: a fake with the client's method surface scripts the
heartbeat commands (run… → finalize → serve → exit) and progresses launches and
benchmarks instantly, so the loop's full lifecycle runs in milliseconds. This is
the test that says the policy is contract-correct.
"""

from __future__ import annotations

from autotune_policy import SearchLoop
from strategies.random_search import RandomSearch

MANIFEST = {
    "contract": {"version": "1.1"},
    "session_id": 17,
    "campaign_id": 42,
    "policy": {"name": "random-search", "image": "reg/random:0.1.0", "version": "0.1.0"},
    "model": {"engine": "sglang", "image": "reg/sglang:v1", "model_path": "/model",
              "served_model_name": "m", "extra_env": {}, "extra_volumes": {}},
    "hardware": {"machine": "gpu-01", "gpu_type": "H100", "gpu_indices": [0, 1],
                 "ports": [28200, 28201], "gpus_visible_in_container": True},
    "objective": {"screen": {"target_metric": "perf.output_tpm_card_norm",
                             "direction": "max", "redlines": []}},
    "search_space": {"base": {"mem_fraction_static": 0.9},
                     "grid": {"tp_size": [1, 2]},
                     "range": {"chunked_prefill_size": {"min": 2048, "max": 8192, "step": 2048}}},
    "space_policy": {"deviations": "allowed_if_declared"},
    "production": {"engine_args": {}, "launch_command": ""},
    "time": {"started_at": None, "search_deadline": None, "hard_deadline": None,
             "heartbeat_interval_s": 30, "heartbeat_timeout_s": 180, "tick_s": 10},
    "contenders": {"max": 2, "validation_suite": "verify", "approx_minutes_each": 60},
    "services": {"launch": True, "benchmarks": {"screen": {"slug": "s", "approx_minutes": 5}},
                 "state": True},
    "prior": {"contenders": []},
}


def _objective(engine_args: dict) -> float:
    # Deterministic and monotone, so ranking is checkable: bigger tp + chunk = better.
    return 1000.0 + 100 * engine_args.get("tp_size", 0) + 0.01 * engine_args.get(
        "chunked_prefill_size", 0)


class FakePlatform:
    def __init__(self, manifest, target_trials=4):
        self.manifest = manifest
        self.target = target_trials
        self.plan = None
        self.state: dict = {}
        self.events: list = []
        self.launches: dict = {}
        self.benchmarks: dict = {}
        self.contender_rows: list = []
        self.trials = 0
        self.finalized_called = False
        self.served: list = []
        self.serving_launch_args: list = []
        self._seq = 0
        self.heartbeat_phases: list = []
        self.heartbeat_statuses: list = []
        self.last_coverage: dict | None = None

    # -- session ---
    def get_session(self):
        return self.manifest

    def _command(self):
        if self.served:
            return "exit", None, None
        if self.finalized_called:
            return ("serve", self.contender_rows[0]["id"], self.manifest["hardware"]["ports"][0]) \
                if self.contender_rows else ("exit", None, None)
        # Mirror the platform: an explicit "exhausted" signal finalizes now.
        if "exhausted" in self.heartbeat_statuses or self.trials >= self.target:
            return "finalize", None, None
        return "run", None, None

    def heartbeat(self, *, phase, message="", progress=None, status="",
                  failure_class="", coverage=None, handshake=None):
        self.heartbeat_phases.append(phase)
        self.heartbeat_statuses.append(status)
        if coverage is not None:
            self.last_coverage = coverage
        cmd, cid, port = self._command()
        return {"command": cmd, "search_deadline": None, "hard_deadline": None,
                "contender_id": cid, "port": port}

    def put_plan(self, *, tuning, extras, notes):
        self.plan = {"tuning": tuning, "extras": extras, "notes": notes}

    def finalized(self):
        self.finalized_called = True

    # -- launches ---
    def launch(self, *, engine_args, gpu_indices, port, idempotency_key):
        self._seq += 1
        lid = self._seq
        self.launches[lid] = {"id": lid, "status": "serving", "config": dict(engine_args),
                              "endpoint_url": f"http://x:{port}", "environment": {"engine_version": "1"}}
        if self.finalized_called:
            self.serving_launch_args.append(engine_args)
        return {"id": lid, "status": "queued"}

    def get_launch(self, launch_id):
        return self.launches[launch_id]

    def release(self, launch_id):
        self.launches[launch_id]["status"] = "released"
        return self.launches[launch_id]

    # -- benchmarks ---
    def bench_launch(self, launch_id, *, suite, idempotency_key):
        self.trials += 1
        self._seq += 1
        bid = self._seq
        cfg = self.launches[launch_id]["config"]
        self.benchmarks[bid] = {
            "id": bid, "status": "succeeded", "suite": suite, "trial_id": 900 + bid,
            "summary": {"objective_value": _objective(cfg), "feasible": True,
                        "constraints": [], "breaches": []},
            "config": cfg, "environment": {"engine_version": "1"},
        }
        return {"id": bid, "status": "queued"}

    def get_benchmark(self, benchmark_id):
        return self.benchmarks[benchmark_id]

    # -- contenders ---
    def put_contenders(self, contenders):
        self.contender_rows = [{**c, "id": 500 + i} for i, c in enumerate(contenders)]
        return {"contenders": self.contender_rows}

    def get_contenders(self):
        return self.contender_rows

    def contender_serving(self, contender_id):
        self.served.append(contender_id)
        return {"id": contender_id}

    # -- state / events ---
    def get_state(self):
        return self.state

    def put_state(self, blob):
        self.state = blob

    def event(self, level, message):
        self.events.append((level, message))


def test_a_full_night_runs_to_exit():
    fake = FakePlatform(MANIFEST, target_trials=4)
    loop = SearchLoop(RandomSearch(max_trials=10), sleep=lambda _s: None)

    rc = loop.run(fake)

    assert rc == 0
    # handshake + plan happened
    assert fake.plan is not None
    assert "tp_size" in fake.plan["tuning"] and "chunked_prefill_size" in fake.plan["tuning"]
    # it searched, then finalized, then served the top contender, then exited
    assert fake.trials >= 4
    assert fake.finalized_called
    assert fake.served == [fake.contender_rows[0]["id"]]
    assert "validating" in fake.heartbeat_phases


def test_contenders_are_ranked_best_first_and_complete():
    fake = FakePlatform(MANIFEST, target_trials=4)
    loop = SearchLoop(RandomSearch(max_trials=10), sleep=lambda _s: None)
    loop.run(fake)

    rows = fake.contender_rows
    assert 1 <= len(rows) <= MANIFEST["contenders"]["max"]
    scores = [r["evidence"]["screen_objective"] for r in rows]
    assert scores == sorted(scores, reverse=True), "contenders must be best-first"
    top = rows[0]["launch_spec"]
    assert top["image"] == MANIFEST["model"]["image"]
    assert top["engine_args"], "a contender's launch_spec must carry a complete config"


def test_the_served_contender_is_the_top_ranked_config():
    fake = FakePlatform(MANIFEST, target_trials=4)
    loop = SearchLoop(RandomSearch(max_trials=10), sleep=lambda _s: None)
    loop.run(fake)

    # validation delegated-launched exactly the rank-1 contender's config
    assert fake.serving_launch_args, "nothing was served for validation"
    assert fake.serving_launch_args[-1] == fake.contender_rows[0]["launch_spec"]["engine_args"]


def test_state_is_written_for_warm_start():
    fake = FakePlatform(MANIFEST, target_trials=4)
    SearchLoop(RandomSearch(max_trials=10), sleep=lambda _s: None).run(fake)
    assert fake.state.get("seen"), "the strategy should persist seen configs for the next night"


def test_running_out_of_configs_signals_exhausted():
    """When the strategy has nothing left, the loop tells the platform so
    explicitly — the cooperative fast-path — rather than idling to the deadline."""
    # Target far above what the small strategy will produce, so the ONLY way
    # this night ends is the exhausted signal driving the fake to finalize.
    fake = FakePlatform(MANIFEST, target_trials=999)
    rc = SearchLoop(RandomSearch(max_trials=3), sleep=lambda _s: None).run(fake)
    assert rc == 0
    assert "exhausted" in fake.heartbeat_statuses
    assert fake.finalized_called and fake.served


def test_a_crash_in_the_strategy_signals_error():
    """An uncaught exception in the search logic becomes an 'error' signal and a
    non-zero exit — the platform is told we gave up, not left to time us out."""
    class Boom(RandomSearch):
        def next_config(self):
            raise ValueError("kaboom")

    fake = FakePlatform(MANIFEST, target_trials=4)
    rc = SearchLoop(Boom(max_trials=10), sleep=lambda _s: None).run(fake)
    assert rc == 1
    assert "error" in fake.heartbeat_statuses


def test_coverage_is_reported_automatically():
    """The author writes no coverage code; the loop attaches per-axis coverage
    built from the configs measured, keyed in the platform's vocabulary."""
    fake = FakePlatform(MANIFEST, target_trials=4)
    SearchLoop(RandomSearch(max_trials=10), sleep=lambda _s: None).run(fake)
    assert fake.last_coverage is not None
    params = {d["param"] for d in fake.last_coverage["dimensions"]}
    assert "tp_size" in params and "chunked_prefill_size" in params
    tp = next(d for d in fake.last_coverage["dimensions"] if d["param"] == "tp_size")
    assert tp["tried_values"], "coverage should record which tp_size values were tried"

"""The fault menu.

Each scenario drives the platform down one handling path — the happy path, or a
specific way a real policy goes wrong: a bug in the search code, a config the
platform must reject, a container that dies, a policy that goes quiet, a policy
that beats forever while doing nothing. One image emits any of them; the
campaign picks which via the ``CHAOS_SCENARIO`` env var.

Two shapes of fault:

  * Strategy-level — a normal ``Strategy`` that misbehaves at one point. The
    real SDK loop runs it, so these exercise exactly what the loop does with a
    buggy strategy (turn a crash into an ``error`` signal, a rejected config
    into a recorded failure, an empty proposal into ``exhausted``).
  * Below-the-loop — a raw-client routine, for faults the loop is designed to
    prevent: never heartbeating (wedged), heartbeating without ever launching
    (idle watchdog), dying mid-beat (container death), a flaky socket.

Every scenario maps to a platform behaviour worth asserting — see README.md.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable

from autotune_policy import (
    PolicyClient,
    PolicyError,
    SearchLoop,
    Strategy,
    TrialResult,
    build_contender,
    run_policy,
)
from autotune_policy import space as space_mod

Runner = Callable[[PolicyClient], int]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


# -- a small honest strategy, and the buggy variants of it ----------------------


class HappyStrategy(Strategy):
    """A tiny real search: measure a few distinct configs, keep the best as one
    contender. The baseline the fault strategies below deviate from."""

    def __init__(self, *, max_trials: int = 3):
        self.max_trials = max_trials
        self.space: dict = {}
        self.image = ""
        self.direction = "max"
        self.seen: set[str] = set()
        self.results: list[TrialResult] = []
        self._proposed = 0
        self._rng = random.Random(0)

    def setup(self, manifest: dict) -> None:
        self.space = manifest.get("search_space") or {}
        self.image = (manifest.get("model") or {}).get("image", "")
        screen = (manifest.get("objective") or {}).get("screen") or {}
        self.direction = (screen.get("direction") or "max").lower()
        self._rng = random.Random(manifest.get("session_id", 0))

    def plan(self) -> dict:
        return {"tuning": space_mod.swept_keys(self.space), "extras": [],
                "notes": "chaos: happy path"}

    def next_config(self) -> dict | None:
        if self._proposed >= self.max_trials:
            return None
        for _ in range(100):
            config = space_mod.sample(self.space, self._rng)
            key = space_mod.canonical_key(config)
            if key not in self.seen:
                self.seen.add(key)
                self._proposed += 1
                return config
        return None

    def observe(self, result: TrialResult) -> None:
        self.results.append(result)

    def contenders(self) -> list[dict]:
        best = sorted((r for r in self.results if r.ok),
                      key=lambda r: r.objective, reverse=self.direction != "min")
        return [
            build_contender(rank=i + 1, engine_args=r.config, image=self.image,
                            evidence={"screen_objective": r.objective},
                            trial_ids=[r.trial_id] if r.trial_id else [])
            for i, r in enumerate(best[:1])
        ]

    def status(self) -> tuple[str, float | None]:
        return (f"chaos: {len(self.results)}/{self.max_trials}", None)


class CrashInNextConfig(HappyStrategy):
    """A bug in the proposal code. The loop turns it into an `error` signal and
    a non-zero exit — the platform finalizes and validates what exists."""

    def next_config(self) -> dict | None:
        if self._proposed >= 1:
            raise RuntimeError("chaos: simulated bug in next_config()")
        return super().next_config()


class CrashInObserve(HappyStrategy):
    """A bug in the learning code, after a real measurement is in hand."""

    def observe(self, result: TrialResult) -> None:
        raise RuntimeError("chaos: simulated bug in observe()")


class CrashInSetup(HappyStrategy):
    """A bug before the first heartbeat ever goes out. No `error` signal is
    possible — the process dies during setup, and the platform sees a container
    that exited before it ever spoke (container_died / ready_timeout)."""

    def setup(self, manifest: dict) -> None:
        raise RuntimeError("chaos: simulated bug in setup() — dies before handshake")


class ExhaustImmediately(HappyStrategy):
    """Nothing to try at all. The loop sends `exhausted` on the first search
    beat; the platform finalizes with zero contenders."""

    def next_config(self) -> dict | None:
        return None


class ProposeBadConfig(HappyStrategy):
    """Propose a config carrying a placement flag the platform owns. The launch
    endpoint 422s it; the loop records a `bad_config` failure and moves on."""

    def next_config(self) -> dict | None:
        if self._proposed >= 1:
            return None
        self._proposed += 1
        return {"port": 31999, "tp_size": 1}  # `port` is placement → hard 422


class ProposeDeviation(HappyStrategy):
    """Step outside the declared space with an undeclared parameter. The
    platform does NOT reject it — it badges the trial as a deviation and runs
    it. Declaring it in `extras` is the honest version; this omits that on
    purpose to exercise the louder undeclared badge."""

    def next_config(self) -> dict | None:
        if self._proposed >= 1:
            return None
        self._proposed += 1
        base = dict(self.space.get("base") or {})
        base["enable_torch_compile"] = True  # not in any declared axis
        base["tp_size"] = 1
        return base


class ProposeBadContender(HappyStrategy):
    """Do a real trial, then PUT a schema-valid contender whose engine_args
    carry a placement flag the platform owns (`port`). The platform rejects THAT
    entry per-entry and echoes why, WITHOUT losing the session — the
    rejected-contender echo path. (Contrast a hard schema 422, e.g. an empty
    image, which fails the whole request; that is just another 422.)"""

    def contenders(self) -> list[dict]:
        if not self.results:
            return []
        return [{"rank": 1, "launch_spec": {
            "engine_args": {"port": 31999, "tp_size": 1},
            "image": self.image or "mock:latest"}}]


_STRATEGIES: dict[str, Callable[[], Strategy]] = {
    "happy": HappyStrategy,
    "crash_next_config": CrashInNextConfig,
    "crash_observe": CrashInObserve,
    "crash_setup": CrashInSetup,
    "exhaust_immediately": ExhaustImmediately,
    "bad_config": ProposeBadConfig,
    "deviate": ProposeDeviation,
    "bad_contender": ProposeBadContender,
}


# -- below-the-loop faults: raw-client routines ---------------------------------


class _NoServeLoop(SearchLoop):
    """Reaches validation with a contender, then refuses to serve it — never
    POSTs /serving. The platform waits out the serve budget and skips the
    contender ('policy never served it')."""

    def _serve(self, client, manifest, contender_id, port) -> None:
        return  # deliberately do nothing


def _handshake(client: PolicyClient) -> dict:
    manifest = client.get_session()
    client.heartbeat(phase="starting", message="chaos", handshake={
        "sdk_version": "chaos-0.1.0", "contract_version": "1.1", "capabilities": ["serve"]})
    client.put_plan(tuning=[], extras=[], notes="chaos: below-the-loop fault")
    return manifest


def run_idle(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Heartbeat forever, never launch, never signal exhausted — the exact
    behaviour the productivity watchdog exists to catch. Obeys the platform once
    it finalizes us on strikes, so the container still exits cleanly."""
    _handshake(client)
    beats = _env_int("CHAOS_BEATS", 240)
    interval = _env_int("CHAOS_INTERVAL_S", 10)
    finalized = False
    for _ in range(beats):
        hb = client.heartbeat(phase="searching", message="chaos: idle on purpose (no work)")
        cmd = hb.get("command", "run")
        if cmd in ("exit", "abort"):
            return 0
        if cmd == "finalize" and not finalized:
            client.finalized()
            finalized = True
        sleep(interval)
    return 0


def run_wedge(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Beat once, then go silent while alive — the platform declares us wedged
    at 2x the heartbeat timeout. The sleep is bounded only so a local run ends;
    on the real platform the container is reaped first."""
    manifest = _handshake(client)
    client.heartbeat(phase="searching", message="chaos: about to go silent")
    timeout = int((manifest.get("time") or {}).get("heartbeat_timeout_s") or 180)
    sleep(_env_int("CHAOS_WEDGE_S", timeout * 3))
    return 0


def run_exit_hard(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Die abruptly mid-search — a segfault/OOM-kill stand-in. The platform sees
    the container gone and fails us toward validation (container_died)."""
    _handshake(client)
    client.heartbeat(phase="searching", message="chaos: about to hard-exit")
    os._exit(137)  # SIGKILL-style code; no cleanup, no error signal


def run_flaky_network(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Run the happy path over a socket that fails a fraction of calls. Tests
    that the SDK rides out transient network errors (and the platform tolerates
    the gaps) rather than dying on the first blip."""
    rng = random.Random(_env_int("CHAOS_SEED", 1))
    rate = _env_int("CHAOS_FLAKE_PCT", 30) / 100.0
    real_open = client._open

    def flaky_open(req, timeout=None):
        if rng.random() < rate:
            import urllib.error
            raise urllib.error.URLError("chaos: simulated network blip")
        return real_open(req, timeout=timeout)

    client._open = flaky_open
    return run_policy(SearchLoop(HappyStrategy(), sleep=sleep), client=client)


# -- process / state-machine probes ---------------------------------------------
# Not crashes — the wrong-order calls and constraint violations a developer hits
# while building a policy. Each PROBEs one platform guardrail and checks it
# answers with the right status, so a policy author (and the platform) can trust
# the contract's error contract. The session exits non-zero if any guardrail
# misbehaves, and every probe narrates its result to the events stream.


def _probe(client: PolicyClient, label: str, expect: int, fn) -> bool:
    """Run a deliberately-bad call; pass iff the platform refuses it with
    `expect`. A success, or the wrong status, is a failed probe."""
    try:
        fn()
        client.event("error", f"PROBE {label}: expected HTTP {expect}, got success")
        return False
    except PolicyError as exc:
        ok = exc.status == expect
        client.event("info" if ok else "error",
                     f"PROBE {label}: HTTP {exc.status} ({type(exc).__name__}) "
                     f"expected {expect} — {'OK' if ok else 'MISMATCH'}")
        return ok


def _finish_probes(client: PolicyClient, results: list[tuple[str, bool]],
                   *, sleep=time.sleep) -> int:
    passed = sum(1 for _, ok in results if ok)
    client.event("info", f"probes: {passed}/{len(results)} guardrails behaved as expected")
    # Nothing to search — signal exhausted (the idiomatic "I'm done") and obey
    # the platform until it releases us, so the session ends cleanly as done /
    # policy_exhausted instead of looking like a crash or a deadline stop. These
    # are diagnostics, and a passing run should read as a passing run.
    for _ in range(30):
        try:
            hb = client.heartbeat(phase="searching", message="probes complete",
                                  status="exhausted")
        except PolicyError:
            break
        command = hb.get("command")
        if command in ("exit", "abort"):
            break
        if command == "finalize":
            try:
                client.finalized()
            except PolicyError:
                pass
        sleep(2)
    return 0 if passed == len(results) else 1


def run_hw_constraints(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Configs that violate the machine's hardware, and card-accounting slips —
    each must be a 422, never a launch that silently lands wrong."""
    manifest = _handshake(client)
    gpus = (manifest.get("hardware") or {}).get("gpu_indices") or list(range(8))
    n = len(gpus)
    results = [
        # A config that needs more cards than the machine has.
        ("oversized_tp", _probe(client, "oversized_tp", 422, lambda: client.launch(
            engine_args={"tp_size": n * 2}, gpu_indices=gpus[:1], port=None,
            idempotency_key="hw-oversized"))),
        # cards_for(tp=2)=2, but only one card requested — a miscount.
        ("card_mismatch", _probe(client, "card_mismatch", 422, lambda: client.launch(
            engine_args={"tp_size": 2}, gpu_indices=gpus[:1], port=None,
            idempotency_key="hw-mismatch"))),
        # The right count of cards, but not ones this session was allocated.
        ("cards_outside_allocation", _probe(client, "cards_outside_allocation", 422,
            lambda: client.launch(engine_args={"tp_size": 2}, gpu_indices=[9998, 9999],
                                  port=None, idempotency_key="hw-outside"))),
    ]
    return _finish_probes(client, results, sleep=sleep)


def run_bench_unserved(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Bench things you cannot bench: a launch you already tore down, an unknown
    suite, a launch id that never existed. The developer who benches before the
    engine is up hits exactly these."""
    manifest = _handshake(client)
    hw = manifest.get("hardware") or {}
    gpus = (hw.get("gpu_indices") or [0])[:1]  # tp=1 => 1 card
    port = (hw.get("ports") or [None])[0]
    launch = client.launch(engine_args={"tp_size": 1, "mock_latency_ms": 100},
                           gpu_indices=gpus, port=port, idempotency_key="bu-launch")
    lid = launch["id"]
    client.release(lid)  # tear it down, THEN try to bench the corpse
    results = [
        ("bench_released_launch", _probe(client, "bench_released_launch", 409,
            lambda: client.bench_launch(lid, suite="screen", idempotency_key="bu-b1"))),
        ("bench_unknown_suite", _probe(client, "bench_unknown_suite", 422,
            lambda: client.bench_launch(lid, suite="does-not-exist", idempotency_key="bu-b2"))),
        ("bench_bogus_launch_id", _probe(client, "bench_bogus_launch_id", 404,
            lambda: client.bench_launch(999999, suite="screen", idempotency_key="bu-b3"))),
    ]
    return _finish_probes(client, results, sleep=sleep)


def run_gpu_packing(client: PolicyClient, *, sleep=time.sleep) -> int:
    """Does the platform actually pack? Fill the machine with tp=2 engines on
    DISTINCT card pairs — every one should be accepted — then one more must be
    refused 429. (The reference policy always asks for the first cards_for
    cards, which is why naive packing 'did not work'; the platform is fine.)"""
    manifest = _handshake(client)
    hw = manifest.get("hardware") or {}
    gpus = hw.get("gpu_indices") or list(range(8))
    pairs = [gpus[i:i + 2] for i in range(0, len(gpus) - 1, 2)]  # [0,1],[2,3],...
    held: list[int] = []
    results: list[tuple[str, bool]] = []
    for i, pair in enumerate(pairs):
        try:
            lp = client.launch(engine_args={"tp_size": 2}, gpu_indices=pair, port=None,
                               idempotency_key=f"pack-{i}")
            held.append(lp["id"])
            client.event("info", f"PACK {i}: accepted on cards {pair} (launch {lp['id']})")
        except PolicyError as exc:
            client.event("error", f"PACK {i}: cards {pair} REFUSED HTTP {exc.status}")
    results.append((f"packed_{len(pairs)}_engines", len(held) == len(pairs)))
    client.event("info", f"packing: {len(held)}/{len(pairs)} tp=2 engines fit on {len(gpus)} cards")
    # Machine full: the next launch must be refused for capacity, not accepted.
    results.append(("overcommit_refused_429", _probe(client, "overcommit", 429,
        lambda: client.launch(engine_args={"tp_size": 2}, gpu_indices=gpus[:2], port=None,
                              idempotency_key="pack-over"))))
    for lid in held:  # give the cards back
        try:
            client.release(lid)
        except PolicyError:
            pass
    return _finish_probes(client, results, sleep=sleep)


_RAW: dict[str, Callable[..., int]] = {
    "idle": run_idle,
    "wedge": run_wedge,
    "exit_hard": run_exit_hard,
    "flaky_network": run_flaky_network,
    "hw_constraints": run_hw_constraints,
    "bench_unserved": run_bench_unserved,
    "gpu_packing": run_gpu_packing,
}


# -- dispatch -------------------------------------------------------------------


def scenarios() -> list[str]:
    return sorted([*_STRATEGIES, *_RAW, "serve_fail"])


def runner_for(scenario: str, *, sleep=time.sleep) -> Runner:
    """A callable ``(client) -> exit_code`` for the named scenario."""
    if scenario in _STRATEGIES:
        strat = _STRATEGIES[scenario]
        return lambda client: run_policy(SearchLoop(strat(), sleep=sleep), client=client)
    if scenario == "serve_fail":
        return lambda client: run_policy(_NoServeLoop(HappyStrategy(), sleep=sleep), client=client)
    if scenario in _RAW:
        routine = _RAW[scenario]
        return lambda client: routine(client, sleep=sleep)
    raise SystemExit(
        f"unknown CHAOS_SCENARIO {scenario!r}; choose one of: {', '.join(scenarios())}"
    )


__all__ = ["PolicyError", "runner_for", "scenarios"]

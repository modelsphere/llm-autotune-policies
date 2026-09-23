"""The SearchLoop: the standard night, so your strategy never speaks HTTP.

It satisfies every MUST in the contract — dial-out only, heartbeat on cadence,
declare a plan, keep contenders current, obey the heartbeat command
(run/finalize/serve/exit/abort), and tear engines down when told. It drives one
delegated engine at a time: propose → launch → screen-benchmark → hand the
result to the strategy → refresh contenders. Commands that arrive mid-launch or
mid-benchmark interrupt cleanly (the engine is released, the command obeyed).

A strategy that wants to serve its own engines or run many at once can subclass
this; the reference random-search policy does not need to.
"""

from __future__ import annotations

import time

from . import space
from .client import (
    CONTRACT_VERSION,
    ConfigRejected,
    NoCapacity,
    PhaseError,
    PolicyClient,
    PolicyError,
    SessionOver,
)
from .strategy import Strategy, TrialResult

_ACTIVE = "run"
_DONE_LAUNCH = {"serving", "failed", "released"}
_DONE_BENCH = {"succeeded", "failed", "killed"}


class SearchLoop:
    def __init__(self, strategy: Strategy, *, sdk_version: str = "0.1.0",
                 sleep=time.sleep, poll_seconds: float | None = None):
        self.strategy = strategy
        self.sdk_version = sdk_version
        self._sleep = sleep
        self._poll_override = poll_seconds
        self._seq = 0
        self._active_launches: set[int] = set()
        self._poll = 30.0
        self._state_enabled = False
        self._space: dict = {}
        # Every config measured this night (successes and failures alike), used
        # to auto-build the self-reported coverage the platform badges alongside
        # its own derived view. The strategy never has to assemble it.
        self._observed_configs: list[dict] = []
        # A config the strategy already proposed (and thus recorded as "seen")
        # but that we could not launch this tick for a transient reason — no
        # free cards/ports while a just-released engine tears down. It is NOT a
        # measured trial; hold it and try it again rather than discard it, or a
        # capacity blip silently drops points from the sweep.
        self._retry: dict | None = None

    # -- entry point -----------------------------------------------------------
    def run(self, client: PolicyClient) -> int:
        manifest = client.get_session()
        self._poll = self._poll_override or float(
            (manifest.get("time") or {}).get("heartbeat_interval_s") or 30
        )
        self._state_enabled = bool((manifest.get("services") or {}).get("state"))
        self._space = manifest.get("search_space") or {}
        self.strategy.setup(manifest)

        hb = client.heartbeat(phase="starting", message="handshake", handshake={
            "sdk_version": self.sdk_version, "contract_version": CONTRACT_VERSION,
            "capabilities": ["serve"],
        })
        plan = self.strategy.plan()
        client.put_plan(tuning=plan.get("tuning", []), extras=plan.get("extras", []),
                        notes=plan.get("notes", ""))
        self.strategy.load_state(client.get_state())

        try:
            while True:
                command = hb.get("command", _ACTIVE)
                if command in ("exit", "abort"):
                    self._teardown(client)
                    return 0
                if command == "finalize":
                    hb = self._finalize(client)
                    continue
                if command == "serve":
                    self._serve(client, manifest, hb.get("contender_id"), hb.get("port"))
                    hb = self._beat(client, "validating",
                                    f"serving contender {hb.get('contender_id')}")
                    continue
                # "run", or an unknown command the contract says to treat as run.
                hb = self._search_once(client, manifest)
        except SessionOver:
            raise  # the platform is done with us — a clean exit, not an error
        except Exception as exc:  # noqa: BLE001 — surface ANY strategy bug as a signal
            # A crash in the search logic is not silent death: tell the platform
            # we gave up (so it finalizes and validates what we already found)
            # rather than making it wait out the heartbeat timeout.
            self._signal_error(client, exc)
            return 1

    # -- one unit of search ----------------------------------------------------
    def _search_once(self, client: PolicyClient, manifest: dict) -> dict:
        # A config held back by a transient capacity rejection comes first, so
        # it is measured rather than lost to the blip that deferred it.
        config = self._retry or self.strategy.next_config()
        self._retry = None
        if config is None:
            # Nothing left to explore. Signal exhaustion explicitly so the
            # platform finalizes now instead of waiting out the deadline; the
            # idle watchdog would eventually catch us, but saying so is the
            # cooperative thing to do.
            return self._beat(client, "searching",
                              self._status_msg() + " (nothing left to try)",
                              progress=1.0, status="exhausted", coverage=self._coverage())

        hw = manifest.get("hardware") or {}
        # Request exactly the cards this config's parallelism uses, not the
        # whole session allocation — the launch endpoint rejects a mismatch.
        gpus = (hw.get("gpu_indices") or [])[:space.cards_for(config)]
        port = (hw.get("ports") or [None])[0]
        self._seq += 1
        tag = f"t{self._seq}"

        try:
            launch = client.launch(engine_args=config, gpu_indices=gpus, port=port,
                                   idempotency_key=f"{tag}-launch")
        except ConfigRejected as exc:
            self._observed_configs.append(config)
            self.strategy.observe(TrialResult(config=config, objective=None, feasible=False,
                                              failure_class="bad_config", error=str(exc)))
            return self._beat(client, "searching", f"{tag}: rejected ({exc})")
        except NoCapacity:
            # Transient: a just-released engine is still holding its port/cards.
            # Keep the config and retry it instead of burning it — and wait a
            # beat first, so the worker has time to tear the old one down rather
            # than us busy-spinning launch calls against a port that is not free
            # yet (the run loop calls us straight back on an early return).
            self._retry = config
            hb = self._beat(client, "searching", f"{tag}: no free cards yet, will retry")
            if hb.get("command", _ACTIVE) == _ACTIVE:
                self._sleep(self._poll)
            return hb

        launch_id = launch["id"]
        self._active_launches.add(launch_id)
        launch, hb = self._wait(client, lambda: client.get_launch(launch_id), _DONE_LAUNCH,
                                f"{tag}: launching")
        if hb.get("command") != _ACTIVE:
            self._release(client, launch_id)
            return hb
        if launch.get("status") != "serving":
            self._observed_configs.append(launch.get("config") or config)
            self.strategy.observe(TrialResult(
                config=config, objective=None, feasible=False,
                failure_class=launch.get("failure_class") or "launch_failed",
                error=launch.get("error", ""), raw=launch))
            self._release(client, launch_id)
            self._refresh(client)
            return self._beat(client, "searching",
                              f"{tag}: {launch.get('failure_class') or 'failed'}")

        # Serving. The echoed config is the canonical truth — record trials
        # against it, not against what we asked for.
        canonical = launch.get("config") or config
        try:
            bench = client.bench_launch(launch_id, suite="screen", idempotency_key=f"{tag}-bench")
        except PhaseError as exc:
            self._release(client, launch_id)
            return self._beat(client, "searching", f"{tag}: bench refused ({exc})")

        bench_id = bench["id"]
        bench, hb = self._wait(client, lambda: client.get_benchmark(bench_id), _DONE_BENCH,
                               f"{tag}: benchmarking")
        self._release(client, launch_id)
        # If the benchmark finished, record it — even if a stop command just
        # arrived, a measurement in hand is not one to throw away.
        if bench.get("status") in _DONE_BENCH:
            summary = bench.get("summary") or {}
            self._observed_configs.append(canonical)
            self.strategy.observe(TrialResult(
                config=canonical,
                objective=summary.get("objective_value"),
                feasible=bool(summary.get("feasible")),
                failure_class=bench.get("failure_class", ""),
                trial_id=bench.get("trial_id"),
                error=bench.get("error", ""),
                raw=bench,
            ))
            self._refresh(client)
        if hb.get("command") != _ACTIVE:
            return hb
        msg, progress = self.strategy.status()
        return self._beat(client, "searching", msg or self._status_msg(),
                          progress=progress, coverage=self._coverage())

    # -- finalize + validation -------------------------------------------------
    def _finalize(self, client: PolicyClient) -> dict:
        self._teardown(client)
        self._refresh(client)  # last word on contenders before they freeze
        try:
            client.finalized()
        except PhaseError:
            pass  # already finalized on a retry
        return self._beat(client, "validating", "finalized; awaiting validation")

    def _serve(self, client: PolicyClient, manifest: dict, contender_id, port) -> None:
        """Serve exactly the commanded contender on the commanded port, then
        report it serving. We re-launch its recorded spec via delegated launch —
        the platform then health-checks and runs the full replay itself."""
        if contender_id is None:
            return
        spec = next((c for c in client.get_contenders() if c.get("id") == contender_id), None)
        if spec is None:
            client.event("error", f"serve: contender {contender_id} not found")
            return
        launch_spec = spec.get("launch_spec") or {}
        engine_args = launch_spec.get("engine_args") or {}
        gpus = ((manifest.get("hardware") or {}).get("gpu_indices") or [])[:space.cards_for(engine_args)]
        launch = client.launch(engine_args=engine_args, gpu_indices=gpus, port=port,
                               idempotency_key=f"serve-{contender_id}")
        launch_id = launch["id"]
        self._active_launches.add(launch_id)
        launch, _ = self._wait(client, lambda: client.get_launch(launch_id), _DONE_LAUNCH,
                               f"serving contender {contender_id}", stop_on_command=False)
        if launch.get("status") == "serving":
            client.contender_serving(contender_id)
        else:
            client.event("error",
                         f"serve: contender {contender_id} did not come up "
                         f"({launch.get('failure_class') or launch.get('status')})")

    # -- helpers ---------------------------------------------------------------
    def _wait(self, client: PolicyClient, poll, done: set, message: str,
              *, stop_on_command: bool = True):
        """Poll ``poll()`` until its status is terminal, heartbeating each round.
        Returns (last_object, last_heartbeat). If a non-run command arrives and
        ``stop_on_command``, returns early so the caller can obey it."""
        while True:
            obj = poll()
            hb = self._beat(client, "searching", message)
            if stop_on_command and hb.get("command") != _ACTIVE:
                return obj, hb
            if obj.get("status") in done:
                return obj, hb
            self._sleep(self._poll)

    def _beat(self, client: PolicyClient, phase: str, message: str,
              progress: float | None = None, *, status: str = "",
              coverage: dict | None = None) -> dict:
        # The heartbeat is the lifeline, but so is every other call: the client
        # rides out transient network blips and platform restarts on all of
        # them, so this is a plain call.
        return client.heartbeat(phase=phase, message=message, progress=progress,
                                status=status, coverage=coverage)

    def _coverage(self) -> dict | None:
        """The self-reported coverage to attach to a heartbeat: whatever the
        strategy overrides, else auto-built from the configs measured so far."""
        declared = self.strategy.coverage()
        if declared is not None:
            return declared
        if not self._observed_configs:
            return None
        return space.coverage_from_trials(self._space, self._observed_configs)

    def _signal_error(self, client: PolicyClient, exc: Exception) -> None:
        """Best-effort: tell the platform the search logic crashed, then release
        anything we were holding. Never raises — we are already failing."""
        try:
            client.heartbeat(phase="searching", message=f"policy error: {exc}"[:400],
                             status="error", failure_class=type(exc).__name__)
        except PolicyError:
            pass
        try:
            self._teardown(client)
        except PolicyError:
            pass

    def _refresh(self, client: PolicyClient) -> None:
        try:
            client.put_contenders(self.strategy.contenders())
        except PhaseError:
            pass  # frozen at finalize
        if self._state_enabled:
            try:
                client.put_state(self.strategy.dump_state())
            except PolicyError:
                pass  # state is a convenience, never worth failing a night over

    def _release(self, client: PolicyClient, launch_id: int) -> None:
        self._active_launches.discard(launch_id)
        try:
            client.release(launch_id)
        except PolicyError:
            pass

    def _teardown(self, client: PolicyClient) -> None:
        for launch_id in list(self._active_launches):
            self._release(client, launch_id)

    def _status_msg(self) -> str:
        return self.strategy.status()[0] or "searching"

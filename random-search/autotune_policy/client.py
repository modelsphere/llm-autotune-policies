"""HTTP client for the LLM AutoTune Policy Session API (contract v1.0).

Stdlib only, on purpose: a policy container should need nothing but Python to
phone home. Every call carries ``X-API-Key``; the POSTs that create work carry
an ``Idempotency-Key`` so a retry after a platform restart never double-launches
(the platform returns the original result for a repeated key).

The full contract is docs/api/policy-contract.md in the platform repo, and the
always-current schema is at ``<api>/api/docs``. This client maps one method to
one endpoint and returns the parsed JSON; it does not interpret it — that is the
loop's and the strategy's job.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CONTRACT_VERSION = "1.1"
_BASE_PATH = "/api/policy/v1"


class PolicyError(Exception):
    """Any non-2xx from the platform. ``status`` is the HTTP code (0 = network)."""

    def __init__(self, status: int, message: str, body: dict | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body or {}


class SessionOver(PolicyError):
    """401 or 410 — the session is terminal for us. Exit."""


class PhaseError(PolicyError):
    """409 — the current phase forbids this call. Re-read the heartbeat command."""


class ConfigRejected(PolicyError):
    """422 — a safety rejection (cards, malformed value). Fix the config, don't retry it."""


class NoCapacity(PolicyError):
    """429 — no free cards/ports inside our allocation. Release something or wait."""


_STATUS_EXC = {401: SessionOver, 410: SessionOver, 409: PhaseError,
               422: ConfigRejected, 429: NoCapacity}


@dataclass
class PolicyConfig:
    """The three environment variables every policy container boots with."""

    api_url: str
    api_key: str
    session_id: int

    @classmethod
    def from_env(cls, env: dict | None = None) -> PolicyConfig:
        env = env if env is not None else os.environ
        url = (env.get("AUTOTUNE_API_URL") or "").rstrip("/")
        key = env.get("AUTOTUNE_API_KEY") or ""
        sid = env.get("AUTOTUNE_SESSION_ID") or ""
        missing = [n for n, v in (("AUTOTUNE_API_URL", url), ("AUTOTUNE_API_KEY", key),
                                  ("AUTOTUNE_SESSION_ID", sid)) if not v]
        if missing:
            raise RuntimeError(f"missing policy env: {', '.join(missing)}")
        return cls(api_url=url, api_key=key, session_id=int(sid))


# Failures worth retrying: a network blip (status 0, from URLError) and the
# gateway errors a platform restart shows through a proxy. A 4xx is a decision
# (auth, phase, safety, capacity) that a retry cannot change, so it is never
# retried — the loop handles 429 with its own backoff.
_TRANSIENT = frozenset({0, 502, 503, 504})


class PolicyClient:
    """One method per endpoint. Raises the typed errors above on 4xx.

    Every call retries on transient failures on its own, so no method here (nor
    the loop that calls them) has to. This is safe for all of them: the POSTs
    that create work carry an ``Idempotency-Key``, so a retry after a lost
    response returns the original result rather than doubling the work, and
    every other call is naturally idempotent.
    """

    def __init__(self, config: PolicyConfig, *, timeout: float = 30.0, opener=None,
                 retries: int = 5, backoff: float = 1.0, backoff_cap: float = 8.0,
                 sleep=time.sleep):
        self.config = config
        self._base = config.api_url + _BASE_PATH
        self._timeout = timeout
        self._open = opener or urllib.request.urlopen
        self._retries = retries
        self._backoff = backoff
        self._backoff_cap = backoff_cap
        self._sleep = sleep

    def _request(self, method: str, path: str, body: Any = None,
                 idempotency_key: str | None = None) -> Any:
        attempt = 0
        while True:
            try:
                return self._attempt(method, path, body, idempotency_key)
            except PolicyError as exc:
                if exc.status in _TRANSIENT and attempt < self._retries:
                    self._sleep(min(self._backoff * (2 ** attempt), self._backoff_cap))
                    attempt += 1
                    continue
                raise

    def _attempt(self, method: str, path: str, body: Any,
                 idempotency_key: str | None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._base + path, data=data, method=method)
        req.add_header("X-API-Key", self.config.api_key)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if idempotency_key:
            req.add_header("Idempotency-Key", idempotency_key)
        try:
            with self._open(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                parsed = json.loads(raw) if raw else {}
            except ValueError:
                parsed = {"detail": raw.decode(errors="replace")}
            message = parsed.get("detail") or parsed.get("message") or exc.reason or "error"
            raise _STATUS_EXC.get(exc.code, PolicyError)(exc.code, str(message), parsed) from exc
        except urllib.error.URLError as exc:
            # A transient network blip or a platform restart; _request retries.
            raise PolicyError(0, f"network error: {exc.reason}") from exc

    # -- session ---------------------------------------------------------------
    def get_session(self) -> dict:
        return self._request("GET", "/session")

    def heartbeat(self, *, phase: str, message: str = "", progress: float | None = None,
                  status: str = "", failure_class: str = "",
                  coverage: dict | None = None, handshake: dict | None = None) -> dict:
        payload: dict[str, Any] = {"phase": phase, "message": message}
        if progress is not None:
            payload["progress"] = progress
        if status:
            payload["status"] = status  # "exhausted" | "error"
        if failure_class:
            payload["failure_class"] = failure_class
        if coverage is not None:
            payload["coverage"] = coverage
        if handshake:
            payload.update(handshake)
        return self._request("POST", "/session/heartbeat", payload)

    def put_plan(self, *, tuning: list[str], extras: list[dict] | None = None,
                 notes: str = "") -> None:
        self._request("PUT", "/session/plan",
                      {"tuning": tuning, "extras": extras or [], "notes": notes})

    def finalized(self) -> None:
        self._request("POST", "/session/finalized", {})

    # -- delegated launches ----------------------------------------------------
    def launch(self, *, engine_args: dict, gpu_indices: list[int], port: int | None,
               idempotency_key: str) -> dict:
        body: dict[str, Any] = {"engine_args": engine_args, "gpu_indices": gpu_indices}
        if port is not None:
            body["port"] = port
        return self._request("POST", "/launches", body, idempotency_key=idempotency_key)

    def get_launch(self, launch_id: int) -> dict:
        return self._request("GET", f"/launches/{launch_id}")

    def release(self, launch_id: int) -> dict:
        return self._request("DELETE", f"/launches/{launch_id}")

    # -- benchmarks ------------------------------------------------------------
    def bench_launch(self, launch_id: int, *, suite: str, idempotency_key: str) -> dict:
        return self._request("POST", f"/launches/{launch_id}/benchmarks",
                             {"suite": suite}, idempotency_key=idempotency_key)

    def bench_self(self, *, port: int, engine_args: dict, gpu_indices: list[int],
                   suite: str, idempotency_key: str) -> dict:
        return self._request("POST", "/benchmarks",
                             {"port": port, "engine_args": engine_args,
                              "gpu_indices": gpu_indices, "suite": suite},
                             idempotency_key=idempotency_key)

    def get_benchmark(self, benchmark_id: int) -> dict:
        return self._request("GET", f"/benchmarks/{benchmark_id}")

    # -- trials / contenders ---------------------------------------------------
    def report_trial(self, trial: dict, *, idempotency_key: str) -> dict:
        return self._request("POST", "/trials", trial, idempotency_key=idempotency_key)

    def put_contenders(self, contenders: list[dict]) -> dict:
        return self._request("PUT", "/contenders", {"contenders": contenders})

    def get_contenders(self) -> list[dict]:
        return self._request("GET", "/contenders")

    def contender_serving(self, contender_id: int) -> dict:
        return self._request("POST", f"/contenders/{contender_id}/serving", {})

    # -- state / events --------------------------------------------------------
    def get_state(self) -> dict:
        try:
            return self._request("GET", "/state") or {}
        except PolicyError as exc:
            if exc.status == 404:
                return {}  # first night — nothing written yet
            raise

    def put_state(self, blob: dict) -> None:
        self._request("PUT", "/state", blob)

    def event(self, level: str, message: str) -> None:
        self._request("POST", "/events", {"level": level, "message": message})

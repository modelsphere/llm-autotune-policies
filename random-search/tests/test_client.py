"""The client's built-in retry: every call rides out a transient blip, none
retries a decision (4xx). No sleeps, no network — a scripted opener."""

from __future__ import annotations

import io
import urllib.error

import pytest

from autotune_policy.client import (
    ConfigRejected,
    NoCapacity,
    PolicyClient,
    PolicyConfig,
    PolicyError,
)


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _http_error(code: int):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(b'{"detail":"no"}'))


def _client(opener):
    return PolicyClient(
        PolicyConfig(api_url="http://x", api_key="k", session_id=1),
        opener=opener, sleep=lambda _s: None,  # no real waiting in tests
    )


def test_a_network_blip_is_retried_until_it_succeeds():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("blip")
        return _Resp(b'{"ok": true}')

    out = _client(opener).get_session()
    assert out == {"ok": True}
    assert calls["n"] == 3  # two failures ridden out, third succeeded


def test_a_gateway_error_during_a_restart_is_retried():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)  # platform restarting behind the proxy
        return _Resp(b"{}")

    _client(opener).heartbeat(phase="searching")
    assert calls["n"] == 2


def test_a_4xx_is_a_decision_and_is_never_retried():
    for code, exc_type in ((422, ConfigRejected), (429, NoCapacity)):
        calls = {"n": 0}

        def opener(req, timeout=None, _code=code, _calls=calls):
            _calls["n"] += 1
            raise _http_error(_code)

        with pytest.raises(exc_type) as caught:
            _client(opener).launch(engine_args={}, gpu_indices=[0], port=None,
                                   idempotency_key="k1")
        assert caught.value.status == code
        assert calls["n"] == 1  # not retried


def test_retries_are_bounded_and_then_it_gives_up():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("down")

    client = PolicyClient(
        PolicyConfig(api_url="http://x", api_key="k", session_id=1),
        opener=opener, sleep=lambda _s: None, retries=3,
    )
    with pytest.raises(PolicyError) as caught:
        client.get_session()
    assert caught.value.status == 0
    assert calls["n"] == 4  # the first try plus `retries` more

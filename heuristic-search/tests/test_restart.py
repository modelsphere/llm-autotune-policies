"""Persist through the platform API and restart with a fresh policy process."""
import json
from copy import deepcopy

import pytest

from autotune_policy import SearchLoop
from autotune_policy.space import canonical_key
from autotune_policy.strategy import TrialResult
from strategies.heuristic_search import HeuristicSearch, HeuristicSearchV2
from tests.test_heuristic_search import manifest
from tests.test_night import FakePlatform


@pytest.mark.parametrize("cls", [HeuristicSearch, HeuristicSearchV2])
@pytest.mark.parametrize("new_session", [False, True])
def test_restart_retains_results_and_skips_completed_trials(cls, new_session):
    spec = manifest({"grid": {"max_running_requests": [8, 16, 32, 64, 128]}})
    first = FakePlatform(spec, target_trials=2)
    original = cls(max_trials=2)
    assert SearchLoop(original, sleep=lambda _: None).run(first) == 0
    saved = json.loads(json.dumps(first.state))
    if new_session:
        spec = deepcopy(spec)
        spec["session_id"] += 1
    restored = cls(max_trials=4)
    restored.setup(spec)
    restored.load_state(saved)
    assert restored.observations == original.observations
    assert restored.contenders() == original.contenders()
    if cls is HeuristicSearchV2:
        assert restored.samples == original.samples
        assert restored.center == original.center
        assert restored.radius == original.radius
    completed = {canonical_key(r.config) for r in original.observations}
    assert canonical_key(restored.next_config()) not in completed
    assert restored.proposed == (1 if new_session else 3)


@pytest.mark.parametrize("cls", [HeuristicSearch, HeuristicSearchV2])
def test_restart_preserves_failed_trials_and_retries_unfinished_proposal(cls):
    spec = manifest({"grid": {"max_running_requests": [8, 16, 32, 64]}})
    original = cls()
    original.setup(spec)
    failed = original.next_config()
    original.observe(TrialResult(failed, None, False, failure_class="oom"))
    unfinished = original.next_config()
    restored = cls()
    restored.setup(spec)
    restored.load_state(json.loads(json.dumps(original.dump_state())))
    assert restored.observations == original.observations
    assert canonical_key(failed) in restored.seen
    assert restored.next_config() == unfinished


@pytest.mark.parametrize("cls", [HeuristicSearch, HeuristicSearchV2])
def test_legacy_seed_still_loads(cls):
    spec = manifest({"grid": {"max_running_requests": [8, 16, 32, 64]}})
    obj = cls()
    obj.setup(spec)
    obj.load_state({"strategy": obj.state_strategy, "context": obj.context,
                    "seed": {"max_running_requests": 64}})
    assert obj.observations == []
    assert obj.next_config() == {"max_running_requests": 64}


@pytest.mark.parametrize("cls", [HeuristicSearch, HeuristicSearchV2])
def test_abrupt_stop_then_loop_restart_does_not_repeat_completed_trials(cls):
    class Stopped(BaseException):
        pass

    class RestartPlatform(FakePlatform):
        stop = True

        def __init__(self, spec):
            super().__init__(spec, target_trials=4)
            self.keys = set()
            self.configs = []

        def launch(self, **kwargs):
            if not self.finalized_called:
                if self.stop and self.trials == 2:
                    raise Stopped()
                key = kwargs["idempotency_key"]
                assert key not in self.keys, "a restarted loop reused an old launch request key"
                self.keys.add(key)
                self.configs.append(canonical_key(kwargs["engine_args"]))
            return super().launch(**kwargs)

        def put_state(self, blob):
            self.state = json.loads(json.dumps(blob))

    spec = manifest({"grid": {"max_running_requests": [8, 16, 32, 64, 128]}})
    fake = RestartPlatform(spec)
    with pytest.raises(Stopped):
        SearchLoop(cls(max_trials=8), sleep=lambda _: None).run(fake)
    assert not fake.finalized_called
    fake.stop = False
    restored = cls(max_trials=8)
    assert SearchLoop(restored, sleep=lambda _: None).run(fake) == 0
    assert len(fake.configs) == len(set(fake.configs))
    assert len(restored.observations) == fake.trials
    assert fake.finalized_called


def test_rejected_config_is_checkpointed_before_exit():
    from autotune_policy.client import ConfigRejected

    class RejectedPlatform(FakePlatform):
        rejected = False

        def launch(self, **kwargs):
            self.rejected = True
            raise ConfigRejected(422, {"error": "bad config"})

        def _command(self):
            return ("exit", None, None) if self.rejected else super()._command()

    spec = manifest({"grid": {"max_running_requests": [8, 16, 32, 64]}})
    fake = RejectedPlatform(spec)
    assert SearchLoop(HeuristicSearchV2(), sleep=lambda _: None).run(fake) == 0
    assert fake.state["observations"][0]["failure_class"] == "bad_config"


def test_state_write_failure_is_visible(caplog):
    from autotune_policy.client import PolicyError

    class BrokenState(FakePlatform):
        def put_state(self, blob):
            raise PolicyError(503, {"error": "unavailable"})

    fake = BrokenState(manifest({"grid": {"max_running_requests": [16, 32]}}), target_trials=1)
    assert SearchLoop(HeuristicSearchV2(), sleep=lambda _: None).run(fake) == 0
    assert "Failed to persist strategy state" in caplog.text

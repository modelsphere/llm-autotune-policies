"""TPE against the campaign's space vocabulary.

Whether TPE optimizes well is Optuna's business; these tests pin the seams.
The space maps into distributions and back without inventing or losing a
config, float steps snap to what the space declares, proposals are
deterministic for the same history, and the learning signal flows: better
regions draw proposals, crashing regions repel them, and infrastructure
failures teach nothing.
"""

from __future__ import annotations

from autotune_policy.space import canonical_key
from autotune_policy.strategy import TrialResult
from strategies.optuna_space import TIED_PREFIX, decode, distributions, encode, enumerate_space
from strategies.tpe import PENALTY_SLACK, TPE

SPACE = {
    "base": {"mem_fraction_static": 0.9},
    "grid": {"schedule_policy": ["fcfs", "lpm"]},
    "tied": [{"tp_size": [1, 2, 4], "dp_size": [4, 2, 1]}],
    "range": {"chunked_prefill_size": {"min": 2048, "max": 8192, "step": 2048}},
}


def _keys(configs) -> set[str]:
    return {canonical_key(c) for c in configs}


def _strategy(space, *, direction="max", seed=7, **kwargs) -> TPE:
    s = TPE(seed=seed, **kwargs)
    s.setup({"search_space": space, "session_id": 1, "model": {"image": "img"},
             "objective": {"screen": {"direction": direction}}})
    return s


def _measured(config, value, constraints=None, feasible=True) -> TrialResult:
    summary = {"objective_value": value, "feasible": feasible}
    if constraints is not None:
        summary["constraints"] = list(constraints)
    return TrialResult(config=config, objective=value, feasible=feasible,
                       raw={"summary": summary})


def _crashed(config, failure_class="oom") -> TrialResult:
    return TrialResult(config=config, objective=None, feasible=False,
                       failure_class=failure_class)


def _proposals(strategy: TPE, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        config = strategy.next_config()
        if config is None:
            break
        out.append(config)
    return out


# -- the space -------------------------------------------------------------------


def test_distributions_cover_every_axis_kind():
    dists = distributions(SPACE)
    assert set(dists) == {"schedule_policy", f"{TIED_PREFIX}0", "chunked_prefill_size"}
    assert dists["schedule_policy"].choices == ("fcfs", "lpm")
    # The tied group is ONE index over its rows, not two free columns.
    assert (dists[f"{TIED_PREFIX}0"].low, dists[f"{TIED_PREFIX}0"].high) == (0, 2)
    assert dists["chunked_prefill_size"].step == 2048


def test_float_ranges_are_continuous_and_integer_ranges_stepped():
    dists = distributions({"range": {
        "mem": {"min": 0.7, "max": 0.9, "step": 0.05},
        "chunk": {"min": 2048, "max": 8192, "step": 2048},
    }})
    assert dists["mem"].step is None
    assert dists["chunk"].step == 2048


def test_encode_then_decode_gives_back_every_config_of_the_space():
    configs = enumerate_space(SPACE)
    assert len(configs) == 2 * 3 * 4
    for config in configs:
        assert decode(encode(config, SPACE), SPACE) == config


def test_decode_snaps_float_arithmetic_to_declared_values():
    space = {"range": {"mem": {"min": 0.7, "max": 0.9, "step": 0.05}}}
    drifted = decode({"mem": 0.7999999999999999}, space)
    assert drifted == {"mem": 0.8}
    assert _keys([drifted]) <= _keys(enumerate_space(space))


def test_encode_keeps_only_what_the_space_still_declares():
    moved = {"schedule_policy": "retired", "chunked_prefill_size": 999999}
    assert encode(moved, SPACE) == {}
    half = {"schedule_policy": "lpm", "chunked_prefill_size": 999999}
    assert encode(half, SPACE) == {"schedule_policy": "lpm"}


def test_conditions_are_applied_to_proposals():
    space = {
        "grid": {"speculative_algorithm": ["off", "EAGLE"], "speculative_num_steps": [2, 4]},
        "conditions": {"speculative_num_steps": {"speculative_algorithm": ["EAGLE"]}},
    }
    proposals = _proposals(_strategy(space), 10)
    assert _keys(proposals) <= _keys(enumerate_space(space))
    for p in proposals:
        if p["speculative_algorithm"] == "off":
            assert "speculative_num_steps" not in p


# -- proposing -------------------------------------------------------------------


def test_proposals_stay_inside_the_space_and_never_repeat():
    proposals = _proposals(_strategy(SPACE), 12)
    assert len(proposals) == 12
    assert _keys(proposals) <= _keys(enumerate_space(SPACE))
    assert len(_keys(proposals)) == 12


def test_the_same_history_proposes_the_same_config():
    """What makes a restart safe: the study is rebuilt from the observations,
    and the seed follows them."""
    first, second = _strategy(SPACE), _strategy(SPACE)
    for s in (first, second):
        s.observe(_measured(enumerate_space(SPACE)[0], 10.0, [-1.0]))
    assert first.next_config() == second.next_config()


def test_a_full_space_runs_out_and_says_so():
    s = _strategy(SPACE)
    proposals = _proposals(s, 100)
    assert len(proposals) == len(enumerate_space(SPACE))
    assert s.next_config() is None


def test_max_trials_is_a_budget():
    s = _strategy(SPACE, max_trials=3)
    assert len(_proposals(s, 10)) == 3


def test_a_space_with_nothing_to_vary_proposes_the_base_once():
    s = _strategy({"base": {"tp_size": 8}})
    assert s.next_config() == {"tp_size": 8}
    s.observe(_measured({"tp_size": 8}, 1.0))
    assert s.next_config() is None


def test_prior_contenders_are_not_measured_again():
    s = TPE(seed=1)
    top = {"mem_fraction_static": 0.9, "schedule_policy": "lpm", "tp_size": 4,
           "dp_size": 1, "chunked_prefill_size": 8192}
    s.setup({"search_space": SPACE, "session_id": 2, "model": {},
             "prior": {"contenders": [{"launch_spec": {"engine_args": top}}]}})
    assert canonical_key(top) not in _keys(_proposals(s, 23))


# -- learning ----------------------------------------------------------------------


def test_learns_toward_the_better_region():
    """Past startup, proposals concentrate where measured values are high:
    what separates TPE from random search."""
    space = {"range": {"x": {"min": 0, "max": 100, "step": 5}}}
    peak = 60
    s = _strategy(space)
    for x in (0, 15, 30, 45, 60, 75, 90):
        s.observe(_measured({"x": x}, 100.0 - abs(x - peak), [-1.0]))
    xs = [c["x"] for c in _proposals(s, 6)]
    near = [x for x in xs if abs(x - peak) <= 25]
    assert len(near) >= len(xs) // 2, f"proposals ignore the peak: {xs}"


def test_minimizing_learns_toward_low_values():
    space = {"range": {"x": {"min": 0, "max": 100, "step": 5}}}
    s = _strategy(space, direction="min")
    for x in (0, 15, 30, 45, 60, 75, 90):
        s.observe(_measured({"x": x}, float(x), [-1.0]))
    xs = [c["x"] for c in _proposals(s, 6)]
    assert sum(x <= 40 for x in xs) >= len(xs) // 2, f"proposals ignore the minimum: {xs}"


def test_avoids_the_region_that_crashes():
    """Config-induced failures are observations: told the high-memory region
    OOMs, the search mostly stays out of it, even though measured values
    climb right up to the cliff."""
    space = {"range": {"mem": {"min": 0.7, "max": 0.95, "step": 0.01}}}
    s = _strategy(space)
    for m in (0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84):
        s.observe(_measured({"mem": m}, m * 100, [-1.0]))
    for m in (0.86, 0.88, 0.9, 0.92, 0.94):
        s.observe(_crashed({"mem": m}))
    mems = [c["mem"] for c in _proposals(s, 8)]
    crashed = [m for m in mems if m >= 0.86]
    # Uniform draws would land about 38% in the crash zone (10 of 26 values).
    assert len(crashed) <= len(mems) // 3, f"still proposing into the cliff: {mems}"


def test_a_crash_is_recorded_as_a_violated_constraint():
    s = _strategy(SPACE)
    config = enumerate_space(SPACE)[0]
    s.observe(_crashed(config, "oom"))
    assert s.observations == [
        {"config": config, "value": None, "constraints": [PENALTY_SLACK]}]


def test_an_infrastructure_failure_teaches_nothing():
    s = _strategy(SPACE)
    config = enumerate_space(SPACE)[0]
    s.observe(_crashed(config, "ssh_timeout"))
    s.observe(_crashed(config, "launch_failed"))
    assert s.observations == []
    # But the config is not proposed again tonight.
    assert config not in _proposals(s, 23)


def test_redline_slacks_ride_along_and_are_finite():
    s = _strategy(SPACE)
    config = enumerate_space(SPACE)[0]
    s.observe(_measured(config, 5.0, [12.5, float("inf")], feasible=False))
    assert s.observations[0]["constraints"] == [12.5, PENALTY_SLACK]


def test_history_from_an_edited_space_is_skipped_not_fatal():
    s = _strategy(SPACE)
    s.observe(_measured({"retired_param": True}, 3.0))
    assert len(_proposals(s, 3)) == 3


def test_contenders_are_the_best_feasible_results():
    s = _strategy(SPACE)
    configs = enumerate_space(SPACE)
    s.observe(_measured(configs[0], 10.0))
    s.observe(_measured(configs[1], 30.0, feasible=False))
    s.observe(_measured(configs[2], 20.0))
    rows = s.contenders()
    assert [r["evidence"]["screen_objective"] for r in rows] == [20.0, 10.0]
    assert rows[0]["launch_spec"]["engine_args"] == configs[2]

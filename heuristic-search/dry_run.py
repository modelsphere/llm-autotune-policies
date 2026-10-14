"""Offline platform with an explicitly synthetic, configurable response surface."""
from __future__ import annotations

import math
import operator
import random
from copy import deepcopy

PREFIX = "perf_guidellm_sweep."
TTFT = PREFIX + "ttft_p99"
TPM = PREFIX + "total_tpm_card_norm"
TPS = PREFIX + "c1.request_output_tps"
SLO = PREFIX + "reported_level_meets_slo"
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
DEFAULT_MODEL = {
    "baseline": {"max_running_requests": 32, "chunked_prefill_size": 4096,
                 "prefill_max_requests": 4, "mem_fraction_static": 0.85,
                 "mamba_full_memory_ratio": 0.5, "cuda_graph_max_bs_decode": 32,
                 "schedule_conservativeness": 1.0},
    "metrics": {TTFT: 9.0, TPM: 12000.0, TPS: 80.0},
    # metric = baseline metric * product((parameter / baseline parameter) ** exponent)
    "effects": {
        "max_running_requests": {TTFT: -0.45, TPM: 0.55, TPS: -0.15},
        "chunked_prefill_size": {TTFT: -0.25, TPM: 0.15, TPS: -0.10},
        "prefill_max_requests": {TTFT: -0.15, TPM: 0.10},
        "mem_fraction_static": {TTFT: -0.3, TPM: 0.5},
        "mamba_full_memory_ratio": {TTFT: -0.15, TPM: 0.20},
        "cuda_graph_max_bs_decode": {TPS: 0.10, TPM: 0.05},
        "schedule_conservativeness": {TTFT: 0.20, TPM: -0.15},
    },
    "saturation_concurrency": 64,
    "queue_penalty": 0.8,
    "ttft_slo": 8.0,
    "oom_pressure": 3.0,
}


class MetricSimulator:
    def __init__(self, manifest, *, model=None, seed=0, noise=0.0):
        if not math.isfinite(noise) or not 0 <= noise < 1:
            raise ValueError("noise must be finite and in [0, 1)")
        self.model = deepcopy(DEFAULT_MODEL)
        for key, value in (model or {}).items():
            if isinstance(value, dict):
                self.model[key].update(deepcopy(value))
            else:
                self.model[key] = value
        self.screen = manifest["objective"]["screen"]
        self.rng = random.Random(seed)
        self.noise = noise
        for key in ("saturation_concurrency", "ttft_slo", "oom_pressure"):
            if not math.isfinite(self.model[key]) or self.model[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.model["queue_penalty"]) or self.model["queue_penalty"] < 0:
            raise ValueError("queue_penalty must be finite and nonnegative")
        for value in self.model["baseline"].values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError("model baseline parameters must be finite and positive")
        for value in self.model["metrics"].values():
            if not math.isfinite(value) or value < 0:
                raise ValueError("model metrics must be finite and nonnegative")
        for param, effects in self.model["effects"].items():
            if param not in self.model["baseline"]:
                raise ValueError(f"missing model baseline for {param}")
            for metric, exponent in effects.items():
                if metric not in self.model["metrics"] or not math.isfinite(exponent):
                    raise ValueError(f"invalid effect for {param}: {metric}")
        required = {self.screen["target_metric"]}
        for rule in self.screen.get("redlines", []):
            if rule["op"] not in OPS or not math.isfinite(rule["value"]):
                raise ValueError("invalid screen redline")
            required.add(rule["metric"])
        missing = required - (self.model["metrics"].keys() | {SLO})
        if missing:
            raise ValueError(f"no simulation model for metrics: {sorted(missing)}")

    def parameters(self, config):
        values = {**self.model["baseline"], **config}
        for key in self.model["baseline"]:
            value = values[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"simulation requires positive numeric {key}, got {value!r}")
        return values

    def is_oom(self, config):
        p = self.parameters(config)
        pressure = (0.55 * p["max_running_requests"] / 32
                    + 0.30 * p["chunked_prefill_size"] / 4096
                    + 0.15 * p["prefill_max_requests"] / 4)
        pressure *= (0.15 / max(0.01, 1 - p["mem_fraction_static"])) ** 0.5
        return pressure > self.model["oom_pressure"]

    def benchmark(self, config):
        p = self.parameters(config)
        metrics = dict(self.model["metrics"])
        for param, effects in self.model["effects"].items():
            ratio = p[param] / self.model["baseline"][param]
            for metric, exponent in effects.items():
                metrics[metric] *= ratio ** exponent
        excess = max(0, p["max_running_requests"] / self.model["saturation_concurrency"] - 1)
        metrics[TTFT] *= 1 + self.model["queue_penalty"] * excess ** 2
        metrics[TPM] /= 1 + 0.3 * excess
        metrics = {k: round(v * (1 + self.rng.uniform(-self.noise, self.noise)), 6)
                   for k, v in metrics.items()}
        metrics[SLO] = int(metrics[TTFT] <= self.model["ttft_slo"])
        constraints, breaches = [], []
        for rule in self.screen.get("redlines", []):
            value, limit = metrics[rule["metric"]], rule["value"]
            constraints.append(value - limit if rule["op"] in ("<", "<=") else limit - value)
            if not OPS[rule["op"]](value, limit):
                breaches.append({**rule, "actual": value})
        return {"status": "succeeded", "metrics": metrics,
                "summary": {"objective_value": metrics[self.screen["target_metric"]],
                            "feasible": not breaches, "constraints": constraints,
                            "breaches": breaches}}


class DryRunPlatform:
    """Client-compatible in-memory platform; no network, engines, or wall-clock waits."""
    def __init__(self, manifest, *, model=None, seed=0, noise=0.0):
        self.manifest = deepcopy(manifest)
        self.simulator = MetricSimulator(manifest, model=model, seed=seed, noise=noise)
        self.launches, self.benchmarks = {}, {}
        self.contenders, self.trace, self.events = [], [], []
        self.state, self.plan = {}, {}
        self.finished = self.exhausted = False
        self.served = []
        self.coverage = None

    def get_session(self):
        return self.manifest

    def heartbeat(self, **payload):
        self.exhausted |= payload.get("status") in ("exhausted", "error")
        self.coverage = payload.get("coverage") or self.coverage
        if payload.get("status") == "error":
            self.events.append(payload)
        if self.finished:
            pending = [c for c in self.contenders if c["id"] not in self.served]
            if pending:
                return {"command": "serve", "contender_id": pending[0]["id"],
                        "port": (self.manifest.get("hardware", {}).get("ports") or [28200])[0]}
            return {"command": "exit"}
        return {"command": "finalize" if self.exhausted else "run"}

    def put_plan(self, **plan):
        self.plan = plan

    def get_state(self):
        return self.state

    def put_state(self, blob):
        self.state = deepcopy(blob)

    def launch(self, *, engine_args, **kwargs):
        lid = len(self.launches) + 1
        oom = self.simulator.is_oom(engine_args)
        row = {"id": lid, "config": deepcopy(engine_args),
               "status": "failed" if oom else "serving",
               "environment": {"simulated": True}}
        if oom:
            row.update(failure_class="oom", error="synthetic memory pressure exceeds threshold")
            if not self.finished:
                self.trace.append({"stage": "launch", **deepcopy(row)})
        self.launches[lid] = row
        return row

    def get_launch(self, launch_id):
        return self.launches[launch_id]

    def release(self, launch_id):
        self.launches[launch_id]["status"] = "released"

    def bench_launch(self, launch_id, *, suite, **kwargs):
        bid = len(self.benchmarks) + 1
        config = self.launches[launch_id]["config"]
        row = {**self.simulator.benchmark(config), "id": bid, "trial_id": bid,
               "suite": suite, "config": deepcopy(config), "environment": {"simulated": True}}
        self.benchmarks[bid] = row
        self.trace.append({"stage": "benchmark", **deepcopy(row)})
        return row

    def get_benchmark(self, benchmark_id):
        return self.benchmarks[benchmark_id]

    def put_contenders(self, contenders):
        self.contenders = [{**deepcopy(c), "id": i + 1} for i, c in enumerate(contenders)]

    def get_contenders(self):
        return self.contenders

    def finalized(self):
        self.finished = True

    def contender_serving(self, contender_id):
        self.served.append(contender_id)

    def event(self, level, message):
        self.events.append({"level": level, "message": message})

    def report(self):
        return {"simulated": True, "validation": "serve lifecycle only; verify suite not executed",
                "manifest": self.manifest, "model": self.simulator.model,
                "plan": self.plan, "trials": self.trace, "contenders": self.contenders,
                "coverage": self.coverage, "state": self.state, "events": self.events}

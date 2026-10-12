# Heuristic search policy

The entrypoint runs the single `HeuristicSearch` class in
`strategies/heuristic_search.py`. It preserves the constraint-aware search behavior
and accepts existing V2 checkpoints. New checkpoints use the model-neutral
`heuristic-search` identifier; V1 and seed-only checkpoints are not supported.

From this directory:

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run python main.py --dry-run --seed 7 --output /tmp/heuristic-search.json
```

Online runs require `AUTOTUNE_API_URL`, `AUTOTUNE_API_KEY`, and
`AUTOTUNE_SESSION_ID`. Run `uv run python main.py`, or build from the repository root:

```bash
docker build -t llm-autotune-policy-heuristic-search:local heuristic-search/
docker run --rm llm-autotune-policy-heuristic-search:local --dry-run --seed 7
```

Simulation uses synthetic metrics; it does not validate real GPU performance.
The SDK is vendored from the repository root; update it there and run
`scripts/sync-sdk.sh`. CI tests and builds this policy, but does not publish it.

## Heuristic strategy

`strategies/heuristic_search.py` provides `HeuristicSearch`, a deterministic,
budgeted local search for a **previously validated deployment**.
To configure it directly:

```python
from strategies.heuristic_search import HeuristicSearch

run_policy(SearchLoop(HeuristicSearch(workload="long_prefill", max_trials=24)))
```

Profiles: `interactive`, `long_prefill`, `long_decode`, `agent` (default), `batch`, `mixed`.
Profiles prioritize different operational parameters; they do not
infer workload characteristics from a score or enable new engine features.
Only supported knobs present in `search_space` are varied. Topology, backend,
precision, cache mode and speculative decoding remain fixed. Validate those as
separate recipes, including their compatibility with the deployed engine version.

The initial point projects production arguments (optionally overridden with
`baseline={...}`) onto declared values. For missing operational values it uses
conservative seed hints; other dimensions use their first declared value. Supply
a validated baseline, particularly `mamba_full_memory_ratio`: there is no universal
memory-pool ratio. Unknown parameters outside the space are not copied in.
Tied groups move atomically and are fixed if any column is not a supported knob.
Conditions are pruned before deduplication. Overlapping axes, empty domains and
malformed ranges are rejected during setup.

Search measures the baseline, probes adjacent values, and uses local predictions
to rank larger steps within an adaptive trust radius. Once the center is feasible,
a relative improvement above `min_improvement=0.01` moves the search center; all feasible measured results remain eligible as contenders.
OOM feedback prioritizes local reductions of chunk size, concurrency or static
memory, without blacklisting an entire parameter value. The strategy stops on
budget exhaustion or a fully visited local neighborhood, **not** proof of global
optimality. It does not yet perform coupled-axis escape moves or repeat trials
to estimate measurement noise. Use a stable screen suite and platform validation.

State stores a context-checked checkpoint: completed results (including failures,
raw metrics and trial IDs), seen configurations, search center and pending work.
Matching contexts resume without remeasuring completed configurations. A restart
in the same session retains the trial budget counter; a new session receives a
fresh budget while retaining historical evidence. Changed model, hardware, search
space, objectives or strategy settings invalidate incompatible checkpoints.
Coverage remains the loop's current-run measured-axis
coverage, not a claim that the entire Cartesian space has been searched.

You never touch `loop.py` or `client.py` unless you need something exotic
(serving your own engines, many concurrent launches) — subclass `SearchLoop`
then.

## Constraint-aware search

The same class exposes constraint-aware search settings:

```python
from strategies.heuristic_search import HeuristicSearch

run_policy(SearchLoop(HeuristicSearch(
    workload="agent",
    max_trials=24,
    safety_margin=0.05,
    max_step=4,
)))
```

The strategy reads constraints directly from the platform manifest; a separate
constructor constraints mapping is not supported.
The interface is:

```json
{
  "objective": {
    "screen": {
      "target_metric": "perf_guidellm_sweep.total_tpm_card_norm",
      "direction": "maximize",
      "redlines": [
        {"metric": "perf_guidellm_sweep.reported_level_meets_slo", "op": ">=", "value": 1}
      ]
    },
    "verify": {
      "target_metric": "perf_guidellm_sweep.total_tpm_card_norm",
      "direction": "maximize",
      "redlines": [
        {"metric": "perf_guidellm_sweep.reported_level_meets_slo", "op": ">=", "value": 1},
        {"metric": "opencompass.gpqa_diamond", "op": ">=", "value": 0.6}
      ]
    }
  }
}
```

`GET /api/policy/v1/benchmarks/{id}` returns the following shape, passed intact as
`TrialResult.raw`:

```json
{
  "status": "succeeded",
  "summary": {
    "objective_value": 1234.5,
    "feasible": true,
    "constraints": [-120.0],
    "breaches": []
  },
  "metrics": {
    "perf_guidellm_sweep.reported_level_meets_slo": 1,
    "perf_guidellm_sweep.ttft_p99": 7.2,
    "perf_guidellm_sweep.c1.request_output_tps": 80.0
  }
}
```

Metric names are **literal flattened keys** in top-level `metrics`, including
any dots. They are not nested paths and are not under `summary`. `SearchLoop`
passes `summary.objective_value` and `summary.feasible` into the corresponding
`TrialResult` fields; neither strategy recomputes the objective from
`target_metric`. `summary.constraints` and `summary.breaches` remain diagnostics,
not replacements for metric values. Both `maximize/minimize` and legacy
`max/min` directions are accepted.

Search runs only the screen suite and checks `objective.screen.redlines`.
`objective.verify.redlines` are retained in strategy context but left to the
platform's final validation: absent verify-only metrics (such as GPQA) do not
reject screen contenders. Screen contenders are not claims of verify success.
Supported operators are `<`, `<=`, `>`, and `>=`; multiple bounds on one metric
are preserved. Missing or nonfinite screen metrics fail local feasibility closed,
and platform feasibility is also required. SLO misses from completed benchmarks
can guide constraint repair with `feasible=false` and no execution `failure_class`;
failed launches/benchmarks are not fitted as measurements.

To guide the strategy with a continuous TTFT signal, declare a screen redline such as
`{"metric": "perf_guidellm_sweep.ttft_p99", "op": "<", "value": 8.0}` using the
actual metric's units. A `reported_level_meets_slo >= 1` flag alone reveals only
pass/fail; the strategy cannot infer how many seconds a request exceeds the SLO.
For binary flags, `safety_margin=0` avoids predicting an unattainable inward
margin above 1. No undeclared latency threshold is inferred from the flag.

The strategy keeps the search center separate from the best feasible result. Before finding
a feasible point, it accepts reductions in the worst normalized SLO violation,
then total violation as a tie-breaker, even without an objective value. After
finding a feasible center, only feasible objective improvements above
`min_improvement` move it. All measured feasible results compete for contenders,
including smaller objective improvements.

Unknown directions start with adjacent probes. A local secant through the center
and the nearest measured point with identical other axes estimates both objective
and constraint changes. Numeric axes use actual value differences, including
uneven grids. Numeric tied groups use row displacement and are modeled as a whole;
categorical groups are explored without linear prediction. Conditional activation
changes and platform-rewritten configurations are excluded from model fitting.
After every measurement, eligible directions are re-ranked. Feasible predictions
or predicted repairs precede unknown directions,
which precede predicted violations. Within each category, predicted worst/total
violation, objective and shorter step break ties; workload priority breaks
remaining ties. Predictions target an inward margin of `5%` of each limit's
absolute value by default; actual contender acceptance uses the exact hard limit.

The initial modeled step radius is two rows, capped by `max_step`. Successful
moves with prediction errors at most `20%` expand the radius by one; unsuccessful
or inaccurate predictions halve it (minimum one). Unknown directions still use
one-row probes. This is a local linear heuristic, not a statistical confidence
model: it does not estimate measurement noise or guarantee safe experiments.
Predicted-risk directions remain fallback experiments, so SLO violations during
search are possible; **only measured feasible configurations become contenders**.
OOMs trigger local downward repairs. The strategy respects the trial budget,
declared values, fixed topology, tied rows, conditional pruning and deduplication.

Measured worsening can prune further exploration along a single numeric axis.
With all other active parameters unchanged, the strategy compares a new measurement with
the nearest previous measurement on either side. If a screen redline fails and
its metric worsens by at least `pruning_threshold` (default `0.1`) times
`max(abs(limit), abs(previous_measurement), 1e-9)`, candidates strictly beyond
the worse measured point in that direction are skipped before prediction ranking.
Intermediate points and the opposite direction remain eligible. An improving but
still infeasible measurement, small fluctuations, missing metrics, execution
failures, and predictions alone do not trigger pruning. Categorical axes and tied
groups are excluded. Changing another active parameter or conditional activation
reopens exploration in the new context; the original context retains its boundary.
This is a heuristic based on measured evidence, not proof of monotonicity, and can
miss a nonmonotonic recovery farther away. Set the constructor's
`pruning_threshold` to a larger positive value to make pruning more conservative.

Checkpoints also restore the measured samples, center and adaptive
step radii used for predictions. Direction boundaries are rebuilt from those
samples on restore. V1 state and changed constraints or pruning thresholds are
rejected. The strategy can stop at a local optimum and does not perform
coupled-axis escape moves.

Restart recovery requires the platform's `services.state` service and retention
of the campaign's `GET /state` / `PUT /state` blob between container runs. Results
are checkpointed after observations, including rejected configurations. Disabled
state storage and write failures produce warnings. These checkpoints are stored
on the platform, not in the container's local decision log. A forced stop after a
measurement but before its successful checkpoint can still cause that last trial
to be repeated; the SDK currently has no trial-ledger reconciliation endpoint.
Unfinished proposals remain eligible for retry. Search launch/benchmark request
keys are unique per process run to avoid replaying old responses after restart.
Historical evidence is reused only for matching contexts; start a fresh campaign
or clear its saved state when intentionally collecting fresh measurements.

## What the SDK guarantees for you

- **Dial-out only**: the platform never connects into your container; commands
  arrive on the heartbeat response.
- **The clock is the platform's**: `finalize` on the heartbeat means stop; the
  loop tears engines down, freezes contenders, and moves to validation.
- **Crash-safe**: whatever `contenders()` last returned is what gets validated
  if you die — the loop re-PUTs it after every measurement. An uncaught
  exception in your strategy becomes an explicit `error` signal, not silent
  death, so the platform validates what you found instead of timing you out.
- **You can't stall the night**: liveness (heartbeat) and productivity (an idle
  watchdog over the delegated work the platform sees) both wind you down — even
  if you forget to signal `exhausted`, or livelock while still beating.
- **Coverage for free**: the loop reports which part of the space you've tried,
  per axis, built from your measured configs — override `coverage()` only to add
  a completion estimate or deliberate skips.
- **Network-resilient**: the client retries *every* call on a transient blip or
  a platform restart (network error / 502-503-504) with bounded backoff; the
  creating POSTs carry idempotency keys, so a retry after a lost response never
  double-launches. A 4xx is a decision (auth, phase, safety, capacity) and is
  never retried.
- **The verdict is the platform's**, on its pinned verify dataset, of what you
  serve — nothing you report ranks a leaderboard.

## Test

```bash
uv run --extra dev pytest      # a full night against an in-memory platform, no network
uv run --extra dev ruff check autotune_policy/ strategies/ tests/
```

The bundled [`.gitlab-ci.yml`](.gitlab-ci.yml) runs exactly this on every push,
and builds + pushes the image to harbor on a version tag. Because the fake night
exercises the whole contract, a green check means a change to `strategies/`
didn't break the protocol — change the algorithm freely.

## Build & run

The image is stdlib-only, so there's nothing to install:

```bash
docker build -t <registry>/autotune-policy-random-search:0.1.0 .
docker push  <registry>/autotune-policy-random-search:0.1.0
```

Register it as a policy on the platform (name + image), then start a campaign
with `policy_id` set. The platform injects `AUTOTUNE_API_URL`,
`AUTOTUNE_API_KEY`, `AUTOTUNE_SESSION_ID` and the policy takes it from there.

> Restricted network: set `--build-arg BASE=<your-registry>/python:3.12-slim` if
> Docker Hub isn't reachable.

## Contract version

Built against policy contract **1.1** (`/api/policy/v1`) — adds the `exhausted`
/ `error` heartbeat signals, self-reported `coverage`, and the productivity
watchdog. Both sides ignore unknown JSON fields, so new manifest fields and
services are non-breaking, and a 1.0 image still runs unchanged.

## 离线 dry run：用模拟 metrics 驱动策略

无需平台、GPU 或 `AUTOTUNE_*` 环境变量：

```bash
uv run python main.py --dry-run
uv run python main.py --dry-run --max-trials 40 --seed 7 --noise 0.03 --output /tmp/search.json
uv run python main.py --dry-run --manifest examples/dry_run_manifest.json --sim-model my-model.json
```

默认使用当前入口的 `HeuristicSearch`，通过真实 `SearchLoop` 执行配置生成、
模拟 launch/benchmark、observe、候选更新、finalize 和 serve 流程。每轮配置打印到
终端；`dry-run-report.json` 保存完整扁平 metrics、summary、OOM、候选、覆盖率、
状态及模型假设。报告显式标记 `simulated`；仅模拟验证阶段的 serve 协议，不执行 verify
数据集，不声称候选通过真实验证。不加 `--dry-run` 仍连接真实平台。

默认场景以 TTFT=9 秒、吞吐=12000 tokens/min/card 为基准，TTFT 上限为 8 秒。
参数影响采用 `基准指标 × ∏(参数值 / 基准参数值)^指数`：

- 提高 `max_running_requests`：中低并发 TTFT 下降、吞吐上升，单请求输出 TPS 下降。
  超过 64 并发后追加二次排队惩罚，使 TTFT 回升，并压低吞吐增长。
- 提高 chunk 或 prefill 并发：改善预填充效率，同时增加合成内存压力。
- 提高静态显存占比：提高吞吐，但减少内存余量，可能触发模拟 OOM。
- mamba 比例、CUDA graph batch、调度保守度也有显式指数假设，见
  `autotune_policy/dry_run.py` 的 `DEFAULT_MODEL`。

这些方向和量级是可修改的实验假设，不是硬件性能预测；不同负载下真实方向可能相反。
默认不加噪声；`--noise 0.03` 表示每项连续指标独立乘以 ±3% 均匀噪声，固定 seed
可复现整个搜索。离散 SLO 标记由加噪后的 TTFT 与 `ttft_slo` 比较产生。

例如 `my-model.json` 可以让并发上升时 TTFT 直接上升：

```json
{
  "effects": {
    "max_running_requests": {
      "perf_guidellm_sweep.ttft_p99": 0.4,
      "perf_guidellm_sweep.total_tpm_card_norm": 0.3
    }
  },
  "saturation_concurrency": 96,
  "queue_penalty": 1.2,
  "oom_pressure": 2.5,
  "ttft_slo": 8.0
}
```

模型顶层字典按键覆盖默认值；`effects` 中指定参数的整组指数被替换，未指定参数
保留默认值。`baseline`、`metrics` 可增加自定义参数基准和指标，`effects` 再声明其关系。
未建模的配置参数保持中性，不自动猜测其影响。模型中的基准是固定校准点，不随
manifest 的 production 配置改变。默认参数模型只接受正数，不支持引擎的负数哨兵值。
自定义 manifest 的 screen 目标及每条 redline 都必须有模型指标，否则启动时明确报错；
约束严格区分 `<` 与 `<=`，只检查 screen，不把 verify-only 指标当作搜索约束。

## 终端日志

默认以 DEBUG 级别输出到 stderr，包含启动、心跳阶段、试验结果、资源释放和网络重试；
搜索循环异常会输出完整堆栈。Docker 会收集这些日志：

```bash
docker logs -f <容器名或ID>
```

镜像默认设置 `AUTOTUNE_LOG_LEVEL=DEBUG`，包含详细配置和轮询状态日志。
可通过容器环境变量 `AUTOTUNE_LOG_LEVEL=INFO` 或启动参数 `--log-level INFO`
减少输出，命令行参数优先。支持 DEBUG、INFO、WARNING、ERROR、CRITICAL。
本地示例：

```bash
uv run python main.py --dry-run --log-level DEBUG --output /tmp/search.json
```

镜像入口使用 `python -u`，避免 `print()` 输出缓冲。代码修改后需重新构建并启动镜像。
终端日志与下面的 JSONL 决策日志分别输出。

检查平台指标读取时，INFO 日志中的 `platform benchmark_id=...` 会输出平台返回的
完整 `metrics` 和 `summary`；策略的 `Decision ... observation` 输出实际参与决策的
约束指标、目标值、平台/本地可行性、缺失或无效指标、相对违规程度及此前中心点。
`proposal` 输出所参考的中心点指标和预测值，`outcome` 输出中心点、最优点和搜索半径。
目标值读取自 `summary.objective_value`，约束指标按 `metrics` 中完整扁平键名读取，
仅接受有限数值；字符串不会自动转换，缺失或无效的必需指标会输出 WARNING。
DEBUG 额外输出预测所参考的历史测量、外推比例及每个候选的排序依据。
可通过试验日志的 `trial_id` 与决策日志关联，再通过 `decision_id` 对应提案和观测；
JSONL 观测也保存 `raw_metrics`、`metric_context`、此前中心指标和预测值供离线核对。

## 本地决策日志

当前 `main.py` 使用的启发式策略在在线运行和 dry run 时都会追加写入
`logs/decisions.jsonl`，每行一个 JSON 对象。可自定义位置：

```bash
uv run python main.py --dry-run --decision-log /tmp/decisions.jsonl
```

每次选出配置立即写入 `proposal`：包含相对搜索中心的参数前后值、完整配置、
选择原因、阶段、可用的预测指标和各轴搜索半径。首轮基线通常没有参数变化；
无局部预测时 `prediction` 为 `null`。接收测量或启动失败结果后写入
`observation`：包含实测指标、可行性、错误、是否移动中心及是否更新最优结果。
搜索半径的键为内部轴编号；参数名和实际变化见 `changes`。

触发方向剪枝时额外写入 `direction_pruned`，包含轴编号、方向（数值增大为 `1`、
减小为 `-1`）、边界、前后配置和实测值、红线及恶化阈值；仅剪掉边界以外的点。
控制台同时输出 `Direction pruned`，DEBUG 日志记录实际跳过的候选。

用 `run_id` + `decision_id` 关联记录，时间戳为 UTC。文件跨运行追加，
每条记录写完即关闭文件；中途退出可能留下尚无结果的提案。容量重试不会生成
新决策记录。非有限数值保存为 `null`；写入失败会输出错误但不终止搜索。
默认 `logs/` 已被 Git 忽略。容器中需将日志路径指向挂载目录才能在容器删除后保留。
直接实例化 `HeuristicSearch` 时，可传入 `decision_log=DecisionLog(path)` 启用；

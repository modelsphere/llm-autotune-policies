| 参数 | 控制什么 | 调大后的主要影响 | 调小后的主要影响 |
|---|---|---|---|
| `max_running_requests` | 同时运行的请求数上限 | 可能提高总吞吐，也增加显存压力和单请求生成延迟 | 降低并发压力，但可能增加排队 |
| `chunked_prefill_size` | 每个 prefill 分块的 token 上限 | 长输入处理可能更高效，但单次计算和显存峰值更大 | 降低 prefill 显存峰值，但长输入处理可能变慢 |
| `prefill_max_requests` | 单个 prefill batch 的请求数上限 | 允许更多输入请求一起处理 | 限制一次进入 prefill 的请求数量 |
| `mem_fraction_static` | 静态分配显存的比例，主要用于权重和缓存池 | 扩大缓存容量，但挤压临时计算和 CUDA Graph 的显存余量 | 留出更多运行时显存，但减少缓存容量 |
| `cuda_graph_max_bs_decode` | decode 使用 CUDA Graph 的最大 batch size | 让更大 batch 有机会使用 Graph 加速，但增加显存开销 | 缩小 Graph 覆盖范围，节省显存 |
| `schedule_conservativeness` | 调度器接纳新请求的保守程度 | 更保守，有助于减少缓存不足导致的请求回退 | 更积极接纳请求，可能提高利用率，也可能增加回退 |
| `mamba_full_memory_ratio` | Mamba 状态内存与 full KV cache 内存之比 | 在固定预算下，分配更偏向 Mamba 状态 | 分配更偏向 full KV cache |



mem_fraction_static：粗略理解为 权重 + kv cache，剩下的用于运行时的cache，设置得太大可能会导致oom
所以它的影响是：
- 调大：缓存池能容纳更多 token，有机会承载更多并发或更长上下文；但运行时余量减少，可能在 prefill、decode 或 Graph 捕获时 OOM。
- 调小：运行时更宽裕，但缓存容量下降，可能更早排队或回退请求，降低吞吐。
- 缓存本来就够用时：调大通常不会直接让单个请求计算更快。它主要改变容量，不改变计算精度。


mamba_full_memory_ratio：合理范围应该大致是：
| `r` | Mamba 占两个池总预算 | KV cache 占两个池总预算 |
|---|---:|---:|
| 0.25 | 20% | 80% |
| 0.5 | 33.3% | 66.7% |
| 0.9 | 47.4% | 52.6% |
| 1 | 50% | 50% |
| 2 | 66.7% | 33.3% |
| 4 | 80% | 20% |


这个文件实现了一个 **根据工作负载选择调参顺序、由粗到细进行局部搜索** 的策略 `HeuristicSearch`。

它负责“下一次测哪个配置”和“如何根据结果调整搜索”，实际启动模型、运行 benchmark 由外部 `SearchLoop` 完成。它不穷举所有参数组合，也不保证找到全局最优。

下面按源码顺序解释所有函数，包括两个内部函数。

**先理解三个常量**

- `PROFILES`：不同工作负载对应的可调参数及优先顺序。例如 `agent` 优先考虑 `mamba_full_memory_ratio`，其次是 `max_running_requests`。
- `COMMON`：所有工作负载都可以调的公共参数。
- `DEFAULTS`：生成初始配置时的偏好值。最终仍必须选择搜索空间中声明的合法值，不会额外加入这些默认值。

**1. [`__init__()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:47)：保存策略设置**

接收四个参数：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `workload` | `"agent"` | 决定可调参数和搜索优先级 |
| `max_trials` | `24` | 最多提出多少个配置进行测试 |
| `min_improvement` | `0.01` | 移动搜索中心所需的相对改善阈值 |
| `baseline` | `None` | 显式指定初始配置的参数偏好 |

检查工作负载是否支持、预算是否为正、改善阈值是否有限且非负，并深拷贝 `baseline`。

这里还没有初始化搜索空间，后续需要调用 `setup()`。

**2. [`setup()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:58)：初始化一次搜索**

从 `manifest` 读取搜索空间、模型配置、目标方向（最大化或最小化）和候选数量，然后：

1. 生成 `context`，用于判断历史保存的配置是否适用于当前环境。
2. 调用 `_axes()`，把搜索空间转换为统一的“轴”结构。
3. 筛选可调轴，并按工作负载优先级排序。
4. 生成初始配置，偏好覆盖顺序为：

   ```text
   DEFAULTS < production.engine_args < baseline
   ```

5. 调用 `_project()`，把偏好映射到合法配置。
6. 清空历史记录、去重集合和计数器，把初始配置放进待测队列。

几个核心状态：

| 状态 | 含义 |
|---|---|
| `anchor` | 当前搜索中心，保留完整参数 |
| `best` | 达到改善阈值后被接受的结果 |
| `pending` | 等待提出的配置及原因 |
| `inflight` | 已提出、但尚未收到结果的完整配置 |
| `seen` | 已提出或已观测配置的去重标识 |
| `observations` | 本轮收到的全部测试结果 |
| `phase` | 当前阶段：`baseline`、`coarse`、`fine` |

只有包含多个候选行、且所有参数都在当前工作负载支持范围内的轴，才会参与搜索。**一个绑定组中只要有不支持的参数，整个组就固定不动。**

**3. [`_axes()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:92)：统一搜索空间的表示**

将三类搜索空间统一转换为 `list[list[dict]]`：

```python
# grid：一个参数独立变化
[[{"max_running_requests": 16},
  {"max_running_requests": 32}]]

# tied：多个参数按行绑定，一起变化
[[{"max_running_requests": 16, "chunked_prefill_size": 2048},
  {"max_running_requests": 32, "chunked_prefill_size": 4096}]]
```

具体处理：

- `grid`：每个参数变成一个轴，每个候选值变成一行。
- `tied`：将各列按下标配对，形成一个绑定轴，不进行笛卡尔积组合。
- `range`：根据 `min/max/step` 展开成离散候选值，使用 `Decimal` 避免浮点累计误差。
- 单参数数值轴按数值升序排列；多参数绑定轴保留声明顺序。

它还检查空候选、绑定列长度不一致、非法范围、范围过大以及不同轴重复声明同一参数等问题。

内部的 [`add(rows)`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:96) 负责检查参数是否与已有轴重叠，然后登记参数并追加新轴。

**4. [`_project()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:132)：把偏好配置映射到合法配置**

先复制 `search_space.base`，然后在每个轴上选择距离 `preferred` 最近的一行，合并成完整配置。

例如：

```text
期望 max_running_requests = 30
声明候选 = [16, 32, 64]
最终选择 = 32
```

内部的 [`distance(row)`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:135) 计算距离：

- 数值：`abs(value - target) / max(abs(target), 1e-9)`。
- 非数值：相同为 `0`，不同为 `1`。
- 绑定行：累加各参数的距离。
- 偏好中未指定的参数：不贡献距离。

距离相同时选择轴中的第一行。因此，完全没有偏好时，单参数数值轴会选择排序后的最小值。

**5. [`plan()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:147)：声明调参计划**

返回：

```python
{
    "tuning": [...],  # 实际准备调整的参数名
    "extras": [],     # 不引入搜索空间之外的参数
    "notes": "...",   # 策略、预算和局部搜索说明
}
```

用于向外部系统说明策略会调整什么。它本身不生成配置，也不执行测试。

**6. [`_neighbors()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:152)：生成邻居配置**

围绕输入配置，按优先级依次调整每个可调轴。每个候选只改变一个轴；如果是绑定轴，就一起修改该行中的多个参数。

步长按**候选下标**计算：

```python
coarse=True:  stride = max(1, (候选行数 - 1) // 2)
coarse=False: stride = 1
```

例如候选是 `[8, 16, 32, 64, 128]`，当前值为 `32`：

- 粗搜索：步长 `2`，提出 `8`、`128`。
- 细搜索：步长 `1`，提出 `16`、`64`。

越界的邻居直接跳过。返回 `(配置, 提案原因)` 的列表。

工作负载只决定**先调整哪个轴**，最终增大还是减小由测试结果决定。

**7. [`next_config()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:167)：取出下一份待测配置**

这是外部循环获取配置的入口。

处理步骤：

1. 如果上一份配置还没调用 `observe()`，抛出异常。
2. 从 `pending` 队头取出配置。
3. 用 `prune_inactive()` 删除条件未满足的参数。
4. 用 `canonical_key()` 去重；重复配置直接跳过。
5. 保存完整配置到 `inflight`，增加提案计数，返回裁剪后的配置。

队列耗尽时推进阶段：

```text
baseline → coarse → fine → 结束
```

如果预算耗尽，或者细搜索候选已经处理完，返回 `None`。

这里保留完整的 `inflight`，是为了之后还能基于完整轴结构生成邻居；实际交给平台的则是裁剪后的配置。

**8. [`observe()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:189)：接收测试结果并调整搜索**

这是搜索决策的核心。先清除 `inflight`，保存结果，再根据结果分支处理。

对于有效结果，要求 `result.ok` 且目标值有限。其中 `ok` 表示有目标值、满足可行性要求且没有失败类别。

随后计算改善量：

```text
最大化目标：gain = 新值 - 旧值
最小化目标：gain = 旧值 - 新值
```

如果是第一个有效结果，或者：

```python
gain > min_improvement * max(abs(old), 1e-9)
```

就接受这个结果作为新的 `best`，将对应完整配置设为 `anchor`，**丢弃原有待测队列，从新中心重新进行粗搜索**。

默认阈值是 `1%`。例如最大化目标时旧值为 `100`，新值必须严格大于 `101` 才会移动中心。

对于 OOM 失败：

- 围绕失败配置生成相邻配置。
- 只保留降低以下参数的候选：
  - `chunked_prefill_size`
  - `max_running_requests`
  - `mem_fraction_static`
- 所有被修改值必须仍然大于 `0`。
- 将修复候选插到待测队列前面。

这是局部修复，不会把某个参数值永久拉黑。其他失败只记录，不进行专门修复。

**9. [`_ranked()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:217)：对有效测试结果排名**

从全部 `observations` 中：

1. 排除失败、不可行和非有限目标值。
2. 按目标方向排序。
3. 按配置去重，同配置保留排序最靠前的结果。
4. 返回前 `max_contenders` 个。

注意：**排名不受 `min_improvement` 限制。** 一个改善不足以移动搜索中心的结果，仍然可以成为最终第一名。

**10. [`contenders()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:225)：生成提交给平台的候选配置**

调用 `_ranked()`，再用 `build_contender()` 将结果包装为平台需要的格式，包括：

- 排名和引擎参数；
- 模型镜像、环境变量和挂载；
- 测试记录中的运行环境；
- 目标值和工作负载等证据；
- 对应的 `trial_id`。

这个函数负责构造数据，不直接发网络请求。

**11. [`status()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:234)：报告搜索状态**

返回 `(状态文字, 进度比例)`。

文字包括工作负载、当前阶段、已收到结果数、最大预算和最近提案原因。

进度使用：

```python
min(1.0, proposed / max_trials)
```

它表示提案预算的使用比例。因此即使搜索已经提前结束，进度也可能不到 `100%`；这不是搜索空间覆盖率。

**12. [`dump_state()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:239)：保存下一轮的起点**

返回三个字段：

```python
{
    "strategy": "k3-heuristic-v1",
    "context": ...,
    "seed": ...,
}
```

`seed` 优先使用实际排名第一的配置；如果没有有效结果，则使用当前 `anchor`。

它不会保存历史分数、完整测试记录或搜索队列，因此不是完整的断点续跑快照。

**13. [`load_state()`](/Users/4paradigm/Documents/git/autotune-search-policy/strategies/heuristic_search.py:244)：加载历史配置作为热启动基线**

检查：

- 策略标识是否匹配；
- `context` 是否与当前搜索一致；
- `seed` 是否为字典。

满足条件时，将历史配置重新投影到合法空间，并作为第一份待测配置；不匹配则忽略。

它应在 `setup()` 后、开始提出配置前使用。**历史配置会重新测量，旧分数不会沿用。**

这些函数在外部通常按下面的顺序配合：

```python
strategy.setup(manifest)
strategy.load_state(previous_state)
plan = strategy.plan()

while (config := strategy.next_config()) is not None:
    result = benchmark(config)  # 示意：由外部执行测量
    strategy.observe(result)
    candidates = strategy.contenders()
    status = strategy.status()

state = strategy.dump_state()
```

搜索过程就是：**测试基线 → 尝试粗邻居 → 有足够改善就移动中心并重启粗搜索 → 没有改善则尝试细邻居 → 无新候选或预算耗尽时停止。**
# 基于三层染色体 GA 的 1 UGV + M UAV 协同配送算法实现指南（修正版）

## 0. 本版修改重点

本文件是在原 `ga_mmce_codex_implementation_guide.md` 基础上的修正版。核心修改是：

```text
旧设计：
    Chromosome 1：sequence
    Chromosome 2：assignment

新设计：
    Chromosome 1：sequence
    Chromosome 2：assignment
    Chromosome 3：rendezvous
```

其中：

```text
sequence   ：订单处理顺序
assignment ：每个订单由卡车 / 哪架无人机 / 哪种模式执行
rendezvous ：协同节点选择，即无人机从哪个仓库/充换电站起飞，送完后在哪个仓库/充换电站回收
```

本版最重要的原则是：

```text
GA 决定路径结构和协同节点。
GreedyMMCE 只复用底层物理计算能力，不复用完整贪心调度决策。
```

因此，GA Decoder **禁止直接调用**：

```python
GreedyMMCE.dispatch(...)
GreedyMMCE.dispatch_incremental(...)
GreedyMMCE.dispatch_replan_current_state(...)
GreedyMMCE._dispatch_impl(...)
GreedyMMCE._allocate_order(...)
GreedyMMCE._try_mode_b_with_waiting(...)
GreedyMMCE._try_mode_b(...)
GreedyMMCE._try_mode_c(...)
GreedyMMCE._try_mode_a(...)
GreedyMMCE._build_truck_route(...)
```

原因是这些函数会重新做：

```text
1. 订单 deadline 排序；
2. 模式 B_WAIT / B / C / A 贪心选择；
3. launch_station_id 贪心选择；
4. recovery_station_id 贪心选择；
5. 卡车最近邻路径构建；
6. 充换电站顺路插入。
```

如果 GA Decoder 调用这些函数，GA 的 `sequence`、`assignment` 和 `rendezvous` 会被贪心逻辑覆盖，GA 会退化成外层包装器。

---

# 1. 总体实现路线

当前任务不是重写环境、实体或前端，而是在已有项目基础上新增一个真正参与协同路径优化的 GA solver。

已有基础：

```text
1. greedy_mmce.py 已经实现完整贪心基线；
2. 环境与实体状态机已经实现；
3. 前端展示已经实现；
4. 订单、卡车、无人机、充换电站实体已经存在；
5. GA 只需要作为新的 solver 接入现有调度系统。
```

本版推荐架构：

```text
GA Individual
    sequence
    assignment
    rendezvous
        ↓
GA Decoder
    按 GA 指定顺序、模式、起飞点、回收点解码
        ↓
PhysicalEvaluator
    固定决策可行性评估
    复用 greedy_mmce 中的距离、能耗、OSM 路径、评分工具
        ↓
PlanBuilder
    生成 DispatchPlan，保持前端兼容
        ↓
Fitness
        ↓
GA selection / crossover / mutation
```

推荐新增目录：

```text
backend/solver/ga_mmce/
    __init__.py
    config.py
    chromosome.py
    operators.py
    population.py
    adapters.py
    physical_evaluator.py
    decoder.py
    fitness.py
    solver.py
    dynamic_rescheduler.py
```

与原版相比，新增核心文件：

```text
physical_evaluator.py
```

它负责固定起飞点、固定回收点、固定模式下的物理可行性评估。

---

# 2. 阶段划分

建议分四阶段推进。

```text
阶段 1：静态三层 GA 可跑通
    输入当前订单和环境状态
    GA 显式决定 sequence + assignment + rendezvous
    输出 DispatchPlan
    暂时不处理动态订单

阶段 2：接入 decision_engine.py
    通过配置选择 solver = ga_mmce
    前端无需改动
    GA 输出结构与 greedy_mmce.py 兼容

阶段 3：事件触发式动态重调度
    新订单到达时，冻结已完成和执行中任务
    对剩余订单 + 新订单重新 GA
    warm start 保留上一轮个体的未完成部分

阶段 4：局部搜索 / 修复增强
    对不可行 rendezvous 做局部修复
    可选加入 2-opt / station mutation / repair operator
```

---

# 3. Step 0：让 Codex 先阅读现有接口

## 3.1 目标

先不要修改代码，让 Codex 读取现有文件，确认：

```text
1. GreedyMMCE 的主入口；
2. DispatchPlan / AllocationResult / TruckRoute / DroneRoute 数据结构；
3. decision_engine.py 如何调用 solver；
4. 前端依赖哪些字段；
5. Order / Drone / Truck / SwapStation 的关键字段；
6. greedy_mmce.py 中哪些函数是贪心决策，哪些函数是底层物理工具。
```

## 3.2 必须区分的函数

### 3.2.1 GA 禁止调用的贪心决策函数

```python
GreedyMMCE.dispatch
GreedyMMCE.dispatch_incremental
GreedyMMCE.dispatch_replan_current_state
GreedyMMCE._dispatch_impl
GreedyMMCE._allocate_order
GreedyMMCE._try_mode_b_with_waiting
GreedyMMCE._try_mode_b
GreedyMMCE._try_mode_c
GreedyMMCE._try_mode_a
GreedyMMCE._build_truck_route
GreedyMMCE._check_energy_feasible
```

其中 `_check_energy_feasible` 也不建议直接用于 GA，因为它会在 recovery pool 中自动选择回收点；GA 需要的是“指定 recover_node 后只判断可不可行”。

### 3.2.2 GA 可以复用的底层物理/工具函数

```python
GreedyMMCE._load_road_graph
GreedyMMCE._road_dist
GreedyMMCE._dist
GreedyMMCE._flight_energy
GreedyMMCE._uav_energy_wh
GreedyMMCE._truck_energy_wh
GreedyMMCE._recalculate_plan_route_costs
GreedyMMCE.build_incremental_route_from_stops
```

特别注意：

```text
build_incremental_route_from_stops(truck, ordered_stops, current_time)
```

这个函数是按给定停靠顺序重建 OSM 路线，不会做最近邻重排，适合 GA Decoder 使用。

## 3.3 给 Codex 的提示词

```text
请先不要修改任何代码。

请阅读以下文件：
- backend/solver/greedy_mmce.py
- backend/solver/decision_engine.py
- backend/config/loader.py
- backend/core/entities/drone.py
- backend/core/entities/truck.py
- backend/core/entities/swap_station.py
- backend/core/entities/primitives.py

请输出：
1. greedy_mmce.py 的主调度入口函数名称、参数、返回值结构。
2. DispatchPlan、AllocationResult、TruckRoute、TruckRouteNode、DroneRoute 的字段。
3. decision_engine.py 是如何选择和调用 solver 的。
4. 前端依赖的调度结果字段有哪些。
5. Order / Drone / Truck / SwapStation 的关键字段名。
6. greedy_mmce.py 中哪些函数属于贪心决策层，GA Decoder 禁止调用。
7. greedy_mmce.py 中哪些函数属于物理工具层，GA 可以复用。
8. build_incremental_route_from_stops 是否可以按给定 ordered_stops 生成 OSM 路线。

只做代码阅读和总结，不要修改文件。
```

## 3.4 验收标准

Codex 应该输出一份接口总结，并明确：

```text
GA Decoder 不允许调用完整 greedy dispatch。
GA Decoder 只能调用物理工具层，或新建固定决策 physical_evaluator。
```

---

# 4. Step 1：新增 GA 配置文件

## 4.1 目标

让 GA 参数可配置，尤其新增第三层染色体的变异概率。

## 4.2 新增文件

```text
backend/solver/ga_mmce/config.py
```

## 4.3 推荐代码

```python
from dataclasses import dataclass


@dataclass
class GAConfig:
    population_size: int = 80
    generations: int = 120
    elite_ratio: float = 0.08
    tournament_k: int = 3

    crossover_rate: float = 0.9
    mutation_rate_sequence: float = 0.2
    mutation_rate_assignment: float = 0.15
    mutation_rate_rendezvous: float = 0.20

    use_greedy_seed: bool = True
    use_truck_only_seed: bool = True
    use_obl_seed: bool = True

    enable_repair: bool = True
    repair_penalty_factor: float = 1000.0

    big_m: float = 1e9

    weight_completion: float = 1.0
    weight_delay: float = 10.0
    weight_energy: float = 0.1
    weight_waiting: float = 5.0
    weight_infeasible: float = 100000.0
    weight_truck_distance: float = 1.0
    weight_uav_distance: float = 1.0

    max_runtime_seconds: float | None = None
    random_seed: int | None = 42
    verbose: bool = False

    # 是否允许 C 模式无人机送完后落在充换电站；若 False，则 C 必须回仓。
    allow_depot_drone_recover_at_station: bool = True

    # 若卡车与无人机回收同步失败，是否作为软惩罚而不是直接判死。
    soft_rendezvous_violation: bool = True

    truck_wait_max_s: float = 10.0
```

## 4.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/config.py。

实现 GAConfig dataclass。

除了基础 GA 参数外，请新增：
- mutation_rate_rendezvous
- enable_repair
- repair_penalty_factor
- weight_truck_distance
- weight_uav_distance
- allow_depot_drone_recover_at_station
- soft_rendezvous_violation
- truck_wait_max_s

不要修改其他文件。
```

---

# 5. Step 2：设计三层染色体结构

## 5.1 目标

实现三层编码：

```text
Chromosome 1：sequence
Chromosome 2：assignment
Chromosome 3：rendezvous
```

含义：

```text
sequence[k]
    第 k 个被解码的订单 ID

assignment[k]
    该订单的模式与执行载具：A / B_<drone_id> / C_<drone_id>

rendezvous[k]
    对应订单的协同节点：
        A：None
        B：{"launch": <depot_or_station_id>, "recover": <depot_or_station_id>}
        C：{"launch": <depot_id>, "recover": <depot_or_station_id>}
```

## 5.2 新增文件

```text
backend/solver/ga_mmce/chromosome.py
```

## 5.3 推荐代码

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Rendezvous = dict[str, str] | None


@dataclass
class Individual:
    sequence: list[str]
    assignment: list[str]
    rendezvous: list[Rendezvous]
    fitness: float = float("inf")
    decoded_plan: Any | None = None
    penalties: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        n = len(self.sequence)
        if len(self.assignment) != n:
            raise ValueError("sequence and assignment length mismatch")
        if len(self.rendezvous) != n:
            raise ValueError("sequence and rendezvous length mismatch")
        if len(set(self.sequence)) != n:
            raise ValueError("sequence contains duplicated order ids")

        for gene, rv in zip(self.assignment, self.rendezvous):
            if gene == "A":
                if rv is not None:
                    raise ValueError("A mode must use rendezvous=None")
            elif gene.startswith("B_"):
                if not isinstance(rv, dict):
                    raise ValueError("B mode must use rendezvous dict")
                if not rv.get("launch") or not rv.get("recover"):
                    raise ValueError("B mode rendezvous must include launch and recover")
            elif gene.startswith("C_"):
                if not isinstance(rv, dict):
                    raise ValueError("C mode must use rendezvous dict")
                if rv.get("launch") != "DEPOT" and not rv.get("launch", "").startswith("depot"):
                    raise ValueError("C mode launch must be DEPOT/depot id")
                if not rv.get("recover"):
                    raise ValueError("C mode rendezvous must include recover")
            else:
                raise ValueError(f"unknown assignment gene: {gene}")


def make_gene_pool(drone_ids: list[str | int]) -> list[str]:
    raise RuntimeError(
        "make_gene_pool(drone_ids) is deprecated because it creates both B_ and C_ "
        "genes for every drone. Use make_gene_pool_by_location(truck_drone_ids, "
        "depot_drone_ids) with current physical state instead."
    )


def make_gene_pool_by_location(
    truck_drone_ids: list[str | int],
    depot_drone_ids: list[str | int],
) -> list[str]:
    truck_ids = [str(uid).strip() for uid in truck_drone_ids]
    depot_ids = [str(uid).strip() for uid in depot_drone_ids]
    overlap = set(truck_ids) & set(depot_ids)
    if overlap:
        raise ValueError(f"drone cannot be both truck-docked and depot-ready: {sorted(overlap)}")

    return ["A"] + [f"B_{uid}" for uid in truck_ids] + [f"C_{uid}" for uid in depot_ids]


def make_node_pool(depot_ids: list[str], station_ids: list[str]) -> list[str]:
    return list(depot_ids) + list(station_ids)


def normalize_depot_id(depot_ids: list[str]) -> str:
    if depot_ids:
        return depot_ids[0]
    return "DEPOT"
```

## 5.4 给 Codex 的提示词

```text
请修改/新增 backend/solver/ga_mmce/chromosome.py。

实现三层染色体 Individual：
- sequence: list[str]
- assignment: list[str]
- rendezvous: list[dict[str, str] | None]
- fitness
- decoded_plan
- penalties

validate() 必须检查：
1. 三条染色体长度一致；
2. sequence 不重复；
3. A 的 rendezvous 必须为 None；
4. B_<uid> 的 rendezvous 必须包含 launch 和 recover；
5. C_<uid> 的 launch 必须是 DEPOT ，recover 必须存在；
6. B/C 的无人机 ID 必须匹配 `UAV-TEST-01` 到 `UAV-TEST-12`；
7. 不允许未知 gene。

同时实现：
- make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)
- make_gene_pool(drone_ids) 仅保留为废弃保护接口，调用时应抛错，防止每架无人机同时生成 B/C。
- make_node_pool(depot_ids, station_ids)
- normalize_depot_id(depot_ids)
```

---

# 6. Step 3：实现三层染色体交叉、变异、选择算子

## 6.1 目标
请新增/修改 backend/solver/ga_mmce/operators.py，并根据当前 chromosome.py 的实际实现适配三层染色体算子。

当前 chromosome.py 已经实现：
- Individual(sequence, assignment, rendezvous)
- Individual.validate()
- Individual.validate_with_context(truck_drone_ids, depot_drone_ids, valid_drone_ids=None, support_node_ids=None)
- make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)
- make_node_pool(depot_ids, station_ids)
- normalize_depot_id(depot_ids)

重要设计约束：
1. 当前采用版本 A 动态重调度策略。
2. 每次重调度时，上游会根据当前物理状态重新生成 gene_pool。
3. gene_pool 已经满足：
   - A
   - B_<truck_drone_id>
   - C_<depot_drone_id>
4. operators.py 不负责判断无人机是否在卡车或仓库。
5. operators.py 不负责从实体状态中提取无人机池。
6. operators.py 只从外部传入的 gene_pool 中随机采样 assignment。
7. operators.py 只从外部传入的 support_node_ids 中随机采样 launch/recover。
8. support_node_ids 由上游提前生成，通常等于 depot ids + station ids。

请不要在 operators.py 中使用：
- make_gene_pool(...)
- make_gene_pool_by_location(...)
- normalize_depot_id(...)
- make_node_pool(...)
- depot_ids 参数
- station_ids 参数

请实现以下函数：

1. make_random_rendezvous_for_gene(gene, support_node_ids, allow_c_recover_station=True)

函数规则：
- gene == "A":
    return None

- gene.startswith("B_"):
    返回：
    {
        "launch": random.choice(support_node_ids),
        "recover": random.choice(support_node_ids),
    }

- gene.startswith("C_"):
    launch 必须固定为 "DEPOT"。
    如果 support_node_ids 中没有 "DEPOT"，但存在 depot id，例如 "depot-1" 或 "DEPOT-1"，则使用 support_node_ids 中第一个以 depot/DEPOT 开头的节点作为 launch。
    如果 allow_c_recover_station=True:
        recover 从 support_node_ids 中随机选择。
    如果 allow_c_recover_station=False:
        recover 只能从 depot 节点中随机选择。
    返回：
    {
        "launch": depot_node,
        "recover": selected_recover_node,
    }

- 对未知 gene 抛出 ValueError。

请额外实现一个辅助函数：
    find_depot_node(support_node_ids)

规则：
- 优先返回 "DEPOT"。
- 否则返回第一个 str(node).upper().startswith("DEPOT") 或 str(node).lower().startswith("depot") 的节点。
- 如果找不到 depot 节点，抛出 ValueError。

2. order_crossover(p1, p2)

要求：
- sequence 使用 OX 顺序交叉。
- 交叉前调用 p1.validate() 和 p2.validate()。
- 检查两个父代 sequence 长度一致。
- 检查 set(p1.sequence) == set(p2.sequence)，否则抛出 ValueError。
- assignment 和 rendezvous 必须按订单 ID 对齐继承。
- 不能简单按下标复制 assignment/rendezvous。
- 对每个 child_seq 中的订单，从 base_map 或 donor_map 中以 0.5 概率继承该订单对应的 gene 和 rendezvous。
- 返回两个 Individual。
- 子代生成后调用 child.validate()。

3. mutate(ind, gene_pool, support_node_ids, p_seq, p_assign, p_rendezvous, allow_c_recover_station=True)

要求：
- mutate 开始前调用 ind.validate()。
- gene_pool 必须非空，且必须包含 "A"。
- support_node_ids 必须非空。
- sequence swap mutation：
    - 当 random.random() < p_seq 且订单数 >= 2 时，随机交换两个位置。
    - sequence、assignment、rendezvous 三条染色体对应位置必须一起交换。
- assignment mutation：
    - 对每个位置，如果 random.random() < p_assign：
        - new_gene = random.choice(gene_pool)
        - ind.assignment[i] = new_gene
        - ind.rendezvous[i] = make_random_rendezvous_for_gene(new_gene, support_node_ids, allow_c_recover_station)
    - assignment mutation 后必须同步重建该订单的 rendezvous。
- rendezvous mutation：
    - 对每个位置，如果 random.random() < p_rendezvous：
        - 不改变 sequence[i]
        - 不改变 assignment[i]
        - 只重新生成 ind.rendezvous[i]
        - 使用 make_random_rendezvous_for_gene(ind.assignment[i], support_node_ids, allow_c_recover_station)
- mutate 结束后调用 ind.validate()。

4. tournament_select(population, k)

要求：
- population 不能为空，否则抛出 ValueError。
- 从 population 中随机采样 min(k, len(population)) 个个体。
- 按 fitness 升序选择最优个体。
- 返回该个体的 deepcopy。

请保证 operators.py 不访问实体状态，不读取 truck/drone/station 对象。
B/C 无人机归属约束由上游 gene_pool 和后续 validate_with_context / PhysicalEvaluator 保证。

---

# 7. Step 4：实现种群初始化

## 7.1 目标

初始种群必须生成合法的三层染色体，并与当前已实现的 `chromosome.py` / `operators.py` 接口保持一致。

当前版本采用 **版本 A 动态重调度策略**：

```text
上游 adapter / solver 在每次重调度时根据当前物理状态生成：
    order_ids
    gene_pool
    support_node_ids

population.py 只消费这些简单列表，不访问 truck / drone / depot / station 实体对象。
```

其中：

```text
gene_pool:
    已由上游根据当前位置生成，通常来自 make_gene_pool_by_location(...)
    形如：
        A
        B_<truck_docked_drone_id>
        C_<depot_ready_drone_id>

support_node_ids:
    已由上游生成，通常为 depot ids + station ids
    用于随机生成 rendezvous 的 launch / recover
```

包括：

```text
1. 随机解；
2. 纯卡车解；
3. greedy seed 转换而来的三层个体；
4. OBL 反向学习个体。
```

重要边界：

```text
population.py 不负责判断无人机是否在卡车或仓库。
population.py 不负责从实体状态中提取无人机池。
population.py 不调用 make_gene_pool(...)。
population.py 不调用 make_gene_pool_by_location(...)。
population.py 不接收 depot_ids / station_ids 参数。
population.py 只从外部传入的 gene_pool 中随机采样 assignment。
population.py 只从外部传入的 support_node_ids 中随机采样 rendezvous。
```

## 7.2 新增文件

```text
backend/solver/ga_mmce/population.py
```

## 7.3 推荐代码

```python
from __future__ import annotations

import copy
import random

from .chromosome import Individual
from .operators import make_random_rendezvous_for_gene


def _check_inputs(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
) -> None:
    if len(set(order_ids)) != len(order_ids):
        raise ValueError("order_ids contains duplicated order ids")
    if not gene_pool:
        raise ValueError("gene_pool must not be empty")
    if "A" not in gene_pool:
        raise ValueError('gene_pool must include "A"')
    if not support_node_ids:
        raise ValueError("support_node_ids must not be empty")


def make_random_individual(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual:
    _check_inputs(order_ids, gene_pool, support_node_ids)

    seq = list(order_ids)
    random.shuffle(seq)
    assignment: list[str] = []
    rendezvous = []

    for _ in seq:
        gene = random.choice(gene_pool)
        assignment.append(gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                gene,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def make_truck_only_individual(order_ids: list[str]) -> Individual:
    ind = Individual(
        sequence=list(order_ids),
        assignment=["A"] * len(order_ids),
        rendezvous=[None] * len(order_ids),
    )
    ind.validate()
    return ind


def make_obl_individual(
    base: Individual,
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual:
    base.validate()
    _check_inputs(base.sequence, gene_pool, support_node_ids)

    seq = list(reversed(base.sequence))
    assignment: list[str] = []
    rendezvous = []

    for gene in reversed(base.assignment):
        candidates = [candidate for candidate in gene_pool if candidate != gene]
        new_gene = random.choice(candidates or gene_pool)

        assignment.append(new_gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                new_gene,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    ind = Individual(seq, assignment, rendezvous)
    ind.validate()
    return ind


def _copy_seed_if_valid(
    seed: Individual,
    order_set: set[str],
    gene_set: set[str],
) -> Individual | None:
    copied = copy.deepcopy(seed)
    try:
        copied.validate()
    except ValueError:
        return None

    if set(copied.sequence) != order_set:
        return None
    if any(gene not in gene_set for gene in copied.assignment):
        return None
    return copied


def initialize_population(
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    pop_size: int,
    greedy_seed: Individual | None = None,
    warm_start: list[Individual] | None = None,
    use_truck_only_seed: bool = True,
    use_obl_seed: bool = True,
    allow_c_recover_station: bool = True,
) -> list[Individual]:
    _check_inputs(order_ids, gene_pool, support_node_ids)
    if pop_size <= 0:
        return []

    order_set = set(order_ids)
    gene_set = set(gene_pool)
    population: list[Individual] = []

    if warm_start:
        for seed in warm_start:
            copied = _copy_seed_if_valid(seed, order_set, gene_set)
            if copied is not None:
                population.append(copied)

    if greedy_seed is not None:
        copied = _copy_seed_if_valid(greedy_seed, order_set, gene_set)
        if copied is not None:
            population.append(copied)

    if use_truck_only_seed:
        population.append(make_truck_only_individual(order_ids))

    if use_obl_seed and population:
        population.append(
            make_obl_individual(
                population[0],
                gene_pool,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    while len(population) < pop_size:
        population.append(
            make_random_individual(
                order_ids,
                gene_pool,
                support_node_ids,
                allow_c_recover_station,
            )
        )

    return population[:pop_size]
```

## 7.4 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/population.py。

要求：
1. 所有初始化个体都必须包含 sequence、assignment、rendezvous 三层。
2. A 模式 rendezvous=None。
3. B 模式随机生成 launch/recover。
4. C 模式 rendezvous 通过 operators.make_random_rendezvous_for_gene(...) 生成，launch 固定为 DEPOT 或 support_node_ids 中第一个 depot 前缀节点，recover 根据配置可为 depot 或 station。
5. initialize_population 支持 greedy_seed 和 warm_start。
6. initialize_population 参数使用 order_ids、gene_pool、support_node_ids，不再使用 drone_ids、depot_ids、station_ids。
7. 不要访问真实环境状态，不读取 truck/drone/station/depot 对象。
8. 不要调用 make_gene_pool(...)、make_gene_pool_by_location(...)、make_node_pool(...)、normalize_depot_id(...)。
9. greedy_seed / warm_start 必须 deepcopy 后再加入种群，并且要 validate；若订单集合与当前 order_ids 不一致，或 assignment 不属于当前 gene_pool，则跳过。
10. 所有新建个体返回前都必须调用 validate()。
```

---

# 8. Step 5：写 GA 与现有环境之间的适配层

## 8.1 目标

GA 模块不应该直接猜实体字段名。需要 adapter 把现有系统状态转换成 GA 当前这一轮需要的简单上下文池。

当前 `chromosome.py` 已经实现：

```text
Individual(sequence, assignment, rendezvous)
Individual.validate()
Individual.validate_with_context(
    truck_drone_ids,
    depot_drone_ids,
    valid_drone_ids=None,
    support_node_ids=None,
)
make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)
make_node_pool(depot_ids, station_ids)
```

因此 adapter 的职责不再是只返回 `drone_ids`。如果 adapter 只返回全量无人机 ID，再由 solver 调用旧 `make_gene_pool(drone_ids)`，就会重新产生错误基因：

```text
每架无人机同时拥有 B_<uid> 和 C_<uid>
```

这与当前物理约束冲突：

```text
B_<uid>：uid 必须是当前已经停在卡车平台上的无人机，例如 truck.docked_drones。
C_<uid>：uid 必须是当前位于仓库且可从仓库起飞的无人机。
```

adapter 应该提供：

```text
1. 当前待优化订单集合 order_ids
2. 当前可用于 B 模式的车载无人机集合 truck_drone_ids
3. 当前可用于 C 模式的仓库无人机集合 depot_drone_ids
4. 系统内全部无人机 ID all_drone_ids
5. 仓库节点 ID depot_ids
6. 充换电站节点 ID station_ids
7. GA 可用 support_node_ids
8. 当前重调度轮次可用 gene_pool
9. 状态深拷贝
10. greedy plan -> 三层 Individual 的转换
```

当前需要特别注意：

```text
make_gene_pool(drone_ids) 已废弃，不允许 solver.py 继续调用。
solver.py 必须使用 make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)。
validate() 当前会检查 UAV-TEST-01 ~ UAV-TEST-12 格式。
validate_with_context(...) 可额外检查 B/C 无人机池和 support_node_ids。
make_location_gene(...) 只适合静态初始配置，不适合动态重调度时判断无人机真实位置。
```

## 8.2 新增文件

```text
backend/solver/ga_mmce/adapters.py
```

## 8.3 需要实现的函数

```python
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from .chromosome import (
    Individual,
    make_gene_pool_by_location,
    make_node_pool,
)
from .operators import (
    find_depot_node,
    make_random_rendezvous_for_gene,
)


@dataclass
class GAAdapterContext:
    order_ids: list[str]
    truck_drone_ids: list[str]
    depot_drone_ids: list[str]
    all_drone_ids: list[str]
    depot_ids: list[str]
    station_ids: list[str]
    support_node_ids: list[str]
    gene_pool: list[str]


def extract_active_order_ids(state) -> list[str]:
    """返回需要进入 GA 的订单 ID，排除 completed / cancelled / locked。"""


def extract_truck_drone_ids(state) -> list[str]:
    """
    返回当前可用于 B 模式的无人机 ID。

    典型来源：
    - truck.docked_drones
    - 且无人机当前未被 locked action / running route 占用
    """


def extract_depot_drone_ids(state) -> list[str]:
    """
    返回当前可用于 C 模式的无人机 ID。

    判断不能只看 home_type。
    应参考当前物理状态：无人机位于仓库、空闲、可从仓库起飞、未被 locked action 占用。
    """


def extract_all_drone_ids(state) -> list[str]:
    """返回系统内全部无人机 ID，仅用于上下文校验和诊断。"""


def extract_truck_ids(state) -> list[str]:
    """返回可用卡车 ID。本问题通常只有一辆。"""


def extract_depot_ids(state) -> list[str]:
    """返回仓库 ID。"""


def extract_station_ids(state) -> list[str]:
    """返回充换电站 ID。"""


def extract_support_node_ids(state) -> list[str]:
    """返回 GA 可选协同节点，通常等于 depot ids + station ids。"""


def build_ga_context(state) -> GAAdapterContext:
    order_ids = extract_active_order_ids(state)
    truck_drone_ids = extract_truck_drone_ids(state)
    depot_drone_ids = extract_depot_drone_ids(state)
    all_drone_ids = extract_all_drone_ids(state)
    depot_ids = extract_depot_ids(state)
    station_ids = extract_station_ids(state)
    support_node_ids = make_node_pool(depot_ids, station_ids)
    gene_pool = make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)

    return GAAdapterContext(
        order_ids=order_ids,
        truck_drone_ids=truck_drone_ids,
        depot_drone_ids=depot_drone_ids,
        all_drone_ids=all_drone_ids,
        depot_ids=depot_ids,
        station_ids=station_ids,
        support_node_ids=support_node_ids,
        gene_pool=gene_pool,
    )


def clone_state_for_decode(state):
    """返回 state 深拷贝，确保 decode 不污染真实环境。"""


def greedy_plan_to_individual(
    greedy_plan,
    order_ids: list[str],
    gene_pool: list[str],
    support_node_ids: list[str],
    allow_c_recover_station: bool = True,
) -> Individual | None:
    """
    将 greedy plan 转换成三层 Individual。
    可从 AllocationResult 中读取：mode、drone_id、launch_station_id、recovery_station_id。
    若字段缺失，则用 make_random_rendezvous_for_gene(...) 修复。

    重要：
    - 不允许生成 gene_pool 之外的 assignment。
    - 如果 greedy 的无人机不在当前 gene_pool 中，该订单回退为 A。
    """
```

## 8.4 上下文池提取规则

```text
extract_truck_drone_ids(state):
    从 truck.docked_drones 提取当前已经停在卡车平台上的无人机。
    若存在多辆卡车，则合并所有当前可参与 GA 的 truck.docked_drones。
    排除正在执行 locked action、正在飞行、正在服务未完成订单的无人机。

extract_depot_drone_ids(state):
    从当前真实状态判断位于仓库且可起飞的无人机。
    不能只看 home_type == DEPOT。
    需要排除：
        正在飞行
        在充换电站
        刚被卡车回收但尚未可用
        正在换电
        locked action 占用

extract_all_drone_ids(state):
    返回系统中的全部无人机 ID。
    用于 Individual.validate_with_context(..., valid_drone_ids=all_drone_ids)。

extract_support_node_ids(state):
    support_node_ids = depot_ids + station_ids。
    后续 operators.py 只从 support_node_ids 中随机生成 launch/recover。

build_ga_context(state):
    gene_pool = make_gene_pool_by_location(truck_drone_ids, depot_drone_ids)。
    solver.py 和 population.py 都不应该再调用 make_gene_pool(drone_ids)。
```

## 8.5 greedy_plan_to_individual 规则

```text
alloc.mode == "A":
    assignment = "A"
    rendezvous = None

alloc.mode in ("B", "B_WAIT", "B_DYNAMIC"):
    assignment = f"B_{alloc.drone_id}"
    如果 assignment 不在当前 gene_pool 中：
        assignment = "A"
        rendezvous = None
    否则：
        rendezvous = {
            "launch": alloc.launch_station_id or alloc.launch_node_id or fallback_support_node,
            "recover": alloc.recovery_station_id or alloc.recover_node_id or fallback_support_node,
        }
        若 launch/recover 不在 support_node_ids 中，则用 make_random_rendezvous_for_gene(...) 修复。

alloc.mode == "C":
    assignment = f"C_{alloc.drone_id}"
    如果 assignment 不在当前 gene_pool 中：
        assignment = "A"
        rendezvous = None
    否则：
        depot_node = find_depot_node(support_node_ids)
        rendezvous = {
            "launch": depot_node,
            "recover": alloc.recovery_station_id or alloc.recover_node_id or depot_node,
        }
        若 recover 不在 support_node_ids 中，则用 make_random_rendezvous_for_gene(...) 修复。
```

推荐辅助逻辑：

```python
def _read_field(record: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _iter_allocations(greedy_plan) -> Iterable[Any]:
    if greedy_plan is None:
        return []
    if isinstance(greedy_plan, dict):
        return greedy_plan.values()
    if isinstance(greedy_plan, list):
        return greedy_plan
    for field_name in ("allocations", "results", "assignments"):
        value = getattr(greedy_plan, field_name, None)
        if value is not None:
            return value.values() if isinstance(value, dict) else value
    return []


def _repair_rendezvous_if_needed(
    gene: str,
    rv: dict[str, str] | None,
    support_node_ids: list[str],
    allow_c_recover_station: bool,
) -> dict[str, str] | None:
    if gene == "A":
        return None
    if not isinstance(rv, dict):
        return make_random_rendezvous_for_gene(
            gene,
            support_node_ids,
            allow_c_recover_station,
        )

    support_set = {str(node) for node in support_node_ids}
    if rv.get("launch") not in support_set or rv.get("recover") not in support_set:
        return make_random_rendezvous_for_gene(
            gene,
            support_node_ids,
            allow_c_recover_station,
        )
    return rv
```

## 8.6 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/adapters.py。

实现：
1. GAAdapterContext dataclass。
2. extract_active_order_ids(state)
3. extract_truck_drone_ids(state)
4. extract_depot_drone_ids(state)
5. extract_all_drone_ids(state)
6. extract_truck_ids(state)
7. extract_depot_ids(state)
8. extract_station_ids(state)
9. extract_support_node_ids(state)
10. build_ga_context(state)
11. clone_state_for_decode(state)
12. greedy_plan_to_individual(greedy_plan, order_ids, gene_pool, support_node_ids, allow_c_recover_station=True)

要求：
- 不要修改实体类。
- clone_state_for_decode 优先使用 copy.deepcopy。
- B 模式无人机池必须来自当前卡车平台上的无人机，例如 truck.docked_drones。
- C 模式无人机池必须来自当前位于仓库且可从仓库起飞的无人机。
- 不要只根据 home_type 推断动态重调度时的 B/C 池。
- 不要返回单一 drone_ids 给 solver 生成 gene_pool。
- solver.py 必须使用 context.gene_pool，不允许调用 make_gene_pool(drone_ids)。
- support_node_ids 必须来自 depot ids + station ids。
- greedy_plan_to_individual 必须返回三层 Individual。
- greedy_plan_to_individual 不允许生成 gene_pool 之外的 assignment。
- greedy_plan_to_individual 生成后必须调用 individual.validate()。
- 若上下文池可用，也应调用 individual.validate_with_context(
      truck_drone_ids,
      depot_drone_ids,
      valid_drone_ids=all_drone_ids,
      support_node_ids=support_node_ids,
  )。
```

---

# 9. Step 6：实现 PhysicalEvaluator，固定协同节点评估

## 9.1 目标

新增 GA 专用物理评估器 `backend/solver/ga_mmce/physical_evaluator.py`。

它只评估 GA 染色体已经指定的固定动作，不重新做任何贪心决策。

```text
输入 GA 已经指定的：
    order_id
    mode
    drone_id
    truck_id
    launch_node_id
    recover_node_id

输出：
    这个固定动作是否可行
    距离、能耗、时间、迟到、等待、惩罚
    需要追加的 truck stop
    需要追加的 drone route
```

特别边界：

```text
允许复用 GreedyMMCE 的底层物理/工具函数：
    _road_dist
    _dist
    _flight_energy
    _uav_energy_wh
    _truck_energy_wh
    build_incremental_route_from_stops

禁止调用任何会重新做贪心决策的函数：
    dispatch
    dispatch_incremental
    dispatch_replan_current_state
    _dispatch_impl
    _allocate_order
    _try_mode_b_with_waiting
    _try_mode_b
    _try_mode_c
    _try_mode_a
    _build_truck_route
    _check_energy_feasible
```

`_check_energy_feasible` 禁止调用，因为它会在 recovery pool 中自动选择回收点。GA 需要的是“给定 recover_node_id 后只判断可不可行”。

## 9.2 新增文件

```text
backend/solver/ga_mmce/physical_evaluator.py
```

## 9.3 Candidate 数据结构

```python
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GACandidate:
    order_id: str
    mode: str
    feasible: bool
    reason: str = ""

    truck_id: str = ""
    drone_id: str = ""
    launch_node_id: str = ""
    recover_node_id: str = ""

    completion_time: float = 0.0
    truck_distance: float = 0.0
    uav_distance: float = 0.0
    truck_energy: float = 0.0
    uav_energy: float = 0.0
    waiting_time: float = 0.0
    lateness: float = 0.0

    cost_dist: float = 0.0
    cost_energy: float = 0.0
    cost_penalty: float = 0.0
    score_total: float = math.inf

    truck_stops: list[dict[str, Any]] = field(default_factory=list)
    drone_route_fragment: Any | None = None
    allocation_fragment: Any | None = None

    truck_final_node_id: str = ""
    truck_final_time: float = 0.0
    drone_final_node_id: str = ""
    drone_final_time: float = 0.0
    drone_energy_used: float = 0.0
    drone_energy_after: float = 0.0
    delivered_order_ids: list[str] = field(default_factory=list)
```

新增的 `*_final_*` 字段用于 Decoder 在 state copy 上逐单推进虚拟位置/时间，避免每个 candidate 都从真实初始状态重新评估。

## 9.4 PhysicalEvaluator 已实现接口

```python
class PhysicalEvaluator:
    def __init__(self, entity_mgr, greedy_helper, config): ...

    def get_node_position(self, node_id: str, state: Any | None = None): ...
    def is_legal_rendezvous_node(self, node_id: str, state: Any | None = None) -> bool: ...
    def validate_drone_for_mode(self, state, mode: str, drone_id: str) -> tuple[bool, str]: ...

    def evaluate_fixed_mode_a(self, state, order_id: str, truck_id: str) -> GACandidate: ...
    def evaluate_fixed_mode_b(
        self,
        state,
        order_id: str,
        truck_id: str,
        drone_id: str,
        launch_node_id: str,
        recover_node_id: str,
    ) -> GACandidate: ...
    def evaluate_fixed_mode_c(self, state, order_id: str, drone_id: str, recover_node_id: str) -> GACandidate: ...

    def score_candidate(self, candidate: GACandidate, order: Any) -> GACandidate: ...
    def apply_candidate(self, state, candidate: GACandidate) -> None: ...
```

实现约定：

```text
state 应该是 Decoder 持有的 deepcopy，不是真实运行 state。
评估函数优先读取 state 上的 orders/trucks/drones/depots/stations。
若缺失，再回退到 entity_mgr。
validate_drone_for_mode 优先读取 _ga_* 虚拟宿主状态，再 fallback 到真实实体字段。
apply_candidate 只写 state copy 上的 _ga_position / _ga_time / _ga_energy / _ga_host_* / _ga_docked_drones / _ga_idle_drones / _ga_plan_fragments 等虚拟字段。
```

## 9.5 节点解析规则

`get_node_position(node_id)` 必须兼容抽象仓库节点：

```text
node_id == "DEPOT"          -> 映射到实际唯一仓库
node_id.startswith("DEPOT") -> 映射到匹配仓库；不存在则回退唯一仓库
node_id.startswith("DEP-")  -> 映射到匹配仓库，例如 DEP-TEST-01
station_id                  -> 从 state.stations / entity_mgr.stations 中读取
```

`is_legal_rendezvous_node(node_id)`：

```text
合法：
    DEPOT
    DEPOT*
    DEP-*
    已存在 depot_id
    已存在 station_id

非法：
    客户点 order_id
    任意非 depot/station 节点
```

## 9.6 无人机模式校验

`validate_drone_for_mode(state, mode, drone_id)`：

```text
通用不可用条件：
    drone 不存在
    非 IDLE / 非 dispatchable
    carrying_order_id 非空
    waiting_recovery_station_id 非空
    has_pending_route 为 True

B 模式：
    优先判断 drone._ga_host_type == "TRUCK"。
    其次判断 truck._ga_docked_drones / drone._ga_transport_truck_id。
    最后 fallback 到真实 truck.docked_drones / drone.transport_truck_id。

C 模式：
    若 drone._ga_host_type == "TRUCK" 或 "STATION"，不可作为 C 仓库起飞无人机。
    优先判断 drone._ga_host_type == "DEPOT"。
    其次判断 depot._ga_idle_drones。
    最后 fallback 到真实 depot.idle_drones 和仓库位置容差。
```

动态重调度时不能只看 `home_type`。`home_type` 只表示静态归属，不等价于当前物理位置。

## 9.7 固定模式 A 评估

`evaluate_fixed_mode_a(state, order_id, truck_id)`：

```text
检查：
    order 是否存在
    truck 是否存在

计算：
    truck_start_pos = truck._ga_position 或 truck.get_location(current_time)
    truck_start_time = truck._ga_time 或 state.current_time
    dist = greedy._road_dist(truck_start_pos, order.delivery_loc)
    arrival = truck_start_time + dist / truck.speed
    completion = arrival + GreedyMMCE.SERVICE_TIME_CUSTOMER
    lateness = max(0, completion - order.deadline)

输出：
    truck_stops 追加 customer stop
    truck_final_node_id = order_id
    truck_final_time = completion
    delivered_order_ids = [order_id]
```

## 9.8 固定模式 B 评估

`evaluate_fixed_mode_b(state, order_id, truck_id, drone_id, launch_node_id, recover_node_id)`：

```text
禁止自动搜索 station。
必须使用 GA 给定的 launch_node_id 和 recover_node_id。
```

检查：

```text
order / truck / drone 是否存在
B drone 当前是否在卡车上或 docked
launch/recover 是否是合法协同节点
payload 是否超过 drone.payload_capacity
固定 launch -> delivery -> recover 的电量是否足够
```

时间计算要点：

```text
truck_dist_to_launch = road_dist(truck_start_pos, launch_pos)
launch_arrival_time = truck_start_time + truck_dist_to_launch / truck.speed
truck_depart_launch_time = launch_arrival_time + TRUCK_DRONE_LAUNCH_TIME

truck_dist_launch_to_recover = road_dist(launch_pos, recover_pos)
truck_recover_arrival = truck_depart_launch_time + truck_dist_launch_to_recover / truck.speed

drone_launch_time = max(truck_depart_launch_time, drone_start_time)
delivery_arrival = drone_launch_time + dist(launch_pos, order.delivery_loc) / drone.cruise_speed
delivery_done = delivery_arrival + drone_service_time
drone_recover_arrival = delivery_done + dist(order.delivery_loc, recover_pos) / drone.cruise_speed

truck_wait = max(0, drone_recover_arrival - truck_recover_arrival)
uav_wait = max(0, truck_recover_arrival - drone_recover_arrival)
```

注意：

```text
truck_recover_arrival 必须从 truck_depart_launch_time 计算，
不能从 launch_arrival_time 直接计算，否则会漏掉起飞服务时间。
```

等待约束：

```text
如果 truck_wait > config.truck_wait_max_s：
    soft_rendezvous_violation == False -> candidate infeasible
    soft_rendezvous_violation == True  -> 作为 waiting_time 惩罚
```

输出：

```text
truck_stops = [launch stop, recover stop]
drone_route_fragment = launch -> delivery -> recover
truck_final_node_id = recover_node_id
truck_final_time = max(truck_recover_arrival, drone_recover_arrival) + recover_time
drone_final_node_id = recover_node_id
drone_final_time = drone_recover_arrival + recover_time
drone_energy_used / drone_energy_after
delivered_order_ids = [order_id]
```

## 9.9 固定模式 C 评估

`evaluate_fixed_mode_c(state, order_id, drone_id, recover_node_id)`：

```text
launch 固定为 depot。
不生成 truck_stop。
```

检查：

```text
order / drone 是否存在
C drone 当前是否在仓库 depot.idle_drones 中
recover_node_id 是否合法
如果 config.allow_depot_drone_recover_at_station == False：
    recover_node_id 必须是 depot
payload 是否超过 drone.payload_capacity
固定 depot -> delivery -> recover 的电量是否足够
```

输出：

```text
drone_route_fragment = depot -> delivery -> recover
drone_final_node_id = recover_node_id
drone_final_time = recover_arrival
drone_energy_used / drone_energy_after
delivered_order_ids = [order_id]
```

## 9.10 评分与状态推进

`score_candidate(candidate, order)`：

```text
只计算单个动作局部分数。
candidate.score_total 不等于 individual.fitness。
最终 Individual fitness 由 Decoder/Fitness 汇总所有 candidate 得到。

individual.fitness =
    sum(candidate.score_total)
    + infeasible_penalty
    + repair_penalty
    + final_return_penalty
    + station_queue_penalty
    + unserved_order_penalty

score_total =
    completion_time * weight_completion
    + truck_distance * weight_truck_distance
    + uav_distance * weight_uav_distance
    + (truck_energy + uav_energy) * weight_energy
    + lateness * weight_delay
    + waiting_time * weight_waiting
```

`apply_candidate(state, candidate)`：

```text
只作用于 Decoder 的 state copy。
更新 truck._ga_position / truck._ga_node_id / truck._ga_time。
更新 truck._ga_docked_drones。
更新 drone._ga_position / drone._ga_node_id / drone._ga_time / drone._ga_energy。
更新 drone._ga_host_type / drone._ga_host_node_id / drone._ga_transport_truck_id / drone._ga_waiting_station_id。
更新 depot._ga_idle_drones。
更新 station._ga_idle_drones / station._ga_waiting_drones / station._ga_queue_state。
给 order 写入 _ga_completed / _ga_completion_time。
向 state._ga_plan_fragments 追加 candidate。
禁止污染真实运行 state。
```

## 9.11 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/physical_evaluator.py。

实现：
1. GACandidate dataclass，并包含状态推进字段：
   - truck_final_node_id
   - truck_final_time
   - drone_final_node_id
   - drone_final_time
   - drone_energy_used
   - drone_energy_after
   - delivered_order_ids
2. PhysicalEvaluator。
3. get_node_position(node_id)，兼容 DEPOT / DEPOT* / DEP-*。
4. is_legal_rendezvous_node(node_id)，只允许 depot/station。
5. validate_drone_for_mode(state, mode, drone_id)。
6. evaluate_fixed_mode_a(...)。
7. evaluate_fixed_mode_b(...)，必须固定使用 launch_node_id/recover_node_id。
8. evaluate_fixed_mode_c(...)，launch 固定 depot。
9. score_candidate(candidate, order)。
10. apply_candidate(state, candidate)。

允许复用：
- _road_dist
- _dist
- _flight_energy
- _uav_energy_wh
- _truck_energy_wh
- build_incremental_route_from_stops

禁止调用：
- dispatch
- dispatch_incremental
- dispatch_replan_current_state
- _dispatch_impl
- _allocate_order
- _try_mode_b_with_waiting
- _try_mode_b
- _try_mode_c
- _try_mode_a
- _build_truck_route
- _check_energy_feasible
```

---

# 10. Step 7：实现 Decoder，按 GA 决策生成 DispatchPlan

## 10.1 目标

Decoder 读取：

```text
individual.sequence
individual.assignment
individual.rendezvous
```

按 GA 指定顺序逐个订单解码，不再让 greedy 重排顺序、重选模式或重选回收点。

## 10.2 新增文件

```text
backend/solver/ga_mmce/decoder.py
```

## 10.3 Decoder 结构

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chromosome import Individual
from .config import GAConfig
from .adapters import GAAdapterContext, clone_state_for_decode
from .physical_evaluator import PhysicalEvaluator


@dataclass
class DecodeResult:
    plan: Any
    objective: float
    # penalties 的 value 统一为“已换算成本”，不要混入 count。
    penalties: dict[str, float]
    penalty_counts: dict[str, int]
    feasible: bool
    candidates: list[Any]
    truck_ordered_stops: list[dict]
    drone_route_fragments: list[Any]
    unserved_order_ids: list[str]
    repaired: bool
    metrics: dict[str, float]


class GADecoder:
    def __init__(self, config: GAConfig, evaluator: PhysicalEvaluator):
        self.config = config
        self.evaluator = evaluator

    def decode(self, individual: Individual, state, context: GAAdapterContext) -> DecodeResult:
        individual.validate_with_context(
            truck_drone_ids=context.truck_drone_ids,
            depot_drone_ids=context.depot_drone_ids,
            valid_drone_ids=context.all_drone_ids,
            support_node_ids=context.support_node_ids,
        )
        state_copy = clone_state_for_decode(state)

        # penalties 存“成本”，penalty_counts 存次数，避免 Fitness 二次猜权重。
        penalties: dict[str, float] = {}
        penalty_counts: dict[str, int] = {}
        allocations = []
        candidates = []
        drone_route_fragments = []
        truck_ordered_stops: list[dict] = []
        total_score = 0.0
        feasible = True
        repaired = False
        unserved_order_ids: list[str] = []
        attempted_order_ids: set[str] = set()

        truck_id = self._select_default_truck_id(state_copy)

        for order_id, gene, rv in zip(
            individual.sequence,
            individual.assignment,
            individual.rendezvous,
        ):
            attempted_order_ids.add(order_id)
            if gene == "A":
                candidate = self.evaluator.evaluate_fixed_mode_a(
                    state_copy,
                    order_id=order_id,
                    truck_id=truck_id,
                )

            elif gene.startswith("B_"):
                drone_id = gene.split("_", 1)[1]
                candidate = self.evaluator.evaluate_fixed_mode_b(
                    state_copy,
                    order_id=order_id,
                    truck_id=truck_id,
                    drone_id=drone_id,
                    launch_node_id=rv["launch"],
                    recover_node_id=rv["recover"],
                )

            elif gene.startswith("C_"):
                drone_id = gene.split("_", 1)[1]
                candidate = self.evaluator.evaluate_fixed_mode_c(
                    state_copy,
                    order_id=order_id,
                    drone_id=drone_id,
                    recover_node_id=rv["recover"],
                )

            else:
                candidate = None

            if candidate is None or not candidate.feasible:
                feasible = False
                reason = candidate.reason if candidate else "unknown_gene"
                penalty_counts[reason] = penalty_counts.get(reason, 0) + 1
                penalties["infeasible_penalty"] = (
                    penalties.get("infeasible_penalty", 0.0)
                    + self.config.weight_infeasible
                )
                unserved_order_ids.append(order_id)
                continue

            # 只追加 GA 指定产生的节点，不做最近邻重排。
            candidates.append(candidate)
            truck_ordered_stops.extend(candidate.truck_stops)
            allocations.append(self._candidate_to_allocation_fragment(candidate))
            if candidate.drone_route_fragment is not None:
                # 同一架无人机可能连续执行多个任务，不能用 drone_id 单值覆盖。
                drone_route_fragments.append(candidate.drone_route_fragment)
            total_score += candidate.score_total

            self.evaluator.apply_candidate(state_copy, candidate)

        missing_order_ids = sorted(set(context.order_ids) - attempted_order_ids)
        if missing_order_ids:
            feasible = False
            unserved_order_ids.extend(missing_order_ids)
            penalty_counts["unserved_order_penalty"] = len(missing_order_ids)
            penalties["unserved_order_penalty"] = (
                penalties.get("unserved_order_penalty", 0.0)
                + len(missing_order_ids) * self.config.weight_infeasible
            )

        closure_penalties = self._append_c_station_closure_stops(
            state_copy,
            truck_id,
            truck_ordered_stops,
        )
        for name, value in closure_penalties.items():
            penalties[name] = penalties.get(name, 0.0) + value
        repaired = repaired or bool(closure_penalties.get("repair_penalty", 0.0))

        truck_routes = self._build_truck_routes_by_given_order(
            state_copy,
            truck_id,
            truck_ordered_stops,
        )
        metrics = self._collect_plan_metrics(
            truck_routes=truck_routes,
            truck_ordered_stops=truck_ordered_stops,
            drone_route_fragments=drone_route_fragments,
            candidates=candidates,
        )
        total_score += metrics.get("closure_route_cost", 0.0)

        plan = self._build_dispatch_plan(
            allocations=allocations,
            truck_routes=truck_routes,
            drone_routes=self._group_drone_route_fragments(drone_route_fragments),
            cost_total=total_score,
            penalties=penalties,
        )

        return DecodeResult(
            plan=plan,
            objective=total_score,
            penalties=penalties,
            penalty_counts=penalty_counts,
            feasible=feasible,
            candidates=candidates,
            truck_ordered_stops=truck_ordered_stops,
            drone_route_fragments=drone_route_fragments,
            unserved_order_ids=unserved_order_ids,
            repaired=repaired,
            metrics=metrics,
        )
```

## 10.4 核心要求

### 10.4.1 不允许 Decoder 调用完整 greedy

禁止：

```python
greedy.dispatch(...)
greedy._dispatch_impl(...)
greedy._allocate_order(...)
greedy._build_truck_route(...)
```

### 10.4.2 卡车路径顺序由 GA 决定

Decoder 根据 `sequence + assignment + rendezvous` 生成：

```python
truck_ordered_stops = [
    {"node_id": "order_1", "node_type": "customer", ...},
    {"node_id": "S1", "node_type": "recovery", "action": "launch", ...},
    {"node_id": "S3", "node_type": "recovery", "action": "recover", ...},
]
```

然后调用：

```python
greedy.build_incremental_route_from_stops(
    truck=truck,
    ordered_stops=truck_ordered_stops,
    current_time=current_time,
)
```

这个函数按给定顺序构建 OSM 路径，不会像 `_build_truck_route` 那样最近邻重排。

### 10.4.3 C 模式落站必须进入闭环修复阶段

当 `config.allow_depot_drone_recover_at_station=True` 时，C 模式允许：

```text
DEPOT -> customer -> station
```

但这只表示“本单配送可结束在 station”，不表示无人机生命周期已经闭环。Decoder 必须在所有订单解码后进入 closure phase，处理这些等待在充换电站的 C 模式无人机。

Decoder 必须明确处理以下状态：

```text
C recover = depot:
    drone._ga_host_type = "DEPOT"
    drone 加入 depot._ga_idle_drones
    生命周期闭环

C recover = station:
    drone._ga_host_type = "STATION"
    drone._ga_waiting_station_id = station_id
    drone 加入 station._ga_waiting_drones
    station._ga_queue_state 记录该无人机占用/等待事件
    drone 不得在同一个 Individual 后续再次作为 C 模式仓库无人机使用
    drone 不得进入本轮 depot_drone_ids
    Decoder closure phase 必须决定是否追加卡车 pickup stop
```

推荐第一版采用确定性闭环修复，而不是让 Fitness 或 Greedy 重新决策：

```text
1. GA 解码阶段只处理 sequence + assignment + rendezvous 指定的订单动作。
2. apply_candidate 将 C 落站无人机标记为 STATION 宿主，并写入 station queue。
3. 所有订单解码完成后，Decoder 调用 _append_c_station_closure_stops(...)。
4. _append_c_station_closure_stops 按确定性规则把 station pickup stop 追加到 truck_ordered_stops 末尾。
5. 同一个 station 上等待的多个无人机合并成一个 pickup stop。
6. pickup stop 必须标记为 repair/closure，不属于原始 GA 订单动作。
7. pickup 后将无人机虚拟宿主改为 TRUCK，并加入 truck._ga_docked_drones。
```

pickup stop 建议结构：

```python
{
    "node_id": station_id,
    "node_type": "station",
    "position": station.location,
    "action": "closure_pickup_c_drone",
    "source": "repair",
    "drone_ids": ["UAV-TEST-03", "UAV-TEST-08"],
}
```

`position` 必须显式写入，除非 `_build_truck_routes_by_given_order(...)` 已经保证能按 `node_id` 从 depot/station 映射到坐标。为了减少路线构建阶段的隐式依赖，第一版推荐 stop 直接带 `position`。

确定性排序建议：

```text
优先级 1：按该 station 最早等待无人机的 _ga_time 升序。
优先级 2：若时间相同，按 station_id 字典序。
禁止：调用 greedy._build_truck_route 或任何会重排 stop 的贪心函数。
```

如果 `config.enable_repair=False`，Decoder 不应追加 pickup stop，而应保留开环状态并追加较大的 `final_return_penalty`。

如果后续要让 GA 主动优化 C 落站回收顺序，有两种升级路径：

```text
方案 A：保留 Decoder repair，但把 station pickup 顺序做成局部搜索/repair operator。
方案 B：扩展 Chromosome 3/4，让 GA 显式编码 C-station drone 的最终回收策略。
```

在这两种升级完成前，C 落站无人机不应视为免费可复用资源；要么被 closure pickup 接回，要么产生开环惩罚。

## 10.5 apply_candidate 状态更新原则

`PhysicalEvaluator.apply_candidate(state_copy, candidate)` 应做最小必要更新。

Decoder 不再维护第二套 `_apply_candidate_state_update`，避免状态推进规则分叉。

```text
A 模式：
    更新卡车虚拟位置、时间；
    标记订单完成。

B 模式：
    更新卡车到 launch/recover 的虚拟时间；
    更新无人机电量、位置、时间；
    将无人机虚拟宿主设置为 TRUCK；
    写入 drone._ga_transport_truck_id = truck_id；
    将 drone_id 加入 truck._ga_docked_drones；
    标记订单完成。

C 模式：
    更新无人机位置、时间、电量；
    如果 recover 是 depot，则设置 drone._ga_host_type = "DEPOT"，并加入 depot._ga_idle_drones；
    如果 recover 是 station，则设置 drone._ga_host_type = "STATION"，并加入 station._ga_waiting_drones / station._ga_queue_state；
    C 落站无人机不应在同一个 Individual 后续作为 C 仓库无人机复用；
    标记订单完成。
```

所有更新都必须只写 `_ga_*` 字段；真实实体字段如 `current_loc`、`docked_drones`、`idle_drones` 不应被修改。

特别注意：C 落站 closure 的状态来源就是 `PhysicalEvaluator.apply_candidate(...)` 写入的虚拟宿主字段。因此它必须维护：

```text
drone._ga_host_type
drone._ga_host_node_id
drone._ga_transport_truck_id
drone._ga_waiting_station_id
truck._ga_docked_drones
depot._ga_idle_drones
station._ga_waiting_drones
station._ga_queue_state
```

如果这些字段缺失，Decoder 后续无法可靠收集等待在 station 的 C 模式无人机。

## 10.6 闭环修复成本建议

Decoder 负责生成闭环修复路线；Fitness 只负责对 Decoder 生成的方案打分。

建议新增 Decoder 私有函数：

```python
def _append_c_station_closure_stops(
    self,
    state_copy,
    truck_id: str,
    truck_ordered_stops: list[dict],
) -> dict[str, float]:
    penalties: dict[str, float] = {}

    waiting_by_station = self._collect_waiting_station_drones(state_copy)
    if not waiting_by_station:
        return penalties

    if not self.config.enable_repair:
        penalties["final_return_penalty"] = len(waiting_by_station) * self.config.weight_infeasible
        return penalties

    for station_id, drone_ids in self._sort_waiting_stations(waiting_by_station):
        station_pos = self.evaluator.get_node_position(station_id, state_copy)
        truck_ordered_stops.append({
            "node_id": station_id,
            "node_type": "station",
            "position": station_pos,
            "action": "closure_pickup_c_drone",
            "source": "repair",
            "drone_ids": drone_ids,
        })
        penalties["repair_penalty"] = penalties.get("repair_penalty", 0.0) + self.config.repair_penalty_factor
        penalties["station_queue_penalty"] = penalties.get("station_queue_penalty", 0.0) + len(drone_ids) * self.config.weight_waiting

    return penalties
```

注意：

```text
repair_penalty:
    表示这个 Individual 需要 Decoder 额外闭环修复，不是 GA 原生动作。

station_queue_penalty:
    表示无人机在 station 等待卡车接回的占用/等待成本。

final_return_penalty:
    仅在无法或不允许 closure pickup 时使用，表示方案仍有无人机未闭环。
```

闭环 pickup stop 被追加到 `truck_ordered_stops` 后，`build_incremental_route_from_stops(...)` 会自然把卡车去充换电站的路程计入最终路线。Fitness 不需要自己添加 station 位置，也不应该修改路线。

但 Decoder 必须在构建路线后把这部分增量成本显式计入：

```text
metrics["closure_route_distance"]
metrics["closure_route_cost"]
metrics["closure_waiting_time"]
```

其中 `closure_route_cost` 必须进入 `DecodeResult.objective`，或者作为 `penalties["repair_penalty"]` 的一部分进入最终 fitness。推荐第一版放进 `metrics` 并加到 `objective`，避免 Fitness 反向解析路线。

## 10.7 V1 暂不实现但需预留的闭环扩展

以下能力不进入 Decoder V1 的代码实现，但需要在状态字段和 metrics 命名上预留扩展空间：

```text
站点换电队列：
    后续可基于 station._ga_queue_state 增加更真实的换电排队、服务时间和电池恢复逻辑。
    V1 只记录 station waiting/queue 状态和 station_queue_penalty。

最终回仓：
    后续可在所有订单和 closure pickup 后，追加卡车/无人机最终返回仓库的显式收尾逻辑。
    V1 只在 enable_repair=False 或闭环失败时使用 final_return_penalty。
```

## 10.8 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/decoder.py。

实现 GADecoder 和 DecodeResult。

关键要求：
1. decode 必须读取三层染色体：sequence + assignment + rendezvous。
2. decode 必须在 state 深拷贝上执行。
3. decode 按 GA sequence 的顺序处理订单，不允许重新按 deadline 排序。
4. decode 按 GA assignment 指定模式，不允许重新选择 B/C/A。
5. decode 按 GA rendezvous 指定 launch/recover，不允许自动搜索最近站点。
6. B 模式调用 evaluate_fixed_mode_b。
7. C 模式调用 evaluate_fixed_mode_c。
8. A 模式调用 evaluate_fixed_mode_a。
9. 卡车 ordered_stops 的顺序由 GA 解码过程产生。
10. 构建卡车路线时调用 build_incremental_route_from_stops，而不是 _build_truck_route。
11. 输出 DispatchPlan 结构必须兼容前端。
12. 不可行时记录 penalty，不能静默跳过。
13. 每个 Individual 必须使用 clone_state_for_decode(state) 的独立副本。
14. Decoder 入口必须调用 individual.validate_with_context(...)，不能只调用 validate()。
15. C 模式 recover 到 station 时，不允许静默把无人机重新加入 depot_drone_ids。
16. C 模式 recover 到 station 时，应保留 station waiting/queue 状态。
17. Decoder 第一版可以在所有订单动作之后追加 C 落站闭环 pickup stops。
18. pickup stops 必须标记 source="repair" / action="closure_pickup_c_drone"，并显式包含 position。
19. Decoder 结束时必须调用 _append_c_station_closure_stops(...) 或等价逻辑处理 C 落站闭环。
20. drone_route_fragments 必须用 list 保存，不能用 drone_id 单值覆盖。
21. penalties 的 value 必须统一为成本；次数另存 penalty_counts。
22. 不可行订单只写 penalties["infeasible_penalty"] 成本和 penalty_counts，不要同时 total_score += big_m。
23. closure pickup 的路线增量成本必须进入 DecodeResult.objective 或 metrics。
24. Fitness 只计算已生成 plan 的分数，不负责新增闭环路线。
25. candidate 转前端 allocation 时必须经过 _candidate_to_allocation_fragment(...)，不要直接把 GACandidate 放进 DispatchPlan。
```

---

# 11. Step 8：实现 Fitness 计算

## 11.1 目标

`fitness.py` 只负责把 Decoder 已经生成的 `DecodeResult` 转成 GA 可比较的标量分数。

重要边界：

```text
Fitness 不负责生成闭环修复路线。
Fitness 不负责追加 C 落站 station pickup stop。
Fitness 不修改 state_copy、plan、truck_ordered_stops、drone route。
Fitness 只读取 DecodeResult.plan / penalties / objective / feasible 等结果并计算最终 individual.fitness。
```

C 模式无人机落到充换电站后的闭环路线，应该由 `decoder.py` 的 closure phase 生成；Fitness 只对这条已生成路线带来的距离、等待、repair penalty、未闭环 penalty 做计分。

## 11.2 新增文件

```text
backend/solver/ga_mmce/fitness.py
```

## 11.3 与 Decoder 的职责分工

```text
PhysicalEvaluator:
    评估单个固定动作。
    产生 candidate.score_total。

Decoder:
    按 Individual 顺序解码全部订单。
    在 state copy 上推进 _ga_* 状态。
    追加 C 落站 closure pickup stops。
    调用 build_incremental_route_from_stops 构建完整卡车路线。
    输出 DecodeResult。

Fitness:
    不再改变路线。
    不再执行 repair。
    只汇总 DecodeResult.objective 与 DecodeResult.penalties。
    返回越小越好的 float。
```

因此：

```text
闭环修复路线 = Decoder 负责生成
闭环修复成本 = Fitness 负责计分
```

## 11.4 推荐结构

```python
from __future__ import annotations

import math
from typing import Any

from .config import GAConfig


def compute_fitness(decode_result: Any, config: GAConfig) -> float:
    """
    纯评分函数。

    不生成 route。
    不修改 plan。
    不修改 state。
    """

    objective = float(getattr(decode_result, "objective", 0.0) or 0.0)
    if not math.isfinite(objective):
        return config.big_m

    penalties = getattr(decode_result, "penalties", {}) or {}

    # penalties 的 value 已经是成本，不再按名字二次乘权重。
    penalty_cost = sum(float(value or 0.0) for value in penalties.values())
    if not math.isfinite(penalty_cost):
        return config.big_m

    # 正常情况下，不可行原因也应已经写入 penalties 成本。
    # 仅当 feasible=False 且 penalties 为空时，补一个兜底成本。
    if not bool(getattr(decode_result, "feasible", False)) and not penalties:
        penalty_cost += float(config.weight_infeasible)

    fitness = objective + penalty_cost
    return fitness if math.isfinite(fitness) else config.big_m
```

推荐做法是：

```text
DecodeResult.objective:
    保存候选动作、卡车路线、闭环 pickup 路线的基础成本。

DecodeResult.penalties:
    保存 penalty 明细和已经换算好的 penalty cost。

DecodeResult.penalty_counts:
    保存 penalty 次数，仅用于日志、调试、可视化，不参与直接求和。

compute_fitness:
    objective + sum(penalties.values())
```

不要一边在 Decoder 中 `total_score += penalty`，一边在 Fitness 中再次 `sum(penalties)`；也不要在 Fitness 中按 penalty 名字再次乘权重，否则会双重惩罚。

## 11.5 必须包含的惩罚项

```text
infeasible_penalty:
    某个 candidate 不可行，或未知 gene。

repair_penalty:
    Decoder 为 C 落站无人机追加 closure pickup stop。

station_queue_penalty:
    C 无人机在充换电站等待卡车接回。

final_return_penalty:
    enable_repair=False 或闭环失败，导致无人机最终仍在 station。

unserved_order_penalty:
    Individual 中订单缺失、重复或最终未完成。
```

## 11.6 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/fitness.py。

实现 compute_fitness(decode_result, config)。

要求：
1. fitness 越小越好。
2. Fitness 是纯函数，不修改 plan/state/route。
3. Fitness 不负责生成 C 落站闭环路线。
4. C 落站闭环 pickup stops 必须由 decoder.py 生成。
5. Fitness 需要计入：
   - DecodeResult.objective
   - sum(DecodeResult.penalties.values())
6. 对缺失字段安全处理。
7. 不要依赖 greedy 的完整 dispatch 结果。
8. 返回 float。
9. 避免重复计分：penalties 已经是成本，Fitness 不应再次乘权重。
10. 增加 inf / nan 兜底，异常个体返回 config.big_m。
11. penalty_counts 只用于日志，不直接参与 fitness。
```

---

# 12. Step 9：实现 GA 主求解器

## 12.1 新增文件

```text
backend/solver/ga_mmce/solver.py
```

## 12.2 核心职责

```text
1. 从 adapter 获取 GAAdapterContext；
2. 使用 context.gene_pool 和 context.support_node_ids 初始化三层种群；
3. 可选使用 greedy seed，但 greedy seed 只用于初始化个体；
4. 循环迭代；
5. 返回最佳 DispatchPlan。
```

当前代码实现还需要补充以下边界：

```text
1. 对外必须兼容 DispatchSolver 协议：
   - dispatch(pending_orders, current_time, bbox, scene_id=None)
   - dispatch_incremental(new_orders, current_time, bbox, scene_id=None)
   - dispatch_replan_current_state(replan_orders, current_time, bbox, scene_id=None)

2. solve(...) 可以作为内部入口，但对外入口应先构造 GAStateView：
   - entity_mgr
   - orders
   - current_time
   - bbox
   - scene_id

3. Decoder 会调用 build_incremental_route_from_stops(...)。
   因此 solver 在评估种群前应调用 GreedyMMCE._load_road_graph(bbox, scene_id)
   预加载 OSM 路网。若路网加载失败，应记录 warning 并允许 Decoder fallback
   到直接距离路线，而不是在 GA 主循环中崩溃。

4. greedy seed 是 best-effort：
   - 允许调用完整 greedy 生成初始种子；
   - 但只能用于 greedy_plan_to_individual(...)；
   - 失败时跳过 seed，不影响 GA 主流程；
   - Decoder 阶段仍禁止调用 greedy dispatch。

5. warm_start 应统一为 list[Individual]。
   如果调用方没有显式传入 warm_start，可以尝试使用 last_best_individual。
   initialize_population 会过滤订单集合或 gene_pool 不匹配的旧个体。

6. evaluate_individual 应保存：
   - ind.fitness = compute_fitness(result, config)
   - ind.decoded_plan = result.plan
   - ind.penalties = result.penalties
   solver 额外可保存 last_best_decode_result 便于调试。
```

## 12.3 推荐代码框架

```python
from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from solver.greedy_mmce import DispatchPlan, GreedyMMCE

from .adapters import (
    build_ga_context,
    greedy_plan_to_individual,
)
from .config import GAConfig
from .decoder import GADecoder
from .fitness import compute_fitness
from .operators import order_crossover, mutate, tournament_select
from .physical_evaluator import PhysicalEvaluator
from .population import initialize_population

logger = logging.getLogger(__name__)


@dataclass
class GAStateView:
    entity_mgr: Any
    orders: dict[str, Any]
    current_time: float
    bbox: dict | None = None
    scene_id: str | None = None


class GAMMCESolver:
    def __init__(self, entity_mgr, config: GAConfig | None = None):
        self.entity_mgr = entity_mgr
        self.config = config or GAConfig()
        self.greedy_helper = GreedyMMCE(entity_mgr)
        self.evaluator = PhysicalEvaluator(entity_mgr, self.greedy_helper, self.config)
        self.decoder = GADecoder(self.config, self.evaluator)
        self.last_best_individual = None
        self.last_best_decode_result = None

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def dispatch(self, pending_orders, current_time, bbox, scene_id=None):
        state = GAStateView(self.entity_mgr, dict(pending_orders), current_time, bbox, scene_id)
        return self.solve(state)

    def dispatch_incremental(self, new_orders, current_time, bbox, scene_id=None):
        state = GAStateView(self.entity_mgr, dict(new_orders), current_time, bbox, scene_id)
        return self.solve(state)

    def dispatch_replan_current_state(self, replan_orders, current_time, bbox, scene_id=None):
        state = GAStateView(self.entity_mgr, dict(replan_orders), current_time, bbox, scene_id)
        return self.solve(state)

    def solve(self, state, warm_start=None):
        started = time.time()

        context = build_ga_context(state)
        order_ids = context.order_ids
        gene_pool = context.gene_pool
        support_node_ids = context.support_node_ids

        if not order_ids:
            return self._empty_plan()

        self._prepare_distance_context(state)

        greedy_seed = None
        if self.config.use_greedy_seed:
            # 注意：greedy 只作为 seed，不能作为 decoder。
            greedy_plan = self._make_greedy_seed_plan(state)
            greedy_seed = greedy_plan_to_individual(
                greedy_plan,
                order_ids,
                gene_pool,
                support_node_ids,
                allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
            )

        population = initialize_population(
            order_ids=order_ids,
            gene_pool=gene_pool,
            support_node_ids=support_node_ids,
            pop_size=self.config.population_size,
            greedy_seed=greedy_seed,
            warm_start=self._normalize_warm_start(warm_start),
            use_truck_only_seed=self.config.use_truck_only_seed,
            use_obl_seed=self.config.use_obl_seed,
            allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
        )

        self._evaluate_population(population, state, context)

        for gen in range(self.config.generations):
            if self._timeout(started):
                break

            population.sort(key=lambda ind: ind.fitness)
            self._log_generation(gen, population)

            new_population = []
            elite_n = max(1, int(self.config.elite_ratio * self.config.population_size))
            new_population.extend(copy.deepcopy(population[:elite_n]))

            while len(new_population) < self.config.population_size:
                p1 = tournament_select(population, self.config.tournament_k)
                p2 = tournament_select(population, self.config.tournament_k)

                if random.random() < self.config.crossover_rate:
                    c1, c2 = order_crossover(p1, p2)
                else:
                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

                for child in (c1, c2):
                    mutate(
                        child,
                        gene_pool=gene_pool,
                        support_node_ids=support_node_ids,
                        p_seq=self.config.mutation_rate_sequence,
                        p_assign=self.config.mutation_rate_assignment,
                        p_rendezvous=self.config.mutation_rate_rendezvous,
                        allow_c_recover_station=self.config.allow_depot_drone_recover_at_station,
                    )
                    self._evaluate_individual(child, state, context)
                    new_population.append(child)
                    if len(new_population) >= self.config.population_size:
                        break

            population = new_population

        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        self.last_best_individual = copy.deepcopy(best)
        self.last_best_decode_result = self._evaluate_individual(best, state, context)
        return best.decoded_plan

    def _evaluate_population(self, population, state, context):
        for ind in population:
            self._evaluate_individual(ind, state, context)

    def _evaluate_individual(self, ind, state, context):
        ind.validate_with_context(
            truck_drone_ids=context.truck_drone_ids,
            depot_drone_ids=context.depot_drone_ids,
            valid_drone_ids=context.all_drone_ids,
            support_node_ids=context.support_node_ids,
        )
        result = self.decoder.decode(ind, state, context)
        ind.fitness = compute_fitness(result, self.config)
        ind.decoded_plan = result.plan
        ind.penalties = result.penalties
        return result

    def _prepare_distance_context(self, state):
        bbox = getattr(state, "bbox", None)
        if not bbox:
            return
        try:
            self.greedy_helper._road_distance_memo.clear()
            self.greedy_helper._load_road_graph(bbox, getattr(state, "scene_id", None))
        except Exception as exc:
            logger.warning("[GA-MMCE] OSM 路网预加载失败，Decoder 将 fallback 到直接距离: %s", exc)

    def _timeout(self, started):
        if self.config.max_runtime_seconds is None:
            return False
        return time.time() - started >= self.config.max_runtime_seconds

    def _make_greedy_seed_plan(self, state):
        # 这里可以调用完整 greedy 生成 seed，但仅用于初始化 Individual。
        # 不能在 GA decoder 中调用 greedy。
        try:
            return self.greedy_helper.dispatch_replan_current_state(
                replan_orders=state.orders,
                current_time=state.current_time,
                bbox=state.bbox,
                scene_id=getattr(state, "scene_id", None),
            )
        except Exception as exc:
            logger.warning("[GA-MMCE] greedy seed 生成失败，跳过 seed: %s", exc)
            return None

    def _normalize_warm_start(self, warm_start):
        seeds = []
        if warm_start is None and self.last_best_individual is not None:
            seeds.append(self.last_best_individual)
        elif warm_start is not None:
            seeds.extend(warm_start if isinstance(warm_start, list) else [warm_start])
        return seeds

    def _log_generation(self, gen, population):
        if not self.config.verbose or gen % 10 != 0:
            return
        best = population[0]
        avg = sum(ind.fitness for ind in population) / max(1, len(population))
        feasible_count = sum(1 for ind in population if not ind.penalties)
        logger.info(
            "[GA-MMCE] gen=%s best=%.2f avg=%.2f feasible=%s penalties=%s",
            gen,
            best.fitness,
            avg,
            feasible_count,
            best.penalties,
        )

    def _empty_plan(self):
        return DispatchPlan(
            allocations=[],
            cost_total=0.0,
            summary={
                "total_orders": 0,
                "feasible": 0,
                "modes": {},
                "dispatch_type": "ga_mmce",
                "cost_breakdown": {"dist": 0.0, "energy": 0.0, "penalty": 0.0},
            },
        )
```

## 12.4 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/solver.py。

实现 GAMMCESolver。

要求：
1. 构造函数接收 entity_mgr 和 GAConfig。
2. 内部可以创建 GreedyMMCE 作为 greedy_helper，但 greedy_helper 只用于：
   - 物理工具函数；
   - greedy seed 初始化。
3. solve(state, warm_start=None) 必须使用三层种群。
4. solve 不能在解码阶段调用 greedy dispatch。
5. mutation 必须同时支持 sequence、assignment、rendezvous。
6. 保存 last_best_individual，用于动态重调度 warm start。
7. 返回值必须兼容前端使用的 DispatchPlan。
8. 对外实现 DispatchSolver 协议的 dispatch / dispatch_incremental / dispatch_replan_current_state。
9. 在 GA 主循环前 best-effort 调用 _load_road_graph(bbox, scene_id)，供 Decoder 构建 OSM 增量路线。
10. greedy seed 失败时跳过，不得中断 GA 主流程。
11. _evaluate_individual 应返回 DecodeResult，并保存 ind.decoded_plan / ind.penalties / ind.fitness。
```

---

# 13. Step 10：接入 decision_engine.py

## 13.1 目标

让系统可以通过配置选择：

```yaml
solver: greedy_mmce
```

或：

```yaml
solver: ga_mmce
```

## 13.2 修改文件

```text
backend/solver/factory.py
frontend/src/stores/system.ts
frontend/src/views/DispatchCenter/index.vue
```

当前项目的 `decision_engine.py` 已经通过 `factory.create_solver(...)` 动态创建求解器，
因此 Step 10 第一版不需要直接修改 `decision_engine.py`。只要在 `factory.py`
注册 `ga_mmce`，前端把 solver 名称传为 `"ga_mmce"`，后端就会创建
`GAMMCESolver(entity_mgr)`。

前端调度中心需要新增“遗传算法”按钮，并把 solver 类型扩展为：

```ts
'greedy' | 'greedy_mmce' | 'greedy_mmce_bi' | 'ga_mmce' | 'market'
```

## 13.3 给 Codex 的提示词

```text
请修改 factory.py 和前端调度中心，使系统支持 solver = "ga_mmce"。

要求：
1. 原有 greedy_mmce 行为保持不变。
2. 当 solver 名称为 "ga_mmce" 时：
   - factory 创建 GAMMCESolver(entity_mgr)
   - decision_engine 继续按 DispatchSolver 协议调用 dispatch / dispatch_incremental
3. GA solver 的输出必须沿用现有前端可识别的 DispatchPlan 结构。
4. 前端 DispatchCenter 新增“遗传算法”按钮，点击后传 solver="ga_mmce"。
5. frontend/src/stores/system.ts 的 solver 联合类型需要包含 "ga_mmce"。
6. 如果当前配置文件没有 solver 字段，默认仍保持原有 solver，不强制切换为 GA。
```

---

# 14. Step 11：实现动态重调度模块

## 14.1 目标

动态订单到达时，调用：

```python
reschedule_on_event(...)
```

完成：

```text
1. 读取当前时刻；
2. 冻结已完成订单；
3. 冻结正在执行订单；
4. 合并剩余订单 + 新订单；
5. 从当前实体状态重新 GA；
6. 使用 last_best_individual 构造 warm start；
7. 返回新计划。
```

## 14.2 新增文件

```text
backend/solver/ga_mmce/dynamic_rescheduler.py
```

## 14.3 warm start 必须保留 rendezvous

```python
def build_warm_start(
    previous_best,
    completed_ids,
    locked_ids,
    new_order_ids,
    gene_pool,
    depot_ids,
    station_ids,
    allow_c_recover_station=True,
):
    from .chromosome import Individual
    from .operators import make_random_rendezvous_for_gene

    excluded = set(completed_ids) | set(locked_ids)

    seq = []
    assignment = []
    rendezvous = []

    if previous_best is not None:
        for oid, gene, rv in zip(
            previous_best.sequence,
            previous_best.assignment,
            previous_best.rendezvous,
        ):
            if oid not in excluded:
                seq.append(oid)
                assignment.append(gene)
                rendezvous.append(rv)

    for oid in new_order_ids:
        c_genes = [g for g in gene_pool if g.startswith("C_")]
        b_genes = [g for g in gene_pool if g.startswith("B_")]
        gene = c_genes[0] if c_genes else (b_genes[0] if b_genes else "A")
        seq.append(oid)
        assignment.append(gene)
        rendezvous.append(
            make_random_rendezvous_for_gene(
                gene,
                depot_ids,
                station_ids,
                allow_c_recover_station,
            )
        )

    return Individual(seq, assignment, rendezvous)
```

## 14.4 给 Codex 的提示词

```text
请新增/修改 backend/solver/ga_mmce/dynamic_rescheduler.py。

要求：
1. reschedule_on_event(state, new_orders, event_time) 是主入口。
2. 不要中断正在执行的物理动作。
3. completed 订单不再进入 GA。
4. locked 订单不再进入 GA，但对应实体在 locked action 完成前不可用。
5. 新订单加入当前订单池。
6. warm start 必须从 previous_best 的三层染色体生成。
7. 新订单插入时，必须同步生成 assignment 和 rendezvous。
8. 最后调用 GAMMCESolver.solve(snapshot, warm_start=[...])。
9. 如果项目已有事件系统或状态推进函数，请复用，不要新写一套仿真器。
```

---

# 15. Step 12：添加单元测试

## 15.1 新增测试文件

```text
tests/solver/test_ga_chromosome.py
tests/solver/test_ga_operators.py
tests/solver/test_ga_population.py
tests/solver/test_ga_physical_evaluator.py
tests/solver/test_ga_decoder_smoke.py
tests/solver/test_ga_dynamic_rescheduler.py
```

## 15.2 测试重点

### 15.2.1 染色体合法性

```text
- sequence / assignment / rendezvous 长度不一致应报错；
- sequence 有重复订单应报错；
- A 的 rendezvous 不是 None 应报错；
- B 缺少 launch 或 recover 应报错；
- C 的 launch 不是 depot 应报错。
```

### 15.2.2 OX 交叉

```text
- 子代订单不重复；
- 子代订单集合等于父代订单集合；
- assignment 与 rendezvous 均按订单 ID 对齐；
- 子代 validate 通过。
```

### 15.2.3 变异

```text
- 不丢订单；
- 不增加订单；
- gene 必须来自 gene_pool；
- rendezvous 仍满足模式约束。
```

### 15.2.4 PhysicalEvaluator

```text
- evaluate_fixed_mode_b 使用指定 launch/recover；
- 不允许自动替换 recovery；
- 电量不足时返回 energy_not_enough；
- 非法站点时返回 illegal_launch_node / illegal_recover_node。
```

### 15.2.5 Decoder smoke test

```text
- 3 个订单、1 辆卡车、1 架无人机、1 个换电站；
- 给定固定 rendezvous；
- GA 能返回 DispatchPlan；
- 每个订单最多服务一次；
- truck route 顺序与 sequence/rendezvous 一致；
- 不调用 greedy.dispatch。
```

### 15.2.6 动态重调度

```text
- 已完成订单不会再次进入 active orders；
- locked 订单不会被重新分配；
- 新订单会进入 active orders；
- warm start 保留旧订单的 rendezvous；
- 新订单生成合法 rendezvous。
```

## 15.3 给 Codex 的提示词

```text
请为 ga_mmce 添加单元测试。

优先测试：
1. Individual.validate 三层约束。
2. order_crossover 是否按订单 ID 对齐 assignment 和 rendezvous。
3. mutate 是否保持三层染色体合法。
4. initialize_population 是否生成合法三层个体。
5. PhysicalEvaluator 是否固定使用 GA 指定 launch/recover。
6. GADecoder 是否不调用 greedy.dispatch。
7. DynamicGARescheduler 是否正确处理 completed/locked/new orders 和 warm start。

如果真实实体构造复杂，可以使用 mock state 或项目已有 fixture。
```

---

# 16. Step 13：添加运行日志和调试信息

## 16.1 推荐日志

每 10 代输出：

```text
generation
best_fitness
avg_fitness
best_penalties
feasible_count
best_sequence
best_assignment
best_rendezvous
```

## 16.2 给 Codex 的提示词

```text
请给 GAMMCESolver 添加调试日志。

要求：
1. 使用项目现有 logging 方式。
2. 每 10 代输出：
   - generation
   - best fitness
   - average fitness
   - feasible individual count
   - best penalties
   - best sequence
   - best assignment
   - best rendezvous
3. 日志开关由 GAConfig.verbose 控制。
4. 不要使用 print。
```

---

# 17. 第一次集成验收

完成上述步骤后，运行：

```bash
pytest tests/solver/test_ga_chromosome.py
pytest tests/solver/test_ga_operators.py
pytest tests/solver/test_ga_population.py
pytest tests/solver/test_ga_physical_evaluator.py
pytest tests/solver/test_ga_decoder_smoke.py
pytest tests/solver/test_ga_dynamic_rescheduler.py
```

然后跑一个小规模场景：

```text
订单数：5
无人机数：1 或 2
换电站：2
动态订单：先不启用
solver：ga_mmce
```

验收标准：

```text
1. 后端不报错；
2. 前端能显示路线；
3. 每个订单只出现一次；
4. GA 的 truck route 顺序与 sequence/rendezvous 解码结果一致；
5. B 模式的 launch/recover 来自 rendezvous，而不是 greedy 自动选择；
6. GA 结果不一定优于 greedy，但必须可解释；
7. 若不可行，penalty 能说明原因。
```

---

# 18. 推荐给 Codex 的总代码任务说明

可以把下面这整段直接发给 Codex：

```text
我要在现有项目中新增一个遗传算法求解器，不要重写环境、实体、前端。

重要修改：
GA 必须使用三层染色体：
1. sequence：订单顺序
2. assignment：模式/载具分配，A / B_<drone_id> / C_<drone_id>
3. rendezvous：协同节点选择
   - A: None
   - B: {"launch": <depot_or_station_id>, "recover": <depot_or_station_id>}
   - C: {"launch": <depot_id>, "recover": <depot_or_station_id>}

背景：
- greedy_mmce.py 是完整贪心调度器，会自己排序、选模式、选起飞站点、选回收点、构建最近邻卡车路径。
- 因此 GA Decoder 不能调用完整 greedy dispatch，否则 GA 会被架空。
- GA 只能复用 greedy_mmce.py 中的底层物理工具函数。

请按以下步骤实现：

1. 新增 backend/solver/ga_mmce/ 包。
2. 新增 config.py，实现 GAConfig，包含 mutation_rate_rendezvous。
3. 新增 chromosome.py，实现三层 Individual。
4. 新增 operators.py，实现 OX 交叉、assignment mutation、rendezvous mutation。
5. 新增 population.py，实现三层种群初始化。
6. 新增 adapters.py，提取 active orders、drone ids、truck ids、depot ids、station ids，并把 greedy plan 转成三层 Individual。
7. 新增 physical_evaluator.py，实现固定决策评估：
   - evaluate_fixed_mode_a
   - evaluate_fixed_mode_b
   - evaluate_fixed_mode_c
   这些函数必须使用 GA 指定的 launch/recover，不允许自动搜索最近站点。
8. 新增 decoder.py，实现 GADecoder：
   - 按 sequence 顺序处理订单；
   - 按 assignment 固定模式；
   - 按 rendezvous 固定起飞/回收节点；
   - 使用 build_incremental_route_from_stops 按给定 ordered_stops 构建卡车路线；
   - 不允许调用 greedy.dispatch / _dispatch_impl / _allocate_order / _build_truck_route。
9. 新增 fitness.py，统一计算适应度。
10. 新增 solver.py，实现 GAMMCESolver.solve(state, warm_start=None)。
11. 修改 decision_engine.py，使配置 solver = "ga_mmce" 时调用 GAMMCESolver。
12. 新增 dynamic_rescheduler.py，实现新订单到达时的事件触发式重调度，并保留 rendezvous warm start。
13. 添加 tests/solver 下的单元测试。
14. 每一步完成后运行测试，确保 greedy 原功能不被破坏。

禁止 GA Decoder 调用：
- GreedyMMCE.dispatch
- GreedyMMCE.dispatch_incremental
- GreedyMMCE.dispatch_replan_current_state
- GreedyMMCE._dispatch_impl
- GreedyMMCE._allocate_order
- GreedyMMCE._try_mode_b_with_waiting
- GreedyMMCE._try_mode_b
- GreedyMMCE._try_mode_c
- GreedyMMCE._try_mode_a
- GreedyMMCE._build_truck_route
- GreedyMMCE._check_energy_feasible

GA 可以复用：
- _road_dist
- _dist
- _flight_energy
- _uav_energy_wh
- _truck_energy_wh
- _recalculate_plan_route_costs
- build_incremental_route_from_stops

重要约束：
- 不要修改前端。
- 不要重写实体状态机。
- 不要复制一套能耗计算。
- GA 解码必须在 state 副本上执行，不能污染真实环境。
- 每个订单必须且只能服务一次。
- 无人机携货阶段不允许中途换电。
- 模式 B 的 launch/recover 只能是仓库或充换电站。
- 模式 B 的回收点不能是客户点。
- 动态重调度不能中断正在执行的飞行或服务动作。
```

---

# 19. MVP 实现建议

原 MVP 里“decoder 暂时调用 greedy 整体 solve”不再推荐，因为这会掩盖 GA 的真实作用。

推荐新 MVP：

```text
MVP 1：
    chromosome.py
    operators.py
    population.py
    三层染色体单元测试

MVP 2：
    physical_evaluator.py
    先实现 evaluate_fixed_mode_a / B / C 的基本可行性判断

MVP 3：
    decoder.py
    使用固定 sequence + assignment + rendezvous 生成 DispatchPlan
    truck route 使用 build_incremental_route_from_stops

MVP 4：
    solver.py
    完成 GA 迭代

MVP 5：
    接入 decision_engine.py

MVP 6：
    dynamic_rescheduler.py + warm start
```

临时允许：

```text
greedy 作为 seed 生成器。
```

临时不允许：

```text
greedy 作为 GA decoder。
```

---

# 20. 最终调用链

静态调度调用链：

```text
decision_engine.py
    ↓
GAMMCESolver.solve(state)
    ↓
initialize_population(sequence + assignment + rendezvous)
    ↓
for each generation:
    ↓
GADecoder.decode(individual, state_copy)
    ↓
PhysicalEvaluator.evaluate_fixed_mode_A/B/C(...)
    ↓
按 GA 指定的 ordered_stops 构建 truck route
    ↓
build_incremental_route_from_stops(...)
    ↓
fitness / objective
    ↓
selection / crossover / mutation
    ↓
best.decoded_plan
    ↓
前端展示
```

动态订单调用链：

```text
dynamic order event
    ↓
DynamicGARescheduler.reschedule_on_event()
    ↓
snapshot current state
    ↓
remove completed orders
    ↓
freeze locked orders
    ↓
insert new orders
    ↓
build warm start with sequence + assignment + rendezvous
    ↓
GAMMCESolver.solve(snapshot, warm_start)
    ↓
new plan
```

---

# 21. 推荐 commit 顺序

```text
commit 1: add GAConfig with rendezvous mutation config
commit 2: add three-layer Individual
commit 3: add GA operators with rendezvous crossover/mutation
commit 4: add population init for three-layer chromosome
commit 5: add adapters for depot/station/truck extraction
commit 6: add physical_evaluator fixed-mode evaluation
commit 7: add GADecoder using fixed decisions
commit 8: add GAMMCESolver
commit 9: integrate decision_engine
commit 10: add physical evaluator and decoder tests
commit 11: add dynamic rescheduler with rendezvous warm start
commit 12: add logging and integration tests
```

最重要的原则：

```text
GA 决定 sequence / assignment / rendezvous。
GreedyMMCE 只作为底层物理工具和 seed 生成器。
不要让 greedy 决定 GA 个体的路径结构。
```

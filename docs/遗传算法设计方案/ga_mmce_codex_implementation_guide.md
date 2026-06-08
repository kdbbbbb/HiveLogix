# 基于 GA 的 1 UGV + M UAV 协同配送算法实现指南

## 0. 目标说明

当前任务不是重新设计环境、实体或前端，而是在已有项目基础上新增一个遗传算法求解器。

已有基础：

- 贪心基线调度方式已经实现；
- 环境与实体状态机已经实现；
- 前端展示已经实现；
- GA 只需要作为新的 solver 接入现有调度系统。

建议采用的核心思想：

```text
GA 负责：
    订单顺序 + 载具/模式分配

Decoder 负责：
    调用已有 greedy_mmce.py 中的物理评估逻辑，生成真实调度计划
```

也就是说，遗传算法不要重复实现无人机能耗、卡车路径、充换电站排队、回收点选择等物理逻辑，而应该复用已有模块。

---

# 1. 总体实现路线

建议分三阶段推进。

```text
阶段 1：静态 GA 可跑通
    输入当前订单和环境状态
    输出和 greedy_mmce.py 一样格式的调度结果
    暂时不处理动态订单

阶段 2：接入 decision_engine.py
    通过配置选择 solver = ga_mmce
    前端无需改动
    GA 输出结构与贪心输出保持一致

阶段 3：事件触发式动态重调度
    新订单到达时，冻结已完成和执行中任务
    对剩余订单 + 新订单重新 GA
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
    decoder.py
    fitness.py
    solver.py
    dynamic_rescheduler.py
```

---

# 2. Step 0：让 Codex 先阅读现有接口

## 2.1 目标

先不要修改代码，让 Codex 读取现有文件，确认：

1. `greedy_mmce.py` 的主入口函数签名；
2. `decision_engine.py` 如何调用 solver；
3. 调度结果返回给前端的数据结构；
4. 订单、无人机、卡车、换电站实体的字段名；
5. 参数从哪里加载。

## 2.2 给 Codex 的提示词

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
2. decision_engine.py 是如何选择和调用 solver 的。
3. 前端依赖的调度结果字段有哪些。
4. Order / Drone / Truck / SwapStation 的关键字段名。
5. 哪些函数可以复用来做：
   - 模式 A 评估
   - 模式 B 评估
   - 模式 C 评估
   - 前瞻能量校验
   - 回收点选择
   - 换电站排队
   - 评分函数

只做代码阅读和总结，不要修改文件。
```

## 2.3 验收标准

Codex 应该输出一份接口总结。

如果它发现 `greedy_mmce.py` 中没有可复用公共函数，需要标记：

```text
需要从 greedy_mmce.py 中抽取公共函数
```

---

# 3. Step 1：新增 GA 配置文件

## 3.1 目标

让 GA 参数可配置，避免硬编码。

## 3.2 新增文件

```text
backend/solver/ga_mmce/config.py
```

## 3.3 推荐代码

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

    use_greedy_seed: bool = True
    use_truck_only_seed: bool = True
    use_obl_seed: bool = True

    big_m: float = 1e9

    weight_completion: float = 1.0
    weight_delay: float = 10.0
    weight_energy: float = 0.1
    weight_waiting: float = 5.0
    weight_infeasible: float = 100000.0

    max_runtime_seconds: float | None = None
    random_seed: int | None = 42
    verbose: bool = False
```

## 3.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/config.py。

实现 GAConfig dataclass，包含：
- population_size
- generations
- elite_ratio
- tournament_k
- crossover_rate
- mutation_rate_sequence
- mutation_rate_assignment
- use_greedy_seed
- use_truck_only_seed
- use_obl_seed
- big_m
- weight_completion
- weight_delay
- weight_energy
- weight_waiting
- weight_infeasible
- max_runtime_seconds
- random_seed
- verbose

不要修改其他文件。
```

---

# 4. Step 2：设计染色体结构

## 4.1 目标

实现双层编码：

```text
染色体 1：任务序列 sequence
染色体 2：载具/模式分配 assignment
```

推荐采用逐订单分配基因，而不是单纯切分点：

```text
A       -> 卡车直递
B_0     -> 卡车无人机 0 执行
B_1     -> 卡车无人机 1 执行
C_0     -> 仓库无人机 0 执行
C_1     -> 仓库无人机 1 执行
```

这种方式更适合动态订单插入和事件触发式重调度。

## 4.2 新增文件

```text
backend/solver/ga_mmce/chromosome.py
```

## 4.3 推荐代码

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Individual:
    sequence: list[str]
    assignment: list[str]
    fitness: float = float("inf")
    decoded_plan: Any | None = None
    penalties: dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.sequence) != len(self.assignment):
            raise ValueError(
                f"sequence length {len(self.sequence)} != "
                f"assignment length {len(self.assignment)}"
            )

        if len(set(self.sequence)) != len(self.sequence):
            raise ValueError("sequence contains duplicated order ids")


def make_gene_pool(drone_ids: list[str | int]) -> list[str]:
    genes = ["A"]
    for uid in drone_ids:
        genes.append(f"B_{uid}")
        genes.append(f"C_{uid}")
    return genes
```

## 4.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/chromosome.py。

实现：
1. Individual dataclass：
   - sequence: list[str]
   - assignment: list[str]
   - fitness: float
   - decoded_plan: Any | None
   - penalties: dict[str, float]
   - validate() 方法：检查长度一致、订单不重复。

2. make_gene_pool(drone_ids)：
   返回 ["A", "B_<id>", "C_<id>", ...]。

不要依赖具体实体类，只做通用染色体结构。
```

---

# 5. Step 3：实现交叉、变异、选择算子

## 5.1 目标

实现 GA 基础算子：

1. OX 顺序交叉；
2. 任务序列交换变异；
3. 载具分配变异；
4. 锦标赛选择。

## 5.2 新增文件

```text
backend/solver/ga_mmce/operators.py
```

## 5.3 推荐代码

```python
import copy
import random

from .chromosome import Individual


def order_crossover(p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
    n = len(p1.sequence)
    if n <= 1:
        return copy.deepcopy(p1), copy.deepcopy(p2)

    a, b = sorted(random.sample(range(n), 2))

    def build_child(base: Individual, donor: Individual) -> Individual:
        child_seq = [None] * n
        child_seq[a:b + 1] = base.sequence[a:b + 1]

        donor_fill = [oid for oid in donor.sequence if oid not in child_seq]
        fill_idx = 0

        for i in range(n):
            if child_seq[i] is None:
                child_seq[i] = donor_fill[fill_idx]
                fill_idx += 1

        base_gene = dict(zip(base.sequence, base.assignment))
        donor_gene = dict(zip(donor.sequence, donor.assignment))

        child_assignment = []
        for oid in child_seq:
            if random.random() < 0.5:
                child_assignment.append(base_gene[oid])
            else:
                child_assignment.append(donor_gene[oid])

        return Individual(sequence=child_seq, assignment=child_assignment)

    return build_child(p1, p2), build_child(p2, p1)


def mutate(ind: Individual, gene_pool: list[str], p_seq: float, p_assign: float) -> None:
    n = len(ind.sequence)

    if n >= 2 and random.random() < p_seq:
        i, j = random.sample(range(n), 2)
        ind.sequence[i], ind.sequence[j] = ind.sequence[j], ind.sequence[i]
        ind.assignment[i], ind.assignment[j] = ind.assignment[j], ind.assignment[i]

    for i in range(n):
        if random.random() < p_assign:
            ind.assignment[i] = random.choice(gene_pool)


def tournament_select(population: list[Individual], k: int) -> Individual:
    candidates = random.sample(population, min(k, len(population)))
    candidates.sort(key=lambda x: x.fitness)
    return copy.deepcopy(candidates[0])
```

## 5.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/operators.py。

实现：
1. order_crossover(p1, p2)
   - 对 sequence 使用 OX 顺序交叉
   - assignment 必须按订单 id 对齐，不能简单按下标复制
   - 返回两个子代 Individual

2. mutate(ind, gene_pool, p_seq, p_assign)
   - p_seq 控制任务序列交换变异
   - p_assign 控制模式/载具分配变异

3. tournament_select(population, k)
   - 返回 fitness 最小的个体深拷贝

请确保交叉后订单不重复、不丢失。
```

---

# 6. Step 4：实现种群初始化

## 6.1 目标

生成初始种群，包括：

1. 随机解；
2. 纯卡车解；
3. 贪心种子解；
4. OBL 反向学习种子。

## 6.2 新增文件

```text
backend/solver/ga_mmce/population.py
```

## 6.3 推荐代码

```python
import random

from .chromosome import Individual, make_gene_pool


def make_random_individual(order_ids: list[str], gene_pool: list[str]) -> Individual:
    seq = list(order_ids)
    random.shuffle(seq)
    assignment = [random.choice(gene_pool) for _ in seq]
    return Individual(seq, assignment)


def make_truck_only_individual(order_ids: list[str]) -> Individual:
    return Individual(
        sequence=list(order_ids),
        assignment=["A"] * len(order_ids),
    )


def make_obl_individual(base: Individual, gene_pool: list[str]) -> Individual:
    seq = list(reversed(base.sequence))
    assignment = []
    for gene in reversed(base.assignment):
        if gene == "A":
            assignment.append(random.choice(gene_pool))
        elif gene.startswith("B_"):
            uid = gene.split("_", 1)[1]
            assignment.append(f"C_{uid}" if f"C_{uid}" in gene_pool else "A")
        elif gene.startswith("C_"):
            uid = gene.split("_", 1)[1]
            assignment.append(f"B_{uid}" if f"B_{uid}" in gene_pool else "A")
        else:
            assignment.append(random.choice(gene_pool))
    return Individual(seq, assignment)


def initialize_population(
    order_ids: list[str],
    drone_ids: list[str | int],
    pop_size: int,
    greedy_seed: Individual | None = None,
    use_truck_only_seed: bool = True,
    use_obl_seed: bool = True,
) -> list[Individual]:
    gene_pool = make_gene_pool(drone_ids)
    population: list[Individual] = []

    if greedy_seed is not None:
        population.append(greedy_seed)

    if use_truck_only_seed:
        population.append(make_truck_only_individual(order_ids))

    if use_obl_seed and population:
        population.append(make_obl_individual(population[0], gene_pool))

    while len(population) < pop_size:
        population.append(make_random_individual(order_ids, gene_pool))

    return population[:pop_size]
```

## 6.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/population.py。

实现：
1. make_random_individual(order_ids, gene_pool)
2. make_truck_only_individual(order_ids)
3. make_obl_individual(base, gene_pool)
4. initialize_population(order_ids, drone_ids, pop_size, greedy_seed=None, use_truck_only_seed=True, use_obl_seed=True)

注意：
- 不要直接访问环境实体。
- 这里只生成染色体。
- greedy_seed 以后由 solver 从 greedy_mmce.py 的结果转换而来。
```

---

# 7. Step 5：写 GA 与现有环境之间的适配层

## 7.1 目标

因为项目已有实体类，不要让 GA 直接猜字段名。需要增加 adapter 文件，把现有实体转换成 GA 需要的简单信息。

## 7.2 新增文件

```text
backend/solver/ga_mmce/adapters.py
```

## 7.3 需要实现的函数

```python
def extract_active_order_ids(env_or_state) -> list[str]:
    """
    返回当前需要调度的订单 id。
    排除 completed / cancelled / already_locked 的订单。
    """


def extract_drone_ids(env_or_state) -> list[str]:
    """
    返回可参与 GA 分配的无人机 id。
    """


def clone_state_for_decode(env_or_state):
    """
    解码个体时必须使用环境副本，不能污染真实环境。
    """


def greedy_plan_to_individual(greedy_plan, order_ids, drone_ids):
    """
    把已有贪心结果转换成 Individual。
    如果无法完全解析，则返回 None。
    """
```

## 7.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/adapters.py。

请根据项目中真实实体字段实现以下函数：
1. extract_active_order_ids(state)
2. extract_drone_ids(state)
3. clone_state_for_decode(state)
4. greedy_plan_to_individual(greedy_plan, order_ids, drone_ids)

要求：
- 不要修改实体类。
- 如果字段名不确定，请根据现有代码实际字段适配。
- clone_state_for_decode 优先使用 copy.deepcopy。
- greedy_plan_to_individual 如果无法解析计划结构，可以先返回 None，并添加 TODO 注释。
```

---

# 8. Step 6：实现 Decoder，但必须复用 greedy_mmce 的物理评估

## 8.1 目标

GA 个体只有：

```text
sequence + assignment
```

Decoder 负责把它变成真实调度计划。

但项目中已有 greedy 基线，已经实现：

- 模式 A/B/C 候选评估；
- 前瞻能量校验；
- 回收点选择；
- 卡车路径构建；
- 评分。

所以 Decoder 不应该重复写能耗和路径细节，而应该复用 `greedy_mmce.py`。

## 8.2 新增文件

```text
backend/solver/ga_mmce/decoder.py
```

## 8.3 Decoder 结构

```python
from dataclasses import dataclass
from typing import Any

from .chromosome import Individual
from .config import GAConfig


@dataclass
class DecodeResult:
    plan: Any
    objective: float
    penalties: dict[str, float]
    feasible: bool


class GADecoder:
    def __init__(self, config: GAConfig):
        self.config = config

    def decode(self, individual: Individual, state) -> DecodeResult:
        """
        核心逻辑：
        1. 拷贝 state
        2. 按 individual.sequence 顺序处理订单
        3. 根据 assignment 选择 A / B_u / C_u
        4. 调用 greedy_mmce 中已有的候选评估函数
        5. 如果指定模式不可行，尝试 repair
        6. 汇总 plan、completion time、penalty、objective
        """
        raise NotImplementedError
```

## 8.4 Decoder 内部流程

```text
for order_id, gene in zip(sequence, assignment):

    if gene == "A":
        尝试模式 A
        如果失败：
            尝试 C 修复
            仍失败则加 infeasible penalty

    elif gene.startswith("B_"):
        指定无人机 u
        尝试模式 B / B_WAIT
        如果电量、载重、回收点不可行：
            尝试 A 或 C 修复
            仍失败则加 infeasible penalty

    elif gene.startswith("C_"):
        指定无人机 u
        尝试模式 C
        如果失败：
            尝试 A 修复
            仍失败则加 infeasible penalty
```

## 8.5 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/decoder.py。

实现 GADecoder 类和 DecodeResult dataclass。

要求：
1. decode(individual, state) 必须在 state 的深拷贝上执行，不能污染真实运行环境。
2. decode 按 individual.sequence 顺序处理订单。
3. 对每个订单读取对应 gene：
   - "A" 调用现有 greedy_mmce.py 中模式 A 的评估/落地逻辑。
   - "B_<uid>" 调用现有模式 B 或 B_WAIT 的评估/落地逻辑。
   - "C_<uid>" 调用现有模式 C 的评估/落地逻辑。
4. 如果 greedy_mmce.py 里这些逻辑目前是私有代码，请先从 greedy_mmce.py 中抽取公共函数，不要复制粘贴一份。
5. 如果指定模式不可行，实现 repair：
   - B 不可行：优先尝试 A，其次 C。
   - C 不可行：尝试 A。
   - A 库存不足：尝试 C。
   - 全部失败：加入 infeasible penalty。
6. decode 返回 DecodeResult：
   - plan：必须和 greedy_mmce.py 当前返回给前端的计划结构兼容。
   - objective：总适应度，越小越好。
   - penalties：字典。
   - feasible：是否无大惩罚。

如果当前 greedy_mmce.py 没有公共候选函数，请先只搭好 decoder 框架，并在 TODO 中标记需要抽取的函数名称。
```

---

# 9. Step 7：从 greedy_mmce.py 抽取可复用函数

## 9.1 目标

贪心代码中候选评估可能写在一个大函数里。GA Decoder 需要复用它们，因此需要重构，而不是复制代码。

## 9.2 建议抽取的函数

```python
evaluate_mode_a_candidate(...)
evaluate_mode_b_candidate(...)
evaluate_mode_c_candidate(...)
apply_mode_a(...)
apply_mode_b(...)
apply_mode_c(...)
score_plan(...)
```

也可以更通用地抽象为：

```python
build_candidate(order, mode, drone_id, state)
apply_candidate(candidate, state)
```

## 9.3 推荐抽象

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    mode: str
    order_id: str
    drone_id: str | None
    feasible: bool
    cost: float
    penalty: float
    reason: str | None
    plan_fragment: Any


def evaluate_candidate(state, order_id, gene) -> Candidate:
    ...


def apply_candidate(state, candidate) -> None:
    ...
```

这样 greedy 和 GA 都可以共用。

## 9.4 给 Codex 的提示词

```text
请重构 backend/solver/greedy_mmce.py，但不要改变它对外的行为。

目标是从当前贪心逻辑中抽取可复用候选评估函数，供 GA decoder 调用。

请新增或抽取：
1. Candidate dataclass，字段包括：
   - mode
   - order_id
   - drone_id
   - feasible
   - cost
   - penalty
   - reason
   - plan_fragment

2. evaluate_candidate(state, order_id, gene_or_mode)
   - gene_or_mode 支持 "A", "B_<uid>", "C_<uid>"
   - 内部复用当前已有的能量校验、回收点选择、换电站逻辑。

3. apply_candidate(state, candidate)
   - 将 candidate 对应的计划片段落到状态副本上。

4. 保持原来的 greedy solver 输出完全不变。

请先运行现有测试，确保 greedy 行为没有改变。
```

---

# 10. Step 8：实现 Fitness 计算

## 10.1 目标

适应度统一在 `fitness.py` 中处理。

## 10.2 新增文件

```text
backend/solver/ga_mmce/fitness.py
```

## 10.3 适应度结构

```python
def compute_fitness(plan, penalties, config):
    """
    objective =
        completion_cost
        + delay_cost
        + energy_cost
        + waiting_cost
        + infeasible_penalty
    """
```

如果现有 greedy 已经有评分函数，优先包装它：

```python
def compute_fitness_from_greedy_score(greedy_score, penalties, config):
    ...
```

## 10.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/fitness.py。

实现 compute_fitness(plan, penalties, config)。

要求：
1. 如果 greedy_mmce.py 已经有评分函数，请优先复用。
2. fitness 越小越好。
3. 至少包含：
   - completion time cost
   - delay penalty
   - energy cost
   - waiting penalty
   - infeasible penalty
4. 对缺失字段要安全处理，不要因为某个 plan 字段不存在直接崩溃。
5. 返回 float。
```

---

# 11. Step 9：实现 GA 主求解器

## 11.1 新增文件

```text
backend/solver/ga_mmce/solver.py
```

## 11.2 核心职责

```text
1. 从环境提取 active orders
2. 提取 drone ids
3. 初始化种群
4. 可选调用 greedy 得到 seed
5. 循环迭代
6. 返回最佳 plan
```

## 11.3 推荐代码框架

```python
import random
import time
import copy

from .config import GAConfig
from .chromosome import make_gene_pool
from .population import initialize_population
from .operators import order_crossover, mutate, tournament_select
from .decoder import GADecoder
from .adapters import (
    extract_active_order_ids,
    extract_drone_ids,
    greedy_plan_to_individual,
)


class GAMMCESolver:
    def __init__(self, config: GAConfig | None = None, greedy_solver=None):
        self.config = config or GAConfig()
        self.greedy_solver = greedy_solver
        self.decoder = GADecoder(self.config)

        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def solve(self, state, warm_start=None):
        started = time.time()

        order_ids = extract_active_order_ids(state)
        drone_ids = extract_drone_ids(state)
        gene_pool = make_gene_pool(drone_ids)

        if not order_ids:
            return self._empty_plan(state)

        greedy_seed = None
        if self.config.use_greedy_seed and self.greedy_solver is not None:
            greedy_plan = self.greedy_solver.solve(copy.deepcopy(state))
            greedy_seed = greedy_plan_to_individual(
                greedy_plan=greedy_plan,
                order_ids=order_ids,
                drone_ids=drone_ids,
            )

        population = initialize_population(
            order_ids=order_ids,
            drone_ids=drone_ids,
            pop_size=self.config.population_size,
            greedy_seed=greedy_seed,
            use_truck_only_seed=self.config.use_truck_only_seed,
            use_obl_seed=self.config.use_obl_seed,
        )

        if warm_start:
            population = warm_start + population
            population = population[:self.config.population_size]

        self._evaluate_population(population, state)

        for gen in range(self.config.generations):
            if self._timeout(started):
                break

            population.sort(key=lambda ind: ind.fitness)
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

                mutate(
                    c1,
                    gene_pool,
                    self.config.mutation_rate_sequence,
                    self.config.mutation_rate_assignment,
                )
                mutate(
                    c2,
                    gene_pool,
                    self.config.mutation_rate_sequence,
                    self.config.mutation_rate_assignment,
                )

                for child in (c1, c2):
                    self._evaluate_individual(child, state)
                    new_population.append(child)
                    if len(new_population) >= self.config.population_size:
                        break

            population = new_population

        population.sort(key=lambda ind: ind.fitness)
        best = population[0]
        return best.decoded_plan

    def _evaluate_population(self, population, state):
        for ind in population:
            self._evaluate_individual(ind, state)

    def _evaluate_individual(self, ind, state):
        ind.validate()
        result = self.decoder.decode(ind, state)
        ind.fitness = result.objective
        ind.decoded_plan = result.plan
        ind.penalties = result.penalties

    def _timeout(self, started: float) -> bool:
        if self.config.max_runtime_seconds is None:
            return False
        return time.time() - started >= self.config.max_runtime_seconds

    def _empty_plan(self, state):
        return {
            "status": "ok",
            "solver": "ga_mmce",
            "actions": [],
            "score": 0.0,
        }
```

## 11.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/solver.py。

实现 GAMMCESolver。

要求：
1. 构造函数接收 GAConfig 和可选 greedy_solver。
2. solve(state) 的返回格式必须与 greedy_mmce.py 当前 solver 返回格式兼容。
3. solve 内部流程：
   - extract_active_order_ids
   - extract_drone_ids
   - make_gene_pool
   - 可选 greedy seed
   - initialize_population
   - evaluate
   - generations 循环
   - elite
   - tournament selection
   - OX crossover
   - mutation
   - 返回 best.decoded_plan
4. decode 必须由 GADecoder 完成。
5. 不要在 solve 中直接修改真实 state。
6. solve 支持可选 warm_start 参数。
```

---

# 12. Step 10：接入 decision_engine.py

## 12.1 目标

让系统可以通过配置选择：

```yaml
solver: greedy_mmce
```

或：

```yaml
solver: ga_mmce
```

## 12.2 修改文件

```text
backend/solver/decision_engine.py
backend/config/loader.py
```

可能还需要修改：

```text
backend/config/*.yaml
```

## 12.3 给 Codex 的提示词

```text
请修改 decision_engine.py 和配置加载逻辑，使系统支持 solver = "ga_mmce"。

要求：
1. 原有 greedy_mmce 行为保持不变。
2. 当配置 solver.name 或 solver.type 为 "ga_mmce" 时：
   - 创建 GAMMCESolver
   - 传入 GAConfig
   - 如有必要，同时创建 greedy solver 作为 seed 生成器
3. GA solver 的输出必须沿用现有前端可识别的计划结构。
4. 不要修改前端。
5. 如果当前配置文件没有 solver 字段，请添加默认值，默认仍使用 greedy_mmce。
```

---

# 13. Step 11：实现动态重调度模块

## 13.1 目标

动态订单到达时，调用：

```python
reschedule_on_event(...)
```

它需要完成：

```text
1. 读取当前时刻
2. 冻结已完成订单
3. 冻结正在执行的订单
4. 合并剩余订单 + 新订单
5. 从当前实体状态重新 GA
6. 返回新计划
```

## 13.2 新增文件

```text
backend/solver/ga_mmce/dynamic_rescheduler.py
```

## 13.3 推荐代码框架

```python
import copy

from .solver import GAMMCESolver


class DynamicGARescheduler:
    def __init__(self, ga_solver: GAMMCESolver):
        self.ga_solver = ga_solver
        self.previous_best_individual = None

    def reschedule_on_event(self, state, new_orders, event_time):
        """
        事件触发式重调度。

        state:
            当前真实环境状态

        new_orders:
            新接入订单

        event_time:
            动态订单到达时刻
        """

        # 1. 推进或读取当前状态
        snapshot = self._snapshot_state(state, event_time)

        # 2. 标记已完成和执行中订单
        completed_order_ids = self._get_completed_orders(snapshot)
        locked_order_ids = self._get_locked_orders(snapshot)

        # 3. 注入新订单
        self._add_new_orders(snapshot, new_orders)

        # 4. 将 completed / locked 从 GA active set 中排除
        self._mark_orders_for_replanning(
            snapshot,
            completed_order_ids,
            locked_order_ids,
        )

        # 5. 重新 GA
        new_plan = self.ga_solver.solve(snapshot)

        return new_plan

    def _snapshot_state(self, state, event_time):
        snapshot = copy.deepcopy(state)
        # 如果已有仿真器推进函数，应调用已有函数
        # snapshot.advance_to(event_time)
        return snapshot

    def _get_completed_orders(self, snapshot):
        # 根据项目真实 order.status 判断
        return []

    def _get_locked_orders(self, snapshot):
        # 正在飞行、正在服务、正在换电、卡车正在执行的订单
        return []

    def _add_new_orders(self, snapshot, new_orders):
        # 根据项目现有订单管理器插入
        pass

    def _mark_orders_for_replanning(self, snapshot, completed, locked):
        # 将 completed 和 locked 排除出 active orders
        pass
```

## 13.4 给 Codex 的提示词

```text
请新增 backend/solver/ga_mmce/dynamic_rescheduler.py。

实现 DynamicGARescheduler。

要求：
1. reschedule_on_event(state, new_orders, event_time) 是主入口。
2. 不要中断正在执行的物理动作。
3. completed 订单不再进入 GA。
4. locked 订单不再进入 GA，但对应实体在 locked action 完成前不可用。
5. 新订单加入当前订单池。
6. 最后调用 GAMMCESolver.solve(snapshot) 得到新计划。
7. 如果项目已有事件系统或状态推进函数，请复用，不要新写一套仿真器。
```

---

# 14. Step 12：实现 warm start

## 14.1 目标

动态重调度时，不要完全随机初始化。应把上一轮最优个体的剩余订单作为种子。

## 14.2 修改文件

```text
backend/solver/ga_mmce/solver.py
backend/solver/ga_mmce/dynamic_rescheduler.py
```

## 14.3 新增函数

```python
def build_warm_start(previous_best, completed_ids, locked_ids, new_order_ids, gene_pool):
    excluded = set(completed_ids) | set(locked_ids)

    seq = []
    assignment = []

    for oid, gene in zip(previous_best.sequence, previous_best.assignment):
        if oid not in excluded:
            seq.append(oid)
            assignment.append(gene)

    for oid in new_order_ids:
        # 默认给动态订单一个较合理初值：优先 C 或 B
        seq.append(oid)
        c_genes = [g for g in gene_pool if g.startswith("C_")]
        assignment.append(c_genes[0] if c_genes else "A")

    return Individual(seq, assignment)
```

## 14.4 给 Codex 的提示词

```text
请为 GA 动态重调度增加 warm start。

要求：
1. GAMMCESolver.solve 支持可选参数 warm_start: list[Individual] | None。
2. initialize_population 时优先放入 warm_start 个体。
3. DynamicGARescheduler 保存上一轮最佳 Individual。
4. 新订单到达时：
   - 从 previous_best 中删除 completed 和 locked 订单
   - 插入 new_orders
   - 生成 warm_start
5. 若 previous_best 不存在，则正常随机初始化。
```

---

# 15. Step 13：添加单元测试

## 15.1 目标

先测试 GA 自身，不要直接跑大仿真。

## 15.2 新增测试文件

```text
tests/solver/test_ga_chromosome.py
tests/solver/test_ga_operators.py
tests/solver/test_ga_population.py
tests/solver/test_ga_decoder_smoke.py
tests/solver/test_ga_dynamic_rescheduler.py
```

## 15.3 测试重点

### 15.3.1 染色体合法性

```text
- sequence 和 assignment 长度不一致应报错
- sequence 有重复订单应报错
```

### 15.3.2 OX 交叉

```text
- 子代订单不重复
- 子代订单集合等于父代订单集合
- assignment 长度一致
```

### 15.3.3 变异

```text
- 不丢订单
- 不增加订单
- gene 必须来自 gene_pool
```

### 15.3.4 Decoder smoke test

```text
- 用 3 个订单、1 辆卡车、1 架无人机、1 个换电站
- GA 能返回一个非空计划
- 所有订单最多服务一次
```

### 15.3.5 动态重调度

```text
- 已完成订单不会再次进入 active orders
- locked 订单不会被重新分配
- 新订单会进入 active orders
```

## 15.4 给 Codex 的提示词

```text
请为 ga_mmce 添加单元测试。

优先测试：
1. Individual.validate
2. order_crossover
3. mutate
4. initialize_population
5. GAMMCESolver 的 smoke test
6. DynamicGARescheduler 的 completed/locked/new orders 逻辑

测试中如果真实实体构造复杂，可以先使用项目已有 fixture；如果没有 fixture，请创建最小 mock state。
```

---

# 16. Step 14：添加运行日志和调试信息

## 16.1 目标

GA 很容易“看起来在跑，但不知道为什么解不好”。必须加日志。

## 16.2 推荐日志

每 10 代输出：

```text
generation
best_fitness
avg_fitness
best_penalties
feasible_count
```

## 16.3 修改文件

```text
backend/solver/ga_mmce/solver.py
```

## 16.4 给 Codex 的提示词

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
3. 日志开关由 config 控制，例如 GAConfig.verbose。
4. 不要使用 print。
```

---

# 17. 第一次集成验收

完成上述步骤后，运行这些测试：

```bash
pytest tests/solver/test_ga_chromosome.py
pytest tests/solver/test_ga_operators.py
pytest tests/solver/test_ga_population.py
pytest tests/solver/test_ga_decoder_smoke.py
pytest tests/solver/test_ga_dynamic_rescheduler.py
```

然后跑一次小规模场景：

```text
订单数：5
无人机数：1 或 2
换电站：1
动态订单：先不启用
solver：ga_mmce
```

验收标准：

```text
1. 后端不报错。
2. 前端能显示路线。
3. 每个订单只出现一次。
4. GA 结果不一定优于 greedy，但必须可行。
5. 若不可行，penalty 要能解释原因。
```

---

# 18. 推荐给 Codex 的总代码任务说明

可以把下面这整段直接发给 Codex：

```text
我要在现有项目中新增一个遗传算法求解器，不要重写环境、实体、前端。

背景：
- 现有 backend/solver/greedy_mmce.py 已经实现多模式贪心调度，包括模式 A、B、C、B_WAIT 的候选评估、前瞻能量校验、回收点选择、卡车路径构建和评分。
- 现有 decision_engine.py 负责调用 solver。
- 现有实体包括 drone.py、truck.py、swap_station.py、primitives.py。
- 前端已经能展示现有 solver 输出，所以 GA 输出必须兼容现有计划结构。

请按以下步骤实现：

1. 新增 backend/solver/ga_mmce/ 包。
2. 新增 config.py，实现 GAConfig。
3. 新增 chromosome.py，实现 Individual 和 make_gene_pool。
4. 新增 operators.py，实现 OX 交叉、变异、锦标赛选择。
5. 新增 population.py，实现随机种群、纯卡车种子、OBL 种子、greedy seed 注入。
6. 新增 adapters.py，负责从现有 state 中提取 active orders、drone ids，并把 greedy plan 转 Individual。
7. 新增 decoder.py，实现 GADecoder。decoder 必须复用 greedy_mmce.py 中已有模式评估逻辑，不要复制一份物理规则。
8. 如 greedy_mmce.py 当前没有公共候选评估函数，请先重构 greedy_mmce.py，抽取 evaluate_candidate 和 apply_candidate，同时保持 greedy 原行为不变。
9. 新增 fitness.py，统一计算适应度。
10. 新增 solver.py，实现 GAMMCESolver.solve(state)，返回格式必须兼容现有前端。
11. 修改 decision_engine.py，使配置 solver = "ga_mmce" 时调用 GAMMCESolver，默认仍保持 greedy_mmce。
12. 新增 dynamic_rescheduler.py，实现新订单到达时的事件触发式重调度。
13. 添加 tests/solver 下的单元测试。
14. 每一步完成后运行测试，确保 greedy 原功能不被破坏。

重要约束：
- 不要修改前端。
- 不要重写实体状态机。
- 不要复制一套能耗计算。
- GA 解码必须在 state 副本上执行，不能污染真实环境。
- 每个订单必须且只能服务一次。
- 无人机携货阶段不允许中途换电。
- 模式 B 的回收点只能是仓库或充换电站，不能是客户点。
- 动态重调度不能中断正在执行的飞行或服务动作。
```

---

# 19. 最小可运行版本 MVP

如果想尽快看到 GA 跑起来，可以让 Codex 先实现 MVP。

```text
MVP 1：
    chromosome.py
    operators.py
    population.py
    solver.py 框架

MVP 2：
    decoder.py 暂时调用 greedy 的整体 solve
    用 GA sequence 控制订单顺序
    assignment 暂时只参与惩罚或简单模式选择

MVP 3：
    decoder.py 改为逐订单调用 evaluate_candidate

MVP 4：
    接入 decision_engine.py

MVP 5：
    加 dynamic_rescheduler.py
```

这样不会一开始就卡在大规模重构上。

---

# 20. 最终调用链

静态调度调用链：

```text
decision_engine.py
    ↓
GAMMCESolver.solve(state)
    ↓
initialize_population()
    ↓
for each generation:
    ↓
GADecoder.decode(individual, state_copy)
    ↓
greedy_mmce.evaluate_candidate()
    ↓
greedy_mmce.apply_candidate()
    ↓
fitness.compute_fitness()
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
build warm start
    ↓
GAMMCESolver.solve(snapshot)
    ↓
new plan
```

---

# 21. 推荐 commit 顺序

```text
commit 1: add GAConfig and Individual
commit 2: add GA operators and population init
commit 3: add adapters
commit 4: refactor greedy candidate functions
commit 5: add GADecoder
commit 6: add GAMMCESolver
commit 7: integrate decision_engine
commit 8: add tests
commit 9: add dynamic rescheduler
commit 10: add warm start and logging
```

最重要的原则：

```text
先让静态 GA 返回和 greedy 一样的数据结构，再做动态重调度。
```

这样前端和实体系统都不用大改，风险最低。

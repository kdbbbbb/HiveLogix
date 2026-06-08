# Decision Engine 复用接入说明

本文档面向“新算法接入者”，说明如何让 ALNS/GA/DRL 等算法复用现有 `DispatchDecisionEngine` 的执行与展示链路。

## 1. 目标

`DispatchDecisionEngine` 负责“执行层”：

1. 调用求解器得到 `DispatchPlan`
2. 更新订单状态机（pending/assigned/...）
3. 下发卡车/无人机路径
4. 输出前端可视化所需数据

因此新算法只需要专注“求解层”，返回标准 `DispatchPlan` 即可。

## 2. 你需要实现什么

新算法类需要实现与 `DispatchSolver` 一致的方法签名：

```python
def dispatch(
    self,
    pending_orders: dict[str, Order],
    current_time: float,
    bbox: dict,
    scene_id: str | None = None,
) -> DispatchPlan:
    ...
```

建议参考：

- `backend/solver/interfaces.py`
- `backend/solver/greedy_mmce.py`

## 3. DispatchPlan 最小契约

`DispatchDecisionEngine` 依赖以下字段：

1. `allocations`: `AllocationResult` 列表
2. `truck_routes`: `dict[str, TruckRoute]`
3. `summary`: 至少包含 `total_orders`、`feasible`、`modes`
4. `cost_total`: 总成本

`AllocationResult` 常用字段（建议完整填写）：

1. `order_id`, `vehicle_id`, `mode`, `distance`, `feasible`, `reason`
2. 无人机场景：`drone_id`, `recovery_station_id`
3. `B_WAIT` 额外：`launch_station_id`, `launch_time`, `wait_duration`
4. 评分分解：`score_total`, `cost_dist`, `cost_energy`, `cost_penalty`

## 4. 注册新算法

在系统启动阶段（例如 app 初始化或模块导入时）注册：

```python
from solver.factory import register_solver
from solver.algorithms.alns_solver import ALNSSolver

register_solver("alns", lambda entity_mgr: ALNSSolver(entity_mgr))
```

## 5. 运行时切换算法

### 5.1 代码内切换

```python
engine = DispatchDecisionEngine(entity_mgr, order_mgr)
engine.set_solver("alns")
```

### 5.2 API 切换

`POST /api/sim/dispatch` 请求体支持：

```json
{
  "solver": "alns",
  "bbox": {"minx": 0, "miny": 0, "maxx": 1, "maxy": 1},
  "scene_id": "default_test_4x4km"
}
```

若算法名未注册，会返回 400 并附带 `available_solvers`。

## 6. 推荐目录结构

```text
backend/solver/
  decision_engine.py
  interfaces.py
  factory.py
  greedy_mmce.py
  algorithms/
    alns_solver.py
    ga_solver.py
    drl_solver.py
```

## 7. 接入检查清单

1. 新算法已实现 `dispatch(...) -> DispatchPlan`
2. 已通过 `register_solver("name", builder)` 注册
3. `summary` 已返回 `modes` 和成本信息
4. `B_WAIT` 场景字段完整（如使用）
5. `/api/sim/dispatch` 指定 `solver` 可成功切换

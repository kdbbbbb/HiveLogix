# Phase 7 Run04 Postmortem And Run05 Adjustments

## 背景

`phase7_20260429_run04` 暴露出的一个核心问题是：

- `benchmark` 评估在当前实现中是单条 deterministic replay；
- 它从 `update=10` 到 `update=140` 基本不变；
- 但旧版 early-stop 仍把 `benchmark_no_improve_evals` 作为主要 patience 信号之一；
- 同时 best checkpoint selection 也把 benchmark 的 `mean_episode_end_t_sec` / `mean_total_reward` 纳入排序。

这带来两个直接后果：

1. `benchmark` 明明缺少区分度，却会持续推动 `benchmark_no_improve_evals` 增长；
2. checkpoint selection 的主要目标本应是 `stochastic_high / stochastic_medium` 泛化表现，但旧逻辑里 benchmark 的排序信息混入过深，语义不够干净。

Run05 的目标是把 benchmark 降级为 guardrail：

- 用它约束 checkpoint 不能明显破坏固定 benchmark；
- 但不再让它主导 patience，也不再让它在 guardrail 通过后继续参与主排序。

## 本次修改

### 1. 重写 checkpoint selection 语义

修改文件：

- `backend/training/train_cmrappo.py`

新增：

- `_build_benchmark_guardrail_key()`

语义：

- benchmark 现在只比较两个 guardrail 指标：
  - `sum_timeout_order_count`
  - `all_orders_cleared` 次数

即：

- 优先无 timeout；
- 优先完整清空订单；
- 一旦 guardrail 一样，benchmark 的 `mean_total_reward` 和 `mean_episode_end_t_sec` 不再参与 best checkpoint 主排序。

### 2. 调整 `_build_eval_selection_key()`

旧版逻辑：

- benchmark timeout
- benchmark all_cleared
- stochastic_high timeout
- stochastic_high reward
- stochastic_medium reward
- benchmark episode_end_t
- benchmark total_reward
- stochastic_high fallback

新版逻辑：

- benchmark guardrail
- stochastic_high timeout
- stochastic_high reward
- stochastic_high fallback
- stochastic_medium reward
- stochastic_medium timeout
- stochastic_medium fallback

也就是说：

- benchmark 只负责“能不能进主比较”；
- 真正决定 best checkpoint 的主体，改成 `stochastic_high`，其次 `stochastic_medium`。

### 3. 删除 benchmark patience 参与 early-stop

修改文件：

- `backend/training/train_cmrappo.py`

修改点：

- 移除 `best_benchmark_key`
- 移除 `benchmark_no_improve_evals`
- `_should_stop_early()` 不再接收 `benchmark_no_improve_evals`
- stop reason 中不再记录 `benchmark_no_improve`

新版 early-stop 只依赖：

- `early_stop_min_evals`
- `stochastic_high_no_improve_evals`
- `recent_eval_value_losses`

这意味着：

- 只有当 `stochastic_high` 长时间没有新高，且 critic 的 `value_loss` 也没有出现新低时，才会提前停训；
- benchmark 不再因为“天然恒定”而虚假推动 early-stop。

### 4. 更新训练配置语义

修改文件：

- `backend/config/rh_alns_cmrappo.yaml`

修改内容：

- 删除 `early_stop_benchmark_patience`
- 更新注释，明确：
  - best checkpoint 以 `stochastic_high / stochastic_medium` 为主目标
  - benchmark 仅作 guardrail
  - early-stop 只由 `stochastic_high` patience + `value_loss` 窗口共同决定

### 5. 更新单元测试

修改文件：

- `backend/training/test_phase7_model_runtime.py`

调整内容：

- 新测试验证：
  - benchmark guardrail 相同的情况下，selection key 由 stochastic 指标主导；
  - 即使 stochastic reward 更高，只要 benchmark guardrail 变差，也不能赢过基线；
  - early-stop 只看 `stochastic_high_no_improve_evals` 与 `value_loss`，不再看 benchmark patience。

## 修改原因

### 原因 1：benchmark 当前不是高信息量指标

当前 benchmark 是：

- deterministic replay
- 单条 episode
- 在多数 update 上几乎不变

这类信号适合做回归保护，不适合做 patience 主驱动。

如果继续拿它做 patience：

- 会把“评估集无区分度”误判成“训练长时间无改进”；
- 提前停训的触发条件会被 benchmark 虚假满足。

### 原因 2：best checkpoint 应该面向泛化负载

当前训练输入是 poisson，泛化能力主要体现在：

- `stochastic_high`
- `stochastic_medium`

因此 best checkpoint selection 应该优先看：

- 高强度下是否更稳；
- 中强度下是否没有明显退化；

而 benchmark 更适合作为：

- “不能破坏固定基准场景”的硬约束。

### 原因 3：checkpoint selection 与 early-stop 需要职责分离

这次调整后，两者职责更清晰：

- checkpoint selection：
  - benchmark 负责 guardrail
  - stochastic 负责主排序

- early-stop：
  - 只判断 stochastic 主目标是否停滞
  - 再结合 critic `value_loss` 是否还在创新低

这样可以避免：

- 固定 benchmark 同时污染“挑 best checkpoint”和“决定是否停训”两条链路。

## 影响范围

受影响文件：

- `backend/training/train_cmrappo.py`
- `backend/config/rh_alns_cmrappo.yaml`
- `backend/training/test_phase7_model_runtime.py`

## 验证结果

已完成的验证：

1. `python -m unittest backend.training.test_phase7_model_runtime`
   - 通过

本次验证重点覆盖：

- selection key 新语义
- early-stop 新语义
- 与现有 Phase 7 runtime 测试集的兼容性

## 对 Run05 的预期影响

预期 Run05 相比 Run04 会有以下变化：

1. 更不容易因为 benchmark 恒定而触发伪早停。
2. `policy_best.pt` 更可能对应真正的 stochastic 泛化最优点，而不是被 benchmark 次要排序项干扰。
3. postmortem 时看到的 early-stop reason 会更直接反映高强度随机评估是否停滞。

## 备注

`eval total_reward` 闭环修复已在单独文档中记录：

- `docs/ppo算法方案/phase7_eval_total_reward_closure_fix_20260429.md`

该修复与本次 early-stop / checkpoint selection 重写相互独立，但建议在 Run05 前一并采用。

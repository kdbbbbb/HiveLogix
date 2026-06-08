from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass
class GAConfig:
    population_size: int = 120
    generations: int = 100
    min_generations: int = 80
    early_stopping_patience: int = 60
    improvement_tolerance: float = 1e-3
    elite_ratio: float = 0.08
    tournament_k: int = 3

    crossover_rate: float = 0.9
    mutation_rate_sequence: float = 0.2
    mutation_rate_assignment: float = 0.15
    mutation_rate_rendezvous: float = 0.20

    use_greedy_seed: bool = True
    use_truck_only_seed: bool = True
    use_obl_seed: bool = True
    use_balanced_initialization: bool = True
    use_b_seeded_initialization: bool = True
    use_warm_start: bool = True
    warm_start_mutations: int = 0
    reopt_window_size: int | None = None
    reopt_horizon_seconds: float | None = None

    # GA 动态委托专用：周期性把临期且尚未开始配送的订单改为仓库无人机直递。
    urgent_rescue_enabled: bool = True
    urgent_rescue_check_interval_s: float = 30.0
    urgent_rescue_deadline_window_s: float = 240.0
    urgent_rescue_max_orders_per_check: int = 6

    mutation_mode_prob_a: float = 0.40
    mutation_mode_prob_b: float = 0.35
    mutation_mode_prob_c: float = 0.25

    enable_repair: bool = True
    repair_penalty_factor: float = 1000.0

    big_m: float = 1e9

    # 计划级完成时间权重：使用本轮所有订单的最大完成时间，而不是逐单累加。
    weight_completion: float = 0.5
    weight_delay: float = 80.0
    # 距离已经通过能耗模型间接体现，目标函数中不再重复计入距离成本。
    weight_energy: float = 0.02
    weight_waiting: float = 0.5
    weight_infeasible: float = 100000.0
    weight_truck_distance: float = 0.0
    weight_uav_distance: float = 0.0

    # 鼓励空地协同：每接受一个 B 模式订单，从目标函数中扣减该奖励值。
    air_ground_mode_reward: float = 400.0

    # Static GA future-friendliness terms. These only affect the initial full
    # GA plan, nudging the truck backbone toward later dynamic work that will be
    # inserted by the greedy incremental solver.
    future_preview_enabled: bool = True
    future_preview_max_orders: int = 40
    future_terminal_position_weight: float = 0.02
    future_route_proximity_reward: float = 30.0
    future_route_proximity_radius_m: float = 900.0
    future_station_coverage_reward: float = 90.0
    future_station_coverage_radius_m: float = 1200.0

    # 初始静态 GA 中同一架无人机重复承接多单的软惩罚；
    # 只改变初始全局计划的分布偏好，不禁止后续回收后继续复用无人机。
    drone_reuse_penalty_factor: float = 1200.0

    max_runtime_seconds: float | None = None
    random_seed: int | None = 42
    verbose: bool = False
    log_interval: int = 10
    diagnostics_enabled: bool = True
    diagnostics_dir: str = "logs"
    save_evolution_csv: bool = True
    save_evolution_plots: bool = True
    b_candidate_precheck: bool = True
    diagnostics_label: str = "static"

    # 是否允许 C 模式无人机送完后落在充换电站；若 False，则 C 必须回仓。
    allow_depot_drone_recover_at_station: bool = True

    # 若卡车与无人机回收同步失败，是否作为软惩罚而不是直接判死。
    soft_rendezvous_violation: bool = True

    truck_wait_max_s: float = 60.0

    # GA 初始静态调度的虚拟装载状态：卡车出仓时携带 1-8、11 号无人机，
    # 仓库保留 9、10、12 号无人机。该设置只影响 GA 内部 _ga_* 状态。
    initial_drone_layout_enabled: bool = True
    initial_drone_layout_max_time_s: float = 1e-6
    initial_truck_drone_ids: tuple[str, ...] = (
        "UAV-TEST-01",
        "UAV-TEST-02",
        "UAV-TEST-03",
        "UAV-TEST-08",
        "UAV-TEST-10",
        "UAV-TEST-11",
    )
    initial_depot_drone_ids: tuple[str, ...] = (
        "UAV-TEST-09",
        "UAV-TEST-04",
        "UAV-TEST-05",
        "UAV-TEST-06",
        "UAV-TEST-07",
        "UAV-TEST-12",
    )


STATIC_GA_CONFIG: dict[str, Any] = {
    "population_size": 120,
    "max_generations": 100,
    "min_generations": 80,
    "early_stopping_patience": 60,
    "improvement_tolerance": 1e-3,
    "log_interval": 10,
    "enable_csv": True,
    "enable_png": True,
    "use_warm_start": True,
    "diagnostics_label": "static",
}


DYNAMIC_GA_CONFIG: dict[str, Any] = {
    "population_size": 30,
    "max_generations": 60,
    "min_generations": 15,
    "early_stopping_patience": 10,
    "improvement_tolerance": 1e-6,
    "log_interval": 5,
    "time_budget_seconds": 3.0,
    "enable_csv": True,
    "enable_png": False,
    "use_warm_start": True,
    "warm_start_mutations": 10,
    "reopt_window_size": 8,
    "diagnostics_label": "dynamic",
}


_CONFIG_ALIASES = {
    "max_generations": "generations",
    "time_budget_seconds": "max_runtime_seconds",
    "enable_csv": "save_evolution_csv",
    "enable_png": "save_evolution_plots",
}


def make_ga_config(
    overrides: GAConfig | dict[str, Any] | None = None,
    base: GAConfig | None = None,
) -> GAConfig:
    """Return a GAConfig, accepting public dict aliases used by API callers."""
    if overrides is None:
        return replace(base) if base is not None else GAConfig()
    if isinstance(overrides, GAConfig):
        return replace(overrides)

    values = asdict(base) if base is not None else asdict(GAConfig())
    valid_fields = set(values)
    for raw_key, value in dict(overrides).items():
        key = _CONFIG_ALIASES.get(str(raw_key), str(raw_key))
        if key in valid_fields:
            values[key] = value
    return GAConfig(**values)

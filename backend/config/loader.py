#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — 配置文件加载器

职责：
  - 以 YAML 格式加载 config/ 目录下的参数文件
  - 提供强类型化的数据结构，避免调用方直接操作裸字典
  - 首次加载后缓存结果，重复调用无 I/O 开销

使用方式：
    from config.loader import load_drone_params, load_solver_energy_params
    params = load_drone_params()
    print(params.light.k1)
    print(load_solver_energy_params().truck_energy_kwh_per_km)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ── 配置文件路径 ──────────────────────────────────────────────────────────────
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════════════════
# 强类型配置数据类
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DroneTypeParams:
    """单一无人机型号的全参数集合（不可变，确保运行期不被意外修改）。"""

    # 气动功率参数
    k1: float                 # 诱导功率系数 [W / kg^1.5]
    k2: float                 # 废阻功率系数 [W / (m/s)^3]

    # 飞行性能
    cruise_speed: float       # 巡航速度 [m/s]
    payload_capacity: float   # 最大载重 [kg]
    empty_weight: float       # 机体自重 [kg]

    # 能量
    battery_capacity_j: float # 满电电量 [J]
    safe_margin_ratio: float  # 安全余量比例 [0-1]

    @property
    def safe_margin_j(self) -> float:
        """换算为绝对安全余量（焦耳）。"""
        return self.battery_capacity_j * self.safe_margin_ratio


@dataclass(frozen=True)
class DroneParamsConfig:
    """承载所有无人机型号参数的顶层数据类。"""

    light: DroneTypeParams
    heavy: DroneTypeParams
    solver_energy: "SolverEnergyParams"


@dataclass(frozen=True)
class SolverEnergyParams:
    """求解器统一参数（评分 + 业务时长）。"""

    c_dist_et: float
    c_dist_uav: float
    c_energy_et: float
    c_energy_uav: float
    lambda_time: float

    # 能耗模型参数
    truck_energy_kwh_per_km: float
    uav_energy_model: str
    uav_alpha_wh_per_kg_km: float
    allow_moving_truck_launch: bool

    # 业务时长参数 [s]
    truck_service_time_order_s: float
    drone_service_time_order_s: float
    truck_drone_launch_time_s: float
    truck_drone_recover_time_s: float

    @property
    def truck_energy_wh_per_meter(self) -> float:
        """kWh/km -> Wh/m。数值相同（例如 0.75 kWh/km = 0.75 Wh/m）。"""
        return self.truck_energy_kwh_per_km


# ══════════════════════════════════════════════════════════════════════════════
# 加载逻辑
# ══════════════════════════════════════════════════════════════════════════════

def _load_yaml(filepath: str) -> dict[str, Any]:
    """
    读取 YAML 文件，返回原始字典。

    依赖 PyYAML（`pip install pyyaml`）。若未安装则抛出清晰的错误提示。
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "缺少 PyYAML 依赖，请执行：pip install pyyaml"
        ) from exc

    with open(filepath, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"配置文件格式错误，期望顶层为 mapping: {filepath}")

    return data


def _parse_drone_type(raw: dict[str, Any], label: str) -> DroneTypeParams:
    """从原始字典构造 DroneTypeParams，并校验必要字段。"""
    required = {
        "k1", "k2", "cruise_speed", "payload_capacity",
        "empty_weight", "battery_capacity_j", "safe_margin_ratio",
    }
    missing = required - raw.keys()
    if missing:
        raise KeyError(
            f"drone_params.yaml 中 [{label}] 缺少必要字段: {sorted(missing)}"
        )

    params = DroneTypeParams(
        k1=float(raw["k1"]),
        k2=float(raw["k2"]),
        cruise_speed=float(raw["cruise_speed"]),
        payload_capacity=float(raw["payload_capacity"]),
        empty_weight=float(raw["empty_weight"]),
        battery_capacity_j=float(raw["battery_capacity_j"]),
        safe_margin_ratio=float(raw["safe_margin_ratio"]),
    )

    # 基础合理性校验
    if not (0.0 < params.safe_margin_ratio < 1.0):
        raise ValueError(
            f"[{label}] safe_margin_ratio 必须在 (0, 1) 范围内，"
            f"当前值: {params.safe_margin_ratio}"
        )
    if params.cruise_speed <= 0:
        raise ValueError(f"[{label}] cruise_speed 必须为正数")
    if params.battery_capacity_j <= 0:
        raise ValueError(f"[{label}] battery_capacity_j 必须为正数")

    return params


def _parse_solver_energy(raw: dict[str, Any]) -> SolverEnergyParams:
    """
    解析求解器评分参数。

    为兼容历史配置，允许缺省并使用默认值。
    """
    if not isinstance(raw, dict):
        raw = {}

    params = SolverEnergyParams(
        c_dist_et=float(raw.get("c_dist_et", 1.0)),
        c_dist_uav=float(raw.get("c_dist_uav", 1.0)),
        c_energy_et=float(raw.get("c_energy_et", 1.0)),
        c_energy_uav=float(raw.get("c_energy_uav", 1.0)),
        lambda_time=float(raw.get("lambda_time", 1.0)),
        truck_energy_kwh_per_km=float(raw.get("truck_energy_kwh_per_km", 0.75)),
        uav_energy_model=str(raw.get("uav_energy_model", "physics")).strip().lower(),
        uav_alpha_wh_per_kg_km=float(raw.get("uav_alpha_wh_per_kg_km", 0.24)),
        allow_moving_truck_launch=bool(raw.get("allow_moving_truck_launch", False)),
        truck_service_time_order_s=float(raw.get("truck_service_time_order_s", 60.0)),
        drone_service_time_order_s=float(raw.get("drone_service_time_order_s", 30.0)),
        truck_drone_launch_time_s=float(raw.get("truck_drone_launch_time_s", 10.0)),
        truck_drone_recover_time_s=float(raw.get("truck_drone_recover_time_s", 10.0)),
    )

    if params.truck_energy_kwh_per_km <= 0:
        raise ValueError("[solver_energy] truck_energy_kwh_per_km 必须为正数")
    if params.uav_energy_model not in {"physics", "alpha"}:
        raise ValueError("[solver_energy] uav_energy_model 仅支持 physics 或 alpha")
    if params.uav_alpha_wh_per_kg_km <= 0:
        raise ValueError("[solver_energy] uav_alpha_wh_per_kg_km 必须为正数")
    if params.lambda_time < 0:
        raise ValueError("[solver_energy] lambda_time 不能为负数")
    if params.truck_service_time_order_s < 0:
        raise ValueError("[solver_energy] truck_service_time_order_s 不能为负数")
    if params.drone_service_time_order_s < 0:
        raise ValueError("[solver_energy] drone_service_time_order_s 不能为负数")
    if params.truck_drone_launch_time_s < 0:
        raise ValueError("[solver_energy] truck_drone_launch_time_s 不能为负数")
    if params.truck_drone_recover_time_s < 0:
        raise ValueError("[solver_energy] truck_drone_recover_time_s 不能为负数")

    return params


@lru_cache(maxsize=1)
def load_drone_params() -> DroneParamsConfig:
    """
    加载并缓存无人机气动参数。

    线程安全：lru_cache 在 CPython 中对单个返回对象是原子化的；
    config 对象使用 frozen=True dataclass，不存在竞态修改风险。

    Returns:
        DroneParamsConfig — 包含 light / heavy 两个型号的全参数集合

    Raises:
        FileNotFoundError: drone_params.yaml 不存在
        KeyError: 必要字段缺失
        ValueError: 字段值不合法
    """
    filepath = os.path.join(_CONFIG_DIR, "drone_params.yaml")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"无人机参数配置文件未找到: {filepath}"
        )

    raw = _load_yaml(filepath)
    logger.info("已加载无人机气动参数配置: %s", filepath)

    return DroneParamsConfig(
        light=_parse_drone_type(raw.get("light_drone", {}), "light_drone"),
        heavy=_parse_drone_type(raw.get("heavy_drone", {}), "heavy_drone"),
        solver_energy=_parse_solver_energy(raw.get("solver_energy", {})),
    )


@lru_cache(maxsize=1)
def load_solver_energy_params() -> SolverEnergyParams:
    """加载并缓存求解器评分参数。"""
    return load_drone_params().solver_energy

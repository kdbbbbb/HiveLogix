#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HiveLogix — PPO 训练过程前端遥测运行时。

该模块不接管训练逻辑，只把 `train_cmrappo()` 已有的 event_hook
桥接到前端遥测 WebSocket。训练仍使用配置默认场景和训练配置中的订单源。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from api.websockets.telemetry import broadcast_tick
from training.frontend_runtime_adapter import TrainingTelemetryBridge
from training.scene_loader import DEFAULT_CONFIG_PATH
from training.train_cmrappo import train_cmrappo


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_SCENE_ID = "default_test_4x4km"
DEFAULT_TRAINING_OUTPUT_ROOT = (
    REPO_ROOT / "backend" / "weights" / "rh_alns_cmrappo"
)


@dataclass(frozen=True)
class FrontendTrainingActivation:
    config_path: Path
    output_dir: Path
    scene_id: str
    render_interval_sec: float
    render_every_n_steps: int


def build_frontend_training_activation(
    payload: Mapping[str, Any],
) -> FrontendTrainingActivation:
    config_raw = payload.get("config_path", str(DEFAULT_CONFIG_PATH))
    if not isinstance(config_raw, str) or not config_raw.strip():
        raise ValueError("config_path 不能为空")

    scene_id = str(
        payload.get("scene_id", DEFAULT_TRAINING_SCENE_ID)
    ).strip() or DEFAULT_TRAINING_SCENE_ID
    if scene_id != DEFAULT_TRAINING_SCENE_ID:
        raise ValueError(
            "第一版 PPO 训练可视化仅支持默认场景: "
            f"requested={scene_id}, supported={DEFAULT_TRAINING_SCENE_ID}"
        )

    output_raw = payload.get("output_dir")
    if isinstance(output_raw, str) and output_raw.strip():
        output_dir = _resolve_output_path(output_raw)
    else:
        output_dir = (
            DEFAULT_TRAINING_OUTPUT_ROOT
            / datetime.now().strftime("frontend_train_%Y%m%d_%H%M%S")
        )

    render_interval_sec = float(payload.get("render_interval_sec", 0.25))
    if render_interval_sec < 0:
        raise ValueError("render_interval_sec 不能为负数")

    render_every_n_steps = int(payload.get("render_every_n_steps", 1))
    if render_every_n_steps < 1:
        raise ValueError("render_every_n_steps 必须 >= 1")

    return FrontendTrainingActivation(
        config_path=_resolve_existing_path(config_raw),
        output_dir=output_dir,
        scene_id=scene_id,
        render_interval_sec=render_interval_sec,
        render_every_n_steps=render_every_n_steps,
    )


def _resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (
        Path.cwd() / path,
        REPO_ROOT / path,
        REPO_ROOT / "backend" / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _resolve_output_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


class FrontendPPOTrainingRuntime:
    """后台运行 `train_cmrappo()`，并把训练 env 快照推送到前端。"""

    def __init__(self, *, activation: FrontendTrainingActivation) -> None:
        self._activation = activation
        self._bridge = TrainingTelemetryBridge(
            policy_name="cmrappo_training",
            checkpoint_path=str(activation.output_dir / "policy.pt"),
        )
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._env: Any | None = None
        self._last_payload: dict[str, Any] | None = None
        self._last_emit_wall = 0.0
        self._running = False
        self._completed = False
        self._error: str | None = None
        self._result: dict[str, Any] | None = None
        self._episode_id: int | None = None
        self._global_step = 0
        self._update_idx = 0
        self._started_at_wall_ms = 0
        self._finished_at_wall_ms: int | None = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def current_time(self) -> float:
        with self._lock:
            if self._env is None:
                return 0.0
            return float(self._env.build_runtime_state_view().t_now)

    @property
    def speed_ratio(self) -> float:
        return 1.0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("PPO 训练已经在运行")
            if self._completed:
                raise RuntimeError("当前 PPO 训练任务已结束，请重新创建 runtime")
            self._running = True
            self._started_at_wall_ms = int(time.time() * 1000)
            self._thread = threading.Thread(
                target=self._run,
                name="FrontendPPOTrainingRuntime",
                daemon=True,
            )
            self._thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": self._running or self._completed or self._error is not None,
                "running": self._running,
                "completed": self._completed,
                "error": self._error,
                "output_dir": str(self._activation.output_dir),
                "config_path": str(self._activation.config_path),
                "scene_id": self._activation.scene_id,
                "episode_id": self._episode_id,
                "global_step": self._global_step,
                "update_idx": self._update_idx,
                "started_at_wall_ms": self._started_at_wall_ms,
                "finished_at_wall_ms": self._finished_at_wall_ms,
                "result": self._result,
            }

    def build_full_snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._env is None:
                return {
                    "type": "FULL_SNAPSHOT",
                    "payload": {
                        "sim_time": 0.0,
                        "is_running": self._running,
                        "speed_ratio": 1.0,
                        "sim_start_wall_ms": self._started_at_wall_ms,
                        "entities": {
                            "depots": [],
                            "stations": [],
                            "trucks": [],
                            "drones": [],
                        },
                        "orders": [],
                        "paths": {"trucks": [], "drones": []},
                        "dispatch_chains": [],
                        "recent_decision_events": [],
                        "latest_event_seq": 0,
                        "stats": self._training_stats(),
                    },
                }
            snapshot = self._bridge.build_full_snapshot(
                env=self._env,
                is_running=self._running,
                speed_ratio=1.0,
                sim_start_wall_ms=self._started_at_wall_ms,
                deterministic=False,
                order_source_mode="poisson",
            )
            snapshot["payload"]["stats"].update(self._training_stats())
            return snapshot

    def _run(self) -> None:
        try:
            result = train_cmrappo(
                output_dir=self._activation.output_dir,
                config_path=self._activation.config_path,
                event_hook=self._event_hook,
            )
            with self._lock:
                self._result = dict(result)
                self._completed = True
        except Exception as exc:
            logger.exception("[frontend_training_runtime] PPO 训练失败")
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._running = False
                self._finished_at_wall_ms = int(time.time() * 1000)
                payload = self._last_payload
            if payload is not None:
                payload = dict(payload)
                payload["stats"] = {
                    **dict(payload.get("stats") or {}),
                    **self._training_stats(),
                }
                broadcast_tick(payload)

    def _event_hook(self, event_name: str, payload: Mapping[str, Any]) -> None:
        env = payload.get("env")
        if env is None:
            return

        global_step = int(payload.get("global_step", self._global_step) or 0)
        update_idx = int(payload.get("update_idx", self._update_idx) or 0)
        episode_id_raw = payload.get("episode_id")
        episode_id = None if episode_id_raw is None else int(episode_id_raw)

        with self._lock:
            self._env = env
            self._global_step = global_step
            self._update_idx = update_idx
            self._episode_id = episode_id

        if not self._should_emit(event_name=event_name, global_step=global_step):
            return

        tick = self._bridge.build_tick_payload(
            env=env,
            speed_ratio=1.0,
            deterministic=False,
            order_source_mode="poisson",
        )
        tick["stats"].update(self._training_stats())
        with self._lock:
            self._last_payload = tick
            self._last_emit_wall = time.monotonic()
        broadcast_tick(tick)

    def _should_emit(self, *, event_name: str, global_step: int) -> bool:
        if event_name in {"train_reset", "train_episode_end"}:
            return True
        if event_name != "train_step":
            return False
        if global_step % self._activation.render_every_n_steps != 0:
            return False
        interval = self._activation.render_interval_sec
        if interval <= 0:
            return True
        return (time.monotonic() - self._last_emit_wall) >= interval

    def _training_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_training": True,
                "training_running": self._running,
                "training_completed": self._completed,
                "training_error": self._error,
                "training_output_dir": str(self._activation.output_dir),
                "training_scene_id": self._activation.scene_id,
                "training_global_step": self._global_step,
                "training_update_idx": self._update_idx,
                "training_episode_id": self._episode_id,
            }

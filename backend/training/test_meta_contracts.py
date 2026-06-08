#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meta.json contract tests.

运行方式：
  python -m unittest backend.training.test_meta_contracts
"""

from __future__ import annotations

from pathlib import Path
import unittest

from .contracts import ActionSpaceMeta, PolicyMeta


class TestMetaContracts(unittest.TestCase):
    def test_default_factorized_head_order_has_no_recovery_head(self) -> None:
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            self.skipTest("缺少 PyYAML，跳过 YAML 契约测试")

        config_path = Path(__file__).resolve().parents[1] / "config" / "rh_alns_cmrappo.yaml"
        with config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        head_order = tuple(raw["action_space"]["factorized_head_order"])
        self.assertEqual(head_order, ("root", "order", "mode"))
        self.assertNotIn("recovery", head_order)

    def test_pool_lstm_v2_policy_meta_does_not_require_nhead(self) -> None:
        meta = PolicyMeta(
            encoder_type="pool_lstm_v2",
            d_model=130,
            ff_dim=256,
            dropout=0.0,
            lstm_hidden=128,
            lstm_layers=1,
            hist_len=6,
            max_order_tokens=32,
            use_plan_version_delta=True,
            use_is_riding_truck_flag=True,
            use_drone_source_type_flag=True,
            critic_mode="centralized_train_only",
            inference_mode="greedy",
        )

        self.assertIsNone(meta.nhead)

    def test_action_space_meta_rejects_recovery_head(self) -> None:
        with self.assertRaises(ValueError):
            ActionSpaceMeta(
                type="factorized",
                factorized_head_order=("root", "order", "mode", "recovery"),
                policy_modes=("WAIT", "B", "C"),
                planner_modes=("A", "B", "C"),
                enable_wait_action=True,
                include_mode_a_in_policy=False,
            )

    def test_attention_policy_meta_still_validates_nhead(self) -> None:
        with self.assertRaises(ValueError):
            PolicyMeta(
                encoder_type="attn_lstm_lite",
                d_model=130,
                nhead=8,
                ff_dim=256,
                dropout=0.0,
                lstm_hidden=128,
                lstm_layers=1,
                hist_len=6,
                max_order_tokens=32,
                use_plan_version_delta=True,
                use_is_riding_truck_flag=True,
                use_drone_source_type_flag=True,
                critic_mode="centralized_train_only",
                inference_mode="greedy",
            )


if __name__ == "__main__":
    unittest.main()

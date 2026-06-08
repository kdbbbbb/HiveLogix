#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run random Mode-B-only batch rollout evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.batch_rollout_eval_common import (
    choose_random_mode_b_actions,
    parse_common_args,
    print_summary,
    run_batch_rollout_eval,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_common_args(
        argv,
        description="Random Mode-B-only batch rollout eval.",
    )
    summary = run_batch_rollout_eval(
        args=args,
        strategy_name="random_mode_b",
        choose_actions=choose_random_mode_b_actions,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()

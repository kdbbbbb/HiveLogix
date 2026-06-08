#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run teacher-only batch rollout evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


from training.batch_rollout_eval_common import (
    choose_teacher_actions,
    parse_common_args,
    print_summary,
    run_batch_rollout_eval,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_common_args(
        argv,
        description="Teacher-only batch rollout eval.",
    )
    summary = run_batch_rollout_eval(
        args=args,
        strategy_name="teacher_only",
        choose_actions=choose_teacher_actions,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()

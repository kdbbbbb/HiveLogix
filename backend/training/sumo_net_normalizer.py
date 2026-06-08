#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMO net.xml 规范化工具。

我们导出的 OSM→SUMO net.xml 是最小可用格式，部分连接关系在 GUI 载入时
会被更严格地校验。这里借助 `netconvert` 做一次无损重写，修正 SUMO 内部
的连接/逻辑索引，避免 GUI 因 network error 直接退出。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def resolve_netconvert(sumo_gui_bin: str) -> str | None:
    sibling = Path(sumo_gui_bin).resolve().with_name("netconvert")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("netconvert")


def sumocfg_net_path(sumocfg: Path) -> Path | None:
    root = ET.parse(sumocfg).getroot()
    for elem in root.iter():
        if elem.tag.endswith("net-file"):
            value = elem.get("value")
            if value:
                return (sumocfg.parent / value).resolve()
    return None


def normalize_net_with_netconvert(*, sumocfg: Path, netconvert_bin: str | None) -> bool:
    if not netconvert_bin:
        return False
    net_path = sumocfg_net_path(sumocfg)
    if not net_path or not net_path.is_file():
        return False

    tmp_path = net_path.with_suffix(".netconvert.tmp.xml")
    result = subprocess.run(
        [
            netconvert_bin,
            "-s",
            str(net_path),
            "-o",
            str(tmp_path),
            "--ignore-errors",
            "--ignore-errors.connections",
        ],
        capture_output=True,
        text=True,
        cwd=str(sumocfg.parent),
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    tmp_path.replace(net_path)
    return True

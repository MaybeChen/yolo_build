"""Runtime dependency loading helpers."""
from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any


def load_yolo() -> Callable[..., Any]:
    """Import and return :class:`ultralytics.YOLO` with actionable errors."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        if exc.name != "ultralytics":
            raise
        raise SystemExit(
            "未找到 Python 包 ultralytics。请确认你正在使用安装依赖的同一个 Python 解释器运行脚本：\n"
            f"  {sys.executable} -m pip install -r requirements.txt\n"
            f"  {sys.executable} scripts/train.py --data configs/data.example.yaml\n"
            "如果你使用虚拟环境，请先激活该环境后再运行脚本。"
        ) from exc
    except ImportError as exc:
        message = str(exc)
        if "libGL.so.1" in message:
            raise SystemExit(
                "ultralytics/opencv 导入失败：缺少系统库 libGL.so.1。\n"
                "Linux/容器环境可安装系统依赖，例如：apt-get update && apt-get install -y libgl1。\n"
                "如果不能安装系统库，可尝试把 opencv-python 替换为 opencv-python-headless。"
            ) from exc
        raise SystemExit(
            "ultralytics 导入失败。依赖可能已安装在其他 Python 环境，或缺少运行时系统库。\n"
            f"当前 Python: {sys.executable}\n"
            f"原始错误: {message}"
        ) from exc
    return YOLO

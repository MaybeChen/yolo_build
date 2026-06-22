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
            "未在当前运行脚本的 Python 环境中找到 ultralytics。\n"
            f"当前运行脚本的 Python 是：{sys.executable}\n"
            "如果这个路径是某个应用或插件自带的 Python（例如 opencode-dependence），"
            "说明脚本不是由你在 PyCharm 中配置的项目解释器启动的。\n"
            "请在 PyCharm 的 Run/Debug Configuration 中确认 Python interpreter 选择的是你的项目解释器，"
            "或先激活虚拟环境后再运行：\n"
            "  python -m pip install -r requirements.txt\n"
            "  python scripts/train.py --data configs/data.example.yaml"
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

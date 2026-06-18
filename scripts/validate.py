#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from typing import Iterable

from ultralytics import YOLO

from yolo_build.visualize import save_annotated_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a YOLO model and save annotated images with predicted boxes."
    )
    parser.add_argument("--weights", required=True, help="Trained YOLO weights, for example best.pt.")
    parser.add_argument("--data", help="Optional dataset YAML. If set, validation metrics are saved.")
    parser.add_argument("--source", required=True, help="Image file, image directory, or glob pattern.")
    parser.add_argument("--output", default="runs/val_annotated", help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--device", default=None, help="Device, for example '0' or 'cpu'.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    return parser.parse_args()


def iter_images(source: str) -> list[Path]:
    source_path = Path(source)
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        return [source_path]
    if source_path.is_dir():
        return sorted(
            path for path in source_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )

    matches = sorted(
        Path(path) for path in glob.glob(source, recursive=True) if Path(path).suffix.lower() in IMAGE_SUFFIXES
    )
    if matches:
        return matches
    raise FileNotFoundError(f"No images found for source: {source}")


def save_metrics(model: YOLO, args: argparse.Namespace, output_dir: Path) -> None:
    if not args.data:
        return

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        project=str(output_dir),
        name="metrics",
    )
    summary = {
        "box_map": float(metrics.box.map),
        "box_map50": float(metrics.box.map50),
        "box_map75": float(metrics.box.map75),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation metrics saved to: {metrics_path}")


def annotate_images(model: YOLO, images: Iterable[Path], args: argparse.Namespace, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    names = model.names
    for image_path in images:
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
        class_ids = [int(value) for value in result.boxes.cls.cpu().tolist()] if result.boxes is not None else []
        confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []

        output_path = images_dir / image_path.name
        save_annotated_image(output_path=output_path, image_path=image_path, boxes=boxes, class_ids=class_ids, confidences=confidences, names=names)
        print(f"Annotated image saved to: {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights_path}")

    model = YOLO(str(weights_path))
    save_metrics(model, args, output_dir)
    annotate_images(model, iter_images(args.source), args, output_dir)


if __name__ == "__main__":
    main()

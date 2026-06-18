from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

Box = Sequence[float]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a stable BGR color for a class id."""
    palette = (
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    )
    return palette[class_id % len(palette)]


def draw_detections(
    image: np.ndarray,
    boxes: Sequence[Box],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    names: Mapping[int, str] | Sequence[str] | None = None,
) -> np.ndarray:
    """Draw detection boxes on an image and return a copy.

    Args:
        image: OpenCV image in BGR format.
        boxes: XYXY boxes in pixel coordinates.
        class_ids: Class id for each box.
        confidences: Confidence score for each box.
        names: Optional mapping/list used to render class names.
    """
    annotated = image.copy()
    height, width = annotated.shape[:2]

    for box, class_id, confidence in zip(boxes, class_ids, confidences, strict=True):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        color = color_for_class(int(class_id))
        label_name = _class_name(names, int(class_id))
        label = f"{label_name} {confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y1 = max(0, y1 - text_height - baseline - 4)
        label_y2 = label_y1 + text_height + baseline + 4
        label_x2 = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def save_annotated_image(
    image_path: str | Path,
    output_path: str | Path,
    boxes: Sequence[Box],
    class_ids: Sequence[int],
    confidences: Sequence[float],
    names: Mapping[int, str] | Sequence[str] | None = None,
) -> Path:
    """Load an image, draw detections, and save it."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    annotated = draw_detections(image, boxes, class_ids, confidences, names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), annotated)
    if not ok:
        raise ValueError(f"Unable to write image: {output_path}")
    return output_path


def _class_name(names: Mapping[int, str] | Sequence[str] | None, class_id: int) -> str:
    if names is None:
        return str(class_id)
    if isinstance(names, Mapping):
        return names.get(class_id, str(class_id))
    if 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)

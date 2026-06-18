from pathlib import Path

import cv2
import numpy as np

from yolo_build.visualize import draw_detections, save_annotated_image


def test_draw_detections_changes_pixels() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    annotated = draw_detections(
        image,
        boxes=[(10, 10, 80, 80)],
        class_ids=[0],
        confidences=[0.95],
        names={0: "object"},
    )

    assert annotated.shape == image.shape
    assert np.any(annotated != image)


def test_save_annotated_image_writes_file(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    output_path = tmp_path / "annotated.jpg"
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    saved_path = save_annotated_image(
        image_path=image_path,
        output_path=output_path,
        boxes=[(5, 5, 40, 40)],
        class_ids=[1],
        confidences=[0.8],
        names={1: "target"},
    )

    assert saved_path == output_path
    assert output_path.exists()
    annotated = cv2.imread(str(output_path))
    assert annotated is not None
    assert np.any(annotated != image)

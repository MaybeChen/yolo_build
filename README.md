# YOLO 训练 + 验证工程

这是一个基于 [Ultralytics YOLO](https://docs.ultralytics.com/) 的最小可用训练、验证与可视化工程。工程支持：

- 使用 YOLO 数据集配置训练目标检测模型。
- 对验证集或指定图片目录运行验证/推理。
- 将检测框、类别名和置信度直接绘制到输入图片上，输出可直观看效果的标注图片。

## 目录结构

```text
.
├── configs/
│   └── data.example.yaml     # YOLO 数据集配置示例
├── scripts/
│   ├── train.py              # 训练入口
│   └── validate.py           # 验证/推理并保存画框结果
├── src/yolo_build/
│   ├── __init__.py
│   └── visualize.py          # 画框工具
├── tests/
│   └── test_visualize.py
├── requirements.txt
└── README.md
```

## 环境安装

建议使用 Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 如需 GPU 训练，请按你的 CUDA 版本安装合适的 PyTorch，再安装本项目依赖。

## 准备数据集

Ultralytics YOLO 检测数据集通常使用如下结构：

```text
datasets/my_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

标签文件为 YOLO 格式，每行：

```text
<class_id> <x_center> <y_center> <width> <height>
```

所有坐标均为相对图片宽高的归一化值。

复制并修改数据集配置：

```bash
cp configs/data.example.yaml configs/data.yaml
```

将 `path`、`train`、`val` 和 `names` 改成你的数据集路径与类别。

## 训练

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train \
  --name exp
```

训练完成后，权重一般保存在：

```text
runs/train/exp/weights/best.pt
```

## 验证并输出画框图片

对验证集评估并将结果画在原图上：

```bash
python scripts/validate.py \
  --weights runs/train/exp/weights/best.pt \
  --data configs/data.yaml \
  --source datasets/my_dataset/images/val \
  --output runs/val_annotated \
  --conf 0.25 \
  --iou 0.7
```

输出目录中会包含：

- `metrics.json`：当提供 `--data` 时保存验证指标摘要。
- `images/`：带检测框、类别和置信度的图片。

如果只想对若干图片看效果，也可以将 `--source` 指向单张图片或图片目录。

## 常用参数

### `scripts/train.py`

- `--data`：YOLO 数据集 YAML。
- `--model`：预训练模型或已有权重，例如 `yolo11n.pt`、`yolov8n.pt`。
- `--epochs`：训练轮数。
- `--imgsz`：输入尺寸。
- `--batch`：批大小。
- `--device`：设备，例如 `0`、`cpu`。
- `--project` / `--name`：输出目录。

### `scripts/validate.py`

- `--weights`：训练得到的权重文件。
- `--data`：可选，提供后会额外运行 YOLO 验证并保存指标。
- `--source`：输入图片、图片目录或 glob。
- `--output`：可视化结果输出目录。
- `--conf`：置信度阈值。
- `--iou`：NMS IoU 阈值。
- `--max-det`：单张图最多检测数量。

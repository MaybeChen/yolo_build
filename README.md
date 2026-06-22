# YOLO 训练 + 验证工程

这是一个基于 [Ultralytics YOLO](https://docs.ultralytics.com/) 的最小可用训练、验证与可视化工程。工程支持：

- 使用 YOLO 数据集配置在 CPU 上训练 YOLO26 目标检测模型。
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

本工程默认按 CPU 训练/验证配置，脚本默认参数为 `--device cpu`。如果未来需要 GPU，只需显式传入 `--device 0` 或其他 CUDA 设备编号。

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

## 基模与权重目录建议

建议把下载的 YOLO26 预训练基模放在 `models/pretrained/` 目录下，例如：

```text
models/pretrained/yolo26s.pt
```

训练脚本的 `--model` 既可以写模型名 `yolo26s.pt`，由 Ultralytics 自动查找/下载，也可以写本地路径：

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model models/pretrained/yolo26s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train \
  --name exp
```

推荐目录分工：

- `models/pretrained/`：存放官方预训练基模，例如 `yolo26s.pt`。
- `runs/train/<name>/weights/`：训练过程自动输出的权重，例如 `best.pt`、`last.pt`。
- `models/exports/`：可选，存放后续导出的 ONNX、TensorRT、OpenVINO 等部署格式。

模型权重通常较大，已在 `.gitignore` 中忽略 `*.pt`，因此建议只提交目录占位文件，不把大权重文件提交到 Git。

## 训练

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model yolo26s.pt \
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

## 增训建议

后续增训通常不要再从官方基模重新全量训练，而是优先使用上一轮训练得到的权重继续训练：

- 训练被中断或想严格接着同一次实验跑：使用 `runs/train/<name>/weights/last.pt` 并开启 `--resume`。
- 已完成一轮训练，后续加入新数据或调参继续微调：优先使用上一轮 `best.pt` 作为 `--model`。
- 如果新增数据类别或分布变化很大，建议把旧数据和新数据合并后一起训练，避免模型只学新数据而遗忘旧数据。

示例：基于上一轮最佳权重继续增训：

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model runs/train/exp/weights/best.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --project runs/train \
  --name exp_finetune
```

示例：训练中断后恢复同一次训练：

```bash
python scripts/train.py \
  --data configs/data.yaml \
  --model runs/train/exp/weights/last.pt \
  --resume
```

简单判断：

- 从零搭建或数据彻底换了：用 `models/pretrained/yolo26s.pt`。
- 同一个业务场景持续补数据：用上一轮 `best.pt` 增训。
- 上次训练没跑完：用 `last.pt` + `--resume`。

## AnyLabeling 导出 YOLO 检测标签

AnyLabeling 里通常不会直接显示名为 `Detection` 的导出按钮；在你截图展示的导出列表中，本工程的普通水平矩形框检测应选择 **`YOLO HBB`**。

用于本工程的检测数据请注意：

1. 标注形状要使用矩形框（Rectangle / Bounding Box），不要用 Polygon、Brush、SAM mask。
2. 导出菜单选择 `YOLO HBB`，而不是 `YOLO OBB`、`YOLO Seg`、`YOLO Pose`、COCO/VOC/Mask。
3. 配置文件可使用 `configs/classes.txt`，类别顺序必须和 `configs/data.yaml` 里的 `names` 保持一致。
4. 导出后应得到每张图片对应的 `.txt` 标签文件，每行格式为 `class_id x_center y_center width height`，坐标是 0 到 1 的归一化值。
5. 将导出的标签整理到 `labels/train`、`labels/val`，图片放到 `images/train`、`images/val`。

如果你的目标只是普通矩形框检测，请选 **YOLO HBB**。`YOLO OBB` 是旋转框，`YOLO Seg` 是分割多边形，`YOLO Pose` 是姿态关键点，均不是当前工程默认的检测框训练格式。

截图里的常见选项含义：

- `YOLO HBB`：水平矩形框检测，适合本工程。
- `YOLO OBB`：旋转框检测，适合倾斜文本、遥感目标等旋转目标。
- `YOLO Seg`：实例分割，需要多边形/掩码标签。
- `YOLO Pose`：姿态关键点检测。
- `VOC 检测`、`COCO 检测`：其他检测数据格式，不是本工程默认 YOLO txt 格式。

### 类别名和导出文件命名建议

AnyLabeling 导出时的类别名建议使用稳定的英文小写名称，例如 `person`、`car`、`helmet`，不要使用空格、中文标点或频繁变化的临时名称。类别名必须在三个地方保持一致：

1. AnyLabeling 标注时使用的类别名。
2. `configs/classes.txt` 中的类别顺序。
3. `configs/data.yaml` 中的 `names` 顺序。

例如 `configs/classes.txt`：

```text
person
car
```

对应 `configs/data.yaml`：

```yaml
names:
  0: person
  1: car
```

导出目录名称建议按数据集和版本命名，例如：

```text
datasets/my_dataset_v1/
```

如果后续补标或增训，可以使用：

```text
datasets/my_dataset_v2/
datasets/my_dataset_2026_06/
```

注意：YOLO 标签文件名应和图片文件名一一对应。例如 `0001.jpg` 对应 `0001.txt`，不要手动改乱导出的 `.txt` 文件名。

## 标注结果与 YOLO Seg 说明

当前工程默认是 **目标检测（Detection）** 流程，不是 YOLO Seg 分割流程。验证脚本输出的 `runs/val_annotated/images/` 是把预测框画到原图上的可视化图片，用来肉眼检查效果；它不会导出 YOLO Seg 标签。

两类数据格式不要混淆：

- 检测（Detection）标签：每行是 `class_id x_center y_center width height`，表示矩形框。
- 分割（Segmentation）标签：每行通常是 `class_id x1 y1 x2 y2 ... xn yn`，表示目标轮廓多边形点。

如果你的任务只是框出目标位置，继续使用当前 Detection 工程即可。如果你需要导出或训练 YOLO Seg，需要额外准备分割多边形标签，并把基模切换为带 `-seg` 后缀的分割模型，例如 `yolo26s-seg.pt`。

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

## CPU 与 YOLO26 说明

你的场景是 CPU 训练，因此已经将训练和验证脚本的默认设备改为 `cpu`，日常使用时不需要额外传 `--device cpu`。不过 CPU 训练会明显慢于 GPU，建议先用 `yolo26s.pt`、较小 `--imgsz`、较小 `--batch` 和少量 `--epochs` 跑通流程，再根据耗时提高配置。

基模已切换为 YOLO26 检测模型，默认使用 `yolo26s.pt`。使用 Ultralytics Python API 时，YOLO26 的端到端推理由框架处理，本工程继续从 `result.boxes.xyxy / cls / conf` 读取结果并画框即可。

## 常用参数

### `scripts/train.py`

- `--data`：YOLO 数据集 YAML。
- `--model`：预训练模型或已有权重，默认 `yolo26s.pt`；也可按资源选择更轻量的 `yolo26n.pt`，或更高精度但更慢的 `yolo26m.pt`、`yolo26l.pt`、`yolo26x.pt` 等 YOLO26 尺寸。
- `--epochs`：训练轮数。
- `--imgsz`：输入尺寸。
- `--batch`：批大小。
- `--device`：设备，默认 `cpu`；GPU 可传 `0`。
- `--project` / `--name`：输出目录。

### `scripts/validate.py`

- `--weights`：训练得到的权重文件。
- `--data`：可选，提供后会额外运行 YOLO 验证并保存指标。
- `--source`：输入图片、图片目录或 glob。
- `--output`：可视化结果输出目录。
- `--conf`：置信度阈值。
- `--iou`：NMS IoU 阈值。
- `--max-det`：单张图最多检测数量。

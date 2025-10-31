# SSD 检测器 - 环境变量文档

## 概述
本文档描述了可用于配置 SSD 检测器的所有环境变量，用于车辆识别任务，包括对抗样本生成、攻击评估、防御机制和模型训练。

---

## 核心配置变量

### `PROCESS` (必需)
指定操作模式。
- **类型**: 字符串
- **选项**: `adv`, `attack`, `defend`, `train`
  - `adv`: 生成对抗样本
  - `attack`: 评估模型在攻击下的表现
  - `defend`: 对图像应用防御机制
  - `train`: 训练模型
- **示例**: `PROCESS=adv`

### `MODEL` (必需)
指定模型架构。
- **类型**: 字符串
- **选项**: `ssd300`
- **默认值**: `ssd300`
- **示例**: `MODEL=ssd300`

### `BACKBONE` (必需)
指定骨干网络。
- **类型**: 字符串
- **选项**: `vgg16`
- **默认值**: `vgg16`
- **示例**: `BACKBONE=vgg16`

### `DATA` (必需)
指定数据集类型。
- **类型**: 字符串
- **选项**: `coco`, `voc`
- **默认值**: `coco`
- **示例**: `DATA=coco`

### `CLASS_NUMBER` (必需)
数据集中的类别数量。
- **类型**: 整数
- **选项**: 
  - `80` 用于 COCO 数据集
  - `20` 用于 VOC 数据集
- **默认值**: `80`
- **示例**: `CLASS_NUMBER=80`

---

## 路径配置

### `INPUT_PATH` (必需)
包含模型权重和数据的输入目录路径。
- **类型**: 字符串
- **默认值**: `./input`
- **结构**: 
  - `${INPUT_PATH}/model/`: 包含模型权重文件 (.pt, .pth)
  - `${INPUT_PATH}/data/`: 包含数据集目录
- **示例**: `INPUT_PATH=/project/input`

### `OUTPUT_PATH` (必需)
用于保存结果的输出目录路径。
- **类型**: 字符串
- **默认值**: `./output`
- **结构**:
  - `${OUTPUT_PATH}/adv_images/`: 生成的对抗样本（adv 模式）
  - `${OUTPUT_PATH}/defended_images/`: 防御后的图像（defend 模式）
  - 模型检查点（train 模式）
- **示例**: `OUTPUT_PATH=/project/output`

### `CFG_PATH` (可选)
配置文件目录路径。
- **类型**: 字符串
- **默认值**: `./cfgs`
- **示例**: `CFG_PATH=/project/cfgs`

---

## 攻击配置

### `ATTACK_METHOD` (adv/attack 模式必需)
指定对抗攻击方法。
- **类型**: 字符串
- **选项**: `fgsm`, `pgd`, `bim`, `cw`, `deepfool`
  - `fgsm`: 快速梯度符号法
  - `pgd`: 投影梯度下降
  - `bim`: 基本迭代法
  - `cw`: Carlini & Wagner
  - `deepfool`: DeepFool 攻击
- **默认值**: `fgsm`
- **示例**: `ATTACK_METHOD=pgd`

### `EPSILON` (可选)
攻击的最大扰动幅度。
- **类型**: 浮点数
- **范围**: 0.0 到 1.0 (通常为 0.001 到 0.1)
- **默认值**: `0.031` (8/255)
- **示例**: `EPSILON=0.031`

### `STEP_SIZE` (可选)
迭代攻击的步长（PGD, BIM）。
- **类型**: 浮点数
- **范围**: 0.0 到 EPSILON
- **默认值**: `0.008` (2/255)
- **示例**: `STEP_SIZE=0.008`

### `MAX_ITERATIONS` (可选)
迭代攻击的最大迭代次数。
- **类型**: 整数
- **范围**: 1 到 1000
- **默认值**: `50`
- **示例**: `MAX_ITERATIONS=20`

### `RANDOM_START` (可选)
是否对攻击使用随机初始化。
- **类型**: 布尔值
- **选项**: `true`, `false`
- **默认值**: `false`
- **示例**: `RANDOM_START=true`

### `LOSS_FUNCTION` (可选)
攻击优化的损失函数。
- **类型**: 字符串
- **默认值**: `CrossEntropy`
- **示例**: `LOSS_FUNCTION=CrossEntropy`

### `OPTIMIZATION_METHOD` (可选)
攻击的优化方法。
- **类型**: 字符串
- **默认值**: `Adam`
- **示例**: `OPTIMIZATION_METHOD=Adam`

---

## 防御配置

### `DEFEND_METHOD` (defend 模式必需)
指定防御机制。
- **类型**: 字符串
- **选项**: `scale`, `comp`
  - `scale`: 基于缩放的防御（先缩小再放大）
  - `comp`: JPEG 压缩防御
- **默认值**: `scale`
- **示例**: `DEFEND_METHOD=comp`

---

## 训练配置

### `EPOCHS` (可选，train 模式必需)
训练轮数。
- **类型**: 整数
- **范围**: 1 到 1000
- **默认值**: `100`
- **示例**: `EPOCHS=50`

### `BATCH` (可选，train 模式必需)
训练批次大小。
- **类型**: 整数
- **范围**: 1 到 128
- **默认值**: `8`
- **示例**: `BATCH=16`

---

## 硬件配置

### `DEVICE` (可选)
GPU 设备索引或 CPU。
- **类型**: 整数
- **范围**: -1 (CPU), 0+ (GPU 索引)
- **默认值**: `0`
- **注意**: 防御模式设置为 `-1` 或使用 CPU
- **示例**: `DEVICE=0`

### `WORKERS` (可选)
数据加载器工作进程数。
- **类型**: 整数
- **范围**: 0 到 16
- **默认值**: `0`
- **示例**: `WORKERS=4`

---

## 任务配置

### `TASK` (可选)
模型的任务类型。
- **类型**: 字符串
- **选项**: `detect`
- **默认值**: `detect`
- **示例**: `TASK=detect`

---

## 使用示例

### 示例 1: 生成 FGSM 对抗样本
```bash
docker run --rm --gpus all \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=fgsm \
  -e EPSILON=0.031 \
  -e DEVICE=0 \
  nudt_ssd:latest
```

### 示例 2: 评估 PGD 攻击
```bash
docker run --rm --gpus all \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=attack \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  -e DEVICE=0 \
  nudt_ssd:latest
```

### 示例 3: 应用缩放防御
```bash
docker run --rm \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=ssd300 \
  -e DEFEND_METHOD=scale \
  nudt_ssd:latest
```

### 示例 4: 训练模型
```bash
docker run --rm --gpus all \
  -v /data/input:/project/input:ro \
  -v /data/output:/project/output:rw \
  -e PROCESS=train \
  -e MODEL=ssd300 \
  -e EPOCHS=100 \
  -e BATCH=8 \
  -e DEVICE=0 \
  nudt_ssd:latest
```

---

## SSE 输出格式

系统输出服务器推送事件（SSE）格式的消息：

```
event: input_path_validated
data: {"status": "success", "message": "Input path is valid and complete.", "file_name": "/project/input"}

event: input_data_validated
data: {"status": "success", "message": "Input data file is valid and complete.", "file_name": "/project/input/data/coco/"}

event: input_model_validated
data: {"status": "success", "message": "Input model file is valid and complete.", "file_name": "/project/input/model/ssd300.pth"}

event: output_path_validated
data: {"status": "success", "message": "Output path is valid and complete.", "file_name": "/project/output"}

event: adv_samples_gen_validated
data: {"status": "success", "message": "adv sample is generated.", "file_name": "/project/output/adv_images/adv_image_0_0.jpg"}

event: clean_samples_gen_validated
data: {"status": "success", "message": "clean sample is generated.", "file_name": "/project/output/defended_images/image_0.jpg"}
```

---

## 注意事项

1. **GPU 使用**: 大多数进程（adv, attack, train）需要 GPU。使用 docker run 时添加 `--gpus all` 标志。
2. **防御模式**: 防御操作默认在 CPU 上运行以确保稳定性。
3. **文件权限**: 确保挂载的卷具有适当的读写权限。
4. **输入结构**: 输入路径必须包含：
   - `model/` 目录，至少包含一个模型文件
   - `data/` 目录，至少包含一个包含图像的数据集子目录
5. **输出结构**: 结果根据进程模式保存在相应的子目录中。

---


# SSD 车辆识别检测器

## 概述
本项目实现了一个基于 VGG16 骨干网络的 SSD（单次多框检测器），用于车辆识别任务。它支持对抗样本生成、攻击评估、防御机制和模型训练。

## 功能特性
- **对抗攻击方法**: FGSM, PGD, BIM, C&W, DeepFool
- **防御机制**: 基于缩放的防御、JPEG 压缩防御
- **模型架构**: SSD300 配合 VGG16 骨干网络
- **SSE 输出**: 服务器推送事件格式，用于实时进度跟踪
- **Docker 支持**: 容器化部署，支持 GPU

## 项目结构
```
nudt_ssd/
├── main.py                      # 主入口点
├── Dockerfile                   # Docker 配置文件
├── requirements.txt             # Python 依赖
├── ENV_VARIABLES.md            # 环境变量文档
├── docker_run_scripts.sh       # Docker 运行示例脚本
├── ssd_detector/               # SSD 检测器实现
│   ├── __init__.py
│   └── main.py
├── attacks/                    # 攻击实现
│   ├── __init__.py
│   └── attacks.py
├── defends/                    # 防御实现
│   ├── __init__.py
│   └── defends.py
├── train/                      # 训练实现
│   ├── __init__.py
│   └── trainer.py
└── utils/                      # 工具函数
    ├── __init__.py
    ├── sse.py                  # SSE 输出函数
    └── yaml_rw.py              # YAML 读写
```

## 快速开始

### 构建 Docker 镜像
```bash
cd nudt_ssd
docker build -t nudt_ssd:latest .
```

### 运行容器

#### 1. 生成对抗样本
```bash
docker run --rm --gpus all \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=adv \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=fgsm \
  -e EPSILON=0.031 \
  nudt_ssd:latest
```

#### 2. 评估攻击
```bash
docker run --rm --gpus all \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=attack \
  -e MODEL=ssd300 \
  -e ATTACK_METHOD=pgd \
  -e EPSILON=0.031 \
  -e STEP_SIZE=0.008 \
  -e MAX_ITERATIONS=10 \
  nudt_ssd:latest
```

#### 3. 应用防御
```bash
docker run --rm \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=defend \
  -e MODEL=ssd300 \
  -e DEFEND_METHOD=scale \
  nudt_ssd:latest
```

#### 4. 训练模型
```bash
docker run --rm --gpus all \
  -v /path/to/input:/project/input:ro \
  -v /path/to/output:/project/output:rw \
  -e PROCESS=train \
  -e MODEL=ssd300 \
  -e EPOCHS=100 \
  -e BATCH=8 \
  nudt_ssd:latest
```

## 输入目录结构
```
input/
├── model/
│   └── ssd300.pth              # 预训练模型权重
└── data/
    └── dataset_name/           # 数据集目录
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## 输出目录结构
```
output/
├── adv_images/                 # 生成的对抗样本（adv 模式）
│   ├── adv_image_0_0.jpg
│   └── ...
├── defended_images/            # 防御后的图像（defend 模式）
│   ├── image1.jpg
│   └── ...
└── ssd_model.pth              # 训练后的模型（train 模式）
```

## 环境变量
查看 [ENV_VARIABLES.md](ENV_VARIABLES.md) 了解所有环境变量的详细文档。

### 关键变量
- `PROCESS`: 操作模式 (adv, attack, defend, train)
- `MODEL`: 模型架构 (ssd300)
- `ATTACK_METHOD`: 攻击方法 (fgsm, pgd, bim, cw, deepfool)
- `DEFEND_METHOD`: 防御方法 (scale, comp)
- `EPSILON`: 扰动幅度 (默认: 0.031)
- `DEVICE`: GPU 设备索引 (默认: 0)

## 攻击方法

### FGSM (快速梯度符号法)
快速单步攻击方法。
```bash
-e ATTACK_METHOD=fgsm -e EPSILON=0.031
```

### PGD (投影梯度下降)
带投影的迭代攻击。
```bash
-e ATTACK_METHOD=pgd -e EPSILON=0.031 -e STEP_SIZE=0.008 -e MAX_ITERATIONS=10
```

### BIM (基本迭代法)
迭代 FGSM 变体。
```bash
-e ATTACK_METHOD=bim -e EPSILON=0.031 -e STEP_SIZE=0.008 -e MAX_ITERATIONS=20
```

### C&W (Carlini & Wagner)
基于优化的攻击。
```bash
-e ATTACK_METHOD=cw -e MAX_ITERATIONS=50
```

### DeepFool
最小扰动攻击。
```bash
-e ATTACK_METHOD=deepfool -e MAX_ITERATIONS=50
```

## 防御方法

### 缩放防御
将图像缩小再放大以去除扰动。
```bash
-e DEFEND_METHOD=scale
```

### 压缩防御
应用 JPEG 压缩以去除扰动。
```bash
-e DEFEND_METHOD=comp
```

## SSE 输出格式
系统以 SSE（服务器推送事件）格式输出进度：
```
event: input_path_validated
data: {"status": "success", "message": "Input path is valid and complete.", "file_name": "/project/input"}

event: adv_samples_gen_validated
data: {"status": "success", "message": "adv sample is generated.", "file_name": "/project/output/adv_images/adv_image_0_0.jpg"}
```

## 系统要求
- Python 3.8
- PyTorch 2.4.0
- torchvision 0.19.0
- CUDA 12.1 (用于 GPU 支持)
- 查看 [requirements.txt](requirements.txt) 了解完整列表

## 测试
使用提供的测试脚本验证实现：
```bash
python image-test.py
```

## Docker 运行脚本
查看 [docker_run_scripts.sh](docker_run_scripts.sh) 了解不同配置下运行容器的更多示例。

## 注意事项
1. adv、attack 和 train 模式需要 GPU
2. 防御模式默认在 CPU 上运行
3. 确保输入目录包含模型权重和数据集
4. 所有输出遵循 SSE 格式，便于与后端 API 集成


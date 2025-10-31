## 1. 环境准备节点 API

### 1.1 加载车辆识别模型
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-attack/load-model-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "method_type": "车辆识别攻击",
    "algorithm_type": "对抗样本攻击",
    "task_type": "模型加载",
    "task_name": "车辆识别模型加载",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "model_config": {
      "target_model": "YOLOv5",
      "model_type": "object_detection"
    },
    "hardware_config": {
      "deployment_type": "single_gpu",
      "gpu_config": {
        "gpu_devices": [0],
        "memory_per_device": "24GB",
        "gpu_utilization_threshold": 0.8
      },
      "parallel_strategy": {
        "strategy_type": "none",
        "data_parallel_config": {
          "replication_count": 1,
          "gradient_accumulation": 1,
          "synchronization_mode": "sync",
          "load_balancing": false
        },
        "model_parallel_config": {
          "tensor_parallel_degree": 1,
          "pipeline_parallel_degree": 1,
          "device_map": "auto",
          "memory_optimization": false
        },
        "distributed_config": {
          "backend": "nccl",
          "init_method": "env://",
          "world_size": 1,
          "rank": 0
        }
      },
      "multi_gpu_config": {
        "enabled": false,
        "gpu_topology": "single_node",
        "interconnect_bandwidth": "high",
        "collective_operations": {
          "all_reduce_algorithm": "ring",
          "broadcast_algorithm": "tree"
        }
      }
    },
    "inference_config": {
      "batch_size": 8,
      "max_detections": 100,
      "nms_threshold": 0.45,
      "scale_factor": 0.00392,
      "mean_values": [0, 0, 0],
      "std_values": [1, 1, 1],
      "parallel_inference": {
        "enabled": false,
        "num_streams": 1,
        "stream_priority": "default"
      }
    },
    "performance_config": {
      "warmup_iterations": 10,
      "benchmark_mode": false,
      "optimization_level": "O1",
      "tensorrt_optimization": false,
      "memory_efficient": true,
      "multi_gpu_optimization": {
        "enabled": false,
        "load_balancing_strategy": "round_robin",
        "dynamic_batching": false
      }
    }
  }
}
```
# 车辆识别模型加载接口参数介绍

## callback_params（回调参数）

**task_run_id**：任务运行唯一标识符，用于追踪和管理特定任务实例

**method_type**：方法类型，标识为车辆识别攻击，表明这是针对车辆识别模型的攻击演练

**algorithm_type**：算法类型，标识为对抗样本攻击，指定使用的攻击技术类别

**task_type**：任务类型，标识为模型加载，说明当前执行的是模型加载阶段

**task_name**：具体任务名称，车辆识别模型加载，提供更详细的任务描述

**parent_task_id**：父任务ID，用于在复杂工作流中建立任务关联关系

**user_name**：执行任务的用户名，用于审计和权限管理

## business_params（业务参数）

### scene_instance_id
场景实例ID，唯一标识当前攻击演练的工作环境实例

### model_config（模型配置）
**target_model**：目标模型名称，指定要加载的车辆识别模型为YOLOv5

**model_type**：模型类型，标识为目标检测模型，说明模型功能类别

### hardware_config（硬件配置）
**deployment_type**：部署类型，设置为单GPU部署模式

**gpu_config**：GPU配置参数
- **gpu_devices**：使用的GPU设备ID列表，当前配置为使用第0号GPU
- **memory_per_device**：每个GPU设备的显存限制，设置为24GB
- **gpu_utilization_threshold**：GPU利用率阈值，设置为0.8即80%

**parallel_strategy**：并行策略配置
- **strategy_type**：策略类型，设置为none表示不使用并行策略
- **data_parallel_config**：数据并行配置
  - **replication_count**：模型副本数量，设置为1表示单副本
  - **gradient_accumulation**：梯度累积步数，设置为1表示不累积
  - **synchronization_mode**：同步模式，设置为sync表示同步更新
  - **load_balancing**：负载均衡，设置为false表示不启用
- **model_parallel_config**：模型并行配置
  - **tensor_parallel_degree**：张量并行度，设置为1表示不分割张量
  - **pipeline_parallel_degree**：流水线并行度，设置为1表示不分割模型层
  - **device_map**：设备映射策略，设置为auto表示自动分配
  - **memory_optimization**：内存优化，设置为false表示不启用
- **distributed_config**：分布式配置
  - **backend**：通信后端，设置为nccl表示使用NCCL通信库
  - **init_method**：初始化方法，设置为env://表示使用环境变量初始化
  - **world_size**：全局进程数，设置为1表示单进程
  - **rank**：当前进程排名，设置为0表示主进程

**multi_gpu_config**：多GPU配置
- **enabled**：是否启用多GPU，设置为false表示不启用
- **gpu_topology**：GPU拓扑结构，设置为single_node表示单节点
- **interconnect_bandwidth**：互联带宽，设置为high表示高带宽
- **collective_operations**：集合操作配置
  - **all_reduce_algorithm**：全归约算法，设置为ring表示使用环算法
  - **broadcast_algorithm**：广播算法，设置为tree表示使用树形算法

### inference_config（推理配置）
**batch_size**：批处理大小，设置为8表示每次处理8张图像

**max_detections**：最大检测数量，设置为100表示每张图像最多检测100个目标

**nms_threshold**：非极大值抑制阈值，设置为0.45用于过滤重叠检测框

**scale_factor**：尺度因子，设置为0.00392用于图像像素值归一化

**mean_values**：均值参数，设置为[0,0,0]表示不进行均值减法

**std_values**：标准差参数，设置为[1,1,1]表示不进行标准差归一化

**parallel_inference**：并行推理配置
- **enabled**：是否启用并行推理，设置为false表示不启用
- **num_streams**：流数量，设置为1表示使用单个CUDA流
- **stream_priority**：流优先级，设置为default表示使用默认优先级

### performance_config（性能配置）
**warmup_iterations**：预热迭代次数，设置为10次用于稳定性能

**benchmark_mode**：基准测试模式，设置为false表示不启用性能基准测试

**optimization_level**：优化级别，设置为O1表示中等优化级别

**tensorrt_optimization**：TensorRT优化，设置为false表示不启用TensorRT加速

**memory_efficient**：内存效率，设置为true表示启用内存优化

**multi_gpu_optimization**：多GPU优化配置
- **enabled**：是否启用多GPU优化，设置为false表示不启用
- **load_balancing_strategy**：负载均衡策略，设置为round_robin表示轮询调度
- **dynamic_batching**：动态批处理，设置为false表示不启用动态批处理

```json
OUTPUT:
{
  "resp_code": 0,
  "resp_msg": "环境初始化成功",
  "data": {
    "environment_id": "env_content_audit_20240115001",
    "model_status": {
      "model_name": "BERT-ContentFilter",
      "model_type": "discriminative",
      "model_version": "v2.1",
      "model_size": "440MB",
      "load_time": "2.3s",
      "status": "loaded",
      "inference_endpoint": "127.0.0.1:19001/api-ai-server/inference/env_content_audit_20240115001"
    },
    "resource_usage": {
      "gpu_memory_used": "2.1GB",
      "cpu_usage": "35%",
      "memory_usage": "4.2GB",
      "disk_usage": "1.5GB"
    }
  },
  "time_stamp": "2024/07/01-09:19:32:679"
}
```

***根据实际情况，可以不需要卡，只支持单卡***
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "method_type": "车辆识别攻击",
    "algorithm_type": "对抗样本攻击",
    "task_type": "模型加载",
    "task_name": "车辆识别模型加载",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "model_config": {
      "target_model": "YOLOv5",
      "model_type": "object_detection"
    },
    "hardware_config": {
      "deployment_type": "multi_gpu",
      "gpu_config": {
        "gpu_devices": [0, 1, 2, 3],
        "memory_per_device": "24GB",
        "gpu_utilization_threshold": 0.8
      },
      "parallel_strategy": {
        "strategy_type": "data_parallel",
        "data_parallel_config": {
          "replication_count": 4,
          "gradient_accumulation": 1,
          "synchronization_mode": "sync",
          "load_balancing": true
        },
        "model_parallel_config": {
          "tensor_parallel_degree": 1,
          "pipeline_parallel_degree": 1,
          "device_map": "auto",
          "memory_optimization": false
        },
        "distributed_config": {
          "backend": "nccl",
          "init_method": "env://",
          "world_size": 4,
          "rank": 0
        }
      },
      "multi_gpu_config": {
        "enabled": true,
        "gpu_topology": "single_node",
        "interconnect_bandwidth": "high",
        "collective_operations": {
          "all_reduce_algorithm": "ring",
          "broadcast_algorithm": "tree"
        }
      }
    },
    "inference_config": {
      "batch_size": 32,
      "max_detections": 100,
      "nms_threshold": 0.45,
      "scale_factor": 0.00392,
      "mean_values": [0, 0, 0],
      "std_values": [1, 1, 1],
      "parallel_inference": {
        "enabled": true,
        "num_streams": 4,
        "stream_priority": "high"
      }
    },
    "performance_config": {
      "warmup_iterations": 20,
      "benchmark_mode": false,
      "optimization_level": "O1",
      "tensorrt_optimization": false,
      "memory_efficient": true,
      "multi_gpu_optimization": {
        "enabled": true,
        "load_balancing_strategy": "round_robin",
        "dynamic_batching": true
      }
    }
  }
}
```
# 多GPU车辆识别模型加载接口参数介绍

## callback_params（回调参数）

**task_run_id**：任务运行唯一标识符，用于在多GPU环境中追踪分布式任务执行

**method_type**：方法类型，标识为车辆识别攻击，表明这是针对车辆识别模型的分布式攻击演练

**algorithm_type**：算法类型，标识为对抗样本攻击，指定在多GPU环境下使用的攻击技术

**task_type**：任务类型，标识为模型加载，说明当前执行的是多GPU模型加载阶段

**task_name**：具体任务名称，车辆识别模型加载，描述多GPU环境下的加载任务

**parent_task_id**：父任务ID，用于在分布式工作流中建立任务关联关系

**user_name**：执行任务的用户名，用于多用户环境下的权限管理和资源分配

## business_params（业务参数）

### user_name
重复的用户名字段，在多GPU环境中用于双重验证和审计追踪

### scene_instance_id
场景实例ID，唯一标识当前多GPU攻击演练的分布式工作环境

### model_config（模型配置）
**target_model**：目标模型名称，指定要在多GPU上加载的YOLOv5模型

**model_type**：模型类型，标识为目标检测模型，说明模型适合在多GPU上进行数据并行推理

### hardware_config（硬件配置）
**deployment_type**：部署类型，设置为multi_gpu表示多GPU部署模式

**gpu_config**：GPU配置参数
- **gpu_devices**：使用的GPU设备ID列表，配置为使用4个GPU：[0, 1, 2, 3]
- **memory_per_device**：每个GPU设备的显存限制，设置为24GB确保充足显存
- **gpu_utilization_threshold**：GPU利用率阈值，设置为0.8确保各卡负载均衡

**parallel_strategy**：并行策略配置
- **strategy_type**：策略类型，设置为data_parallel表示采用数据并行策略
- **data_parallel_config**：数据并行详细配置
  - **replication_count**：模型副本数量，设置为4表示在4个GPU上各放置一个模型副本
  - **gradient_accumulation**：梯度累积步数，设置为1表示每个批次都更新梯度
  - **synchronization_mode**：同步模式，设置为sync表示同步梯度更新
  - **load_balancing**：负载均衡，设置为true启用各GPU间负载均衡
- **model_parallel_config**：模型并行配置
  - **tensor_parallel_degree**：张量并行度，设置为1表示不分割模型张量
  - **pipeline_parallel_degree**：流水线并行度，设置为1表示不分割模型层
  - **device_map**：设备映射策略，设置为auto自动分配模型组件
  - **memory_optimization**：内存优化，设置为false在数据并行中不需要特殊内存优化
- **distributed_config**：分布式训练配置
  - **backend**：通信后端，设置为nccl使用NVIDIA Collective Communications Library
  - **init_method**：初始化方法，设置为env://使用环境变量进行进程组初始化
  - **world_size**：全局进程数，设置为4对应4个GPU进程
  - **rank**：当前进程排名，设置为0表示主进程负责协调

**multi_gpu_config**：多GPU高级配置
- **enabled**：是否启用多GPU，设置为true启用多GPU功能
- **gpu_topology**：GPU拓扑结构，设置为single_node表示所有GPU在单个节点内
- **interconnect_bandwidth**：互联带宽，设置为high假设GPU间有高带宽互联
- **collective_operations**：集合操作算法配置
  - **all_reduce_algorithm**：全归约算法，设置为ring使用环算法进行梯度同步
  - **broadcast_algorithm**：广播算法，设置为tree使用树形算法进行参数广播

### inference_config（推理配置）
**batch_size**：批处理大小，设置为32充分利用多GPU的计算能力

**max_detections**：最大检测数量，设置为100保持与单GPU配置一致

**nms_threshold**：非极大值抑制阈值，设置为0.45用于多GPU检测结果融合

**scale_factor**：尺度因子，设置为0.00392用于多GPU统一的图像预处理

**mean_values**：均值参数，设置为[0,0,0]在多GPU环境下保持预处理一致性

**std_values**：标准差参数，设置为[1,1,1]确保各GPU预处理结果一致

**parallel_inference**：并行推理配置
- **enabled**：是否启用并行推理，设置为true启用多流并行推理
- **num_streams**：流数量，设置为4为每个GPU分配一个CUDA流
- **stream_priority**：流优先级，设置为high确保推理任务获得高优先级

### performance_config（性能配置）
**warmup_iterations**：预热迭代次数，设置为20次用于多GPU环境性能稳定

**benchmark_mode**：基准测试模式，设置为false在生产环境中不启用基准测试

**optimization_level**：优化级别，设置为O1在多GPU环境下平衡性能与稳定性

**tensorrt_optimization**：TensorRT优化，设置为false在多GPU数据并行中暂不启用

**memory_efficient**：内存效率，设置为true在多GPU环境中优化内存使用

**multi_gpu_optimization**：多GPU专用优化配置
- **enabled**：是否启用多GPU优化，设置为true启用多GPU特定优化
- **load_balancing_strategy**：负载均衡策略，设置为round_robin使用轮询调度平衡各GPU负载
- **dynamic_batching**：动态批处理，设置为true根据各GPU负载动态调整批处理大小


## 3. 生成对抗样本节点 API (SSE流式响应)

### 3.1 生成对抗样本
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-attack/generate-adversarial-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304",
    "method_type": "车辆识别攻击",
    "algorithm_type": "FGSM攻击",
    "task_type": "样本生成",
    "task_name": "FGSM对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "FGSM",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 100,
        "sample_selection_strategy": "random"
      },
      "algorithm_parameters": {
        "epsilon": 0.08,
        "step_size": 0.01,
        "max_iterations": 10,
        "targeted": false,
        "target_class": -1,
        "random_start": false,
        "loss_function": "cross_entropy",
        "optimization_method": "gradient_ascent"
      },
      "constraints": {
        "perturbation_norm": "linf",
        "max_perturbation": 0.08,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness"
      ]
    }
  }
}
```

# 对抗样本生成API参数简要说明

## 回调参数
- **task_run_id**: 任务唯一标识，用于追踪执行流程
- **method_type**: 方法类型，如"车辆识别攻击"
- **algorithm_type**: 算法类型，如"FGSM攻击"
- **task_type**: 任务类型，如"样本生成"
- **task_name**: 具体任务名称
- **parent_task_id**: 父任务ID，用于任务链管理
- **user_name**: 执行用户名称

## 业务参数
- **scene_instance_id**: 场景实例ID，关联具体演练场景

## 生成配置
- **attack_algorithm**: 攻击算法名称，如FGSM
- **dataset_config**: 数据集配置
  - **dataset_name**: 数据集名称，如KITTI
  - **dataset_format**: 数据格式，如图像
  - **total_samples**: 数据集总样本数
  - **selected_samples**: 本次使用的样本数
  - **sample_selection_strategy**: 样本选择策略，如随机选择

## 算法参数
- **epsilon**: 扰动强度，控制攻击力度
- **step_size**: 步长，迭代攻击的更新幅度
- **max_iterations**: 最大迭代次数
- **targeted**: 是否为有目标攻击
- **target_class**: 目标类别（有目标攻击时使用）
- **random_start**: 是否随机初始化扰动
- **loss_function**: 损失函数类型
- **optimization_method**: 优化方法

## 约束条件
- **perturbation_norm**: 扰动范数类型
- **max_perturbation**: 最大扰动限制
- **clip_min/max**: 像素值范围限制
- **spatial_constraints**: 空间约束配置

## 模型配置
- **model_name**: 目标模型名称，如YOLOv5
- **model_state**: 模型状态，如已加载

## 监控配置
- **real_time_metrics**: 实时监控指标
- **quality_metrics**: 质量评估指标

基于FGSM对抗样本生成API的格式，我为您生成其他攻击方法的接口INPUT：

## 1. PGD攻击方法

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305",
    "method_type": "车辆识别攻击",
    "algorithm_type": "PGD攻击",
    "task_type": "样本生成",
    "task_name": "PGD对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "PGD",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 100,
        "sample_selection_strategy": "difficult_samples"
      },
      "algorithm_parameters": {
        "epsilon": 0.03,
        "step_size": 0.01,
        "max_iterations": 40,
        "targeted": false,
        "target_class": -1,
        "random_start": true,
        "loss_function": "cross_entropy",
        "optimization_method": "projected_gradient_descent",
        "momentum": 0.9,
        "restart_strategy": "random_restarts",
        "num_restarts": 5
      },
      "constraints": {
        "perturbation_norm": "linf",
        "max_perturbation": 0.03,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time",
        "iteration_progress"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness",
        "convergence_status"
      ]
    }
  }
}
```

## 2. MIM攻击方法

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306",
    "method_type": "车辆识别攻击",
    "algorithm_type": "MIM攻击",
    "task_type": "样本生成",
    "task_name": "MIM对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "MIM",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 100,
        "sample_selection_strategy": "high_confidence"
      },
      "algorithm_parameters": {
        "epsilon": 0.06,
        "step_size": 0.01,
        "max_iterations": 20,
        "targeted": false,
        "target_class": -1,
        "random_start": true,
        "loss_function": "cross_entropy",
        "optimization_method": "momentum_iterative",
        "momentum_factor": 0.9,
        "decay_factor": 1.0,
        "nesterov": false
      },
      "constraints": {
        "perturbation_norm": "linf",
        "max_perturbation": 0.06,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time",
        "momentum_accumulation"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness",
        "transferability"
      ]
    }
  }
}
```

## 3. C&W攻击方法

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307",
    "method_type": "车辆识别攻击",
    "algorithm_type": "C&W攻击",
    "task_type": "样本生成",
    "task_name": "C&W对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "C&W",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 50,
        "sample_selection_strategy": "targeted_samples"
      },
      "algorithm_parameters": {
        "attack_type": "l2",
        "confidence": 0.0,
        "learning_rate": 0.01,
        "max_iterations": 1000,
        "binary_search_steps": 9,
        "initial_const": 0.01,
        "abort_early": true,
        "targeted": true,
        "target_class": 2,
        "loss_function": "cw_loss",
        "optimization_method": "adam"
      },
      "constraints": {
        "perturbation_norm": "l2",
        "max_perturbation": 1.0,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time",
        "binary_search_step",
        "current_const"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness",
        "minimal_perturbation"
      ]
    }
  }
}
```

## 4. DEEPFOOL攻击方法

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3308",
    "method_type": "车辆识别攻击",
    "algorithm_type": "DEEPFOOL攻击",
    "task_type": "样本生成",
    "task_name": "DEEPFOOL对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "DEEPFOOL",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 80,
        "sample_selection_strategy": "boundary_samples"
      },
      "algorithm_parameters": {
        "max_iterations": 50,
        "overshoot": 0.02,
        "step_size": 0.02,
        "targeted": false,
        "num_classes": 4,
        "loss_function": "distance_to_boundary",
        "optimization_method": "iterative_linearization",
        "epsilon": 1e-4
      },
      "constraints": {
        "perturbation_norm": "l2",
        "max_perturbation": 0.5,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time",
        "distance_to_boundary",
        "iteration_count"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness",
        "minimal_perturbation"
      ]
    }
  }
}
```

## 5. BadNet攻击方法

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3309",
    "method_type": "车辆识别攻击",
    "algorithm_type": "BadNet攻击",
    "task_type": "样本生成",
    "task_name": "BadNet后门样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "BadNet",
      "dataset_config": {
        "dataset_name": "KITTI",
        "dataset_format": "image",
        "total_samples": 7481,
        "selected_samples": 200,
        "sample_selection_strategy": "backdoor_injection"
      },
      "algorithm_parameters": {
        "trigger_type": "square_patch",
        "trigger_size": [20, 20],
        "trigger_position": "bottom_right",
        "trigger_pattern": "white_square",
        "poisoning_rate": 0.1,
        "target_class": 0,
        "injection_method": "pixel_pattern",
        "blend_alpha": 0.2,
        "training_epochs": 50,
        "backdoor_strength": 0.8
      },
      "constraints": {
        "perturbation_norm": "spatial",
        "max_perturbation": 0.2,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": true,
          "mask_regions": ["bottom_right_corner"],
          "trigger_visibility": "subtle"
        }
      }
    },
    "model_config": {
      "model_name": "YOLOv5",
      "model_state": "retraining_required"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "trigger_injection_rate",
        "backdoor_success_rate",
        "memory_usage",
        "computation_time",
        "model_retraining_progress"
      ],
      "quality_metrics": [
        "trigger_visibility",
        "clean_accuracy",
        "backdoor_effectiveness",
        "stealthiness"
      ]
    }
  }
}
```

OUTPUT:
```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:00:123", "data": {"event": "process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始FGSM对抗样本生成任务", "log": "[5%] 开始FGSM对抗样本生成任务\n", "details": {"attack_method": "FGSM", "target_model": "YOLOv5", "max_samples": 100, "business_params": {"scene_instance_id": "f54d72a78c264f9bb936954522881e7c", "model_info": {"model_name": "YOLOv5", "model_path": "/models/yolov5/pytorch/weights.pt"}, "dataset_info": {"dataset_name": "KITTI", "dataset_path": "/datasets/KITTI/processed", "sample_type": "image", "total_samples": 7481, "sample_indices": "random_100"}, "generation_config": {"method_type": "车辆识别攻击", "algorithm_type": "FGSM", "adversarial_samples_dir": "/data/adversarial_samples/vehicle_recognition_123456789", "adversarial_samples_name": "adversarial_samples_kitti_fgsm", "original_samples_save_path": "/data/original_samples/vehicle_recognition_123456789", "original_samples_name": "original_samples_kitti", "visualization_dir": "/data/visualizations/vehicle_recognition_123456789", "visualization_name": "perturbation_visualization_kitti", "file_format": "npy", "max_samples": 100, "save_visualizations": true, "sample_selection_strategy": "random", "user_params": {"epsilon": 0.08, "targeted": false, "target_class": null, "norm_type": "inf", "max_iterations": 10, "step_size": 0.01}}}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:01:456", "data": {"event": "model_loaded", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "目标模型加载成功", "log": "[15%] 目标模型YOLOv5加载完成\n", "details": {"model_name": "YOLOv5", "model_path": "/models/yolov5/pytorch/weights.pt", "model_type": "object_detection", "input_shape": [640, 640, 3], "classes": ["car", "truck", "bus", "motorcycle", "person"]}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:02:789", "data": {"event": "dataset_loaded", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "数据集加载完成", "log": "[25%] 加载100个车辆图像样本\n", "details": {"dataset_path": "/datasets/KITTI/processed", "sample_count": 100, "dataset_type": "image", "sample_indices": "random_100", "dataset_info": {"name": "KITTI", "total_samples": 7481, "vehicle_classes": ["car", "truck", "bus", "motorcycle"]}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:03:123", "data": {"event": "generation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 30, "message": "开始生成对抗样本", "log": "[30%] 开始FGSM对抗样本生成，扰动参数epsilon=0.08\n", "details": {"attack_params": {"epsilon": 0.08, "targeted": false, "norm_type": "inf", "max_iterations": 10, "step_size": 0.01}, "batch_size": 20}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:15:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 40, "message": "样本处理中", "log": "[40%] 正在处理样本20/100 - 计算梯度\n", "details": {"current_sample": 20, "total_samples": 100, "step": "gradient_computation", "batch_progress": "1/5"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:27:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 50, "message": "样本处理中", "log": "[50%] 正在处理样本40/100 - 应用扰动\n", "details": {"current_sample": 40, "total_samples": 100, "step": "perturbation_application", "batch_progress": "2/5"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:40:123", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 60, "message": "样本处理中", "log": "[60%] 正在处理样本60/100 - 验证攻击效果\n", "details": {"current_sample": 60, "total_samples": 100, "step": "attack_validation", "batch_progress": "3/5", "current_success_rate": 0.85}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:52:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 70, "message": "样本处理中", "log": "[70%] 正在处理样本80/100 - 应用扰动\n", "details": {"current_sample": 80, "total_samples": 100, "step": "perturbation_application", "batch_progress": "4/5", "current_success_rate": 0.87}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:04:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 80, "message": "样本处理中", "log": "[80%] 正在处理样本100/100 - 最终验证\n", "details": {"current_sample": 100, "total_samples": 100, "step": "final_validation", "batch_progress": "5/5", "current_success_rate": 0.88}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:05:123", "data": {"event": "generation_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 85, "message": "对抗样本生成完成", "log": "[85%] 对抗样本生成完成 - 成功生成88/100样本\n", "details": {"total_samples": 100, "successful_samples": 88, "success_rate": 0.88, "generation_time": "65.0秒", "avg_perturbation": 0.066, "attack_success_rate": 0.88}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:10:456", "data": {"event": "visualization_generated", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "扰动可视化生成完成", "log": "[90%] 扰动可视化图像已生成\n", "details": {"visualization_path": "/data/visualizations/vehicle_recognition_123456789/perturbation_visualization_kitti", "file_format": "png", "sample_count": 88, "visualization_types": ["original_vs_adversarial", "perturbation_heatmap", "detection_comparison"]}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:15:789", "data": {"event": "results_saved", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 95, "message": "结果文件保存完成", "log": "[95%] 对抗样本结果保存完成\n", "details": {"output_files": {"adversarial_samples": "/data/adversarial_samples/vehicle_recognition_123456789/adversarial_samples_kitti_fgsm.npy", "original_samples": "/data/original_samples/vehicle_recognition_123456789/original_samples_kitti.npy", "visualization_files": "/data/visualizations/vehicle_recognition_123456789/perturbation_visualization_kitti.zip", "metadata_file": "/data/adversarial_samples/vehicle_recognition_123456789/generation_metadata.json"}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:20:123", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "样本生成", "task_name": "FGSM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "FGSM对抗样本生成任务完成", "log": "[100%] FGSM对抗样本生成任务完成\n", "details": {"generation_id": "gen_fgsm_vehicle_202407011015", "attack_method": "FGSM", "attack_type": "白盒", "generation_stats": {"total_samples": 100, "successful_samples": 88, "success_rate": 0.88, "attack_success_rate": 0.88, "avg_perturbation_magnitude": 0.066, "generation_time": "80.0秒"}, "quality_metrics": {"avg_l2_norm": 4.32, "avg_linf_norm": 0.066, "original_detection_rate": 1.0, "adversarial_detection_rate": 0.12, "psnr": 32.5, "ssim": 0.92}, "output_files": {"adversarial_samples": "adversarial_samples_kitti_fgsm.zip", "original_samples": "original_samples_kitti.zip", "visualization_files": "perturbation_visualization_kitti.zip", "metadata_file": "generation_metadata.json"}, "adversarial_samples_info": {"sample_count": 88, "format": "numpy_array", "dimensions": [88, 640, 640, 3], "data_type": "float32", "perturbation_range": [0.05, 0.08]}, "original_dataset": "KITTI"}}}
```
***说明：随后一个消息只返回样板名称，不携带路径***


## 4. 执行攻击节点 API (SSE流式响应)

### 4.1 执行车辆识别攻击
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-attack/execute-vehicle-attack-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305",
    "method_type": "车辆识别攻击",
    "algorithm_type": "FGSM攻击",
    "task_type": "攻击执行",
    "task_name": "车辆识别FGSM攻击",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "attack_config": {
      "attack_method": "FGSM",
      "adversarial_samples_name": "adversarial_samples_kitti_fgsm.zip",
      "original_samples_name": "original_samples_kitti.zip",
      "evaluation_mode": "comparison",
      "comparison_baseline": "original"
    },
    "evaluation_config": {
      "attack_effectiveness_metrics": {
        "detection_metrics": {
          "original_detections": true,
          "adversarial_detections": true,
          "detection_drop_rate": true,
          "confidence_reduction": true
        },
        "localization_metrics": {
          "bbox_iou_change": true,
          "missed_detections": true,
          "false_detections": true
        },
        "classification_metrics": {
          "class_change_rate": true,
          "confidence_distribution": true
        }
      },
      "perturbation_metrics": {
        "norm_metrics": {
          "l0_norm": true,
          "l2_norm": true,
          "linf_norm": true,
          "psnr": true,
          "ssim": true
        },
        "visual_metrics": {
          "human_perceptibility": true,
          "structural_similarity": true
        }
      },
      "performance_metrics": {
        "inference_time": true,
        "throughput": true,
        "memory_usage": true,
        "computational_cost": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "current_success_rate",
        "detection_accuracy_drop",
        "confidence_reduction",
        "perturbation_visibility",
        "processing_throughput"
      ],
      "alert_thresholds": {
        "success_rate_alert": 0.7,
        "perturbation_alert": 0.1,
        "performance_degradation": 0.5
      }
    }
  }
}
```
# 执行车辆识别攻击API接口入参说明

## 回调参数 (callback_params)

- **task_run_id**: 任务运行唯一标识符，用于追踪和管理任务执行流程
- **method_type**: 方法类型分类，标识当前任务所属的方法类别
- **algorithm_type**: 具体算法类型，指定使用的攻击算法类别
- **task_type**: 任务类型分类，定义任务的操作类型
- **task_name**: 具体任务名称，详细描述当前执行的任务
- **parent_task_id**: 父任务标识符，用于任务链的关联和管理
- **user_name**: 执行用户名称，记录任务执行者信息

## 业务参数 (business_params)

- **user_name**: 执行用户名称，记录任务执行者信息
- **scene_instance_id**: 场景实例标识符，关联具体的演练场景实例

## 攻击配置 (attack_config)

- **attack_method**: 攻击算法名称，指定使用的对抗攻击算法
- **adversarial_samples_name**: 对抗样本文件名，指定用于攻击的对抗样本文件
- **original_samples_name**: 原始样本文件名，指定用于对比的原始样本文件
- **evaluation_mode**: 评估模式，定义评估的方式
- **comparison_baseline**: 对比基线，指定对比的基准

## 评估配置 (evaluation_config)

### 攻击效果指标 (attack_effectiveness_metrics)

- **detection_metrics**: 检测指标配置
  - **original_detections**: 原始样本检测结果
  - **adversarial_detections**: 对抗样本检测结果
  - **detection_drop_rate**: 检测率下降指标
  - **confidence_reduction**: 置信度下降指标

- **localization_metrics**: 定位指标配置
  - **bbox_iou_change**: 边界框IOU变化指标
  - **missed_detections**: 漏检指标
  - **false_detections**: 误检指标

- **classification_metrics**: 分类指标配置
  - **class_change_rate**: 类别改变率指标
  - **confidence_distribution**: 置信度分布指标

### 扰动指标 (perturbation_metrics)

- **norm_metrics**: 范数指标配置
  - **l0_norm**: L0范数指标
  - **l2_norm**: L2范数指标
  - **linf_norm**: L无穷范数指标
  - **psnr**: 峰值信噪比指标
  - **ssim**: 结构相似性指标

- **visual_metrics**: 视觉指标配置
  - **human_perceptibility**: 人类可感知性指标
  - **structural_similarity**: 结构相似性指标

### 性能指标 (performance_metrics)

- **inference_time**: 推理时间指标
- **throughput**: 吞吐量指标
- **memory_usage**: 内存使用指标
- **computational_cost**: 计算成本指标

## 监控配置 (monitoring_config)

- **real_time_metrics**: 实时监控指标列表
  - **current_success_rate**: 当前成功率
  - **detection_accuracy_drop**: 检测准确率下降
  - **confidence_reduction**: 置信度下降
  - **perturbation_visibility**: 扰动可见性
  - **processing_throughput**: 处理吞吐量

- **alert_thresholds**: 告警阈值配置
  - **success_rate_alert**: 成功率告警阈值
  - **perturbation_alert**: 扰动告警阈值
  - **performance_degradation**: 性能下降告警阈值

***OUTPUT***
```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:00:123", "data": {"event": "attack_process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始车辆识别FGSM攻击执行任务", "log": "[5%] 开始车辆识别FGSM攻击执行任务 - 目标检测模型: YOLOv5\n", "details": {"attack_method": "FGSM", "target_model": "YOLOv5", "total_samples": 100, "batch_size": 10}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:01:456", "data": {"event": "sample_loading", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "攻击样本加载中", "log": "[15%] 加载原始样本: original_samples_kitti.zip，对抗样本: adversarial_samples_kitti_fgsm.zip\n", "details": {"original_samples": "original_samples_kitti.zip", "adversarial_samples": "adversarial_samples_kitti_fgsm.zip", "total_available_samples": 100, "selected_sample_count": 100, "sample_categories": ["car", "truck", "bus", "motorcycle"]}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:02:789", "data": {"event": "sample_preprocessing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "样本预处理完成", "log": "[25%] 样本预处理完成 - 图像标准化、尺寸调整、数据增强\n", "details": {"preprocessing_steps": ["image_normalization", "resize_640x640", "data_augmentation"], "final_sample_count": 100, "category_distribution": {"car": 60, "truck": 20, "bus": 15, "motorcycle": 5}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:03:123", "data": {"event": "model_loading", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 35, "message": "目标检测模型加载完成", "log": "[35%] YOLOv5模型加载成功，准备执行车辆识别\n", "details": {"target_model": "YOLOv5", "model_version": "v6.0", "input_size": "640x640", "confidence_threshold": 0.5, "iou_threshold": 0.45}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:05:456", "data": {"event": "attack_initialization", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 45, "message": "FGSM攻击初始化完成", "log": "[45%] FGSM攻击参数配置完成 - epsilon: 0.03, 迭代次数: 1\n", "details": {"attack_technique": "Fast Gradient Sign Method", "epsilon": 0.03, "iterations": 1, "target_model_state": "loaded", "memory_usage": "2.3GB"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:06:789", "data": {"event": "baseline_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 50, "message": "开始基线评估", "log": "[50%] 对原始样本进行基线检测评估\n", "details": {"evaluation_phase": "baseline", "samples_processed": 0, "total_samples": 100, "current_batch": 1, "total_batches": 10}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:08:123", "data": {"event": "baseline_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 55, "message": "基线评估进行中", "log": "[55%] 批次1/10 - 原始样本检测完成，平均置信度: 0.78\n", "details": {"current_batch": 1, "total_batches": 10, "batch_size": 10, "average_confidence": 0.78, "detection_count": 24}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:12:456", "data": {"event": "adversarial_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 60, "message": "开始对抗样本评估", "log": "[60%] 基线评估完成，开始对抗样本检测评估\n", "details": {"evaluation_phase": "adversarial", "baseline_results": {"total_detections": 245, "average_confidence": 0.76, "mAP": 0.82}, "current_batch": 1, "total_batches": 10}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:15:789", "data": {"event": "adversarial_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 65, "message": "对抗样本评估进行中", "log": "[65%] 批次3/10 - 对抗样本检测，检测率显著下降\n", "details": {"current_batch": 3, "total_batches": 10, "detection_drop_rate": 0.45, "confidence_reduction": 0.32, "successful_attacks": 27}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:18:123", "data": {"event": "real_time_metrics", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 70, "message": "实时监控指标更新", "log": "[70%] 实时指标 - 攻击成功率: 52%, 检测率下降: 45%, 扰动可见性: 低\n", "details": {"current_success_rate": 0.52, "detection_accuracy_drop": 0.45, "confidence_reduction": 0.32, "perturbation_visibility": "low", "processing_throughput": "15.2 samples/sec"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:22:456", "data": {"event": "perturbation_analysis", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 75, "message": "扰动分析完成", "log": "[75%] 扰动指标计算完成 - L2范数: 0.023, PSNR: 38.5dB\n", "details": {"l2_norm": 0.023, "linf_norm": 0.03, "psnr": 38.5, "ssim": 0.92, "human_perceptibility": "imperceptible"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:25:789", "data": {"event": "mid_process_summary", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 80, "message": "攻击评估中期汇总", "log": "[80%] 已完成70个样本评估，当前攻击成功率: 54%\n", "details": {"processed_samples": 70, "total_samples": 100, "current_success_rate": 0.54, "category_breakdown": {"car": 0.58, "truck": 0.52, "bus": 0.45, "motorcycle": 0.35}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:28:123", "data": {"event": "final_evaluation_batch", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "最终批次评估", "log": "[90%] 处理最终批次10/10，完成所有样本评估\n", "details": {"current_batch": 10, "total_batches": 10, "remaining_samples": 10, "estimated_completion_time": "1分钟"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:30:456", "data": {"event": "attack_execution_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 95, "message": "攻击执行完成", "log": "[95%] FGSM攻击评估完成 - 所有100个样本处理完毕\n", "details": {"total_samples": 100, "successful_attacks": 55, "failed_attacks": 45, "overall_success_rate": 0.55, "total_execution_time": "30.3秒"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:32:789", "data": {"event": "results_analysis", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 98, "message": "攻击结果分析完成", "log": "[98%] 攻击结果分析完成，生成详细评估报告\n", "details": {"attack_effectiveness": 0.55, "model_vulnerability_score": 0.62, "detection_metrics": {"detection_drop_rate": 0.48, "confidence_reduction": 0.35, "bbox_iou_change": 0.42}, "perturbation_metrics": {"l2_norm": 0.025, "psnr": 38.2, "ssim": 0.91}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:34:123", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3305", "method_type": "车辆识别攻击", "algorithm_type": "FGSM攻击", "task_type": "攻击执行", "task_name": "车辆识别FGSM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "车辆识别FGSM攻击任务完成", "log": "[100%] 车辆识别FGSM攻击任务完成\n", "details": {"execution_id": "vehicle_attack_202407011435", "attack_method": "FGSM", "target_model": "YOLOv5", "execution_stats": {"total_samples": 100, "successful_attacks": 55, "success_rate": 0.55, "average_inference_time": "0.25秒", "total_execution_time": "34.0秒"}, "effectiveness_analysis": {"detection_drop_rate": 0.48, "confidence_reduction": 0.35, "bbox_iou_change": 0.42, "class_change_rate": 0.12}, "perturbation_analysis": {"l2_norm": 0.025, "linf_norm": 0.03, "psnr": 38.2, "ssim": 0.91, "human_perceptibility": "imperceptible"}, "security_insights": {"critical_vulnerabilities": 6, "high_risk_vulnerabilities": 15, "medium_risk_vulnerabilities": 34, "defense_recommendations": ["adversarial_training", "input_preprocessing", "ensemble_defense", "detection_certification"]}}}}

```

## 5. 释放资源节点 API

### 5.1 释放车辆识别资源
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-attack/release-vehicle-resources-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306",
    "method_type": "车辆识别攻击",
    "algorithm_type": "FGSM攻击",
    "task_type": "资源释放",
    "task_name": "车辆识别资源释放",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "release_config": {
      "models_to_release": ["YOLOv5"],
      "datasets_to_clear": [],  // 改为空数组，不清除原始数据集
      "clear_cache": true,
      "release_memory": true,
      "cleanup_temp_files": false,  // 改为false，保留临时文件中的样本数据
      "preserve_results": true,
      "results_paths": [
        "/results/adversarial/fgsm", 
        "/results/attack_results/fgsm",
        "/results/original_samples/vehicle"  // 新增原始样本路径
      ],
      "preserve_samples_config": {  // 新增样本保留配置
        "preserve_adversarial_samples": true,
        "preserve_original_samples": true,
        "preserve_attack_intermediates": true,
        "samples_retention_days": 30,
        "preserve_attack_artifacts": ["perturbation_maps", "attack_logs"]
      }
    },
    "verification_config": {
      "verify_memory_release": true,
      "verify_gpu_clearance": true,
      "generate_cleanup_report": true,
      "validate_resource_recovery": true,
      "verify_samples_preserved": true,  // 新增样本保留验证
      "verify_attack_results_intact": true  // 新增攻击结果完整性验证
    }
  }
}
```

**OUTPUT**:
```json
{
  "resp_code": 0,
  "resp_msg": "资源释放成功，攻击样本和结果数据已保留",
  "data": {
    "release_id": "release_vehicle_fgsm_202407011450",
    "release_status": {
      "models_released": ["YOLOv5"],
      "datasets_cleared": [],  // 显示未清除任何数据集
      "adversarial_samples_preserved": ["fgsm_adversarial_samples"],  // 新增对抗样本保留状态
      "original_samples_preserved": ["KITTI_original_samples"],  // 新增原始样本保留状态
      "memory_freed": "2.8GB",
      "gpu_memory_cleared": true,
      "cache_cleaned": true,
      "temp_files_removed": false,  // 改为false，反映临时文件保留
      "results_preserved": true,
      "samples_preserved": true  // 新增样本保留状态
    },
    "resource_recovery": {
      "gpu_memory_available": "7.9GB",
      "cpu_usage": "8%",
      "memory_usage": "1.5GB",
      "gpu_utilization": "5%"
    },
    "cleanup_report": {
      "total_models_released": 1,
      "total_memory_freed": "2.8GB",
      "cache_size_cleared": "450MB",
      "temp_files_removed_count": 0,  // 改为0，反映未删除临时文件
      "datasets_cleared_count": 0,  // 改为0
      "samples_preserved_count": 156,  // 新增保留的样本数量
      "attack_artifacts_preserved": ["perturbation_maps", "attack_logs"],  // 新增攻击产物保留
      "cleanup_duration": "3.2秒"
    },
    "preserved_attack_artifacts": {  // 新增攻击产物保留信息
      "adversarial_samples": "/results/adversarial/fgsm",
      "original_samples": "/results/original_samples/vehicle",
      "attack_results": "/results/attack_results/fgsm",
      "perturbation_maps": "/results/attack_results/fgsm/perturbation_maps",
      "attack_logs": "/results/attack_results/fgsm/logs",
      "performance_metrics": "fgsm_attack_metrics_202407011435.json",
      "attack_configuration": "fgsm_attack_config_202407011435.json"
    },
    "sample_access_info": {  // 新增样本访问信息
      "adversarial_samples_count": 88,
      "original_samples_count": 68,
      "attack_success_rate": "92%",
      "sample_viewer_url": "http://127.0.0.1:19001/attack-sample-viewer/fgsm_202407011435",
      "comparison_tool_url": "http://127.0.0.1:19001/attack-comparison/fgsm_202407011435",
      "retention_period": "30 days",
      "access_instructions": "通过攻击样本查看器API访问保留的攻击数据"
    }
  },
  "time_stamp": "2024/07/01-14:50:15:123"
}
```



## 1. 加载模型节点 API

### 1.1 加载车辆识别模型
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-defense/load-model-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
    "method_type": "车辆识别防御",
    "algorithm_type": "防御模型加载",
    "task_type": "模型加载",
    "task_name": "车辆识别防御模型加载",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "model_config": {
      "target_model": "YOLOv5",
      "model_type": "object_detection",
      "defense_model": "image_compression",
      "model_version": "v6.0"
    },
    "hardware_config": {
      "deployment_type": "single_gpu",
      "gpu_config": {
        "gpu_devices": [0],
        "memory_per_device": "8GB",
        "gpu_utilization_threshold": 0.8
      },
      "parallel_strategy": {
        "strategy_type": "none",
        "data_parallel_config": {
          "replication_count": 1,
          "gradient_accumulation": 1,
          "synchronization_mode": "sync",
          "load_balancing": false
        }
      }
    },
    "defense_config": {
      "defense_method": "image_compression",
      "defense_strength": 5,
      "defense_type": "input_preprocessing",
      "quality_factor": 75,
      "compression_type": "jpeg"
    },
    "inference_config": {
      "batch_size": 8,
      "max_detections": 100,
      "nms_threshold": 0.45,
      "confidence_threshold": 0.5,
      "input_size": [640, 640]
    },
    "performance_config": {
      "warmup_iterations": 10,
      "optimization_level": "O1",
      "memory_efficient": true
    }
  }
}
```
# 车辆识别防御模型加载接口参数介绍

## 回调参数 (callback_params)

**task_run_id** - 任务运行唯一标识符，用于在整个系统中追踪和管理特定任务实例的执行状态和生命周期

**method_type** - 方法类型分类，标识当前任务属于车辆识别防御这一大类方法，便于系统进行方法级别的管理和统计

**algorithm_type** - 算法类型细分，具体指定为防御模型加载，区别于攻击模型或其他类型的模型加载操作

**task_type** - 任务类型定义，明确当前执行的是模型加载阶段的操作，用于工作流的状态管理

**task_name** - 具体任务描述，提供更详细的任务说明，便于用户理解和系统日志记录

**parent_task_id** - 父任务标识符，在复杂的工作流场景中建立任务间的层级关系，支持任务链的追踪和管理

**user_name** - 执行用户名称，用于系统审计、权限验证和资源配额管理

## 业务参数 (business_params)

**user_name** - 重复的用户名字段，在业务层面再次确认执行者身份，提供双重验证机制

**scene_instance_id** - 场景实例唯一标识，关联特定的防御演练环境配置和资源分配

### 模型配置 (model_config)

**target_model** - 目标模型指定，当前配置为YOLOv5，这是业界广泛使用的目标检测模型

**model_type** - 模型功能类型，标识为目标检测模型，区别于分类、分割等其他计算机视觉任务

**defense_model** - 防御模型类型，指定使用图像压缩作为防御手段，这是输入预处理类防御的典型代表

**model_version** - 模型版本控制，确保加载特定版本的模型文件，保证实验的可复现性

### 硬件配置 (hardware_config)

**deployment_type** - 部署模式选择，单GPU部署适合中小规模的防御演练场景

**gpu_config** - GPU资源配置，指定使用0号GPU设备，设置8GB显存限制和80%的利用率阈值

**parallel_strategy** - 并行策略配置，当前设置为不使用任何并行策略，保持简单的单进程推理

### 防御配置 (defense_config)

**defense_method** - 防御技术选择，图像压缩通过降低图像质量来消除对抗扰动的影响

**defense_strength** - 防御强度参数，数值5代表中等防御强度，在效果和性能间取得平衡

**defense_type** - 防御机制分类，输入预处理类防御在推理前对输入数据进行处理

**quality_factor** - 图像质量因子，75%的质量压缩能在保持可接受视觉效果的同时有效防御攻击

**compression_type** - 压缩算法选择，JPEG格式提供良好的压缩效率和广泛的兼容性

### 推理配置 (inference_config)

**batch_size** - 批处理大小，设置为8在推理速度和内存使用间取得平衡

**max_detections** - 最大检测数量，限制每张图像最多检测100个目标，防止过度检测

**nms_threshold** - 非极大值抑制阈值，0.45能有效过滤重叠检测框，提高检测精度

**confidence_threshold** - 置信度阈值，0.5过滤掉低置信度的检测结果，提高检测可靠性

**input_size** - 输入图像尺寸，YOLOv5标准输入尺寸640x640，确保模型正常推理

### 性能配置 (performance_config)

**warmup_iterations** - 预热迭代次数，10次预热确保模型推理性能达到稳定状态

**optimization_level** - 优化级别选择，O1级别在性能和模型精度间提供良好平衡

**memory_efficient** - 内存优化标志，启用内存优化减少不必要的内存占用，提高系统稳定性

这套参数配置为车辆识别防御演练提供了完整的模型加载和环境初始化方案，确保后续的对抗样本生成和防御执行能够顺利进行。

**OUTPUT**:
```json
{
  "resp_code": 0,
  "resp_msg": "防御模型加载成功",
  "data": {
    "environment_id": "env_vehicle_defense_20240715001",
    "model_status": {
      "target_model": "YOLOv5",
      "defense_model": "image_compression",
      "model_type": "object_detection",
      "model_version": "v6.0",
      "model_size": "14.4MB",
      "load_time": "1.8s",
      "status": "loaded",
      "defense_status": "active",
      "inference_endpoint": "127.0.0.1:19001/api-ai-server/inference/env_vehicle_defense_20240715001"
    },
    "resource_usage": {
      "gpu_memory_used": "1.2GB",
      "cpu_usage": "25%",
      "memory_usage": "2.8GB",
      "disk_usage": "0.8GB"
    },
    "defense_capabilities": {
      "supported_attacks": ["FGSM", "PGD", "MIM", "C&W", "DeepFool", "BadNet"],
      "defense_metrics": ["detection_accuracy", "robustness_score", "processing_latency"]
    }
  },
  "time_stamp": "2024/07/01-09:19:32:679"
}
```

## 2. 生成对抗样本节点 API (SSE流式响应)

### 2.1 生成对抗样本用于防御测试
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-defense/generate-adversarial-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302",
    "method_type": "车辆识别防御",
    "algorithm_type": "对抗样本生成",
    "task_type": "样本生成",
    "task_name": "防御测试对抗样本生成",
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
        "loss_function": "object_detection_loss",
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
      "target_model": "YOLOv5",
      "model_state": "loaded",
      "defense_model": "none"
    },
    "defense_testing_config": {
      "test_scenarios": ["baseline", "defense_applied"],
      "evaluation_metrics": ["detection_accuracy", "robustness", "processing_time"]
    }
  }
}
```
# 生成对抗样本用于防御测试接口参数介绍

## 回调参数 (callback_params)

**task_run_id** - 任务运行唯一标识符，用于在整个防御演练流程中追踪对抗样本生成任务的执行状态和进度

**method_type** - 方法类型分类，明确标识为车辆识别防御，区别于攻击或其他类型的任务

**algorithm_type** - 算法类型细分，具体指定为对抗样本生成，这是防御测试的关键前置步骤

**task_type** - 任务类型定义，样本生成表明当前执行的是生成用于测试的对抗样本

**task_name** - 具体任务描述，防御测试对抗样本生成清晰说明了任务的目的和性质

**parent_task_id** - 父任务标识符，与模型加载任务建立关联，形成完整的工作流链条

**user_name** - 执行用户名称，用于权限验证、资源分配和操作审计

## 业务参数 (business_params)

**user_name** - 重复的用户名字段，在业务操作层面确认执行者身份

**scene_instance_id** - 场景实例标识，关联特定的防御演练环境和资源配置

### 生成配置 (generation_config)

**attack_algorithm** - 攻击算法选择，FGSM（快速梯度符号法）是经典的对抗攻击方法，计算效率高

#### 数据集配置 (dataset_config)

**dataset_name** - 数据集名称，KITTI是广泛使用的车辆检测数据集，包含丰富的道路场景

**dataset_format** - 数据格式，图像格式适合计算机视觉任务的对抗样本生成

**total_samples** - 数据集总样本数，7481个样本提供了充足的测试基础

**selected_samples** - 选择样本数，100个样本在测试效率和覆盖率间取得平衡

**sample_selection_strategy** - 样本选择策略，随机选择确保样本的代表性和无偏性

#### 算法参数 (algorithm_parameters)

**epsilon** - 扰动强度参数，0.08的扰动强度在攻击效果和视觉质量间提供良好平衡

**step_size** - 步长参数，控制每次迭代的扰动更新幅度

**max_iterations** - 最大迭代次数，限制优化过程的计算复杂度

**targeted** - 攻击类型标志，false表示无目标攻击，旨在降低整体检测性能

**target_class** - 目标类别，-1表示不指定特定类别，进行通用攻击

**random_start** - 随机初始化标志，false表示从原始样本开始生成扰动

**loss_function** - 损失函数类型，目标检测损失针对检测任务的特点进行优化

**optimization_method** - 优化方法选择，梯度上升法是生成对抗样本的标准方法

#### 约束条件 (constraints)

**perturbation_norm** - 扰动范数类型，L无穷范数限制每个像素的最大扰动

**max_perturbation** - 最大扰动限制，确保扰动在可接受范围内

**clip_min/max** - 像素值范围限制，保持生成样本在有效像素值范围内

**spatial_constraints** - 空间约束配置，当前禁用，允许在全图范围内添加扰动

### 模型配置 (model_config)

**target_model** - 目标模型指定，YOLOv5作为攻击的目标检测模型

**model_state** - 模型状态标识，loaded表示模型已成功加载并准备就绪

**defense_model** - 防御模型状态，none表示在生成对抗样本时不应用任何防御

### 防御测试配置 (defense_testing_config)

**test_scenarios** - 测试场景定义，包含基线测试和防御应用两种测试模式

**evaluation_metrics** - 评估指标选择，涵盖检测精度、鲁棒性和处理时间三个维度


## 1. 图片放缩防御 INPUT

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
    "method_type": "车辆识别防御",
    "algorithm_type": "图片放缩防御",
    "task_type": "防御执行",
    "task_name": "车辆识别图片放缩防御",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "defense_config": {
      "defense_method": "image_scaling",
      "adversarial_samples_name": "adversarial_samples_kitti_fgsm.zip",
      "original_samples_name": "original_samples_kitti.zip",
      "defense_strength": 5,
      "scaling_factor": 0.8,
      "interpolation_method": "bilinear",
      "resize_strategy": "downscale_upscale"
    },
    "evaluation_config": {
      "defense_effectiveness_metrics": {
        "detection_recovery_metrics": {
          "recovered_detections": true,
          "detection_recovery_rate": true,
          "confidence_restoration": true,
          "false_positive_impact": true
        },
        "robustness_metrics": {
          "defense_success_rate": true,
          "attack_success_reduction": true,
          "model_accuracy_preservation": true
        },
        "performance_metrics": {
          "processing_latency": true,
          "throughput_impact": true,
          "resource_utilization": true
        }
      },
      "comparison_metrics": {
        "baseline_comparison": true,
        "defense_improvement": true,
        "tradeoff_analysis": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "defense_success_rate",
        "detection_recovery",
        "processing_latency",
        "confidence_restoration",
        "resource_usage"
      ],
      "alert_thresholds": {
        "defense_success_alert": 0.6,
        "latency_alert": 100,
        "accuracy_drop_alert": 0.2
      }
    }
  }
}
```

## 2. PGD防御 INPUT

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
    "method_type": "车辆识别防御",
    "algorithm_type": "PGD防御",
    "task_type": "防御执行",
    "task_name": "车辆识别PGD防御",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "defense_config": {
      "defense_method": "pgd_defense",
      "adversarial_samples_name": "adversarial_samples_kitti_fgsm.zip",
      "original_samples_name": "original_samples_kitti.zip",
      "defense_strength": 5,
      "pgd_parameters": {
        "epsilon": 0.03,
        "step_size": 0.01,
        "num_steps": 7,
        "random_start": true,
        "norm_type": "linf"
      },
      "defense_mode": "adversarial_training",
      "model_robustness": "enhanced"
    },
    "evaluation_config": {
      "defense_effectiveness_metrics": {
        "detection_recovery_metrics": {
          "recovered_detections": true,
          "detection_recovery_rate": true,
          "confidence_restoration": true,
          "false_positive_impact": true
        },
        "robustness_metrics": {
          "defense_success_rate": true,
          "attack_success_reduction": true,
          "model_accuracy_preservation": true,
          "robust_accuracy": true
        },
        "performance_metrics": {
          "processing_latency": true,
          "throughput_impact": true,
          "resource_utilization": true
        }
      },
      "comparison_metrics": {
        "baseline_comparison": true,
        "defense_improvement": true,
        "tradeoff_analysis": true,
        "robustness_comparison": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "defense_success_rate",
        "detection_recovery",
        "processing_latency",
        "confidence_restoration",
        "resource_usage",
        "robust_accuracy"
      ],
      "alert_thresholds": {
        "defense_success_alert": 0.6,
        "latency_alert": 100,
        "accuracy_drop_alert": 0.2,
        "robustness_threshold": 0.7
      }
    }
  }
}
```

## 3. Neural Cleanse防御 INPUT

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
    "method_type": "车辆识别防御",
    "algorithm_type": "Neural Cleanse防御",
    "task_type": "防御执行",
    "task_name": "车辆识别Neural Cleanse防御",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "defense_config": {
      "defense_method": "neural_cleanse",
      "adversarial_samples_name": "adversarial_samples_kitti_fgsm.zip",
      "original_samples_name": "original_samples_kitti.zip",
      "defense_strength": 5,
      "neural_cleanse_parameters": {
        "trigger_detection": true,
        "anomaly_index_threshold": 2.0,
        "pruning_ratio": 0.1,
        "fine_tuning_epochs": 10,
        "backdoor_mitigation": "pruning_finetuning"
      },
      "backdoor_defense": {
        "detect_backdoors": true,
        "mitigate_backdoors": true,
        "validate_cleanliness": true
      }
    },
    "evaluation_config": {
      "defense_effectiveness_metrics": {
        "detection_recovery_metrics": {
          "recovered_detections": true,
          "detection_recovery_rate": true,
          "confidence_restoration": true,
          "false_positive_impact": true
        },
        "robustness_metrics": {
          "defense_success_rate": true,
          "attack_success_reduction": true,
          "model_accuracy_preservation": true,
          "backdoor_detection_rate": true
        },
        "security_metrics": {
          "anomaly_index": true,
          "trigger_detection_accuracy": true,
          "model_cleanliness": true
        },
        "performance_metrics": {
          "processing_latency": true,
          "throughput_impact": true,
          "resource_utilization": true
        }
      },
      "comparison_metrics": {
        "baseline_comparison": true,
        "defense_improvement": true,
        "tradeoff_analysis": true,
        "security_improvement": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "defense_success_rate",
        "detection_recovery",
        "processing_latency",
        "confidence_restoration",
        "resource_usage",
        "anomaly_index",
        "backdoor_detection"
      ],
      "alert_thresholds": {
        "defense_success_alert": 0.6,
        "latency_alert": 100,
        "accuracy_drop_alert": 0.2,
        "anomaly_alert": 3.0,
        "backdoor_detection_alert": 0.8
      }
    }
  }
}
```

## 主要参数说明

### 图片放缩防御特有参数：
- `scaling_factor`: 缩放因子 (0.5-2.0)
- `interpolation_method`: 插值方法 (bilinear, bicubic, nearest)
- `resize_strategy`: 缩放策略 (downscale_upscale, direct_resize)

### PGD防御特有参数：
- `pgd_parameters`: PGD相关参数
  - `epsilon`: 扰动上限
  - `step_size`: 步长
  - `num_steps`: 迭代次数
  - `random_start`: 随机初始化
  - `norm_type`: 范数类型
- `defense_mode`: 防御模式 (adversarial_training, robust_inference)
- `model_robustness`: 模型鲁棒性级别

### Neural Cleanse防御特有参数：
- `neural_cleanse_parameters`: Neural Cleanse算法参数
  - `trigger_detection`: 触发模式检测
  - `anomaly_index_threshold`: 异常指数阈值
  - `pruning_ratio`: 剪枝比例
  - `fine_tuning_epochs`: 微调轮数
  - `backdoor_mitigation`: 后门缓解策略
- `backdoor_defense`: 后门防御配置



**SSE流式响应OUTPUT**:
```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:00:123", "data": {"event": "process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始防御测试对抗样本生成", "log": "[5%] 开始FGSM对抗样本生成用于防御测试\n", "details": {"attack_method": "FGSM", "target_model": "YOLOv5", "defense_method": "image_compression", "max_samples": 100}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:01:456", "data": {"event": "dataset_loading", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "数据集加载完成", "log": "[15%] 加载KITTI数据集100个样本\n", "details": {"dataset_name": "KITTI", "sample_count": 100, "vehicle_classes": ["car", "truck", "bus", "motorcycle"]}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:02:789", "data": {"event": "model_preparation", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "目标模型准备完成", "log": "[25%] YOLOv5模型准备就绪，开始生成对抗样本\n", "details": {"model_status": "ready", "attack_parameters": {"epsilon": 0.08, "targeted": false}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:03:123", "data": {"event": "generation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 30, "message": "开始生成对抗样本", "log": "[30%] 开始FGSM对抗样本生成，扰动参数epsilon=0.08\n", "details": {"attack_params": {"epsilon": 0.08, "targeted": false, "norm_type": "inf"}, "batch_size": 20}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:15:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 40, "message": "样本处理中", "log": "[40%] 正在处理样本20/100 - 计算梯度\n", "details": {"current_sample": 20, "total_samples": 100, "step": "gradient_computation", "batch_progress": "1/5"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:27:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 50, "message": "样本处理中", "log": "[50%] 正在处理样本40/100 - 应用扰动\n", "details": {"current_sample": 40, "total_samples": 100, "step": "perturbation_application", "batch_progress": "2/5"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:40:123", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 60, "message": "样本处理中", "log": "[60%] 正在处理样本60/100 - 验证攻击效果\n", "details": {"current_sample": 60, "total_samples": 100, "step": "attack_validation", "batch_progress": "3/5", "current_success_rate": 0.85}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:52:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 70, "message": "样本处理中", "log": "[70%] 正在处理样本80/100 - 应用扰动\n", "details": {"current_sample": 80, "total_samples": 100, "step": "perturbation_application", "batch_progress": "4/5", "current_success_rate": 0.87}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:04:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 80, "message": "样本处理中", "log": "[80%] 正在处理样本100/100 - 最终验证\n", "details": {"current_sample": 100, "total_samples": 100, "step": "final_validation", "batch_progress": "5/5", "current_success_rate": 0.88}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:05:123", "data": {"event": "generation_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 85, "message": "对抗样本生成完成", "log": "[85%] 对抗样本生成完成 - 成功生成88/100样本\n", "details": {"total_samples": 100, "successful_samples": 88, "success_rate": 0.88, "generation_time": "65.0秒", "avg_perturbation": 0.066, "attack_success_rate": 0.88}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:10:456", "data": {"event": "defense_preparation", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "防御测试准备完成", "log": "[90%] 防御测试样本准备就绪\n", "details": {"adversarial_samples": "adversarial_samples_kitti_fgsm", "original_samples": "original_samples_kitti", "defense_method": "image_compression"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:15:789", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3302", "method_type": "车辆识别防御", "algorithm_type": "对抗样本生成", "task_type": "样本生成", "task_name": "防御测试对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "防御测试对抗样本生成任务完成", "log": "[100%] 防御测试对抗样本生成任务完成\n", "details": {"generation_id": "gen_defense_test_202407011015", "attack_method": "FGSM", "attack_type": "白盒", "defense_method": "image_compression", "generation_stats": {"total_samples": 100, "successful_samples": 88, "success_rate": 0.88, "attack_success_rate": 0.88, "avg_perturbation_magnitude": 0.066, "generation_time": "75.0秒"}, "quality_metrics": {"avg_l2_norm": 4.32, "avg_linf_norm": 0.066, "original_detection_rate": 1.0, "adversarial_detection_rate": 0.12, "psnr": 32.5, "ssim": 0.92}, "output_files": {"adversarial_samples": "adversarial_samples_kitti_fgsm.zip", "original_samples": "original_samples_kitti.zip", "visualization_files": "perturbation_visualization_kitti.zip", "metadata_file": "generation_metadata.json"}, "adversarial_samples_info": {"sample_count": 88, "format": "numpy_array", "dimensions": [88, 640, 640, 3], "data_type": "float32", "perturbation_range": [0.05, 0.08]}, "original_samples_info": {"sample_count": 100, "format": "numpy_array", "dimensions": [100, 640, 640, 3], "data_type": "float32", "dataset_source": "KITTI"}, "original_dataset": "KITTI", "ready_for_defense_test": true}}}
```

## 3. 执行防御节点 API (SSE流式响应)

### 3.1 执行车辆识别防御
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-defense/execute-defense-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303",
    "method_type": "车辆识别防御",
    "algorithm_type": "图片压缩防御",
    "task_type": "防御执行",
    "task_name": "车辆识别图片压缩防御",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "defense_config": {
      "defense_method": "image_compression",
      "adversarial_samples_name": "adversarial_samples_kitti_fgsm.zip",
      "original_samples_name": "original_samples_kitti.zip",
      "defense_strength": 5,
      "quality_factor": 75,
      "compression_type": "jpeg"
    },
    "evaluation_config": {
      "defense_effectiveness_metrics": {
        "detection_recovery_metrics": {
          "recovered_detections": true,
          "detection_recovery_rate": true,
          "confidence_restoration": true,
          "false_positive_impact": true
        },
        "robustness_metrics": {
          "defense_success_rate": true,
          "attack_success_reduction": true,
          "model_accuracy_preservation": true
        },
        "performance_metrics": {
          "processing_latency": true,
          "throughput_impact": true,
          "resource_utilization": true
        }
      },
      "comparison_metrics": {
        "baseline_comparison": true,
        "defense_improvement": true,
        "tradeoff_analysis": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "defense_success_rate",
        "detection_recovery",
        "processing_latency",
        "confidence_restoration",
        "resource_usage"
      ],
      "alert_thresholds": {
        "defense_success_alert": 0.6,
        "latency_alert": 100,
        "accuracy_drop_alert": 0.2
      }
    }
  }
}
```

# 执行车辆识别防御接口参数介绍

## 回调参数 (callback_params)

**task_run_id** - 任务运行唯一标识符，用于在执行防御测试过程中追踪任务状态、管理任务生命周期和记录执行日志

**method_type** - 方法类型分类，明确标识为车辆识别防御，区别于攻击或其他类型的计算机视觉任务

**algorithm_type** - 算法类型细分，具体指定为图片压缩防御，这是输入预处理类防御的典型代表

**task_type** - 任务类型定义，防御执行表明当前正在进行具体的防御机制测试和评估

**task_name** - 具体任务描述，车辆识别图片压缩防御清晰说明了防御对象、防御方法和应用场景

**parent_task_id** - 父任务标识符，与对抗样本生成任务建立关联，形成完整的防御测试工作流

**user_name** - 执行用户名称，用于系统权限验证、资源配额管理和操作审计追踪

## 业务参数 (business_params)

**user_name** - 重复的用户名字段，在业务操作层面提供双重身份验证，确保操作安全性

**scene_instance_id** - 场景实例唯一标识，关联特定的防御演练环境配置、资源分配和实验数据

### 防御配置 (defense_config)

**defense_method** - 防御技术选择，图像压缩通过有损压缩来消除对抗扰动，是简单有效的防御手段

**adversarial_samples_name** - 对抗样本文件名，指定使用FGSM方法在KITTI数据集上生成的对抗样本

**original_samples_name** - 原始样本文件名，提供基线性能评估的基准数据

**defense_strength** - 防御强度参数，数值5代表中等防御强度，在防御效果和图像质量损失间平衡

**quality_factor** - 图像质量因子，75%的质量压缩能在保持可接受视觉效果的同时有效防御攻击

**compression_type** - 压缩算法选择，JPEG格式提供良好的压缩效率和计算性能

### 评估配置 (evaluation_config)

#### 防御效果指标 (defense_effectiveness_metrics)

**检测恢复指标 (detection_recovery_metrics)** - 衡量防御机制恢复被攻击破坏的检测能力
- **recovered_detections** - 恢复的检测数量，统计防御后重新正确检测的目标数量
- **detection_recovery_rate** - 检测恢复率，计算防御后恢复检测的比例
- **confidence_restoration** - 置信度恢复程度，评估防御对检测置信度的提升效果
- **false_positive_impact** - 误检影响分析，监控防御机制可能引入的误检问题

**鲁棒性指标 (robustness_metrics)** - 评估防御机制的整体鲁棒性表现
- **defense_success_rate** - 防御成功率，统计成功抵御攻击的样本比例
- **attack_success_reduction** - 攻击成功率降低程度，量化防御对攻击效果的削弱
- **model_accuracy_preservation** - 模型精度保持度，评估防御对原始检测精度的影响

**性能指标 (performance_metrics)** - 分析防御机制的计算性能开销
- **processing_latency** - 处理延迟，测量防御机制引入的时间开销
- **throughput_impact** - 吞吐量影响，评估防御对系统处理能力的影响
- **resource_utilization** - 资源利用率，监控防御过程中的计算资源消耗

#### 对比指标 (comparison_metrics)

**baseline_comparison** - 基线对比分析，将防御效果与无防御状态进行系统性比较

**defense_improvement** - 防御改进程度，量化防御机制带来的性能提升

**tradeoff_analysis** - 权衡分析，评估防御效果与性能开销之间的平衡关系

### 监控配置 (monitoring_config)

**real_time_metrics** - 实时监控指标列表，在防御执行过程中持续跟踪关键性能参数
- **defense_success_rate** - 实时防御成功率监控
- **detection_recovery** - 实时检测恢复情况追踪
- **processing_latency** - 实时处理延迟监控
- **confidence_restoration** - 实时置信度恢复跟踪
- **resource_usage** - 实时资源使用情况监控

**alert_thresholds** - 告警阈值配置，设置性能指标的临界值用于异常检测
- **defense_success_alert** - 防御成功率告警阈值，低于60%触发告警
- **latency_alert** - 处理延迟告警阈值，超过100毫秒触发告警
- **accuracy_drop_alert** - 精度下降告警阈值，原始精度下降超过20%触发告警


**SSE流式响应OUTPUT**:
```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:00:123", "data": {"event": "defense_process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始车辆识别防御执行任务", "log": "[5%] 开始图片压缩防御执行 - 目标模型: YOLOv5\n", "details": {"defense_method": "image_compression", "target_model": "YOLOv5", "total_samples": 88, "defense_strength": 5}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:01:456", "data": {"event": "defense_initialization", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "防御机制初始化完成", "log": "[15%] 图片压缩防御初始化 - 质量因子: 75, 压缩类型: JPEG\n", "details": {"compression_params": {"quality_factor": 75, "compression_type": "jpeg", "subsampling": "4:2:0"}, "defense_scope": "input_preprocessing"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:02:789", "data": {"event": "baseline_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "开始基线评估", "log": "[25%] 对原始样本进行基线检测评估\n", "details": {"evaluation_phase": "baseline", "samples_processed": 0, "total_samples": 88, "current_batch": 1, "total_batches": 9}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:05:123", "data": {"event": "baseline_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 35, "message": "基线评估进行中", "log": "[35%] 批次3/9 - 原始样本检测完成，平均置信度: 0.82\n", "details": {"current_batch": 3, "total_batches": 9, "batch_size": 10, "average_confidence": 0.82, "detection_count": 28}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:08:456", "data": {"event": "adversarial_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 45, "message": "开始对抗样本评估", "log": "[45%] 基线评估完成，开始对抗样本检测评估\n", "details": {"evaluation_phase": "adversarial", "baseline_results": {"total_detections": 245, "average_confidence": 0.81, "mAP": 0.84}, "current_batch": 1, "total_batches": 9}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:10:789", "data": {"event": "adversarial_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 55, "message": "对抗样本评估进行中", "log": "[55%] 批次4/9 - 对抗样本检测，检测率显著下降\n", "details": {"current_batch": 4, "total_batches": 9, "detection_drop_rate": 0.48, "confidence_reduction": 0.38, "successful_attacks": 42}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:13:123", "data": {"event": "defense_application_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 65, "message": "开始应用防御机制", "log": "[65%] 开始应用图片压缩防御\n", "details": {"defense_application": "image_compression", "quality_factor": 75, "current_batch": 1, "total_batches": 9}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:15:456", "data": {"event": "defense_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 75, "message": "防御效果评估中", "log": "[75%] 批次6/9 - 防御后检测率明显恢复\n", "details": {"current_batch": 6, "total_batches": 9, "detection_recovery_rate": 0.62, "confidence_restoration": 0.45, "defense_success": 55}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:18:789", "data": {"event": "real_time_metrics", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 85, "message": "实时防御指标更新", "log": "[85%] 实时指标 - 防御成功率: 72%, 检测恢复率: 62%, 处理延迟: 15ms\n", "details": {"defense_success_rate": 0.72, "detection_recovery": 0.62, "processing_latency": 15, "confidence_restoration": 0.45, "resource_usage": "中等"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:20:123", "data": {"event": "defense_comparison_analysis", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "防御对比分析完成", "log": "[90%] 防御前后对比分析完成\n", "details": {"performance_comparison": {"baseline_accuracy": 0.84, "adversarial_accuracy": 0.12, "defended_accuracy": 0.72}, "improvement_metrics": {"accuracy_improvement": 0.60, "attack_success_reduction": 0.76}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:22:456", "data": {"event": "defense_execution_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 95, "message": "防御执行完成", "log": "[95%] 图片压缩防御评估完成 - 所有88个样本处理完毕\n", "details": {"total_samples": 88, "successful_defenses": 63, "failed_defenses": 25, "overall_success_rate": 0.72, "total_execution_time": "22.3秒"}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:24:789", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3303", "method_type": "车辆识别防御", "algorithm_type": "图片压缩防御", "task_type": "防御执行", "task_name": "车辆识别图片压缩防御", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "车辆识别防御任务完成", "log": "[100%] 车辆识别图片压缩防御任务完成\n", "details": {"execution_id": "vehicle_defense_202407011435", "defense_method": "image_compression", "target_model": "YOLOv5", "defense_stats": {"total_samples": 88, "successful_defenses": 63, "defense_success_rate": 0.72, "average_processing_time": "0.18秒", "total_execution_time": "24.7秒"}, "effectiveness_analysis": {"detection_recovery_rate": 0.62, "confidence_restoration": 0.45, "attack_success_reduction": 0.76, "model_accuracy_preservation": 0.86}, "performance_analysis": {"processing_latency": 15, "throughput_impact": "轻微", "resource_utilization": "中等"}, "security_assessment": {"robustness_score": 0.72, "vulnerability_reduction": 0.68, "defense_recommendations": ["可尝试更高压缩质量", "结合其他防御方法", "优化预处理参数"]}}}}
```

## 4. 释放资源节点 API

### 4.1 释放车辆识别防御资源
**URL**: `127.0.0.1:19001/api-ai-server/vehicle-recognition-defense/release-resources-v1`  
**Method**: POST  
**INPUT**:
```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3304",
    "method_type": "车辆识别防御",
    "algorithm_type": "资源释放",
    "task_type": "资源释放",
    "task_name": "车辆识别防御资源释放",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "release_config": {
      "models_to_release": ["YOLOv5"],
      "defense_models_to_release": ["image_compression"],
      "datasets_to_clear": ["KITTI"],
      "adversarial_samples_to_clear": [],  // 改为空数组，不清除对抗样本
      "clear_cache": true,
      "release_memory": true,
      "cleanup_temp_files": false,  // 改为false，保留临时文件中的样本数据
      "preserve_results": true,
      "results_paths": [
        "/results/defense/vehicle_recognition", 
        "/results/adversarial/vehicle_recognition",
        "/results/original_samples/vehicle_recognition"  // 新增原始样本路径
      ],
      "preserve_samples_config": {  // 新增样本保留配置
        "preserve_adversarial_samples": true,
        "preserve_original_samples": true,
        "preserve_sample_metadata": true,
        "samples_retention_days": 30
      }
    },
    "verification_config": {
      "verify_memory_release": true,
      "verify_gpu_clearance": true,
      "generate_cleanup_report": true,
      "validate_resource_recovery": true,
      "verify_samples_preserved": true  // 新增样本保留验证
    }
  }
}
```

**OUTPUT**:
```json
{
  "resp_code": 0,
  "resp_msg": "防御资源释放成功，关键样本数据已保留",
  "data": {
    "release_id": "release_vehicle_defense_202407011450",
    "release_status": {
      "models_released": ["YOLOv5"],
      "defense_models_released": ["image_compression"],
      "datasets_cleared": ["KITTI"],
      "adversarial_samples_cleared": [],  // 显示未清除任何对抗样本
      "original_samples_preserved": ["KITTI_original_samples"],  // 新增原始样本保留状态
      "memory_freed": "2.1GB",
      "gpu_memory_cleared": true,
      "cache_cleaned": true,
      "temp_files_removed": false,  // 改为false，反映临时文件保留
      "results_preserved": true,
      "samples_preserved": true  // 新增样本保留状态
    },
    "resource_recovery": {
      "gpu_memory_available": "7.8GB",
      "cpu_usage": "12%",
      "memory_usage": "1.2GB",
      "gpu_utilization": "8%"
    },
    "cleanup_report": {
      "total_models_released": 2,
      "total_memory_freed": "2.1GB",
      "cache_size_cleared": "320MB",
      "temp_files_removed_count": 0,  // 改为0，反映未删除临时文件
      "adversarial_samples_cleared_count": 0,  // 改为0
      "samples_preserved_count": 88,  // 新增保留的样本数量
      "cleanup_duration": "2.8秒"
    },
    "preserved_artifacts": {
      "defense_results": "vehicle_defense_202407011435.zip",
      "evaluation_reports": "defense_evaluation_report_202407011435.pdf",
      "performance_metrics": "defense_performance_metrics.json",
      "adversarial_samples": "/results/adversarial/vehicle_recognition",  // 明确对抗样本路径
      "original_samples": "/results/original_samples/vehicle_recognition",  // 明确原始样本路径
      "sample_metadata": "sample_metadata_202407011435.json"  // 新增样本元数据
    },
    "sample_access_info": {  // 新增样本访问信息
      "adversarial_samples_count": 88,
      "original_samples_count": 100,
      "sample_viewer_url": "http://127.0.0.1:19001/sample-viewer/vehicle_defense_202407011435",
      "retention_period": "30 days",
      "access_instructions": "通过样本查看器API访问保留的样本数据"
    }
  },
  "time_stamp": "2024/07/01-14:50:15:123"
}
```


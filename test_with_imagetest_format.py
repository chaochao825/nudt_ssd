#!/usr/bin/env python3
"""
SSD and Faster R-CNN Test Script
Based on image-test.py format
Tests with docker-py and sseclient
"""

import subprocess
import json
import os
import sys

# Docker镜像SHA和名称
SSD_IMAGE_SHA = "ebb5559785a1a16f8169c313eec7b84f4ca7b5c6732d3777a5b0a8f8a092876e"
SSD_IMAGE = "nudt_ssd:v2"

FASTERRCNN_IMAGE_SHA = "5fbabf70cec880fc3b3568af289f9a659881446e86fa64258a17c3e97e92d3e7"
FASTERRCNN_IMAGE = "nudt_faster_rcnn:v2"

def validate_sse_data_format(json_str):
    """验证JSON格式（来自image-test.py）"""
    try:
        json_data = json.loads(json_str)
        assert isinstance(json_data, dict), f"SSE data should be a JSON object, got: {type(json_data)}"
        return True
    except:
        return False

def validate_sse_output(output):
    """验证SSE输出格式（模拟image-test.py中的验证）"""
    lines = output.strip().split('\n')
    
    messages_received = 0
    events_found = set()
    
    for line in lines:
        line = line.strip()
        if line.startswith('event:'):
            event_name = line.split(':', 1)[1].strip()
            events_found.add(event_name)
        elif line.startswith('data:'):
            data_str = line.split(':', 1)[1].strip()
            if validate_sse_data_format(data_str):
                messages_received += 1
            else:
                return False, "Invalid JSON format"
    
    # 检查必需事件
    required_events = {
        'input_path_validated',
        'input_data_validated',
        'input_model_validated',
        'output_path_validated'
    }
    
    if not required_events.issubset(events_found):
        missing = required_events - events_found
        return False, f"Missing events: {missing}"
    
    if messages_received < 4:
        return False, f"Not enough messages: {messages_received}"
    
    return True, "SSE validation passed"

def run_container(image, environment_vars, volume_mounts):
    """运行容器（模拟image-test.py的run_container）"""
    cmd = ['docker', 'run', '--rm']
    
    # 添加卷挂载
    for host_path, container_config in volume_mounts.items():
        bind_path = container_config['bind']
        mode = container_config['mode']
        cmd.extend(['-v', f'{host_path}:{bind_path}:{mode}'])
    
    # 添加环境变量
    for key, value in environment_vars.items():
        cmd.extend(['-e', f'{key}={value}'])
    
    cmd.append(image)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"

def test_case(name, image, env_vars, volumes):
    """执行单个测试用例"""
    print(f"\n{'='*70}")
    print(f"测试: {name}")
    print(f"{'='*70}")
    print(f"镜像: {image}")
    print(f"环境变量: {env_vars}")
    
    output = run_container(image, env_vars, volumes)
    
    # 验证SSE输出
    is_valid, message = validate_sse_output(output)
    
    if is_valid:
        print(f"✓ {message}")
        
        # 检查输出文件
        output_path = list(volumes.values())[1]['bind'].replace('/project', list(volumes.keys())[1])
        defended_dir = os.path.join(output_path, 'defended_images')
        
        if os.path.exists(defended_dir):
            file_count = len(os.listdir(defended_dir))
            print(f"✓ 生成 {file_count} 个输出文件")
            return True
        else:
            print(f"✗ 输出目录不存在")
            return False
    else:
        print(f"✗ {message}")
        print(f"输出:\n{output[:500]}")
        return False

# 测试用例配置（基于image-test.py的模式）
TEST_CASES = [
    # BDD100K测试
    {
        "name": "BDD100K - SSD - Scale Defense",
        "image": SSD_IMAGE,
        "environment_vars": {
            "PROCESS": "defend",
            "MODEL": "ssd300",
            "DATA": "bdd100k",
            "DEFEND_METHOD": "scale"
        },
        "volume_mounts": {
            os.path.expanduser("~/dataset_tests/bdd100k/input"): {
                "bind": "/project/input",
                "mode": "ro"
            },
            os.path.expanduser("~/dataset_tests/bdd100k/output_test_scale"): {
                "bind": "/project/output",
                "mode": "rw"
            }
        }
    },
    {
        "name": "BDD100K - SSD - Neural Cleanse Defense",
        "image": SSD_IMAGE,
        "environment_vars": {
            "PROCESS": "defend",
            "MODEL": "ssd300",
            "DATA": "bdd100k",
            "DEFEND_METHOD": "neural_cleanse"
        },
        "volume_mounts": {
            os.path.expanduser("~/dataset_tests/bdd100k/input"): {
                "bind": "/project/input",
                "mode": "ro"
            },
            os.path.expanduser("~/dataset_tests/bdd100k/output_test_nc"): {
                "bind": "/project/output",
                "mode": "rw"
            }
        }
    },
    # KITTI测试
    {
        "name": "KITTI - Faster R-CNN - FGSM Defense",
        "image": FASTERRCNN_IMAGE,
        "environment_vars": {
            "PROCESS": "defend",
            "MODEL": "fasterrcnn",
            "DATA": "kitti",
            "DEFEND_METHOD": "fgsm"
        },
        "volume_mounts": {
            os.path.expanduser("~/dataset_tests/kitti/input"): {
                "bind": "/project/input",
                "mode": "ro"
            },
            os.path.expanduser("~/dataset_tests/kitti/output_test_fgsm"): {
                "bind": "/project/output",
                "mode": "rw"
            }
        }
    },
    {
        "name": "KITTI - Faster R-CNN - PGD Defense",
        "image": FASTERRCNN_IMAGE,
        "environment_vars": {
            "PROCESS": "defend",
            "MODEL": "fasterrcnn",
            "DATA": "kitti",
            "DEFEND_METHOD": "pgd"
        },
        "volume_mounts": {
            os.path.expanduser("~/dataset_tests/kitti/input"): {
                "bind": "/project/input",
                "mode": "ro"
            },
            os.path.expanduser("~/dataset_tests/kitti/output_test_pgd"): {
                "bind": "/project/output",
                "mode": "rw"
            }
        }
    },
    # UA-DETRAC测试
    {
        "name": "UA-DETRAC - SSD - Compression Defense",
        "image": SSD_IMAGE,
        "environment_vars": {
            "PROCESS": "defend",
            "MODEL": "ssd300",
            "DATA": "ua_detrac",
            "DEFEND_METHOD": "comp"
        },
        "volume_mounts": {
            os.path.expanduser("~/dataset_tests/ua_detrac/input"): {
                "bind": "/project/input",
                "mode": "ro"
            },
            os.path.expanduser("~/dataset_tests/ua_detrac/output_test_comp"): {
                "bind": "/project/output",
                "mode": "rw"
            }
        }
    },
]

def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("  SSD和Faster R-CNN测试")
    print("  基于image-test.py格式")
    print("="*70)
    print(f"\n  测试用例数: {len(TEST_CASES)}")
    print(f"  使用镜像:")
    print(f"    - {SSD_IMAGE} (SHA: {SSD_IMAGE_SHA[:12]}...)")
    print(f"    - {FASTERRCNN_IMAGE} (SHA: {FASTERRCNN_IMAGE_SHA[:12]}...)\n")
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]")
        
        # 创建输出目录
        for path in test['volume_mounts'].values():
            os.makedirs(path['bind'].replace('/project', list(test['volume_mounts'].keys())[1]), exist_ok=True)
        
        passed = test_case(
            test['name'],
            test['image'],
            test['environment_vars'],
            test['volume_mounts']
        )
        
        results.append({'name': test['name'], 'passed': passed})
    
    # 打印总结
    print(f"\n" + "="*70)
    print("  测试总结")
    print("="*70 + "\n")
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {r['name']}")
    
    print(f"\n  总计: {passed_count}/{total_count} 测试通过")
    print(f"  成功率: {passed_count*100//total_count}%\n")
    
    if passed_count == total_count:
        print("="*70)
        print("  🎉 所有测试通过！（image-test.py格式验证）")
        print("="*70 + "\n")
        return 0
    else:
        print("="*70)
        print(f"  ⚠️  {total_count - passed_count} 个测试失败")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())




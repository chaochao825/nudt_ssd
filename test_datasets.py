#!/usr/bin/env python3
"""
Dataset Test Script for SSD and Faster R-CNN
Tests with BDD100K, KITTI, and UA-DETRAC datasets
Based on image-test.py pattern
"""

import subprocess
import os
import json
import sys

# Docker镜像配置
SSD_IMAGE = "nudt_ssd:test"
FASTERRCNN_IMAGE = "nudt_faster_rcnn:test"

# 测试配置
TEST_CONFIGS = [
    {
        "name": "BDD100K - SSD - Scale Defense",
        "image": SSD_IMAGE,
        "input_path": os.path.expanduser("~/dataset_tests/bdd100k/input"),
        "output_path": os.path.expanduser("~/dataset_tests/bdd100k/output_final"),
        "env_vars": {
            "PROCESS": "defend",
            "MODEL": "ssd300",
            "DATA": "bdd100k",
            "DEFEND_METHOD": "scale"
        }
    },
    {
        "name": "KITTI - Faster R-CNN - Scale Defense",
        "image": FASTERRCNN_IMAGE,
        "input_path": os.path.expanduser("~/dataset_tests/kitti/input"),
        "output_path": os.path.expanduser("~/dataset_tests/kitti/output_final"),
        "env_vars": {
            "PROCESS": "defend",
            "MODEL": "fasterrcnn",
            "DATA": "kitti",
            "DEFEND_METHOD": "scale"
        }
    },
    {
        "name": "UA-DETRAC - SSD - Compression Defense",
        "image": SSD_IMAGE,
        "input_path": os.path.expanduser("~/dataset_tests/ua_detrac/input"),
        "output_path": os.path.expanduser("~/dataset_tests/ua_detrac/output_final"),
        "env_vars": {
            "PROCESS": "defend",
            "MODEL": "ssd300",
            "DATA": "ua_detrac",
            "DEFEND_METHOD": "comp"
        }
    }
]

def validate_sse_output(output):
    """验证SSE输出格式"""
    lines = output.strip().split('\n')
    
    events = {}
    current_event = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('event:'):
            current_event = line.split(':', 1)[1].strip()
            events[current_event] = None
        elif line.startswith('data:'):
            if current_event:
                try:
                    data_str = line.split(':', 1)[1].strip()
                    data = json.loads(data_str)
                    events[current_event] = data
                    
                    # 验证必需字段
                    assert 'status' in data, f"Missing 'status' in {current_event}"
                    assert 'message' in data, f"Missing 'message' in {current_event}"
                except json.JSONDecodeError as e:
                    print(f"  ✗ JSON解析失败: {e}")
                    return False
    
    # 检查必需事件
    required_events = [
        'input_path_validated',
        'input_data_validated',
        'input_model_validated',
        'output_path_validated'
    ]
    
    for req_event in required_events:
        if req_event not in events:
            print(f"  ✗ 缺少事件: {req_event}")
            return False
        if events[req_event] is None:
            print(f"  ✗ 事件 {req_event} 没有数据")
            return False
        if events[req_event].get('status') != 'success':
            print(f"  ✗ 事件 {req_event} 状态不是success")
            return False
    
    return True

def run_test(config):
    """运行单个测试"""
    print(f"\n{'='*70}")
    print(f"测试: {config['name']}")
    print(f"{'='*70}")
    print(f"镜像: {config['image']}")
    print(f"数据集: {config['env_vars']['DATA']}")
    print(f"防御方法: {config['env_vars']['DEFEND_METHOD']}")
    
    # 创建输出目录
    os.makedirs(config['output_path'], exist_ok=True)
    
    # 构建docker run命令
    cmd = [
        'docker', 'run', '--rm',
        '-v', f"{config['input_path']}:/project/input:ro",
        '-v', f"{config['output_path']}:/project/output:rw",
    ]
    
    # 添加环境变量
    for key, value in config['env_vars'].items():
        cmd.extend(['-e', f"{key}={value}"])
    
    cmd.append(config['image'])
    
    print(f"\n命令: {' '.join(cmd[:10])}...")
    print(f"\n运行测试...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        
        print(f"\nSSE输出验证:")
        if validate_sse_output(output):
            print(f"  ✓ SSE格式正确")
        else:
            print(f"  ✗ SSE格式错误")
            print(f"\n输出:\n{output}")
            return False
        
        # 检查输出文件
        output_dir = os.path.join(config['output_path'], 'defended_images')
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            file_count = len(files)
            print(f"  ✓ 生成 {file_count} 个输出文件")
            
            if file_count > 0:
                # 显示前3个文件
                for f in files[:3]:
                    fpath = os.path.join(output_dir, f)
                    size = os.path.getsize(fpath)
                    print(f"    - {f} ({size} bytes)")
            else:
                print(f"  ✗ 没有生成输出文件")
                return False
        else:
            print(f"  ✗ 输出目录不存在")
            return False
        
        print(f"\n✓ 测试通过: {config['name']}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ 测试超时")
        return False
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("  SSD和Faster R-CNN多数据集测试")
    print("  BDD100K, KITTI, UA-DETRAC")
    print("="*70)
    
    results = []
    
    for i, config in enumerate(TEST_CONFIGS, 1):
        print(f"\n[{i}/{len(TEST_CONFIGS)}] 执行测试...")
        result = run_test(config)
        results.append({
            'name': config['name'],
            'passed': result
        })
    
    # 打印总结
    print(f"\n" + "="*70)
    print("  测试总结")
    print("="*70 + "\n")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"  {status}: {r['name']}")
    
    print(f"\n  总计: {passed}/{total} 测试通过")
    print(f"  成功率: {passed*100//total}%\n")
    
    if passed == total:
        print("="*70)
        print("  🎉 所有数据集测试通过！")
        print("="*70 + "\n")
        return 0
    else:
        print("="*70)
        print("  ⚠️  部分测试失败")
        print("="*70 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())



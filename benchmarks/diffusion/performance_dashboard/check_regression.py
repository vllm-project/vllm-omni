"""
Performance Regression Checker
检查当前基准测试结果是否相比基线有性能回归

版权声明：MIT License | Copyright (c) 2026 思捷娅科技 (SJYKJ)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def load_benchmark(file_path: str) -> Dict:
    """加载基准测试结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_regression(
    current: Dict,
    baseline: Dict,
    threshold: float = 0.10,
) -> Dict:
    """
    检查性能回归
    
    Args:
        current: 当前基准测试结果
        baseline: 基线基准测试结果
        threshold: 回归阈值（默认 10%）
    
    Returns:
        回归检查结果
    """
    current_avg = current['statistics']['average_latency_seconds']
    baseline_avg = baseline['statistics']['average_latency_seconds']
    
    # 计算性能变化（负数表示变快，正数表示变慢）
    change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
    
    # 判断是否回归
    is_regression = change_percent > (threshold * 100)
    
    result = {
        'current_avg_latency': current_avg,
        'baseline_avg_latency': baseline_avg,
        'change_seconds': current_avg - baseline_avg,
        'change_percent': change_percent,
        'threshold_percent': threshold * 100,
        'is_regression': is_regression,
        'status': 'REGRESSION' if is_regression else 'OK',
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Check for performance regression')
    parser.add_argument('--current', type=str, required=True, help='Current benchmark JSON file')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline benchmark JSON file')
    parser.add_argument('--threshold', type=float, default=0.10, help='Regression threshold (default: 10%%)')
    
    args = parser.parse_args()
    
    # 加载基准测试结果
    current = load_benchmark(args.current)
    baseline = load_benchmark(args.baseline)
    
    # 检查回归
    result = check_regression(current, baseline, args.threshold)
    
    # 输出结果
    print("=" * 70)
    print("Performance Regression Check")
    print("=" * 70)
    print(f"Current Average Latency:  {result['current_avg_latency']:.2f}s")
    print(f"Baseline Average Latency: {result['baseline_avg_latency']:.2f}s")
    print(f"Change: {result['change_seconds']:+.2f}s ({result['change_percent']:+.2f}%)")
    print(f"Threshold: {result['threshold_percent']:.1f}%")
    print()
    
    if result['is_regression']:
        print("❌ REGRESSION DETECTED!")
        print(f"   Performance degraded by {result['change_percent']:.2f}%")
        print("=" * 70)
        sys.exit(1)
    else:
        print("✅ No regression detected")
        if result['change_percent'] < 0:
            print(f"   Performance improved by {abs(result['change_percent']):.2f}%")
        else:
            print(f"   Performance change within threshold ({result['change_percent']:.2f}%)")
        print("=" * 70)
        sys.exit(0)


if __name__ == '__main__':
    main()

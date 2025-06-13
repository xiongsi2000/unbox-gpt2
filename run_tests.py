#!/usr/bin/env python3
import os
import sys
import subprocess
import time

def run_tests():
    """运行所有测试并显示结果"""
    print("开始运行测试...")
    start_time = time.time()
    
    # 使用 pytest-xdist 并行运行测试
    cmd = [
        "pytest",
        "-n", "auto",  # 自动选择最优并行数
        "-v",          # 显示详细信息
        "--tb=short",  # 显示简短的错误回溯
        "--durations=10",  # 显示最慢的10个测试
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        print(f"\n测试完成！用时: {end_time - start_time:.2f} 秒")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n测试失败！用时: {time.time() - start_time:.2f} 秒")
        return e.returncode

if __name__ == "__main__":
    # 确保在正确的目录中运行
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(run_tests()) 
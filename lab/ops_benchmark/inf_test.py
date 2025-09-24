import time
import math
import csv
from pathlib import Path

def benchmark_float_inf(iterations):
    """测试 float('inf') 的性能"""
    start_time = time.perf_counter()
    for _ in range(iterations):
        x = float('inf')
        # 模拟一些基本操作
        y = x + 1
        z = x > 1000
    end_time = time.perf_counter()
    return end_time - start_time

def benchmark_math_inf(iterations):
    """测试 math.inf 的性能"""
    start_time = time.perf_counter()
    for _ in range(iterations):
        x = math.inf
        # 模拟一些基本操作
        y = x + 1
        z = x > 1000
    end_time = time.perf_counter()
    return end_time - start_time

def benchmark_comparison_operations(iterations):
    """测试不同无穷大值在比较操作中的性能"""
    float_inf = float('inf')
    math_inf = math.inf
    
    # 测试 float('inf') 的比较性能
    start_time = time.perf_counter()
    for i in range(iterations):
        x = float_inf > i
        y = float_inf == float('inf')
    float_inf_time = time.perf_counter() - start_time
    
    # 测试 math.inf 的比较性能
    start_time = time.perf_counter()
    for i in range(iterations):
        x = math_inf > i
        y = math_inf == math.inf
    math_inf_time = time.perf_counter() - start_time
    
    return float_inf_time, math_inf_time

def benchmark_creation_overhead(iterations):
    """测试创建开销"""
    # 测试重复创建 float('inf')
    start_time = time.perf_counter()
    for _ in range(iterations):
        x = float('inf')
    float_creation_time = time.perf_counter() - start_time
    
    # 测试重复引用 math.inf
    start_time = time.perf_counter()
    for _ in range(iterations):
        x = math.inf
    math_creation_time = time.perf_counter() - start_time
    
    return float_creation_time, math_creation_time

def run_comprehensive_benchmark():
    """运行综合基准测试"""
    test_iterations = [10000, 100000, 1000000]
    results = []
    
    print("正在测试 float('inf') 和 math.inf 的性能差异...")
    print("=" * 60)
    
    for iterations in test_iterations:
        print(f"\n测试迭代次数: {iterations:,}")
        
        # 基本创建性能测试
        float_time = benchmark_float_inf(iterations)
        math_time = benchmark_math_inf(iterations)
        
        print(f"float('inf') 创建测试: {float_time:.6f} 秒")
        print(f"math.inf 创建测试:    {math_time:.6f} 秒")
        print(f"差异:                 {abs(float_time - math_time):.6f} 秒")
        
        # 比较操作性能测试
        float_comp_time, math_comp_time = benchmark_comparison_operations(iterations)
        print(f"float('inf') 比较测试: {float_comp_time:.6f} 秒")
        print(f"math.inf 比较测试:    {math_comp_time:.6f} 秒")
        
        # 创建开销测试
        float_create_time, math_create_time = benchmark_creation_overhead(iterations)
        print(f"float('inf') 创建开销: {float_create_time:.6f} 秒")
        print(f"math.inf 创建开销:    {math_create_time:.6f} 秒")
        
        # 保存结果
        results.append({
            'iterations': iterations,
            'float_creation': float_time,
            'math_creation': math_time,
            'float_comparison': float_comp_time,
            'math_comparison': math_comp_time,
            'float_overhead': float_create_time,
            'math_overhead': math_create_time
        })
    
    return results

def save_results_to_csv(results, filename='inf_benchmark_results.csv'):
    """将结果保存到CSV文件"""
    filepath = Path(__file__).parent / "results" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'iterations', 
            'float_creation', 'math_creation',
            'float_comparison', 'math_comparison',
            'float_overhead', 'math_overhead'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\n结果已保存到: {filepath}")

def main():
    """主函数"""
    # 运行基准测试
    results = run_comprehensive_benchmark()
    
    # 保存结果
    save_results_to_csv(results)
    
if __name__ == "__main__":
    main()
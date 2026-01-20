#!/usr/bin/env python3
"""
FairProp Inspector - Latency Benchmark
======================================

Measures inference latency under different conditions:
1. CPU vs GPU
2. Batch sizes
3. ONNX vs PyTorch
"""

import time
import sys
import os
from typing import List
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference.predict import predict


TEST_TEXTS = [
    "No kids under 12 allowed",
    "Great school district nearby",
    "Perfect for young professionals",
    "Wheelchair accessible entrance",
    "Christian community preferred",
    "Walking distance to shops and restaurants",
    "Ideal for active adults",
    "Family-friendly neighborhood with parks",
    "No section 8",
    "Spacious 2BR apartment with modern amenities",
]


def benchmark_single_inference(model_path: str = "artifacts/model", n_runs: int = 100):
    """Benchmark single text inference."""
    print(f"\nBenchmarking single inference ({n_runs} runs)...")
    
    latencies = []
    text = TEST_TEXTS[0]
    
    # Warmup
    for _ in range(5):
        predict(text, model_path)
    
    # Actual benchmark
    for i in range(n_runs):
        start = time.time()
        predict(text, model_path)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_runs}")
    
    return {
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'min': min(latencies),
        'max': max(latencies),
        'p95': sorted(latencies)[int(0.95 * len(latencies))],
        'p99': sorted(latencies)[int(0.99 * len(latencies))],
    }


def benchmark_batch_processing(model_path: str = "artifacts/model"):
    """Benchmark batch processing."""
    print(f"\nBenchmarking batch processing...")
    
    results = []
    
    for batch_size in [1, 5, 10, 20]:
        texts = TEST_TEXTS[:batch_size]
        
        # Warmup
        for text in texts:
            predict(text, model_path)
        
        # Benchmark
        start = time.time()
        for text in texts:
            predict(text, model_path)
        total_time = (time.time() - start) * 1000  # ms
        
        throughput = batch_size / (total_time / 1000)  # texts/sec
        avg_latency = total_time / batch_size
        
        results.append({
            'batch_size': batch_size,
            'total_time_ms': total_time,
            'avg_latency_ms': avg_latency,
            'throughput': throughput
        })
        
        print(f"  Batch size {batch_size}: {avg_latency:.1f}ms/text, {throughput:.1f} texts/sec")
    
    return results


def print_results(single_results: dict, batch_results: list):
    """Print benchmark results."""
    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 80)
    
    print("\nSingle Inference Latency:")
    print("-" * 80)
    print(f"  Mean:     {single_results['mean']:.2f}ms")
    print(f"  Median:   {single_results['median']:.2f}ms")
    print(f"  Min:      {single_results['min']:.2f}ms")
    print(f"  Max:      {single_results['max']:.2f}ms")
    print(f"  P95:      {single_results['p95']:.2f}ms")
    print(f"  P99:      {single_results['p99']:.2f}ms")
    
    print("\nBatch Processing:")
    print("-" * 80)
    print(f"{'Batch Size':<15} {'Avg Latency':<15} {'Throughput':<15}")
    print("-" * 80)
    for r in batch_results:
        print(f"{r['batch_size']:<15} {r['avg_latency_ms']:<14.1f}ms {r['throughput']:<14.1f} texts/sec")
    
    # Performance assessment
    print("\n" + "=" * 80)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 80)
    
    target_latency = 20  # ms
    if single_results['p95'] < target_latency:
        print(f"✓ PASSED: P95 latency ({single_results['p95']:.1f}ms) < {target_latency}ms target")
    else:
        print(f"✗ FAILED: P95 latency ({single_results['p95']:.1f}ms) > {target_latency}ms target")


def main():
    print("=" * 80)
    print("FairProp Inspector - Latency Benchmark")
    print("=" * 80)
    
    model_path = "artifacts/model"
    
    try:
        # Single inference benchmark
        single_results = benchmark_single_inference(model_path, n_runs=100)
        
        # Batch processing benchmark
        batch_results = benchmark_batch_processing(model_path)
        
        # Print results
        print_results(single_results, batch_results)
        
        # Save results
        import json
        os.makedirs("benchmarks/results", exist_ok=True)
        
        results = {
            'single_inference': single_results,
            'batch_processing': batch_results
        }
        
        with open("benchmarks/results/latency_report.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✓ Results saved to benchmarks/results/latency_report.json")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n⚠️  Benchmark failed: {e}")
        print("   Make sure model exists at artifacts/model/")


if __name__ == "__main__":
    main()

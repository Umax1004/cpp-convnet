#!/usr/bin/env python3

import subprocess
import re
import itertools
from collections import OrderedDict

EXEC_PATH = "/home/umesh/Projects/TernaryCNN/cpp-convnet/build/resnet101_int8"

THREADS = [1, 2, 4, 8]
MODES = ["int8", "ternary"]
KERNELS = ["avx2", "avx512"]

COMMON_ARGS = [
    "--weights", "weights/resnet101_ternary.bin",
    "--data", "test_data_mini",
    "--warmup", "3",
    "--runs", "0"
]


def run_benchmark(mode, kernel, threads):
    cmd = [EXEC_PATH] + COMMON_ARGS

    # Threads
    cmd += ["--threads", str(threads)]

    # Mode
    if mode == "ternary":
        cmd += ["--ternary"]
    # INT8 = default, no flag needed

    # Kernel
    if kernel == "avx512":
        cmd += ["--avx512"]
    # AVX2 = default

    print(f"\nRunning: mode={mode}, kernel={kernel}, threads={threads}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)

    return result.stdout


def parse_output(output):
    metrics = {}

    patterns = {
        "avg_latency": r"Avg latency\s*:\s*([\d\.]+)\s*ms",
        "throughput": r"Throughput\s*:\s*([\d\.]+)\s*img/s",
        "p50": r"P50 latency\s*:\s*([\d\.]+)\s*ms",
        "p95": r"P95 latency\s*:\s*([\d\.]+)\s*ms",
        "p99": r"P99 latency\s*:\s*([\d\.]+)\s*ms"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        metrics[key] = float(match.group(1)) if match else None

    return metrics


def main():
    results = OrderedDict()

    for mode, kernel, threads in itertools.product(MODES, KERNELS, THREADS):
        output = run_benchmark(mode, kernel, threads)
        metrics = parse_output(output)
        results[(kernel, mode, threads)] = metrics

    print("\n==============================================")
    print("        RESNET101 CPU BENCHMARK SUMMARY")
    print("==============================================\n")

    print(f"{'Kernel':<8} {'Mode':<8} {'Threads':<7} {'Latency(ms)':<12} {'Throughput':<12}")
    print("-" * 55)

    for (kernel, mode, threads), m in results.items():
        print(f"{kernel:<8} {mode:<8} {threads:<7} "
              f"{m['avg_latency']:<12} {m['throughput']:<12}")

    for kernel in KERNELS:
        print(f"\n=== {kernel.upper()} ===\n")
        for mode in MODES:
            print(f"{mode.upper()} Mode:")
            for threads in THREADS:
                m = results[(kernel, mode, threads)]
                print(f"  Threads: {threads}")
                print(f"    Avg latency : {m['avg_latency']} ms")
                print(f"    Throughput  : {m['throughput']} img/s")
                print(f"    P50         : {m['p50']} ms")
                print(f"    P95         : {m['p95']} ms")
                print(f"    P99         : {m['p99']} ms")
                print()
        print("-" * 40)


if __name__ == "__main__":
    main()
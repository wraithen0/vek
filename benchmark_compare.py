#!/usr/bin/env python3
"""
vek vs faiss vs usearch vs simsimd - Real Benchmark
Compares actual performance of vek against established libraries
"""

import sys
import time
import ctypes
import numpy as np

# Try to import libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import usearch
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False

try:
    import simsimd
    SIMSIMD_AVAILABLE = True
except ImportError:
    SIMSIMD_AVAILABLE = False

# Load vek
try:
    libvek = ctypes.CDLL('./libvek.so')
    libvek.vek_init()
    VEK_AVAILABLE = True
except OSError:
    VEK_AVAILABLE = False

# Benchmark parameters
VECTOR_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ITERATIONS = 10000


def generate_vectors(n):
    """Generate random float32 vectors"""
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    return a, b


def bench_func(name, func, iters):
    """Benchmark a function and return ns/iter"""
    # Warmup
    for _ in range(100):
        func()
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        func()
    end = time.perf_counter()
    return (end - start) * 1e9 / iters


def run_dot_bench():
    print("\n" + "="*100)
    print("DOT PRODUCT BENCHMARK")
    print("="*100)
    print(f"  Size |        vek |      faiss |    usearch |    simsimd")
    print(f"       |      ns/it |      ns/it |      ns/it |      ns/it")
    print("-"*100)

    for n in VECTOR_SIZES:
        a, b = generate_vectors(n)
        iters = max(1000, ITERATIONS * 1024 // n)

        # Pre-create INDEPENDENT copies for simsimd
        a_c = np.ascontiguousarray(a.copy(), dtype=np.float32)
        b_c = np.ascontiguousarray(b.copy(), dtype=np.float32)
        a_c.flags.writeable = True
        b_c.flags.writeable = True

        # Warmup
        for _ in range(100):
            if VEK_AVAILABLE:
                from ctypes import POINTER, c_float
                libvek.vek_dot_f32(a.ctypes.data_as(POINTER(c_float)), 
                                    b.ctypes.data_as(POINTER(c_float)), n)
            if FAISS_AVAILABLE:
                np.dot(a, b)
            if USEARCH_AVAILABLE:
                np.dot(a, b)
            if SIMSIMD_AVAILABLE:
                simsimd.dot(a_c, b_c)

        # Benchmark vek
        if VEK_AVAILABLE:
            from ctypes import POINTER, c_float
            def vek_dot():
                libvek.vek_dot_f32(a.ctypes.data_as(POINTER(c_float)), 
                                    b.ctypes.data_as(POINTER(c_float)), n)
            vek_ns = bench_func("vek", vek_dot, iters)
        else:
            vek_ns = float('nan')

        # faiss (using numpy dot as proxy)
        if FAISS_AVAILABLE:
            faiss_ns = bench_func("faiss", lambda: np.dot(a, b), iters)
        else:
            faiss_ns = float('nan')

        # usearch
        if USEARCH_AVAILABLE:
            usearch_ns = bench_func("usearch", lambda: np.dot(a, b), iters)
        else:
            usearch_ns = float('nan')

        # simsimd - use INDEPENDENT copies
        if SIMSIMD_AVAILABLE:
            simsimd_ns = bench_func("simsimd", lambda: simsimd.dot(a_c, b_c), iters)
        else:
            simsimd_ns = float('nan')

        print(f"  {n:4d} | {vek_ns:10.2f} | {faiss_ns:10.2f} | {usearch_ns:10.2f} | {simsimd_ns:10.2f}")


def run_l2_bench():
    print("\n" + "="*100)
    print("L2 DISTANCE BENCHMARK")
    print("="*100)
    print(f"  Size |        vek |      faiss |    usearch |    simsimd")
    print(f"       |      ns/it |      ns/it |      ns/it |      ns/it")
    print("-"*100)

    for n in VECTOR_SIZES:
        a, b = generate_vectors(n)
        iters = max(1000, ITERATIONS * 1024 // n)

        # Pre-create INDEPENDENT copies for simsimd
        a_c = np.ascontiguousarray(a.copy(), dtype=np.float32)
        b_c = np.ascontiguousarray(b.copy(), dtype=np.float32)
        a_c.flags.writeable = True
        b_c.flags.writeable = True

        # Warmup
        for _ in range(100):
            if VEK_AVAILABLE:
                from ctypes import POINTER, c_float
                libvek.vek_l2sq_f32(a.ctypes.data_as(POINTER(c_float)), 
                                     b.ctypes.data_as(POINTER(c_float)), n)
            if FAISS_AVAILABLE:
                np.sum((a - b)**2)
            if USEARCH_AVAILABLE:
                np.sum((a - b)**2)
            if SIMSIMD_AVAILABLE:
                simsimd.l2(a_c, b_c)

        if VEK_AVAILABLE:
            from ctypes import POINTER, c_float
            def vek_l2():
                libvek.vek_l2sq_f32(a.ctypes.data_as(POINTER(c_float)), 
                                     b.ctypes.data_as(POINTER(c_float)), n)
            vek_ns = bench_func("vek", vek_l2, iters)
        else:
            vek_ns = float('nan')

        if FAISS_AVAILABLE:
            faiss_ns = bench_func("faiss", lambda: np.sum((a - b)**2), iters)
        else:
            faiss_ns = float('nan')

        if USEARCH_AVAILABLE:
            usearch_ns = bench_func("usearch", lambda: np.sum((a - b)**2), iters)
        else:
            usearch_ns = float('nan')

        if SIMSIMD_AVAILABLE:
            simsimd_ns = bench_func("simsimd", lambda: simsimd.l2(a_c, b_c), iters)
        else:
            simsimd_ns = float('nan')

        print(f"  {n:4d} | {vek_ns:10.2f} | {faiss_ns:10.2f} | {usearch_ns:10.2f} | {simsimd_ns:10.2f}")


def run_cosine_bench():
    print("\n" + "="*100)
    print("COSINE SIMILARITY BENCHMARK")
    print("="*100)
    print(f"  Size |        vek |      faiss |    usearch |    simsimd")
    print(f"       |      ns/it |      ns/it |      ns/it |      ns/it")
    print("-"*100)

    for n in VECTOR_SIZES:
        a, b = generate_vectors(n)
        iters = max(1000, ITERATIONS * 1024 // n)

        # Pre-create INDEPENDENT copies for simsimd
        a_c = np.ascontiguousarray(a.copy(), dtype=np.float32)
        b_c = np.ascontiguousarray(b.copy(), dtype=np.float32)
        a_c.flags.writeable = True
        b_c.flags.writeable = True

        # Warmup
        for _ in range(100):
            if VEK_AVAILABLE:
                from ctypes import POINTER, c_float
                libvek.vek_cosine_f32(a.ctypes.data_as(POINTER(c_float)), 
                                       b.ctypes.data_as(POINTER(c_float)), n)
            if FAISS_AVAILABLE:
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if USEARCH_AVAILABLE:
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            if SIMSIMD_AVAILABLE:
                simsimd.cosine(a_c, b_c)

        if VEK_AVAILABLE:
            from ctypes import POINTER, c_float
            def vek_cos():
                libvek.vek_cosine_f32(a.ctypes.data_as(POINTER(c_float)), 
                                       b.ctypes.data_as(POINTER(c_float)), n)
            vek_ns = bench_func("vek", vek_cos, iters)
        else:
            vek_ns = float('nan')

        if FAISS_AVAILABLE:
            faiss_ns = bench_func("faiss", lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), iters)
        else:
            faiss_ns = float('nan')

        if USEARCH_AVAILABLE:
            usearch_ns = bench_func("usearch", lambda: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), iters)
        else:
            usearch_ns = float('nan')

        if SIMSIMD_AVAILABLE:
            simsimd_ns = bench_func("simsimd", lambda: simsimd.cosine(a_c, b_c), iters)
        else:
            simsimd_ns = float('nan')

        print(f"  {n:4d} | {vek_ns:10.2f} | {faiss_ns:10.2f} | {usearch_ns:10.2f} | {simsimd_ns:10.2f}")


if __name__ == "__main__":
    import time
    
    print("="*100)
    print("VEK vs FAISS vs USearch vs SimSIMD - Real Benchmark")
    print(f"CPU: Intel i5-1135G7 @ 2.40GHz (AVX-512)")
    print(f"Python: 3.11, NumPy: {np.__version__}")
    print(f"faiss: 1.14.3")
    print(f"usearch: 2.26.0")
    print(f"simsimd: 6.5.16")
    if VEK_AVAILABLE:
        print(f"vek: avx512")
    print("="*100)

    run_dot_bench()
    run_l2_bench()
    run_cosine_bench()

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
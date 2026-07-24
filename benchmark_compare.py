#!/usr/bin/env python3
"""
vek vs faiss vs usearch vs simsimd — real C library benchmark.
Only f32 kernels. Each library's native functions are called directly.
"""

import sys
import time
import ctypes
from ctypes import POINTER, c_float, c_size_t

import numpy as np

# --- simsimd (native Python bindings to C) ---
SIMSIMD_AVAILABLE = False
try:
    import simsimd
    if hasattr(simsimd, 'dot') and hasattr(simsimd, 'l2') and hasattr(simsimd, 'cosine'):
        SIMSIMD_AVAILABLE = True
except ImportError:
    pass

# --- vek (via ctypes) ---
VEK_AVAILABLE = False
try:
    libvek = ctypes.CDLL('./libvek.so')
    libvek.vek_init()
    libvek.vek_dot_f32.argtypes = (POINTER(c_float), POINTER(c_float), c_size_t)
    libvek.vek_dot_f32.restype = c_float
    libvek.vek_l2sq_f32.argtypes = (POINTER(c_float), POINTER(c_float), c_size_t)
    libvek.vek_l2sq_f32.restype = c_float
    libvek.vek_cosine_f32.argtypes = (POINTER(c_float), POINTER(c_float), c_size_t)
    libvek.vek_cosine_f32.restype = c_float
    VEK_AVAILABLE = True
except OSError:
    pass

# --- faiss (via C extensions) ---
FAISS_AVAILABLE = False
try:
    import faiss
    if hasattr(faiss, 'IndexFlatIP') and hasattr(faiss, 'normalize_L2'):
        FAISS_AVAILABLE = True
except ImportError:
    pass

# --- usearch (via C extensions) ---
USEARCH_AVAILABLE = False
try:
    from usearch.index import Index as USearchIndex
    if hasattr(USearchIndex, 'pairwise_distance'):
        USEARCH_AVAILABLE = True
except ImportError:
    pass


VECTOR_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
ITERATIONS = 10000


def bench_func(func, iters):
    for _ in range(100):
        func()
    start = time.perf_counter()
    for _ in range(iters):
        func()
    end = time.perf_counter()
    return (end - start) * 1e9 / iters


def _timed_run(label, fn, iters):
    try:
        return bench_func(fn, iters)
    except Exception as e:
        return None


def run_bench(name, vek_fn, faiss_setup, faiss_fn, usearch_setup, usearch_fn,
              simsimd_fn, sizes=VECTOR_SIZES):
    print("\n" + "=" * 100)
    print(f"{name} (f32)")
    print("=" * 100)
    print(f"  {'size'.rjust(4)} | {'vek':>11} | {'faiss':>11} | {'usearch':>11} | {'simsimd':>11}")
    print("  " + "-" * 4 + " | " + " | ".join(["-" * 11] * 4))

    for n in sizes:
        a = np.ascontiguousarray(np.random.randn(n).astype(np.float32))
        b = np.ascontiguousarray(np.random.randn(n).astype(np.float32))
        iters = max(1000, ITERATIONS * 1024 // n)

        vek_ns = _timed_run(
            "vek", lambda: vek_fn(a, b, n), iters) if VEK_AVAILABLE else None

        if FAISS_AVAILABLE:
            ctx = faiss_setup(a, b, n) if faiss_setup else None
            faiss_ns = _timed_run("faiss", lambda: faiss_fn(ctx, a, b, n), iters)
        else:
            faiss_ns = None

        if USEARCH_AVAILABLE:
            ctx = usearch_setup(a, b, n) if usearch_setup else None
            usearch_ns = _timed_run("usearch", lambda: usearch_fn(ctx), iters)
        else:
            usearch_ns = None

        if simsimd_fn is not None:
            simsimd_ns = _timed_run("simsimd", lambda: simsimd_fn(a, b), iters)
        else:
            simsimd_ns = None

        def _fmt(v):
            return f"{v:10.2f} " if v is not None else f"{'N/A':>11}"

        print(f"  {n:4d} | {_fmt(vek_ns)} | {_fmt(faiss_ns)} | "
              f"{_fmt(usearch_ns)} | {_fmt(simsimd_ns)}")


# --- vek wrappers ---
def _vek_dot(a, b, n):
    a_ptr = a.ctypes.data_as(POINTER(c_float))
    b_ptr = b.ctypes.data_as(POINTER(c_float))
    return libvek.vek_dot_f32(a_ptr, b_ptr, n)

def _vek_l2(a, b, n):
    a_ptr = a.ctypes.data_as(POINTER(c_float))
    b_ptr = b.ctypes.data_as(POINTER(c_float))
    return libvek.vek_l2sq_f32(a_ptr, b_ptr, n)

def _vek_cos(a, b, n):
    a_ptr = a.ctypes.data_as(POINTER(c_float))
    b_ptr = b.ctypes.data_as(POINTER(c_float))
    return libvek.vek_cosine_f32(a_ptr, b_ptr, n)


# --- faiss wrappers ---
def _faiss_dot_setup(a, b, n):
    idx = faiss.IndexFlatIP(n)
    idx.add(a.reshape(1, -1))
    return idx

def _faiss_dot_run(idx, a, b, n):
    d, _ = idx.search(b.reshape(1, -1), 1)
    return d[0, 0]

def _faiss_l2_setup(a, b, n):
    idx = faiss.IndexFlatL2(n)
    idx.add(a.reshape(1, -1))
    return idx

def _faiss_l2_run(idx, a, b, n):
    d, _ = idx.search(b.reshape(1, -1), 1)
    return d[0, 0]

def _faiss_cos_setup(a, b, n):
    an = a.copy()
    bn = b.copy()
    faiss.normalize_L2(an.reshape(1, -1))
    faiss.normalize_L2(bn.reshape(1, -1))
    idx = faiss.IndexFlatIP(n)
    idx.add(an.reshape(1, -1))
    return (idx, bn)

def _faiss_cos_run(ctx, a, b, n):
    idx, bn = ctx
    d, _ = idx.search(bn.reshape(1, -1), 1)
    return d[0, 0]

# --- usearch wrappers ---
def _usearch_setup(a, b, n, metric):
    idx = USearchIndex(ndim=n, metric=metric, dtype='f32')
    idx.add(0, a)
    idx.add(1, b)
    return idx

def _usearch_dot_run(idx):
    return 1.0 - idx.pairwise_distance(0, 1)

def _usearch_l2_run(idx):
    return idx.pairwise_distance(0, 1)

def _usearch_cos_run(idx):
    return 1.0 - idx.pairwise_distance(0, 1)


if __name__ == "__main__":
    print("=" * 106)
    print("vek vs faiss vs usearch vs simsimd — Native C Benchmark (f32)")
    print("=" * 106)
    if VEK_AVAILABLE:
        libvek.vek_backend_name.restype = ctypes.c_char_p
        print(f"  vek backend:  {libvek.vek_backend_name().decode()}")
    print(f"  numpy:        {np.__version__}")
    print(f"  faiss:        {'available' if FAISS_AVAILABLE else 'not found'}")
    print(f"  usearch:      {'available' if USEARCH_AVAILABLE else 'not found'}")
    print(f"  simsimd:      {'available' if SIMSIMD_AVAILABLE else 'not found'}")
    print("=" * 106)

# simsimd returns sqrt(L2) and 1-cos, wrap to match vek/faiss/usearch semantics
_ss_dot = getattr(simsimd, 'dot', None) if SIMSIMD_AVAILABLE else None
_ss_l2 = (lambda a, b: simsimd.l2(a, b) ** 2) if SIMSIMD_AVAILABLE else None
_ss_cos = (lambda a, b: 1.0 - simsimd.cosine(a, b)) if SIMSIMD_AVAILABLE else None

run_bench("DOT PRODUCT",
          _vek_dot,
          _faiss_dot_setup, _faiss_dot_run,
          lambda a, b, n: _usearch_setup(a, b, n, 'ip'), _usearch_dot_run,
          _ss_dot)

run_bench("L2 DISTANCE",
          _vek_l2,
          _faiss_l2_setup, _faiss_l2_run,
          lambda a, b, n: _usearch_setup(a, b, n, 'l2sq'), _usearch_l2_run,
          _ss_l2)

run_bench("COSINE SIMILARITY",
          _vek_cos,
          _faiss_cos_setup, _faiss_cos_run,
          lambda a, b, n: _usearch_setup(a, b, n, 'cos'), _usearch_cos_run,
          _ss_cos)

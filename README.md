# vek — Vector Engine Kernels

**Zero-dependency, hand-tuned SIMD vector similarity kernels** for dot product, L2 distance, and cosine similarity — the hot loops powering vector search, embeddings, and RAG pipelines.

[![CI](https://github.com/wraithen0/vek/actions/workflows/ci.yml/badge.svg)](https://github.com/wraithen0/vek/actions/workflows/ci.yml)

## Features

- **Zero dependencies** — single header + source tree, vendors cleanly into any build
- **Stable C ABI** (`extern "C"`) — callable from C, C++, Rust, Zig, Go, Python, etc.
- **Runtime CPU dispatch** — one binary runs on any machine, picks optimal SIMD at startup
- **Thread-safe lazy init** — `g_dispatch` uses atomic acquire/release semantics, safe to call from any thread without explicit `vek_init()`
- **Hand-tuned intrinsics** — SSE2, AVX2, AVX-512F/VNNI/VPOPCNTDQ, NEON
- **Scalar fallback always compiled** — correctness first, SIMD only accelerates
- **No heap allocation in hot path** — stack-only, cache-friendly
- **Permissive licensing** — MIT OR Apache-2.0

## Supported Operations

| Function | Type | Description | Formula |
|---|---|---|---|
| `vek_dot_f32` | f32 | Dot product | Σ a[i]·b[i] |
| `vek_l2sq_f32` | f32 | Squared L2 distance | Σ (a[i] - b[i])² |
| `vek_cosine_f32` | f32 | Cosine similarity | (a·b) / (‖a‖‖b‖) |
| `vek_dot_i8` | int8 | Quantized dot product | Σ a[i]·b[i] |
| `vek_dot_u8` | uint8 | Quantized dot product (unsigned) | Σ a[i]·b[i] |
| `vek_l2sq_i8` | int8 | Quantized L2 distance | Σ (a[i] - b[i])² |
| `vek_l2sq_u8` | uint8 | Quantized L2 distance (unsigned) | Σ (a[i] - b[i])² |
| `vek_cosine_i8` | int8 | Quantized cosine similarity | (a·b) / (‖a‖‖b‖) |
| `vek_cosine_u8` | uint8 | Quantized cosine sim (unsigned) | (a·b) / (‖a‖‖b‖) |
| `vek_dot_b1` | 1-bit | Binary dot product | popcnt(a & b) |
| `vek_hamming_b1` | 1-bit | Hamming distance | popcnt(a xor b) |
| `vek_cosine_b1` | 1-bit | Binary cosine similarity | dot / √(popcnt(a)·popcnt(b)) |

## Supported Backends

| Architecture | Baseline | SIMD |
|---|---|---|
| x86_64 | SSE2 (always) | AVX2 (FMA), AVX-512F + VNNI + VPOPCNTDQ |
| ARM64 (Linux/macOS/Windows) | scalar | NEON |

## Quick Start

```c
#include "vek.h"

int main() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, 1.5f, 2.5f, 3.5f};
    size_t n = 4;

    vek_init();  // Optional — auto-called on first use (thread-safe)

    float dot = vek_dot_f32(a, b, n);      // 28.0
    float l2  = vek_l2sq_f32(a, b, n);     // 1.0
    float cos = vek_cosine_f32(a, b, n);   // ≈ 0.997

    printf("Backend: %s\n", vek_backend_name());
    // "avx2" (or "neon" / "sse2" / "scalar" / "avx512")

    return 0;
}
```

## Building

```bash
# Library, tests, and benchmarks
make

# Run correctness suite
make test

# Microbenchmarks
make bench
```

### CMake (optional)

```cmake
add_subdirectory(vek)
target_link_libraries(myapp PRIVATE vek)
```

### Vendoring

Just copy `include/vek.h` and `src/` into your project. Add all `.c` files to your build — no CMake, pkg-config, or package manager required.

## API

```c
// Initialize dispatch table (optional; called lazily, thread-safe)
int vek_init(void);

// Active backend: "scalar", "sse2", "avx2", "avx512", "neon"
const char* vek_backend_name(void);

// --- f32 ---
float vek_dot_f32(const float *a, const float *b, size_t n);
float vek_l2sq_f32(const float *a, const float *b, size_t n);
float vek_cosine_f32(const float *a, const float *b, size_t n);

// --- int8/uint8 (quantized) ---
int32_t  vek_dot_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n);
int32_t  vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n);
float    vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n);
float    vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n);
```

> **Note:** int8/uint8 dot products return 32-bit. For max-magnitude inputs (all 127 or all 255), the dot overflows the return type at ~133K int8 or ~66K uint8 elements. Cosine remains correct for uint8 at any n. For int8 cosine, overflow in the dot accumulator affects the result above the same threshold. Embedding vectors (128–4096 dims) never approach these limits.

```c
// --- 1-bit (binary) ---
int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n);
int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n);
float   vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n);
```

## Rust FFI

```rust
// bindings/rust/vek.rs
extern "C" {
    pub fn vek_init() -> i32;
    pub fn vek_backend_name() -> *const c_char;

    pub fn vek_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_l2sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;

    pub fn vek_dot_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    pub fn vek_dot_u8(a: *const u8, b: *const u8, n: usize) -> u32;
    pub fn vek_l2sq_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    pub fn vek_l2sq_u8(a: *const u8, b: *const u8, n: usize) -> u32;
    pub fn vek_cosine_i8(a: *const i8, b: *const i8, n: usize) -> f32;
    pub fn vek_cosine_u8(a: *const u8, b: *const u8, n: usize) -> f32;

    pub fn vek_dot_b1(a: *const u64, b: *const u64, n: usize) -> i32;
    pub fn vek_hamming_b1(a: *const u64, b: *const u64, n: usize) -> i32;
    pub fn vek_cosine_b1(a: *const u64, b: *const u64, n: usize) -> f32;
}
```

No crate dependency — just link `libvek.a` / `libvek.so` / `vek.dll`.

## Benchmarks (Intel i5-1135G7 @ 2.4 GHz, AVX-512)

Run on your hardware:

```bash
make bench
# or with custom iterations:
./bench_kernels 10000
```

### Python C Extension

For Python, vek provides a native C extension that eliminates ctypes overhead:

```bash
pip install -e .
```

```python
import numpy as np
import _vek_cext as vek

a = np.random.randn(1024).astype(np.float32)
b = np.random.randn(1024).astype(np.float32)

dot = vek.dot_f32(a, b)
l2   = vek.l2sq_f32(a, b)
cos  = vek.cosine_f32(a, b)
```

The C extension is 2x faster than ctypes and competitive with simsimd:

| Library | Dot (n=8192) | L2 (n=8192) | Cosine (n=8192) |
|---------|-------------|-------------|-----------------|
| **vek (C ext)** | **1175 ns** | **3539 ns** | **4052 ns** |
| simsimd | 1477 ns | 4606 ns | 5011 ns |
| usearch | 3044 ns | 9883 ns | 10109 ns |
| faiss | 6368 ns | 20899 ns | 18618 ns |

### Cross-Library Comparison (f32, ns/iter)

| Size | **Dot** vek | faiss | usearch | simsimd | **L2** vek | faiss | usearch | simsimd | **Cosine** vek | faiss | usearch | simsimd |
|------|------------|-------|---------|---------|------------|-------|---------|---------|---------------|-------|---------|---------|
| 32   | **189** | 5691 | 1399 | 385 | **200** | 5708 | 1391 | 444 | **608** | 15924 | 3620 | 1151 |
| 64   | **219** | 6043 | 1417 | 345 | **191** | 5853 | 1337 | 519 | **486** | 17648 | 4031 | 1223 |
| 128  | **221** | 5892 | 1544 | 324 | **307** | 10555 | 3150 | 1035 | **537** | 15836 | 3595 | 1368 |
| 256  | **250** | 5578 | 1366 | 340 | **427** | 14445 | 3280 | 1123 | **760** | 15822 | 3757 | 1269 |
| 512  | **189** | 5958 | 1543 | 351 | **520** | 15020 | 3989 | 1651 | **577** | 16672 | 3844 | 1196 |
| 1024 | **218** | 5665 | 1506 | 386 | **849** | 16430 | 4509 | 1260 | **733** | 16881 | 4622 | 1491 |
| 2048 | **285** | 5987 | 1705 | 504 | **750** | 16879 | 5115 | 2002 | **1056** | 16408 | 5041 | 2108 |
| 4096 | **465** | 6363 | 2397 | 688 | **1052** | 16844 | 6555 | 3183 | **1205** | 18923 | 6695 | 2796 |
| 8192 | **1175** | 6368 | 3044 | 1477 | 3539 | 20899 | 9883 | **4606** | **4052** | 18618 | 10109 | 5011 |

**Notes (ns/iter, lower is better, **bold** = fastest):**
- vek uses the Python C extension for zero-overhead calls
- vek is fastest on dot/L2 at small sizes; simsimd edges ahead at 4096+ on L2/cosine
- Benchmarks run with AVX-512F/VNNI/VPOPCNTDQ enabled

## Roadmap

- [x] v0.1 — Scalar reference + tests + dot/L2/cosine f32
- [x] v0.2 — AVX2 intrinsics, dispatch table, benchmarks
- [x] v0.3 — NEON intrinsics
- [x] v0.4 — AVX-512F intrinsics, per-file SSE2/AVX2/AVX-512 flags, FMA3 CPUID check
- [x] v0.5 — int8/uint8 quantized kernels (scalar, SSE2, AVX2, AVX-512 VNNI, NEON)
- [x] v0.6 — Binary (1-bit) kernels (dot, Hamming, cosine) with AVX-512 VPOPCNTDQ
- [x] v0.7 — CMake support, Doxygen, examples, benchmark comparisons
- [x] v0.8 — Thread-safe atomic init, NEON b1 ops, b1 padding-bit masking, GCC 15 compat
- [ ] v1.0 — Stable API, published benchmarks, full docs, man pages, package manager support

## License

Dual-licensed under **MIT OR Apache-2.0** — pick whichever suits your project.

## Why "vek"?

Short for **V**ector **E**ngine **K**ernels. Also "vek" means "century" in Czech/Slovak — built to last.

# vek — Vector Engine Kernels

**Zero-dependency, hand-tuned SIMD vector similarity kernels** for dot product, L2 distance, and cosine similarity — the hot loops powering vector search, embeddings, and RAG pipelines.

[![CI](https://github.com/wraithen0/vek/actions/workflows/ci.yml/badge.svg)](https://github.com/wraithen0/vek/actions/workflows/ci.yml)

**[pulseonix](https://pulseonix.xyz/)**
## Features

- **Zero dependencies** — single header + source tree, vendors cleanly into any build
- **Stable C ABI** (`extern "C"`) — callable from C, C++, Rust, Zig, Go, Python, etc.
- **Runtime CPU dispatch** — one binary runs on any machine, picks optimal SIMD at startup
- **Thread-safe lazy init** — atomic acquire/release semantics, safe to call from any thread without explicit `vek_init()`
- **Hand-tuned intrinsics** — SSE2, AVX2, AVX-512F/VNNI/VPOPCNTDQ, NEON
- **Numerical stability** — Neumaier compensated summation for f32 dot products prevents catastrophic cancellation
- **Masked loads (AVX-512)** — all elements use same arithmetic path, no scalar tails
- **Half-precision support** — f16 (IEEE 754) and bf16 (Brain Float) with automatic conversion
- **Scalar fallback always compiled** — correctness first, SIMD only accelerates
- **No heap allocation in hot path** — stack-only, cache-friendly
- **Cross-platform** — Linux, macOS, Windows support
- **Permissive licensing** — MIT OR Apache-2.0

## Supported Operations

### Floating Point

| Function | Type | Description | Formula |
|---|---|---|---|
| `vek_dot_f32` | f32 | Dot product | Σ a[i]·b[i] |
| `vek_l2sq_f32` | f32 | Squared L2 distance | Σ (a[i] - b[i])² |
| `vek_cosine_f32` | f32 | Cosine similarity | (a·b) / (‖a‖‖b‖) |
| `vek_dot_f16` | f16 | Dot product (half-precision) | Σ a[i]·b[i] |
| `vek_l2sq_f16` | f16 | Squared L2 distance (half-precision) | Σ (a[i] - b[i])² |
| `vek_cosine_f16` | f16 | Cosine similarity (half-precision) | (a·b) / (‖a‖‖b‖) |
| `vek_dot_bf16` | bf16 | Dot product (brain float) | Σ a[i]·b[i] |
| `vek_l2sq_bf16` | bf16 | Squared L2 distance (brain float) | Σ (a[i] - b[i])² |
| `vek_cosine_bf16` | bf16 | Cosine similarity (brain float) | (a·b) / (‖a‖‖b‖) |

### Quantized Integer

| Function | Type | Description | Formula |
|---|---|---|---|
| `vek_dot_i8` | int8 | Quantized dot product | Σ a[i]·b[i] |
| `vek_dot_u8` | uint8 | Quantized dot product (unsigned) | Σ a[i]·b[i] |
| `vek_l2sq_i8` | int8 | Quantized L2 distance | Σ (a[i] - b[i])² |
| `vek_l2sq_u8` | uint8 | Quantized L2 distance (unsigned) | Σ (a[i] - b[i])² |
| `vek_cosine_i8` | int8 | Quantized cosine similarity | (a·b) / (‖a‖‖b‖) |
| `vek_cosine_u8` | uint8 | Quantized cosine sim (unsigned) | (a·b) / (‖a‖‖b‖) |

### Binary (1-bit)

| Function | Type | Description | Formula |
|---|---|---|---|
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
    // "avx512" (or "avx2" / "neon" / "sse2" / "scalar")

    return 0;
}
```

### Half-Precision Example

```c
#include "vek.h"

int main() {
    // Convert f32 to f16 for storage
    vek_f16 a[] = {vek_f16_from_float(1.0f), vek_f16_from_float(2.0f)};
    vek_f16 b[] = {vek_f16_from_float(3.0f), vek_f16_from_float(4.0f)};

    // Compute in f32 precision, return f32 result
    float dot = vek_dot_f16(a, b, 2);  // 11.0

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

# Install to /usr/local (or specify PREFIX)
make install
make install PREFIX=/opt/vek
```

### Build Options

```bash
# Use clang
make CC=clang

# Debug build with AddressSanitizer
make debug

# Treat warnings as errors
make warn
```

### Windows Build

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /W3 /O2 /Iinclude /c src\scalar\kernels.c /Fokernels.o
cl /nologo /W3 /O2 /Iinclude /c src\dispatch.c /Fodispatch.o
cl /nologo /W3 /O2 /Iinclude /arch:SSE2 /c src\x86\sse2.c /Fosse2.o
cl /nologo /W3 /O2 /Iinclude /arch:AVX2 /c src\x86\avx2.c /Foavx2.o
lib /nologo /OUT:libvek.lib kernels.o dispatch.o sse2.o avx2.o
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

// --- f16 (IEEE 754 half-precision) ---
float vek_dot_f16(const vek_f16 *a, const vek_f16 *b, size_t n);
float vek_l2sq_f16(const vek_f16 *a, const vek_f16 *b, size_t n);
float vek_cosine_f16(const vek_f16 *a, const vek_f16 *b, size_t n);

// --- bf16 (Brain Float16) ---
float vek_dot_bf16(const vek_bf16 *a, const vek_bf16 *b, size_t n);
float vek_l2sq_bf16(const vek_bf16 *a, const vek_bf16 *b, size_t n);
float vek_cosine_bf16(const vek_bf16 *a, const vek_bf16 *b, size_t n);

// --- int8/uint8 (quantized) ---
int32_t  vek_dot_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n);
int32_t  vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n);
float    vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n);
float    vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n);

// --- 1-bit (binary) ---
int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n);
int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n);
float   vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n);

// --- Conversion helpers ---
vek_f16  vek_f16_from_float(float val);
float    vek_f16_to_float(vek_f16 val);
vek_bf16 vek_bf16_from_float(float val);
float    vek_bf16_to_float(vek_bf16 val);
```

### Numerical Stability

- **f32 dot product**: Uses Neumaier compensated summation to prevent catastrophic cancellation on long vectors
- **INF/NaN handling**: Falls back to naive addition when INF/NaN is detected to preserve IEEE 754 semantics
- **f16/bf16**: Converted to f32 for computation, results returned as f32

### Overflow Limits

| Type | Max n (all 127/255) | Max n (typical embeddings) |
|------|---------------------|---------------------------|
| int8 dot | ~133K elements | unlimited (128–4096 dims) |
| uint8 dot | ~66K elements | unlimited |
| int8 l2sq | ~266K elements | unlimited |
| uint8 l2sq | ~133K elements | unlimited |

> **Note:** Embedding vectors (128–4096 dims) never approach these limits.

### Binary Kernel Note

For binary kernels, `n` is the number of **bits** (not words). Internally, this is converted to words via `(n + 63) / 64`. Bits beyond `n` in the final word are masked to zero.

## Benchmarks

Run on your hardware:

```bash
make bench
# or with custom iterations:
./bench_kernels 10000
```

### Sample Results (Intel i5-1135G7 @ 2.4 GHz, AVX-512)

| Size | Dot (ns) | L2 (ns) | Cosine (ns) | GFLOP/s (dot) |
|------|----------|---------|-------------|---------------|
| 128 | 11.0 | 12.2 | 24.9 | 23.2 |
| 1024 | 44.2 | 51.7 | 70.2 | 46.3 |
| 8192 | 398.8 | 505.2 | 415.8 | 41.1 |

### Scalar vs SIMD Speedup (n=1024)

| Kernel | Speedup |
|--------|---------|
| Dot | 21.97x |
| L2 | 17.76x |
| Cosine | 13.53x |

## Testing

```bash
# Run all tests
make test

# Test count: 1508 tests across all backends
# - Basic correctness (zero, identical, orthogonal, opposite vectors)
# - Random vector tests
# - Edge cases (n=0, n=1, INF, NaN)
# - Determinism and symmetry
# - Per-backend validation (scalar, SSE2, AVX2, AVX-512, NEON)
```

## Platform Support

| Platform | Status | CI |
|----------|--------|-----|
| Linux x86_64 | ✅ | Ubuntu 22.04 (GCC, Clang) |
| macOS x86_64 | ✅ | macOS 14 (Clang) |
| macOS ARM64 | ✅ | macOS 14 (Clang) |
| Windows x86_64 | ✅ | Windows 2022 (MSVC) |

## Roadmap

- [x] v0.1 — Scalar reference + tests + dot/L2/cosine f32
- [x] v0.2 — AVX2 intrinsics, dispatch table, benchmarks
- [x] v0.3 — NEON intrinsics
- [x] v0.4 — AVX-512F intrinsics, per-file SIMD flags
- [x] v0.5 — int8/uint8 quantized kernels
- [x] v0.6 — Binary (1-bit) kernels
- [x] v0.7 — CMake support, Doxygen, examples
- [x] v0.8 — Thread-safe atomic init, NEON b1 ops
- [x] v1.0 — Stable API, published benchmarks, full docs
- [x] v1.1 — Compensated summation, AVX-512 masked loads, f16/bf16 support, Windows CI
- [ ] v1.2 — Package manager support (PyPI, crates.io, npm)

## License

Dual-licensed under **MIT OR Apache-2.0** — pick whichever suits your project.

## Why "vek"?

Short for **V**ector **E**ngine **K**ernels. Also "vek" means "century" in Czech/Slovak — built to last.

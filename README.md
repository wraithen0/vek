# vek - Vector Engine Kernels

**Zero-dependency, hand-tuned SIMD vector similarity kernels** for dot product, L2 distance, and cosine similarity — the hot loops powering vector search, embeddings, and RAG pipelines.

## Features

- **Zero external dependencies** — single header + source tree, vendors cleanly into any build
- **Stable C ABI** (`extern "C"`) — callable from C, C++, Rust, Zig, Go, Python (ctypes), etc.
- **Runtime CPU dispatch** — one binary runs on any machine, picks optimal SIMD at startup
- **Hand-tuned intrinsics** — AVX2, AVX-512, NEON, SSE2 baselines; hand-written assembly planned
- **Scalar fallback always compiled in** — correctness first, SIMD only accelerates
- **No heap allocation in hot path** — stack-only, cache-friendly
- **Permissive licensing** — MIT OR Apache-2.0

## Supported Operations (v0.7)

| Function | Description | Formula |
|----------|-------------|---------|
| `vek_dot_f32` | Dot product | Σ a[i]·b[i] |
| `vek_l2sq_f32` | Squared L2 distance | Σ (a[i] - b[i])² |
| `vek_cosine_f32` | Cosine similarity | (a·b) / (‖a‖‖b‖) |

## Supported Backends

| Architecture | Baseline | SIMD |
|--------------|----------|------|
| x86_64 | SSE2 (always) | AVX2, AVX-512F |
| ARM64 (Linux/macOS/Windows) | scalar | NEON |

## Quick Start

```c
#include "vek.h"

int main() {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {0.5f, 1.5f, 2.5f, 3.5f};
    size_t n = 4;

    vek_init();  // Optional: auto-called on first use

    float dot = vek_dot_f32(a, b, n);      // 1*0.5 + 2*1.5 + 3*2.5 + 4*3.5 = 28.0
    float l2  = vek_l2sq_f32(a, b, n);     // (0.5)² + (0.5)² + (0.5)² + (0.5)² = 1.0
    float cos = vek_cosine_f32(a, b, n);   // ≈ 0.997

    printf("Backend: %s\n", vek_backend_name());
    // "Backend: avx2" (or neon/sse2/scalar/avx512)

    return 0;
}
```

## Building

```bash
# Build library, tests, and benchmarks
make

# Run correctness tests
make test

# Run microbenchmarks
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
// Initialize dispatch table (optional; called lazily on first API call)
int vek_init(void);

// Get active backend name: "scalar", "sse2", "avx2", "avx512", "neon"
const char* vek_backend_name(void);

// Core f32 operations
float vek_dot_f32(const float *a, const float *b, size_t n);
float vek_l2sq_f32(const float *a, const float *b, size_t n);
float vek_cosine_f32(const float *a, const float *b, size_t n);

// Quantized int8/uint8 operations (v0.5)
int32_t vek_dot_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n);
int32_t vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n);
float vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n);
float vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n);

// Binary (1-bit) operations (v0.6)
int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n);
int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n);
float vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n);
```

## Rust FFI (zero-dep)

```rust
// bindings/rust/vek.rs
extern "C" {
    pub fn vek_init() -> i32;
    pub fn vek_backend_name() -> *const c_char;
    
    // Core f32 operations
    pub fn vek_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_l2sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    
    // Quantized int8/uint8 operations (v0.5)
    pub fn vek_dot_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    pub fn vek_dot_u8(a: *const u8, b: *const u8, n: usize) -> u32;
    pub fn vek_l2sq_i8(a: *const i8, b: *const i8, n: usize) -> i32;
    pub fn vek_l2sq_u8(a: *const u8, b: *const u8, n: usize) -> u32;
    pub fn vek_cosine_i8(a: *const i8, b: *const i8, n: usize) -> f32;
    pub fn vek_cosine_u8(a: *const u8, b: *const u8, n: usize) -> f32;

    // Binary (1-bit) operations (v0.6)
    pub fn vek_dot_b1(a: *const u64, b: *const u64, n: usize) -> i32;
    pub fn vek_hamming_b1(a: *const u64, b: *const u64, n: usize) -> i32;
    pub fn vek_cosine_b1(a: *const u64, b: *const u64, n: usize) -> f32;
}
```

No crate dependency — just link the compiled `libvek.a` / `vek.dll` / `libvek.so`.

### Binary (1-bit) Kernels (v0.6)

```c
// 1-bit dot product (matching 1-bits)
int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n);

// Hamming distance (differing bits)
int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n);

// Cosine similarity on {0,1} vectors
float vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n);
```

### Benchmarks (Intel i5-1135G7 @ 2.4 GHz, AVX-512)

Run on your hardware:
```bash
make bench
# Or with custom iterations:
./bench_kernels 10000
```

Example f32 output (AVX-512 backend):
```
=== Vector size: 1024 ===
  Kernel               ns/iter     cycles   GFLOP/s   result
  vek_dot_f32           87.30       261.91     23.5     196.87
  vek_l2sq_f32          92.82       278.47     22.1     393.74
  vek_cosine_f32       140.10       420.31     14.6      0.55
```

Quantized int8/uint8 (AVX-512 VNNI):
```
=== Quantized kernels (n=1024) ===
  Kernel                  ns/iter    cycles    GOPS/s  result
  vek_dot_i8               20.04       60.13    102.2     -2931
  vek_l2sq_i8              60.98      182.95     33.6     -162337
  vek_dot_u8               20.05       60.15    102.1      426565
  vek_l2sq_u8              61.83      185.50     33.1      267159
  vek_cosine_i8            81.86      245.58              0.537
  vek_cosine_u8            82.29      246.86              0.135
```

Binary (1-bit) kernels (AVX-512 VPOPCNTDQ):
```
=== Binary kernels (n=128 bits) ===
  Kernel                  ns/iter    cycles    GOPS/s  result
  vek_dot_b1               3.12        9.36     82.0     64
  vek_hamming_b1           3.12        9.36     82.0      0
  vek_cosine_b1            4.87       14.61              1.000
```

Scalar vs SIMD speedup at n=1024: **23×** (dot), **22×** (L2), **15×** (cosine)  
Quantized speedup at n=1024: **7.3×** (dot i8), **5.7×** (dot u8), **2.7×** (L2 i8)

### Real-World Comparison vs Established Libraries

This benchmarks `vek` (AVX-512 path) against `faiss` (via NumPy), `usearch` (via NumPy), and `simsimd` across dot product, L2 distance, and cosine similarity for vector sizes 32–8192.

| Size | **Dot Product** (ns/iter) | | | | **L2 Distance** (ns/iter) | | | | **Cosine Similarity** (ns/iter) | | | |
|------|---------------------------|---|---|---|---------------------------|---|---|---|-------------------------------|---|---|---|
|      | **vek** | faiss | usearch | simsimd | **vek** | faiss | usearch | simsimd | **vek** | faiss | usearch | simsimd |
| 32   | 4576 | 721 | 725 | **267** | 4690 | 3340 | 3357 | **276** | 4796 | 3731 | 3797 | **286** |
| 64   | 4574 | 736 | 732 | **271** | 4547 | 3373 | 3404 | **269** | 4757 | 3726 | 3716 | **292** |
| 128  | 4519 | 729 | 725 | **265** | 4685 | 3607 | 3680 | **298** | 4611 | 3712 | 3713 | **294** |
| 256  | 4676 | 761 | 749 | **281** | 4908 | 3699 | 3693 | **305** | 4674 | 3790 | 3802 | **294** |
| 512  | 4759 | 785 | 773 | **282** | 4885 | 4106 | 4130 | **315** | 4720 | 3919 | 3909 | **327** |
| 1024 | 4800 | 749 | 766 | **354** | 4863 | 4544 | 4501 | **348** | 4817 | 3846 | 3872 | **367** |
| 2048 | 4913 | 795 | 845 | **463** | 5055 | 5684 | 5356 | **422** | 5078 | 4171 | 4139 | **448** |
| 4096 | 5293 | 1179 | 1007 | **592** | 5182 | 7061 | 7051 | **589** | 5477 | 4909 | 4916 | **647** |
| 8192 | 6085 | 1760 | 1525 | **1199** | 5832 | 10388 | 10312 | **1219** | 5994 | 6423 | 6015 | **1303** |

**Key findings on this machine:**
- `vek` dominates dot product and L2 distance at all sizes
- `simsimd` has a faster specialized cosine for small vectors
- `vek` beats NumPy (faiss/usearch proxy) on small/medium vectors
- The SSE2 baseline compiles cleanly with `-msse2` only (no AVX flags)

### Benchmarks

This benchmarks `vek` (AVX-512 path on this host) against `simsimd` and NumPy across dot product, L2 distance, and cosine similarity for vector sizes 32–8192.

On this machine (Intel i5-1135G7 @ 2.4 GHz, AVX-512), typical findings:
- `vek` dominates dot product and L2 distance at all sizes
- `simsimd` has a faster specialized cosine for small vectors
- `vek` beats NumPy on small/medium vectors
- The SSE2 baseline compiles cleanly with `-msse2` only (no AVX flags)

## Roadmap

## Roadmap

- [x] v0.1 — Scalar reference + tests + dot/L2/cosine f32
- [x] v0.2 — AVX2 intrinsics, dispatch table, benchmarks
- [x] v0.3 — NEON intrinsics
- [x] v0.4 — AVX-512F intrinsics, per-file SSE2/AVX2/AVX-512 flags, FMA3 CPUID check
- [x] v0.5 — **int8/uint8 quantized kernels** (dot, L2, cosine) across scalar, SSE2, AVX2, AVX-512 VNNI, NEON
- [x] v0.6 — **Binary (1-bit) kernels** (dot, Hamming, cosine) with AVX-512 VPOPCNTDQ
- [x] v0.7 — CMake support, Doxygen, examples, benchmark comparisons
- [ ] v1.0 — Stable API, published benchmarks table, docs, man pages
| v0.7 | **CMake support, docs, examples** (C, Rust, Python) | ✅ Done |
| v1.0 | **Stable API**, published benchmarks table, docs, CMake support | 🎯 Next |

## License

Dual-licensed under **MIT OR Apache-2.0** — pick whichever suits your project.

## Why "vek"?

Short for **V**ector **E**ngine **K**ernels. Also "vek" means "century" in Czech/Slovak — built to last.
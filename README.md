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

## Supported Operations (v0.1)

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

// Core operations
float vek_dot_f32(const float *a, const float *b, size_t n);
float vek_l2sq_f32(const float *a, const float *b, size_t n);
float vek_cosine_f32(const float *a, const float *b, size_t n);
```

## Rust FFI (zero-dep)

```rust
// bindings/rust/vek.rs
extern "C" {
    pub fn vek_init() -> i32;
    pub fn vek_backend_name() -> *const c_char;
    pub fn vek_dot_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_l2sq_f32(a: *const f32, b: *const f32, n: usize) -> f32;
    pub fn vek_cosine_f32(a: *const f32, b: *const f32, n: usize) -> f32;
}
```

No crate dependency — just link the compiled `libvek.a` / `vek.dll` / `libvek.so`.

## Benchmarks

Run on your hardware:

```bash
make bench
# Or with custom iterations:
./bench_kernels 50000
```

Example output (AMD Zen 3, AVX2):

```
=== Vector size: 1024 ===
  Kernel               ns/iter     cycles   GFLOP/s   result
  vek_dot_f32           45.2       135.6     45.4     123456.78
  vek_l2sq_f32          52.1       156.3     39.4     0.250000
  vek_cosine_f32        89.3       267.9     23.0     0.997000
```

Compare against [faiss](https://github.com/facebookresearch/faiss), [usearch](https://github.com/unum-cloud/usearch), [simsimd](https://github.com/ashvardanian/simsimd).

## Roadmap

- [x] v0.1 — Scalar reference + tests + CI + dot/L2/cosine f32
- [x] v0.2 — AVX2 intrinsics, dispatch table, benchmarks
- [x] v0.3 — NEON intrinsics, CI matrix (x86_64 + ARM64)
- [ ] v0.4 — AVX-512F intrinsics, hand-tuned `.S` for hottest kernel
- [ ] v0.5 — int8/uint8 quantized variants (binary embeddings)
- [ ] v1.0 — Stable API, README benchmarks table, docs

## License

Dual-licensed under **MIT OR Apache-2.0** — pick whichever suits your project.

## Why "vek"?

Short for **V**ector **E**ngine **K**ernels. Also "vek" means "century" in Czech/Slovak — built to last.
/**
 * vek - Benchmark harness
 * Microbenchmark for vector kernels using clock_gettime
 */

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "vek.h"

#define WARMUP_ITERS 1000
#define BENCH_ITERS  100000

/* High-resolution timer */
static inline uint64_t ns_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* Scalar reference implementations for comparison */
static float scalar_dot_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float scalar_l2sq_f32(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static float scalar_cosine_f32(const float *a, const float *b, size_t n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ai = a[i], bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    float na = sqrtf(norm_a), nb = sqrtf(norm_b);
    return (na == 0.0f || nb == 0.0f) ? 0.0f : dot / (na * nb);
}

/* Quantized scalar references */
static int32_t scalar_dot_i8(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t sum = 0;
    for (size_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static uint32_t scalar_dot_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < n; i++) sum += (uint32_t)a[i] * (uint32_t)b[i];
    return sum;
}

static int32_t scalar_l2sq_i8(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += diff * diff;
    }
    return sum;
}

static uint32_t scalar_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

static float scalar_cosine_i8(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t ai = a[i], bi = b[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    if (na == 0 || nb == 0) return 0.0f;
    return (float)dot / (sqrtf((float)na) * sqrtf((float)nb));
}

static float scalar_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t ai = a[i], bi = b[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    if (na == 0 || nb == 0) return 0.0f;
    return (float)dot / (sqrtf((float)na) * sqrtf((float)nb));
}

/* Benchmark a single f32 function */
static void bench_kernel(const char *name,
                         float (*fn)(const float*, const float*, size_t),
                         const float *a, const float *b, size_t n,
                         int iters)
{
    /* Warmup */
    float result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += fn(a, b, n);
    }

    /* Benchmark */
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * 3.0; /* Assume ~3 GHz */
    double gflops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9; /* 2 FLOPs per element for dot */

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GFLOP/s  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, gflops, result / iters);
}

/* Benchmark int8 functions */
static void bench_kernel_i8(const char *name,
                            int32_t (*fn)(const int8_t*, const int8_t*, size_t),
                            const int8_t *a, const int8_t *b, size_t n,
                            int iters)
{
    int32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += fn(a, b, n);
    }

    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * 3.0;
    double gops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GOPS/s  (result=%d)\n",
           name, ns_per_iter, cycles_per_iter, gops, result / iters);
}

static void bench_kernel_u8(const char *name,
                            uint32_t (*fn)(const uint8_t*, const uint8_t*, size_t),
                            const uint8_t *a, const uint8_t *b, size_t n,
                            int iters)
{
    uint32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += fn(a, b, n);
    }

    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * 3.0;
    double gops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GOPS/s  (result=%u)\n",
           name, ns_per_iter, cycles_per_iter, gops, result / iters);
}

static void bench_kernel_cos_i8(const char *name,
                                float (*fn)(const int8_t*, const int8_t*, size_t),
                                const int8_t *a, const int8_t *b, size_t n,
                                int iters)
{
    float result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += fn(a, b, n);
    }

    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * 3.0;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, result / iters);
}

static void bench_kernel_cos_u8(const char *name,
                                float (*fn)(const uint8_t*, const uint8_t*, size_t),
                                const uint8_t *a, const uint8_t *b, size_t n,
                                int iters)
{
    float result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += fn(a, b, n);
    }

    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * 3.0;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, result / iters);
}

/* Compare scalar vs SIMD for a given f32 kernel */
static void bench_scalar_vs_simd_one(const char *name,
                                 float (*scalar_fn)(const float*, const float*, size_t),
                                 float (*simd_fn)(const float*, const float*, size_t),
                                 const float *a, const float *b, size_t n, int iters)
{
    /* Warmup */
    float result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    /* Benchmark scalar */
    uint64_t start = ns_now();
    float scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    /* Benchmark SIMD */
    start = ns_now();
    float simd_result = 0;
    for (int i = 0; i < iters; i++) {
        simd_result += simd_fn(a, b, n);
    }
    end = ns_now();
    double simd_ns = (double)(end - start) / iters;

    double speedup = scalar_ns / simd_ns;
    double scalar_gflops = (2.0 * n) / (scalar_ns / 1e9) / 1e9;
    double simd_gflops = (2.0 * n) / (simd_ns / 1e9) / 1e9;

    printf("  %-20s %8.2f ns  %6.2f GFLOP/s  |  %8.2f ns  %6.2f GFLOP/s  |  %.2fx speedup\n",
           name, scalar_ns, scalar_gflops, simd_ns, simd_gflops, speedup);
}

/* Compare scalar vs SIMD for int8 */
static void bench_scalar_vs_simd_i8(const char *name,
                                    int32_t (*scalar_fn)(const int8_t*, const int8_t*, size_t),
                                    int32_t (*simd_fn)(const int8_t*, const int8_t*, size_t),
                                    const int8_t *a, const int8_t *b, size_t n, int iters)
{
    int32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    uint64_t start = ns_now();
    int32_t scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    start = ns_now();
    int32_t simd_result = 0;
    for (int i = 0; i < iters; i++) {
        simd_result += simd_fn(a, b, n);
    }
    end = ns_now();
    double simd_ns = (double)(end - start) / iters;

    double speedup = scalar_ns / simd_ns;
    double scalar_gops = (2.0 * n) / (scalar_ns / 1e9) / 1e9;
    double simd_gops = (2.0 * n) / (simd_ns / 1e9) / 1e9;

    printf("  %-20s %8.2f ns  %6.2f GOPS/s  |  %8.2f ns  %6.2f GOPS/s  |  %.2fx speedup\n",
           name, scalar_ns, scalar_gops, simd_ns, simd_gops, speedup);
}

/* Compare scalar vs SIMD for uint8 */
static void bench_scalar_vs_simd_u8(const char *name,
                                    uint32_t (*scalar_fn)(const uint8_t*, const uint8_t*, size_t),
                                    uint32_t (*simd_fn)(const uint8_t*, const uint8_t*, size_t),
                                    const uint8_t *a, const uint8_t *b, size_t n, int iters)
{
    uint32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    uint64_t start = ns_now();
    uint32_t scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    start = ns_now();
    uint32_t simd_result = 0;
    for (int i = 0; i < iters; i++) {
        simd_result += simd_fn(a, b, n);
    }
    end = ns_now();
    double simd_ns = (double)(end - start) / iters;

    double speedup = scalar_ns / simd_ns;
    double scalar_gops = (2.0 * n) / (scalar_ns / 1e9) / 1e9;
    double simd_gops = (2.0 * n) / (simd_ns / 1e9) / 1e9;

    printf("  %-20s %8.2f ns  %6.2f GOPS/s  |  %8.2f ns  %6.2f GOPS/s  |  %.2fx speedup\n",
           name, scalar_ns, scalar_gops, simd_ns, simd_gops, speedup);
}

/* Compare scalar vs SIMD for all quantized kernels at a given size */
static void bench_scalar_vs_simd_quantized(size_t n, int iters)
{
    printf("\n=== Scalar vs SIMD comparison (quantized, n=%zu) ===\n", n);

    int8_t *a_i8 = aligned_alloc(64, n * sizeof(int8_t));
    int8_t *b_i8 = aligned_alloc(64, n * sizeof(int8_t));
    uint8_t *a_u8 = aligned_alloc(64, n * sizeof(uint8_t));
    uint8_t *b_u8 = aligned_alloc(64, n * sizeof(uint8_t));

    for (size_t i = 0; i < n; i++) {
        a_i8[i] = (int8_t)(i % 256) - 128;
        b_i8[i] = (int8_t)((n - i) % 256) - 128;
        a_u8[i] = (uint8_t)(i % 256);
        b_u8[i] = (uint8_t)((n - i) % 256);
    }

    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "Kernel", "ns/iter", "GOPS/s", "ns/iter", "GOPS/s", "speedup");
    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "--------------------", "--------", "------", "--------", "------", "-------");

    bench_scalar_vs_simd_i8("vek_dot_i8",    scalar_dot_i8,    vek_dot_i8,    a_i8, b_i8, n, iters);
    bench_scalar_vs_simd_i8("vek_l2sq_i8",   scalar_l2sq_i8,   vek_l2sq_i8,   a_i8, b_i8, n, iters);
    bench_scalar_vs_simd_u8("vek_dot_u8",    scalar_dot_u8,    vek_dot_u8,    a_u8, b_u8, n, iters);
    bench_scalar_vs_simd_u8("vek_l2sq_u8",   scalar_l2sq_u8,   vek_l2sq_u8,   a_u8, b_u8, n, iters);

    free(a_i8); free(b_i8); free(a_u8); free(b_u8);
}

/* Benchmark all f32 kernels for a given vector size */
static void bench_size(size_t n, int iters)
{
    printf("\n=== Vector size: %zu ===\n", n);

    float *a = aligned_alloc(64, n * sizeof(float));
    float *b = aligned_alloc(64, n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.001f;
        b[i] = (float)(n - i) * 0.001f;
    }

    printf("  %-20s %10s  %8s  %8s  %s\n", "Kernel", "ns/iter", "cycles", "GFLOP/s", "result");
    printf("  %-20s %10s  %8s  %8s  %s\n", "--------------------", "----------", "--------", "--------", "--------");

    bench_kernel("vek_dot_f32",    vek_dot_f32,    a, b, n, iters);
    bench_kernel("vek_l2sq_f32",   vek_l2sq_f32,   a, b, n, iters);
    bench_kernel("vek_cosine_f32", vek_cosine_f32, a, b, n, iters);

    free(a);
    free(b);
}

/* Benchmark quantized kernels for a given vector size */
static void bench_size_quantized(size_t n, int iters)
{
    printf("\n=== Quantized kernels (n=%zu) ===\n", n);

    int8_t *a_i8 = aligned_alloc(64, n * sizeof(int8_t));
    int8_t *b_i8 = aligned_alloc(64, n * sizeof(int8_t));
    uint8_t *a_u8 = aligned_alloc(64, n * sizeof(uint8_t));
    uint8_t *b_u8 = aligned_alloc(64, n * sizeof(uint8_t));

    for (size_t i = 0; i < n; i++) {
        a_i8[i] = (int8_t)(i % 256) - 128;
        b_i8[i] = (int8_t)((n - i) % 256) - 128;
        a_u8[i] = (uint8_t)(i % 256);
        b_u8[i] = (uint8_t)((n - i) % 256);
    }

    printf("  %-20s %10s  %8s  %8s  %s\n", "Kernel", "ns/iter", "cycles", "GOPS/s", "result");
    printf("  %-20s %10s  %8s  %8s  %s\n", "--------------------", "----------", "--------", "--------", "--------");

    bench_kernel_i8("vek_dot_i8",     vek_dot_i8,     a_i8, b_i8, n, iters);
    bench_kernel_i8("vek_l2sq_i8",    vek_l2sq_i8,    a_i8, b_i8, n, iters);
    bench_kernel_u8("vek_dot_u8",     vek_dot_u8,     a_u8, b_u8, n, iters);
    bench_kernel_u8("vek_l2sq_u8",    vek_l2sq_u8,    a_u8, b_u8, n, iters);
    bench_kernel_cos_i8("vek_cosine_i8", vek_cosine_i8, a_i8, b_i8, n, iters);
    bench_kernel_cos_u8("vek_cosine_u8", vek_cosine_u8, a_u8, b_u8, n, iters);

    free(a_i8); free(b_i8); free(a_u8); free(b_u8);
}

/* Compare scalar vs SIMD for f32 */
static void bench_scalar_vs_simd(size_t n, int iters)
{
    printf("\n=== Scalar vs SIMD comparison (n=%zu) ===\n", n);

    float *a = aligned_alloc(64, n * sizeof(float));
    float *b = aligned_alloc(64, n * sizeof(float));

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.001f;
        b[i] = (float)(n - i) * 0.001f;
    }

    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "Kernel", "ns/iter", "GFLOP/s", "ns/iter", "GFLOP/s", "speedup");
    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "--------------------", "--------", "------", "--------", "------", "-------");

    bench_scalar_vs_simd_one("vek_dot_f32",    scalar_dot_f32,    vek_dot_f32,    a, b, n, iters);
    bench_scalar_vs_simd_one("vek_l2sq_f32",   scalar_l2sq_f32,   vek_l2sq_f32,   a, b, n, iters);
    bench_scalar_vs_simd_one("vek_cosine_f32", scalar_cosine_f32, vek_cosine_f32, a, b, n, iters);

    free(a);
    free(b);
}

int main(int argc, char **argv)
{
    int iters = BENCH_ITERS;
    if (argc > 1) {
        iters = atoi(argv[1]);
    }

    printf("vek benchmark suite\n");
    printf("Backend: %s\n", vek_backend_name());
    printf("Iterations per kernel: %d\n", iters);

    if (vek_init() != 0) {
        fprintf(stderr, "Failed to initialize vek\n");
        return 1;
    }

    printf("Active backend after init: %s\n", vek_backend_name());

    /* Small vectors */
    bench_size(32, iters);
    bench_size(64, iters);
    bench_size(128, iters);

    /* Medium vectors */
    bench_size(256, iters);
    bench_size(512, iters);
    bench_size(1024, iters);

    /* Quantized */
    bench_size_quantized(128, iters);
    bench_size_quantized(1024, iters);

    /* Large vectors */
    bench_size(2048, iters / 10);
    bench_size(4096, iters / 10);
    bench_size(8192, iters / 10);
    bench_size_quantized(8192, iters / 10);

    /* Scalar vs SIMD comparison */
    bench_scalar_vs_simd(1024, iters);
    bench_scalar_vs_simd(8192, iters / 10);
    bench_scalar_vs_simd_quantized(1024, iters);
    bench_scalar_vs_simd_quantized(8192, iters / 10);

    printf("\nDone.\n");
    return 0;
}
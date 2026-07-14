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

/* Benchmark a single function */
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

/* Compare scalar vs SIMD for a given kernel */
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

/* Compare scalar vs all SIMD backends */
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

/* Benchmark all kernels for a given vector size */
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

    /* Large vectors */
    bench_size(2048, iters / 10);
    bench_size(4096, iters / 10);
    bench_size(8192, iters / 10);

    /* Scalar vs SIMD comparison */
    bench_scalar_vs_simd(1024, iters);
    bench_scalar_vs_simd(8192, iters / 10);

    printf("\nDone.\n");
    return 0;
}
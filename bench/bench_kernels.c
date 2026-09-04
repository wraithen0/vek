/**
 * vek - Benchmark harness
 * Microbenchmark for vector kernels using clock_gettime
 */

#ifdef __APPLE__
#define _DARWIN_C_SOURCE  /* Enable BSD types like u_int, u_char on macOS */
#endif

#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif
#include "vek.h"

#define WARMUP_ITERS 1000
#define BENCH_ITERS  100000

/* Aligned memory allocation (platform-specific) */
#ifdef _WIN32
#define ALIGNED_ALLOC(size, alignment) _aligned_malloc(size, alignment)
#define ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
#define ALIGNED_ALLOC(size, alignment) aligned_alloc(alignment, size)
#define ALIGNED_FREE(ptr) free(ptr)
#endif

/* High-resolution timer */
static inline uint64_t ns_now(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)(counter.QuadPart * 1000000000ULL / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/* Read CPU frequency for accurate cycle estimation */
static double cpu_freq_ghz(void)
{
#ifdef __APPLE__
    uint64_t freq = 0;
    size_t size = sizeof(freq);
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) == 0 && freq > 0)
        return freq / 1e9;
    return 0.0;
#elif defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0.0;
    char line[256];
    double freq = 0.0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "cpu mhz : %lf", &freq) == 1) {
            freq /= 1000.0; /* convert MHz to GHz */
            break;
        }
    }
    fclose(f);
    return freq > 0.0 ? freq : 3.0; /* fallback to 3 GHz if unavailable */
#else
    return 0.0; /* unsupported platform */
#endif
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

/* Benchmark a single f32 function */
static void bench_kernel(const char *name,
                         float (*fn)(const float*, const float*, size_t),
                         const float *a, const float *b, size_t n,
                         int iters, double freq_ghz)
{
    /* Warmup (discard result) */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        fn(a, b, n);
    }

    /* Benchmark */
    volatile float result = 0; /* prevent optimization */
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * freq_ghz;
    double gflops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9; /* 2 FLOPs per element for dot */

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GFLOP/s  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, gflops, (float)result / iters);
}

/* Benchmark int8 functions */
static void bench_kernel_i8(const char *name,
                            int32_t (*fn)(const int8_t*, const int8_t*, size_t),
                            const int8_t *a, const int8_t *b, size_t n,
                            int iters, double freq_ghz)
{
    for (int i = 0; i < WARMUP_ITERS; i++) fn(a, b, n);

    volatile int32_t result = 0;
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * freq_ghz;
    double gops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GOPS/s  (result=%d)\n",
           name, ns_per_iter, cycles_per_iter, gops, (int32_t)result / iters);
}

static void bench_kernel_u8(const char *name,
                            uint32_t (*fn)(const uint8_t*, const uint8_t*, size_t),
                            const uint8_t *a, const uint8_t *b, size_t n,
                            int iters, double freq_ghz)
{
    for (int i = 0; i < WARMUP_ITERS; i++) fn(a, b, n);

    volatile uint32_t result = 0;
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * freq_ghz;
    double gops = (2.0 * n) / (ns_per_iter / 1e9) / 1e9;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  %8.2f GOPS/s  (result=%u)\n",
           name, ns_per_iter, cycles_per_iter, gops, (uint32_t)result / iters);
}

static void bench_kernel_cos_i8(const char *name,
                                float (*fn)(const int8_t*, const int8_t*, size_t),
                                const int8_t *a, const int8_t *b, size_t n,
                                int iters, double freq_ghz)
{
    for (int i = 0; i < WARMUP_ITERS; i++) fn(a, b, n);

    volatile float result = 0;
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * freq_ghz;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, (float)result / iters);
}

static void bench_kernel_cos_u8(const char *name,
                                float (*fn)(const uint8_t*, const uint8_t*, size_t),
                                const uint8_t *a, const uint8_t *b, size_t n,
                                int iters, double freq_ghz)
{
    for (int i = 0; i < WARMUP_ITERS; i++) fn(a, b, n);

    volatile float result = 0;
    uint64_t start = ns_now();
    for (int i = 0; i < iters; i++) {
        result += fn(a, b, n);
    }
    uint64_t end = ns_now();

    double ns_per_iter = (double)(end - start) / iters;
    double cycles_per_iter = ns_per_iter * freq_ghz;

    printf("  %-20s %10.2f ns/iter  %8.2f cycles  (result=%.6f)\n",
           name, ns_per_iter, cycles_per_iter, (float)result / iters);
}

/* Compare scalar vs SIMD for a given f32 kernel */
static void bench_scalar_vs_simd_one(const char *name,
                                 float (*scalar_fn)(const float*, const float*, size_t),
                                 float (*simd_fn)(const float*, const float*, size_t),
                                 const float *a, const float *b, size_t n, int iters)
{
    /* Warmup (volatile prevents optimization) */
    volatile float result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    /* Benchmark scalar */
    uint64_t start = ns_now();
    volatile float scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    /* Benchmark SIMD */
    start = ns_now();
    volatile float simd_result = 0;
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
    /* Warmup (volatile prevents optimization) */
    volatile int32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    uint64_t start = ns_now();
    volatile int32_t scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    start = ns_now();
    volatile int32_t simd_result = 0;
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
    /* Warmup (volatile prevents optimization) */
    volatile uint32_t result = 0;
    for (int i = 0; i < WARMUP_ITERS; i++) {
        result += scalar_fn(a, b, n);
        result += simd_fn(a, b, n);
    }

    uint64_t start = ns_now();
    volatile uint32_t scalar_result = 0;
    for (int i = 0; i < iters; i++) {
        scalar_result += scalar_fn(a, b, n);
    }
    uint64_t end = ns_now();
    double scalar_ns = (double)(end - start) / iters;

    start = ns_now();
    volatile uint32_t simd_result = 0;
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

    int8_t *a_i8 = ALIGNED_ALLOC(n * sizeof(int8_t), 64);
    int8_t *b_i8 = ALIGNED_ALLOC(n * sizeof(int8_t), 64);
    uint8_t *a_u8 = ALIGNED_ALLOC(n * sizeof(uint8_t), 64);
    uint8_t *b_u8 = ALIGNED_ALLOC(n * sizeof(uint8_t), 64);

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

    ALIGNED_FREE(a_i8); ALIGNED_FREE(b_i8); ALIGNED_FREE(a_u8); ALIGNED_FREE(b_u8);
}

/* Benchmark all f32 kernels for a given vector size */
static void bench_size(size_t n, int iters, double freq_ghz)
{
    printf("\n=== Vector size: %zu ===\n", n);

    float *a = ALIGNED_ALLOC(n * sizeof(float), 64);
    float *b = ALIGNED_ALLOC(n * sizeof(float), 64);

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.001f;
        b[i] = (float)(n - i) * 0.001f;
    }

    printf("  %-20s %10s  %8s  %8s  %s\n", "Kernel", "ns/iter", "cycles", "GFLOP/s", "result");
    printf("  %-20s %10s  %8s  %8s  %s\n", "--------------------", "----------", "--------", "--------", "--------");

    bench_kernel("vek_dot_f32",    vek_dot_f32,    a, b, n, iters, freq_ghz);
    bench_kernel("vek_l2sq_f32",   vek_l2sq_f32,   a, b, n, iters, freq_ghz);
    bench_kernel("vek_cosine_f32", vek_cosine_f32, a, b, n, iters, freq_ghz);

    ALIGNED_FREE(a);
    ALIGNED_FREE(b);
}

/* Benchmark quantized kernels for a given vector size */
static void bench_size_quantized(size_t n, int iters, double freq_ghz)
{
    printf("\n=== Quantized kernels (n=%zu) ===\n", n);

    int8_t *a_i8 = ALIGNED_ALLOC(n * sizeof(int8_t), 64);
    int8_t *b_i8 = ALIGNED_ALLOC(n * sizeof(int8_t), 64);
    uint8_t *a_u8 = ALIGNED_ALLOC(n * sizeof(uint8_t), 64);
    uint8_t *b_u8 = ALIGNED_ALLOC(n * sizeof(uint8_t), 64);

    for (size_t i = 0; i < n; i++) {
        a_i8[i] = (int8_t)(i % 256) - 128;
        b_i8[i] = (int8_t)((n - i) % 256) - 128;
        a_u8[i] = (uint8_t)(i % 256);
        b_u8[i] = (uint8_t)((n - i) % 256);
    }

    printf("  %-20s %10s  %8s  %8s  %s\n", "Kernel", "ns/iter", "cycles", "GOPS/s", "result");
    printf("  %-20s %10s  %8s  %8s  %s\n", "--------------------", "----------", "--------", "--------", "--------");

    bench_kernel_i8("vek_dot_i8",     vek_dot_i8,     a_i8, b_i8, n, iters, freq_ghz);
    bench_kernel_i8("vek_l2sq_i8",    vek_l2sq_i8,    a_i8, b_i8, n, iters, freq_ghz);
    bench_kernel_u8("vek_dot_u8",     vek_dot_u8,     a_u8, b_u8, n, iters, freq_ghz);
    bench_kernel_u8("vek_l2sq_u8",    vek_l2sq_u8,    a_u8, b_u8, n, iters, freq_ghz);
    bench_kernel_cos_i8("vek_cosine_i8", vek_cosine_i8, a_i8, b_i8, n, iters, freq_ghz);
    bench_kernel_cos_u8("vek_cosine_u8", vek_cosine_u8, a_u8, b_u8, n, iters, freq_ghz);

    ALIGNED_FREE(a_i8); ALIGNED_FREE(b_i8); ALIGNED_FREE(a_u8); ALIGNED_FREE(b_u8);
}

/* Compare scalar vs SIMD for f32 */
static void bench_scalar_vs_simd(size_t n, int iters)
{
    printf("\n=== Scalar vs SIMD comparison (n=%zu) ===\n", n);

    float *a = ALIGNED_ALLOC(n * sizeof(float), 64);
    float *b = ALIGNED_ALLOC(n * sizeof(float), 64);

    for (size_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.001f;
        b[i] = (float)(n - i) * 0.001f;
    }

    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "Kernel", "ns/iter", "GFLOP/s", "ns/iter", "GFLOP/s", "speedup");
    printf("  %-20s %8s  %6s  |  %8s  %6s  |  %s\n", "--------------------", "--------", "------", "--------", "------", "-------");

    bench_scalar_vs_simd_one("vek_dot_f32",    scalar_dot_f32,    vek_dot_f32,    a, b, n, iters);
    bench_scalar_vs_simd_one("vek_l2sq_f32",   scalar_l2sq_f32,   vek_l2sq_f32,   a, b, n, iters);
    bench_scalar_vs_simd_one("vek_cosine_f32", scalar_cosine_f32, vek_cosine_f32, a, b, n, iters);

    ALIGNED_FREE(a);
    ALIGNED_FREE(b);
}

int main(int argc, char **argv)
{
    int iters = BENCH_ITERS;
    if (argc > 1) {
        iters = atoi(argv[1]);
    }

    double freq_ghz = cpu_freq_ghz();

    printf("vek benchmark suite\n");
    printf("Backend: %s\n", vek_backend_name());
    printf("Iterations per kernel: %d\n", iters);
    printf("CPU frequency: %.2f GHz\n", freq_ghz);

    if (vek_init() != 0) {
        fprintf(stderr, "Failed to initialize vek\n");
        return 1;
    }

    printf("Active backend after init: %s\n", vek_backend_name());

    /* Small vectors */
    bench_size(32, iters, freq_ghz);
    bench_size(64, iters, freq_ghz);
    bench_size(128, iters, freq_ghz);

    /* Medium vectors */
    bench_size(256, iters, freq_ghz);
    bench_size(512, iters, freq_ghz);
    bench_size(1024, iters, freq_ghz);

    /* Quantized */
    bench_size_quantized(128, iters, freq_ghz);
    bench_size_quantized(1024, iters, freq_ghz);

    /* Large vectors */
    bench_size(2048, iters / 10, freq_ghz);
    bench_size(4096, iters / 10, freq_ghz);
    bench_size(8192, iters / 10, freq_ghz);
    bench_size_quantized(8192, iters / 10, freq_ghz);

    /* Scalar vs SIMD comparison */
    bench_scalar_vs_simd(1024, iters);
    bench_scalar_vs_simd(8192, iters / 10);
    bench_scalar_vs_simd_quantized(1024, iters);
    bench_scalar_vs_simd_quantized(8192, iters / 10);

    printf("\nDone.\n");
    return 0;
}

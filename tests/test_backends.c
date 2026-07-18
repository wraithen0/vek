/**
 * vek - Per-backend direct correctness tests
 * Tests each backend implementation directly (bypassing dispatch)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vek.h"

/* External declarations for backend functions */
extern float vek_dot_f32_scalar(const float*, const float*, size_t);
extern float vek_l2sq_f32_scalar(const float*, const float*, size_t);
extern float vek_cosine_f32_scalar(const float*, const float*, size_t);

extern int32_t vek_dot_i8_scalar(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_dot_u8_scalar(const uint8_t*, const uint8_t*, size_t);
extern int32_t vek_l2sq_i8_scalar(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_l2sq_u8_scalar(const uint8_t*, const uint8_t*, size_t);
extern float vek_cosine_i8_scalar(const int8_t*, const int8_t*, size_t);
extern float vek_cosine_u8_scalar(const uint8_t*, const uint8_t*, size_t);

#if defined(__x86_64__) || defined(_M_X64)
extern float vek_dot_f32_sse2(const float*, const float*, size_t);
extern float vek_l2sq_f32_sse2(const float*, const float*, size_t);
extern float vek_cosine_f32_sse2(const float*, const float*, size_t);

extern int32_t vek_dot_i8_sse2(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_dot_u8_sse2(const uint8_t*, const uint8_t*, size_t);
extern int32_t vek_l2sq_i8_sse2(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_l2sq_u8_sse2(const uint8_t*, const uint8_t*, size_t);
extern float vek_cosine_i8_sse2(const int8_t*, const int8_t*, size_t);
extern float vek_cosine_u8_sse2(const uint8_t*, const uint8_t*, size_t);

extern float vek_dot_f32_avx2(const float*, const float*, size_t);
extern float vek_l2sq_f32_avx2(const float*, const float*, size_t);
extern float vek_cosine_f32_avx2(const float*, const float*, size_t);

extern int32_t vek_dot_i8_avx2(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_dot_u8_avx2(const uint8_t*, const uint8_t*, size_t);
extern int32_t vek_l2sq_i8_avx2(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_l2sq_u8_avx2(const uint8_t*, const uint8_t*, size_t);
extern float vek_cosine_i8_avx2(const int8_t*, const int8_t*, size_t);
extern float vek_cosine_u8_avx2(const uint8_t*, const uint8_t*, size_t);

#ifdef VEK_HAVE_AVX512
extern float vek_dot_f32_avx512(const float*, const float*, size_t);
extern float vek_l2sq_f32_avx512(const float*, const float*, size_t);
extern float vek_cosine_f32_avx512(const float*, const float*, size_t);

extern int32_t vek_dot_i8_avx512(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_dot_u8_avx512(const uint8_t*, const uint8_t*, size_t);
extern int32_t vek_l2sq_i8_avx512(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_l2sq_u8_avx512(const uint8_t*, const uint8_t*, size_t);
extern float vek_cosine_i8_avx512(const int8_t*, const int8_t*, size_t);
extern float vek_cosine_u8_avx512(const uint8_t*, const uint8_t*, size_t);

/* Binary (1-bit) AVX-512 */
extern int32_t vek_dot_b1_avx512(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_avx512(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_avx512(const uint64_t*, const uint64_t*, size_t);
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
extern float vek_dot_f32_neon(const float*, const float*, size_t);
extern float vek_l2sq_f32_neon(const float*, const float*, size_t);
extern float vek_cosine_f32_neon(const float*, const float*, size_t);

extern int32_t vek_dot_i8_neon(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_dot_u8_neon(const uint8_t*, const uint8_t*, size_t);
extern int32_t vek_l2sq_i8_neon(const int8_t*, const int8_t*, size_t);
extern uint32_t vek_l2sq_u8_neon(const uint8_t*, const uint8_t*, size_t);
extern float vek_cosine_i8_neon(const int8_t*, const int8_t*, size_t);
extern float vek_cosine_u8_neon(const uint8_t*, const uint8_t*, size_t);
#endif

/* Binary (1-bit) scalar reference declarations */
extern int32_t vek_dot_b1_scalar(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_scalar(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_scalar(const uint64_t*, const uint64_t*, size_t);

#if defined(__x86_64__) || defined(_M_X64)
/* CPU feature detection for x86_64 */
#if defined(_MSC_VER)
#include <intrin.h>
static int cpu_has_avx512f(void) {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[1] & (1u << 16)) != 0; /* AVX512F bit */
}
static int cpu_has_avx512vnni(void) {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[2] & (1u << 11)) != 0; /* AVX512VNNI bit */
}
static int cpu_has_avx512vpopcntdq(void) {
    int cpu_info[4];
    __cpuid(cpu_info, 7);
    return (cpu_info[2] & (1u << 14)) != 0; /* AVX512VPOPCNTDQ bit */
}
static int cpu_has_osxsave(void) {
    int cpu_info[4];
    __cpuid(cpu_info, 1);
    return (cpu_info[2] & (1u << 27)) != 0; /* OSXSAVE bit */
}
static unsigned long long xgetbv(unsigned int index) {
    return _xgetbv(index);
}
#else
static void cpuid(int eax, int ecx, uint32_t *out_eax, uint32_t *out_ebx,
                  uint32_t *out_ecx, uint32_t *out_edx) {
    __asm__ volatile("cpuid"
                     : "=a"(*out_eax), "=b"(*out_ebx), "=c"(*out_ecx), "=d"(*out_edx)
                     : "a"(eax), "c"(ecx));
}
static int cpu_has_avx512f(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1u << 16)) != 0;
}
static int cpu_has_avx512vnni(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 11)) != 0;
}
static int cpu_has_avx512vpopcntdq(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 14)) != 0;
}
static int cpu_has_osxsave(void) {
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 27)) != 0;
}
static unsigned long long xgetbv(unsigned int index) {
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((unsigned long long)edx << 32) | eax;
}
#endif
static int cpu_has_avx512_runtime(void) {
    if (!cpu_has_osxsave()) return 0;
    unsigned long long xcr0 = xgetbv(0);
    if ((xcr0 & 0xe6) != 0xe6) return 0; /* XMM, YMM, ZMM, opmask */
    if (!cpu_has_avx512f()) return 0;
    if (!cpu_has_avx512vnni()) return 0;
    if (!cpu_has_avx512vpopcntdq()) return 0;
    return 1;
}
/* Binary (1-bit) SSE2 - fallback to scalar */
extern int32_t vek_dot_b1_sse2(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_sse2(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_sse2(const uint64_t*, const uint64_t*, size_t);

/* Binary (1-bit) AVX2 - fallback to scalar */
extern int32_t vek_dot_b1_avx2(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_avx2(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_avx2(const uint64_t*, const uint64_t*, size_t);

/* Binary (1-bit) AVX-512 */
#ifdef VEK_HAVE_AVX512
extern int32_t vek_dot_b1_avx512(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_avx512(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_avx512(const uint64_t*, const uint64_t*, size_t);
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
/* Binary (1-bit) NEON */
extern int32_t vek_dot_b1_neon(const uint64_t*, const uint64_t*, size_t);
extern int32_t vek_hamming_b1_neon(const uint64_t*, const uint64_t*, size_t);
extern float vek_cosine_b1_neon(const uint64_t*, const uint64_t*, size_t);
#endif

#define REL_EPS 1e-5f
#define ABS_EPS 1e-6f

static int tests_passed = 0;
static int tests_failed = 0;

static int float_eq_rel(float a, float b, float rel_eps, float abs_eps)
{
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return diff <= abs_eps || diff <= rel_eps * max_val;
}

static int int32_eq(int32_t a, int32_t b)
{
    return a == b;
}

static int uint32_eq(uint32_t a, uint32_t b)
{
    return a == b;
}

static void test_assert_eq_f32(const char *name, float got, float expected)
{
    if (float_eq_rel(got, expected, REL_EPS, ABS_EPS)) {
        printf("  PASS: %s (got %.6f, exp %.6f)\n", name, got, expected);
        tests_passed++;
    } else {
        printf("  FAIL: %s (got %.6f, exp %.6f, diff %.2e)\n",
               name, got, expected, fabsf(got - expected));
        tests_failed++;
    }
}

static void test_assert_eq_i32(const char *name, int32_t got, int32_t expected)
{
    if (int32_eq(got, expected)) {
        printf("  PASS: %s (got %d, exp %d)\n", name, got, expected);
        tests_passed++;
    } else {
        printf("  FAIL: %s (got %d, exp %d)\n", name, got, expected);
        tests_failed++;
    }
}

static void test_assert_eq_u32(const char *name, uint32_t got, uint32_t expected)
{
    if (uint32_eq(got, expected)) {
        printf("  PASS: %s (got %u, exp %u)\n", name, got, expected);
        tests_passed++;
    } else {
        printf("  FAIL: %s (got %u, exp %u)\n", name, got, expected);
        tests_failed++;
    }
}

static void run_basic_tests_f32(const char *backend,
                                float (*dot_fn)(const float*, const float*, size_t),
                                float (*l2sq_fn)(const float*, const float*, size_t),
                                float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Basic f32 correctness ===\n", backend);

    /* Test 1: Zero vectors */
    float a1[] = {0, 0, 0};
    float b1[] = {0, 0, 0};
    test_assert_eq_f32("dot zero", dot_fn(a1, b1, 3), 0.0f);
    test_assert_eq_f32("l2sq zero", l2sq_fn(a1, b1, 3), 0.0f);
    test_assert_eq_f32("cosine zero", cos_fn(a1, b1, 3), 0.0f);

    /* Test 2: Identical vectors */
    float a2[] = {1.0f, 2.0f, 3.0f};
    float b2[] = {1.0f, 2.0f, 3.0f};
    test_assert_eq_f32("dot identical", dot_fn(a2, b2, 3), 14.0f);
    test_assert_eq_f32("l2sq identical", l2sq_fn(a2, b2, 3), 0.0f);
    test_assert_eq_f32("cosine identical", cos_fn(a2, b2, 3), 1.0f);

    /* Test 3: Orthogonal vectors */
    float a3[] = {1.0f, 0.0f};
    float b3[] = {0.0f, 1.0f};
    test_assert_eq_f32("dot orthogonal", dot_fn(a3, b3, 2), 0.0f);
    test_assert_eq_f32("cosine orthogonal", cos_fn(a3, b3, 2), 0.0f);

    /* Test 4: Opposite vectors */
    float a4[] = {1.0f, 2.0f};
    float b4[] = {-1.0f, -2.0f};
    test_assert_eq_f32("dot opposite", dot_fn(a4, b4, 2), -5.0f);
    test_assert_eq_f32("cosine opposite", cos_fn(a4, b4, 2), -1.0f);

    /* Test 5: Linear scaling property */
    float a5[] = {2.0f, 3.0f, 4.0f};
    float b5[] = {5.0f, 6.0f, 7.0f};
    test_assert_eq_f32("dot linear scaling", dot_fn(a5, b5, 3), 56.0f);
}

static void run_basic_tests_i8(const char *backend,
                               int32_t (*dot_fn)(const int8_t*, const int8_t*, size_t),
                               int32_t (*l2sq_fn)(const int8_t*, const int8_t*, size_t),
                               float (*cos_fn)(const int8_t*, const int8_t*, size_t))
{
    printf("\n=== %s: Basic int8 correctness ===\n", backend);

    int8_t a1[] = {1, 2, 3};
    int8_t b1[] = {1, 2, 3};
    test_assert_eq_i32("dot i8 identical", dot_fn(a1, b1, 3), 14);
    test_assert_eq_i32("l2sq i8 identical", l2sq_fn(a1, b1, 3), 0);
    test_assert_eq_f32("cosine i8 identical", cos_fn(a1, b1, 3), 1.0f);

    int8_t a2[] = {-1, 2, -3};
    int8_t b2[] = {1, -2, 3};
    test_assert_eq_i32("dot i8 opposite", dot_fn(a2, b2, 3), -14);
    test_assert_eq_f32("cosine i8 opposite", cos_fn(a2, b2, 3), -1.0f);

    int8_t a3[] = {1, 0};
    int8_t b3[] = {0, 1};
    test_assert_eq_i32("dot i8 orthogonal", dot_fn(a3, b3, 2), 0);
    test_assert_eq_f32("cosine i8 orthogonal", cos_fn(a3, b3, 2), 0.0f);
}

static void run_basic_tests_u8(const char *backend,
                               uint32_t (*dot_fn)(const uint8_t*, const uint8_t*, size_t),
                               uint32_t (*l2sq_fn)(const uint8_t*, const uint8_t*, size_t),
                               float (*cos_fn)(const uint8_t*, const uint8_t*, size_t))
{
    printf("\n=== %s: Basic uint8 correctness ===\n", backend);

    uint8_t a1[] = {1, 2, 3};
    uint8_t b1[] = {1, 2, 3};
    test_assert_eq_u32("dot u8 identical", dot_fn(a1, b1, 3), 14);
    test_assert_eq_u32("l2sq u8 identical", l2sq_fn(a1, b1, 3), 0);
    test_assert_eq_f32("cosine u8 identical", cos_fn(a1, b1, 3), 1.0f);

    uint8_t a2[] = {255, 1, 100};
    uint8_t b2[] = {1, 255, 100};
    test_assert_eq_u32("dot u8 mixed", dot_fn(a2, b2, 3), 255*1 + 1*255 + 100*100);
    test_assert_eq_u32("l2sq u8 mixed", l2sq_fn(a2, b2, 3), (254*254) + (254*254) + 0);

    uint8_t a3[] = {1, 0};
    uint8_t b3[] = {0, 1};
    test_assert_eq_u32("dot u8 orthogonal", dot_fn(a3, b3, 2), 0);
    test_assert_eq_f32("cosine u8 orthogonal", cos_fn(a3, b3, 2), 0.0f);
}

static void run_random_tests_f32(const char *backend,
                                 float (*l2sq_fn)(const float*, const float*, size_t),
                                 float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Random f32 vectors ===\n", backend);
    srand(0xC0FFEE);

    for (size_t n = 1; n <= 100; n++) {
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));

        for (size_t i = 0; i < n; i++) {
            a[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
            b[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
        }

        float cos = cos_fn(a, b, n);
        if (cos < -1.00001f || cos > 1.00001f) {
            printf("  FAIL: cosine out of range for n=%zu: %f\n", n, cos);
            tests_failed++;
        } else {
            tests_passed++;
        }

        float l2sq = l2sq_fn(a, b, n);
        if (l2sq < -1e-6f) {
            printf("  FAIL: l2sq negative for n=%zu: %f\n", n, l2sq);
            tests_failed++;
        } else {
            tests_passed++;
        }

        free(a);
        free(b);
    }

    /* Non-multiple-of-SIMD-width sizes */
    for (size_t n = 1; n < 64; n++) {
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));

        for (size_t i = 0; i < n; i++) {
            a[i] = (float)i * 0.1f;
            b[i] = (float)(n - i) * 0.1f;
        }

        float cos = cos_fn(a, b, n);
        if (cos < -1.00001f || cos > 1.00001f) {
            printf("  FAIL: tail cosine out of range n=%zu: %f\n", n, cos);
            tests_failed++;
        } else {
            tests_passed++;
        }

        free(a);
        free(b);
    }

    printf("%s f32 random tests: %d passed, %d failed\n", backend, tests_passed, tests_failed);
}

static void run_edge_cases_f32(const char *backend,
                               float (*dot_fn)(const float*, const float*, size_t),
                               float (*l2sq_fn)(const float*, const float*, size_t),
                               float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: f32 Edge cases ===\n", backend);

    /* n=1 */
    float a1[] = {5.0f};
    float b1[] = {3.0f};
    test_assert_eq_f32("n=1 dot", dot_fn(a1, b1, 1), 15.0f);
    test_assert_eq_f32("n=1 l2sq", l2sq_fn(a1, b1, 1), 4.0f);
    test_assert_eq_f32("n=1 cosine", cos_fn(a1, b1, 1), 1.0f);

    /* n=0 */
    test_assert_eq_f32("n=0 dot", dot_fn(a1, b1, 0), 0.0f);
    test_assert_eq_f32("n=0 l2sq", l2sq_fn(a1, b1, 0), 0.0f);
    test_assert_eq_f32("n=0 cosine", cos_fn(a1, b1, 0), 0.0f);

    /* Zero vs non-zero */
    float a0[] = {0, 0, 0};
    float b0[] = {1, 2, 3};
    test_assert_eq_f32("zero/non-zero cosine", cos_fn(a0, b0, 3), 0.0f);

    /* NaN propagation */
    float anan[] = {NAN, 1.0f};
    float bnan[] = {1.0f, 2.0f};
    float dot_nan = dot_fn(anan, bnan, 2);
    if (!isnan(dot_nan)) {
        printf("  FAIL: dot with NaN input did not produce NaN\n");
        tests_failed++;
    } else {
        tests_passed++;
    }

    /* INF propagation */
    float ainf[] = {INFINITY, 1.0f};
    float binf[] = {1.0f, 2.0f};
    float dot_inf = dot_fn(ainf, binf, 2);
    if (!isinf(dot_inf)) {
        printf("  FAIL: dot with INF input did not produce INF\n");
        tests_failed++;
    } else {
        tests_passed++;
    }

    printf("%s f32 edge cases: %d passed, %d failed\n", backend, tests_passed, tests_failed);
}

/* Binary (1-bit) tests */
static void run_basic_tests_b1(const char *backend,
                               int32_t (*dot_fn)(const uint64_t*, const uint64_t*, size_t),
                               int32_t (*ham_fn)(const uint64_t*, const uint64_t*, size_t),
                               float (*cos_fn)(const uint64_t*, const uint64_t*, size_t))
{
    printf("\n=== %s: Basic binary (1-bit) correctness ===\n", backend);

    uint64_t a1[] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    uint64_t b1[] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    test_assert_eq_i32("dot b1 identical", dot_fn(a1, b1, 128), 128);
    test_assert_eq_i32("ham b1 identical", ham_fn(a1, b1, 128), 0);
    test_assert_eq_f32("cosine b1 identical", cos_fn(a1, b1, 128), 1.0f);

    uint64_t a2[] = {0xFFFFFFFFFFFFFFFFULL};
    uint64_t b2[] = {0x0000000000000000ULL};
    test_assert_eq_i32("dot b1 opposite", dot_fn(a2, b2, 64), 0);
    test_assert_eq_i32("ham b1 opposite", ham_fn(a2, b2, 64), 64);
    test_assert_eq_f32("cosine b1 opposite", cos_fn(a2, b2, 64), 0.0f);  // zero norm for all-zeros

    uint64_t a3[] = {0xAAAAAAAAAAAAAAAAULL};
    uint64_t b3[] = {0x5555555555555555ULL};
    test_assert_eq_i32("dot b1 orthogonal", dot_fn(a3, b3, 64), 0);
    test_assert_eq_i32("ham b1 orthogonal", ham_fn(a3, b3, 64), 64);
    test_assert_eq_f32("cosine b1 orthogonal", cos_fn(a3, b3, 64), 0.0f);
}

static void test_deterministic(const char *backend,
                               float (*dot_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Determinism ===\n", backend);

    float a[100], b[100];
    for (int i = 0; i < 100; i++) {
        a[i] = (float)i * 0.01f;
        b[i] = (float)(100 - i) * 0.01f;
    }

    float dot1 = dot_fn(a, b, 100);
    float dot2 = dot_fn(a, b, 100);
    float dot3 = dot_fn(a, b, 100);

    if (dot1 == dot2 && dot2 == dot3) {
        printf("  PASS: Deterministic results\n");
        tests_passed++;
    } else {
        printf("  FAIL: Non-deterministic results: %f %f %f\n", dot1, dot2, dot3);
        tests_failed++;
    }
}

static void test_symmetry_f32(const char *backend,
                              float (*dot_fn)(const float*, const float*, size_t),
                              float (*l2sq_fn)(const float*, const float*, size_t),
                              float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: f32 Symmetry ===\n", backend);

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 3.0f, 1.0f, 5.0f, 4.0f};

    test_assert_eq_f32("dot symmetry", dot_fn(a, b, 5), dot_fn(b, a, 5));
    test_assert_eq_f32("l2sq symmetry", l2sq_fn(a, b, 5), l2sq_fn(b, a, 5));
    test_assert_eq_f32("cosine symmetry", cos_fn(a, b, 5), cos_fn(b, a, 5));
}

static void test_cosine_properties_f32(const char *backend,
                                       float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: f32 Cosine properties ===\n", backend);

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    test_assert_eq_f32("cos(a,a)==1", cos_fn(a, a, 4), 1.0f);

    float neg_a[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    test_assert_eq_f32("cos(a,-a)==-1", cos_fn(a, neg_a, 4), -1.0f);

    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    test_assert_eq_f32("cos symmetry", cos_fn(a, b, 4), cos_fn(b, a, 4));

    float ca[] = {2.0f, 4.0f, 6.0f, 8.0f};
    test_assert_eq_f32("cos scale invar", cos_fn(a, b, 4), cos_fn(ca, b, 4));
}

/* Generic test runner for a backend */
#define RUN_BACKEND_TESTS(name, dot_f32, l2sq_f32, cos_f32, \
                           dot_i8, l2sq_i8, cos_i8, \
                           dot_u8, l2sq_u8, cos_u8, \
                           dot_b1, ham_b1, cos_b1) \
    do { \
        run_basic_tests_f32(#name, dot_f32, l2sq_f32, cos_f32); \
        run_random_tests_f32(#name, l2sq_f32, cos_f32); \
        run_edge_cases_f32(#name, dot_f32, l2sq_f32, cos_f32); \
        test_deterministic(#name, dot_f32); \
        test_symmetry_f32(#name, dot_f32, l2sq_f32, cos_f32); \
        test_cosine_properties_f32(#name, cos_f32); \
        run_basic_tests_i8(#name, dot_i8, l2sq_i8, cos_i8); \
        run_basic_tests_u8(#name, dot_u8, l2sq_u8, cos_u8); \
        run_basic_tests_b1(#name, dot_b1, ham_b1, cos_b1); \
    } while (0)

int main(void)
{
    /* Test scalar (always available) */
    RUN_BACKEND_TESTS(scalar,
        vek_dot_f32_scalar, vek_l2sq_f32_scalar, vek_cosine_f32_scalar,
        vek_dot_i8_scalar, vek_l2sq_i8_scalar, vek_cosine_i8_scalar,
        vek_dot_u8_scalar, vek_l2sq_u8_scalar, vek_cosine_u8_scalar,
        vek_dot_b1_scalar, vek_hamming_b1_scalar, vek_cosine_b1_scalar);

#if defined(__x86_64__) || defined(_M_X64)
    /* Test SSE2 */
    RUN_BACKEND_TESTS(sse2,
        vek_dot_f32_sse2, vek_l2sq_f32_sse2, vek_cosine_f32_sse2,
        vek_dot_i8_sse2, vek_l2sq_i8_sse2, vek_cosine_i8_sse2,
        vek_dot_u8_sse2, vek_l2sq_u8_sse2, vek_cosine_u8_sse2,
        vek_dot_b1_sse2, vek_hamming_b1_sse2, vek_cosine_b1_sse2);

    /* Test AVX2 */
    RUN_BACKEND_TESTS(avx2,
        vek_dot_f32_avx2, vek_l2sq_f32_avx2, vek_cosine_f32_avx2,
        vek_dot_i8_avx2, vek_l2sq_i8_avx2, vek_cosine_i8_avx2,
        vek_dot_u8_avx2, vek_l2sq_u8_avx2, vek_cosine_u8_avx2,
        vek_dot_b1_avx2, vek_hamming_b1_avx2, vek_cosine_b1_avx2);

    /* Test AVX-512 - only if compiled in AND CPU supports it at runtime */
#ifdef VEK_HAVE_AVX512
    if (cpu_has_avx512_runtime()) {
        RUN_BACKEND_TESTS(avx512,
            vek_dot_f32_avx512, vek_l2sq_f32_avx512, vek_cosine_f32_avx512,
            vek_dot_i8_avx512, vek_l2sq_i8_avx512, vek_cosine_i8_avx512,
            vek_dot_u8_avx512, vek_l2sq_u8_avx512, vek_cosine_u8_avx512,
            vek_dot_b1_avx512, vek_hamming_b1_avx512, vek_cosine_b1_avx512);
    } else {
        printf("\n=== SKIPPED: avx512 (CPU lacks AVX-512F/VNNI/VPOPCNTDQ or OS support) ===\n");
    }
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    /* Test NEON */
    RUN_BACKEND_TESTS(neon,
        vek_dot_f32_neon, vek_l2sq_f32_neon, vek_cosine_f32_neon,
        vek_dot_i8_neon, vek_l2sq_i8_neon, vek_cosine_i8_neon,
        vek_dot_u8_neon, vek_l2sq_u8_neon, vek_cosine_u8_neon,
        vek_dot_b1_neon, vek_hamming_b1_neon, vek_cosine_b1_neon);
#endif

    printf("\n=== OVERALL SUMMARY ===\n");
    printf("Total: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
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

#if defined(__x86_64__) || defined(_M_X64)
extern float vek_dot_f32_sse2(const float*, const float*, size_t);
extern float vek_l2sq_f32_sse2(const float*, const float*, size_t);
extern float vek_cosine_f32_sse2(const float*, const float*, size_t);

extern float vek_dot_f32_avx2(const float*, const float*, size_t);
extern float vek_l2sq_f32_avx2(const float*, const float*, size_t);
extern float vek_cosine_f32_avx2(const float*, const float*, size_t);

extern float vek_dot_f32_avx512(const float*, const float*, size_t);
extern float vek_l2sq_f32_avx512(const float*, const float*, size_t);
extern float vek_cosine_f32_avx512(const float*, const float*, size_t);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
extern float vek_dot_f32_neon(const float*, const float*, size_t);
extern float vek_l2sq_f32_neon(const float*, const float*, size_t);
extern float vek_cosine_f32_neon(const float*, const float*, size_t);
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

static void test_assert_eq(const char *name, float got, float expected)
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

static void run_basic_tests(const char *backend,
                            float (*dot_fn)(const float*, const float*, size_t),
                            float (*l2sq_fn)(const float*, const float*, size_t),
                            float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Basic correctness ===\n", backend);

    /* Test 1: Zero vectors */
    float a1[] = {0, 0, 0};
    float b1[] = {0, 0, 0};
    test_assert_eq("dot zero", dot_fn(a1, b1, 3), 0.0f);
    test_assert_eq("l2sq zero", l2sq_fn(a1, b1, 3), 0.0f);
    test_assert_eq("cosine zero", cos_fn(a1, b1, 3), 0.0f);

    /* Test 2: Identical vectors */
    float a2[] = {1.0f, 2.0f, 3.0f};
    float b2[] = {1.0f, 2.0f, 3.0f};
    test_assert_eq("dot identical", dot_fn(a2, b2, 3), 14.0f);
    test_assert_eq("l2sq identical", l2sq_fn(a2, b2, 3), 0.0f);
    test_assert_eq("cosine identical", cos_fn(a2, b2, 3), 1.0f);

    /* Test 3: Orthogonal vectors */
    float a3[] = {1.0f, 0.0f};
    float b3[] = {0.0f, 1.0f};
    test_assert_eq("dot orthogonal", dot_fn(a3, b3, 2), 0.0f);
    test_assert_eq("cosine orthogonal", cos_fn(a3, b3, 2), 0.0f);

    /* Test 4: Opposite vectors */
    float a4[] = {1.0f, 2.0f};
    float b4[] = {-1.0f, -2.0f};
    test_assert_eq("dot opposite", dot_fn(a4, b4, 2), -5.0f);
    test_assert_eq("cosine opposite", cos_fn(a4, b4, 2), -1.0f);

    /* Test 5: Linear scaling property */
    float a5[] = {2.0f, 3.0f, 4.0f};
    float b5[] = {5.0f, 6.0f, 7.0f};
    test_assert_eq("dot linear scaling", dot_fn(a5, b5, 3), 56.0f);
}

static void run_random_tests(const char *backend,
                             float (*l2sq_fn)(const float*, const float*, size_t),
                             float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Random vectors ===\n", backend);
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

    printf("%s random tests: %d passed, %d failed\n", backend, tests_passed, tests_failed);
}

static void run_edge_cases(const char *backend,
                           float (*dot_fn)(const float*, const float*, size_t),
                           float (*l2sq_fn)(const float*, const float*, size_t),
                           float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Edge cases ===\n", backend);

    /* n=1 */
    float a1[] = {5.0f};
    float b1[] = {3.0f};
    test_assert_eq("n=1 dot", dot_fn(a1, b1, 1), 15.0f);
    test_assert_eq("n=1 l2sq", l2sq_fn(a1, b1, 1), 4.0f);
    test_assert_eq("n=1 cosine", cos_fn(a1, b1, 1), 1.0f);

    /* n=0 */
    test_assert_eq("n=0 dot", dot_fn(a1, b1, 0), 0.0f);
    test_assert_eq("n=0 l2sq", l2sq_fn(a1, b1, 0), 0.0f);
    test_assert_eq("n=0 cosine", cos_fn(a1, b1, 0), 0.0f);

    /* Zero vs non-zero */
    float a0[] = {0, 0, 0};
    float b0[] = {1, 2, 3};
    test_assert_eq("zero/non-zero cosine", cos_fn(a0, b0, 3), 0.0f);

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

    printf("%s edge cases: %d passed, %d failed\n", backend, tests_passed, tests_failed);
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

static void test_symmetry(const char *backend,
                          float (*dot_fn)(const float*, const float*, size_t),
                          float (*l2sq_fn)(const float*, const float*, size_t),
                          float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Symmetry ===\n", backend);

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 3.0f, 1.0f, 5.0f, 4.0f};

    test_assert_eq("dot symmetry", dot_fn(a, b, 5), dot_fn(b, a, 5));
    test_assert_eq("l2sq symmetry", l2sq_fn(a, b, 5), l2sq_fn(b, a, 5));
    test_assert_eq("cosine symmetry", cos_fn(a, b, 5), cos_fn(b, a, 5));
}

static void test_cosine_properties(const char *backend,
                                   float (*cos_fn)(const float*, const float*, size_t))
{
    printf("\n=== %s: Cosine properties ===\n", backend);

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    test_assert_eq("cos(a,a)==1", cos_fn(a, a, 4), 1.0f);

    float neg_a[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    test_assert_eq("cos(a,-a)==-1", cos_fn(a, neg_a, 4), -1.0f);

    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    test_assert_eq("cos symmetry", cos_fn(a, b, 4), cos_fn(b, a, 4));

    float ca[] = {2.0f, 4.0f, 6.0f, 8.0f};
    test_assert_eq("cos scale invar", cos_fn(a, b, 4), cos_fn(ca, b, 4));
}

int main(void)
{
    /* Test scalar (always available) */
    run_basic_tests("scalar", vek_dot_f32_scalar, vek_l2sq_f32_scalar, vek_cosine_f32_scalar);
    run_random_tests("scalar", vek_l2sq_f32_scalar, vek_cosine_f32_scalar);
    run_edge_cases("scalar", vek_dot_f32_scalar, vek_l2sq_f32_scalar, vek_cosine_f32_scalar);
    test_deterministic("scalar", vek_dot_f32_scalar);
    test_symmetry("scalar", vek_dot_f32_scalar, vek_l2sq_f32_scalar, vek_cosine_f32_scalar);
    test_cosine_properties("scalar", vek_cosine_f32_scalar);

#if defined(__x86_64__) || defined(_M_X64)
    /* Test SSE2 */
    run_basic_tests("sse2", vek_dot_f32_sse2, vek_l2sq_f32_sse2, vek_cosine_f32_sse2);
    run_random_tests("sse2", vek_l2sq_f32_sse2, vek_cosine_f32_sse2);
    run_edge_cases("sse2", vek_dot_f32_sse2, vek_l2sq_f32_sse2, vek_cosine_f32_sse2);
    test_deterministic("sse2", vek_dot_f32_sse2);
    test_symmetry("sse2", vek_dot_f32_sse2, vek_l2sq_f32_sse2, vek_cosine_f32_sse2);
    test_cosine_properties("sse2", vek_cosine_f32_sse2);

    /* Test AVX2 */
    run_basic_tests("avx2", vek_dot_f32_avx2, vek_l2sq_f32_avx2, vek_cosine_f32_avx2);
    run_random_tests("avx2", vek_l2sq_f32_avx2, vek_cosine_f32_avx2);
    run_edge_cases("avx2", vek_dot_f32_avx2, vek_l2sq_f32_avx2, vek_cosine_f32_avx2);
    test_deterministic("avx2", vek_dot_f32_avx2);
    test_symmetry("avx2", vek_dot_f32_avx2, vek_l2sq_f32_avx2, vek_cosine_f32_avx2);
    test_cosine_properties("avx2", vek_cosine_f32_avx2);

    /* Test AVX-512 */
    run_basic_tests("avx512", vek_dot_f32_avx512, vek_l2sq_f32_avx512, vek_cosine_f32_avx512);
    run_random_tests("avx512", vek_l2sq_f32_avx512, vek_cosine_f32_avx512);
    run_edge_cases("avx512", vek_dot_f32_avx512, vek_l2sq_f32_avx512, vek_cosine_f32_avx512);
    test_deterministic("avx512", vek_dot_f32_avx512);
    test_symmetry("avx512", vek_dot_f32_avx512, vek_l2sq_f32_avx512, vek_cosine_f32_avx512);
    test_cosine_properties("avx512", vek_cosine_f32_avx512);
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    /* Test NEON */
    run_basic_tests("neon", vek_dot_f32_neon, vek_l2sq_f32_neon, vek_cosine_f32_neon);
    run_random_tests("neon", vek_dot_f32_neon, vek_l2sq_f32_neon, vek_cosine_f32_neon);
    run_edge_cases("neon", vek_dot_f32_neon, vek_l2sq_f32_neon, vek_cosine_f32_neon);
    test_deterministic("neon", vek_dot_f32_neon);
    test_symmetry("neon", vek_dot_f32_neon, vek_l2sq_f32_neon, vek_cosine_f32_neon);
    test_cosine_properties("neon", vek_cosine_f32_neon);
#endif

    printf("\n=== OVERALL SUMMARY ===\n");
    printf("Total: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
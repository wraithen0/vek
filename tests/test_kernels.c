/**
 * vek - Test harness for vector kernels
 * Correctness testing with relative epsilon comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vek.h"

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

static void run_basic_tests(void)
{
    printf("\n=== Basic correctness tests ===\n");

    /* Test 1: Zero vectors */
    float a1[] = {0, 0, 0};
    float b1[] = {0, 0, 0};
    test_assert_eq("dot zero", vek_dot_f32(a1, b1, 3), 0.0f);
    test_assert_eq("l2sq zero", vek_l2sq_f32(a1, b1, 3), 0.0f);
    test_assert_eq("cosine zero", vek_cosine_f32(a1, b1, 3), 0.0f);

    /* Test 2: Identical vectors */
    float a2[] = {1.0f, 2.0f, 3.0f};
    float b2[] = {1.0f, 2.0f, 3.0f};
    test_assert_eq("dot identical", vek_dot_f32(a2, b2, 3), 14.0f);
    test_assert_eq("l2sq identical", vek_l2sq_f32(a2, b2, 3), 0.0f);
    test_assert_eq("cosine identical", vek_cosine_f32(a2, b2, 3), 1.0f);

    /* Test 3: Orthogonal vectors */
    float a3[] = {1.0f, 0.0f};
    float b3[] = {0.0f, 1.0f};
    test_assert_eq("dot orthogonal", vek_dot_f32(a3, b3, 2), 0.0f);
    test_assert_eq("cosine orthogonal", vek_cosine_f32(a3, b3, 2), 0.0f);

    /* Test 4: Opposite vectors */
    float a4[] = {1.0f, 2.0f};
    float b4[] = {-1.0f, -2.0f};
    test_assert_eq("dot opposite", vek_dot_f32(a4, b4, 2), -5.0f);
    test_assert_eq("cosine opposite", vek_cosine_f32(a4, b4, 2), -1.0f);

    /* Test 5: Scalar multiplication property */
    float a5[] = {2.0f, 3.0f, 4.0f};
    float b5[] = {5.0f, 6.0f, 7.0f};
    test_assert_eq("dot linear scaling", vek_dot_f32(a5, b5, 3), 56.0f);
}

static void run_random_tests(void)
{
    printf("\n=== Random vector tests ===\n");
    srand(0xC0FFEE); /* Deterministic seed */

    for (size_t n = 1; n <= 100; n++) {
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));

        for (size_t i = 0; i < n; i++) {
            a[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f; /* [-10, 10] */
            b[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
        }
        float l2sq = vek_l2sq_f32(a, b, n);
                float cos = vek_cosine_f32(a, b, n);

        /* Verify cosine is in [-1, 1] */
        if (cos < -1.00001f || cos > 1.00001f) {
            printf("  FAIL: cosine out of range for n=%zu: %f\n", n, cos);
            tests_failed++;
        } else {
            tests_passed++;
        }

        /* Verify L2 squared is non-negative */
        if (l2sq < -1e-6f) {
            printf("  FAIL: l2sq negative for n=%zu: %f\n", n, l2sq);
            tests_failed++;
        } else {
            tests_passed++;
        }

        free(a);
        free(b);
    }

    /* Test non-multiple-of-SIMD-width sizes */
    for (size_t n = 1; n < 64; n++) {
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));

        for (size_t i = 0; i < n; i++) {
            a[i] = (float)i * 0.1f;
            b[i] = (float)(n - i) * 0.1f;
        }

        float cos = vek_cosine_f32(a, b, n);

        if (cos < -1.00001f || cos > 1.00001f) {
            printf("  FAIL: tail cosine out of range n=%zu: %f\n", n, cos);
            tests_failed++;
        } else {
            tests_passed++;
        }

        free(a);
        free(b);
    }

    printf("Random tests: %d passed, %d failed\n", tests_passed, tests_failed);
}

static void run_edge_cases(void)
{
    printf("\n=== Edge case tests ===\n");

    /* n=1 */
    float a1[] = {5.0f};
    float b1[] = {3.0f};
    test_assert_eq("n=1 dot", vek_dot_f32(a1, b1, 1), 15.0f);
    test_assert_eq("n=1 l2sq", vek_l2sq_f32(a1, b1, 1), 4.0f);
    test_assert_eq("n=1 cosine", vek_cosine_f32(a1, b1, 1), 1.0f);

    /* n=0 (should return 0) */
    test_assert_eq("n=0 dot", vek_dot_f32(a1, b1, 0), 0.0f);
    test_assert_eq("n=0 l2sq", vek_l2sq_f32(a1, b1, 0), 0.0f);
    test_assert_eq("n=0 cosine", vek_cosine_f32(a1, b1, 0), 0.0f);

    /* One zero vector, one non-zero */
    float a0[] = {0, 0, 0};
    float b0[] = {1, 2, 3};
    test_assert_eq("zero/non-zero cosine", vek_cosine_f32(a0, b0, 3), 0.0f);

    /* NaN handling (should produce NaN if input has NaN) */
    float anan[] = {NAN, 1.0f};
    float bnan[] = {1.0f, 2.0f};
    float dot_nan = vek_dot_f32(anan, bnan, 2);
    if (!isnan(dot_nan)) {
        printf("  FAIL: dot with NaN input did not produce NaN\n");
        tests_failed++;
    } else {
        tests_passed++;
    }

    /* INF handling */
    float ainf[] = {INFINITY, 1.0f};
    float binf[] = {1.0f, 2.0f};
    float dot_inf = vek_dot_f32(ainf, binf, 2);
    if (!isinf(dot_inf)) {
        printf("  FAIL: dot with INF input did not produce INF\n");
        tests_failed++;
    } else {
        tests_passed++;
    }

    printf("Edge cases: %d passed, %d failed\n", tests_passed, tests_failed);
}

static void test_deterministic(void)
{
    printf("\n=== Determinism test ===\n");

    /* Same inputs should always produce same outputs */
    float a[100], b[100];
    for (int i = 0; i < 100; i++) {
        a[i] = (float)i * 0.01f;
        b[i] = (float)(100 - i) * 0.01f;
    }

    float dot1 = vek_dot_f32(a, b, 100);
    float dot2 = vek_dot_f32(a, b, 100);
    float dot3 = vek_dot_f32(a, b, 100);

    if (dot1 == dot2 && dot2 == dot3) {
        printf("  PASS: Deterministic results\n");
        tests_passed++;
    } else {
        printf("  FAIL: Non-deterministic results: %f %f %f\n", dot1, dot2, dot3);
        tests_failed++;
    }
}

static void test_symmetry(void)
{
    printf("\n=== Symmetry tests ===\n");

    float a[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float b[] = {2.0f, 3.0f, 1.0f, 5.0f, 4.0f};

    float dot_ab = vek_dot_f32(a, b, 5);
    float dot_ba = vek_dot_f32(b, a, 5);
    test_assert_eq("dot symmetry", dot_ab, dot_ba);

    float l2sq_ab = vek_l2sq_f32(a, b, 5);
    float l2sq_ba = vek_l2sq_f32(b, a, 5);
    test_assert_eq("l2sq symmetry", l2sq_ab, l2sq_ba);

    float cos_ab = vek_cosine_f32(a, b, 5);
    float cos_ba = vek_cosine_f32(b, a, 5);
    test_assert_eq("cosine symmetry", cos_ab, cos_ba);
}

static void test_cosine_properties(void)
{
    printf("\n=== Cosine similarity properties ===\n");

    /* cos(a, a) == 1 for non-zero a */
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    test_assert_eq("cos(a,a)==1", vek_cosine_f32(a, a, 4), 1.0f);

    /* cos(a, -a) == -1 for non-zero a */
    float neg_a[] = {-1.0f, -2.0f, -3.0f, -4.0f};
    test_assert_eq("cos(a,-a)==-1", vek_cosine_f32(a, neg_a, 4), -1.0f);

    /* cos(a, b) == cos(b, a) */
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    test_assert_eq("cos symmetry", vek_cosine_f32(a, b, 4), vek_cosine_f32(b, a, 4));

    /* cos(c*a, b) == cos(a, b) for c > 0 */
    float ca[] = {2.0f, 4.0f, 6.0f, 8.0f};
    test_assert_eq("cos scale invar", vek_cosine_f32(a, b, 4), vek_cosine_f32(ca, b, 4));
}

static void run_i8_tests(void)
{
    printf("\n=== int8 tests (dispatch) ===\n");

    int8_t a1[] = {1, 2, 3, 4, 5};
    int8_t b1[] = {1, 2, 3, 4, 5};
    int32_t dot = vek_dot_i8(a1, b1, 5);
    int32_t l2 = vek_l2sq_i8(a1, b1, 5);
    float cos = vek_cosine_i8(a1, b1, 5);

    if (dot == 55) { printf("  PASS: dot i8 identical (got %d)\n", dot); tests_passed++; }
    else { printf("  FAIL: dot i8 identical (got %d, exp 55)\n", dot); tests_failed++; }

    if (l2 == 0) { printf("  PASS: l2sq i8 identical (got %d)\n", l2); tests_passed++; }
    else { printf("  FAIL: l2sq i8 identical (got %d, exp 0)\n", l2); tests_failed++; }

    if (float_eq_rel(cos, 1.0f, REL_EPS, ABS_EPS)) { printf("  PASS: cosine i8 identical (got %f)\n", cos); tests_passed++; }
    else { printf("  FAIL: cosine i8 identical (got %f, exp 1.0)\n", cos); tests_failed++; }

    int8_t a2[] = {1, 0};
    int8_t b2[] = {0, 1};
    if (vek_dot_i8(a2, b2, 2) == 0) { printf("  PASS: dot i8 orthogonal\n"); tests_passed++; }
    else { printf("  FAIL: dot i8 orthogonal\n"); tests_failed++; }

    /* Negative values */
    int8_t a3[] = {-1, -2, -3};
    int8_t b3[] = {1, 2, 3};
    if (vek_dot_i8(a3, b3, 3) == -14) { printf("  PASS: dot i8 opposite\n"); tests_passed++; }
    else { printf("  FAIL: dot i8 opposite (got %d, exp -14)\n", vek_dot_i8(a3, b3, 3)); tests_failed++; }
}

static void run_u8_tests(void)
{
    printf("\n=== uint8 tests (dispatch) ===\n");

    uint8_t a1[] = {1, 2, 3, 4, 5};
    uint8_t b1[] = {1, 2, 3, 4, 5};
    uint32_t dot = vek_dot_u8(a1, b1, 5);
    uint32_t l2 = vek_l2sq_u8(a1, b1, 5);
    float cos = vek_cosine_u8(a1, b1, 5);

    if (dot == 55) { printf("  PASS: dot u8 identical (got %u)\n", dot); tests_passed++; }
    else { printf("  FAIL: dot u8 identical (got %u, exp 55)\n", dot); tests_failed++; }

    if (l2 == 0) { printf("  PASS: l2sq u8 identical (got %u)\n", l2); tests_passed++; }
    else { printf("  FAIL: l2sq u8 identical (got %u, exp 0)\n", l2); tests_failed++; }

    if (float_eq_rel(cos, 1.0f, REL_EPS, ABS_EPS)) { printf("  PASS: cosine u8 identical (got %f)\n", cos); tests_passed++; }
    else { printf("  FAIL: cosine u8 identical (got %f, exp 1.0)\n", cos); tests_failed++; }

    uint8_t a2[] = {1, 0};
    uint8_t b2[] = {0, 1};
    if (vek_dot_u8(a2, b2, 2) == 0) { printf("  PASS: dot u8 orthogonal\n"); tests_passed++; }
    else { printf("  FAIL: dot u8 orthogonal\n"); tests_failed++; }
}

static void run_b1_tests(void)
{
    printf("\n=== Binary (1-bit) tests (dispatch) ===\n");

    uint64_t a1[] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    uint64_t b1[] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    int32_t dot = vek_dot_b1(a1, b1, 128);
    int32_t ham = vek_hamming_b1(a1, b1, 128);
    float cos = vek_cosine_b1(a1, b1, 128);

    if (dot == 128) { printf("  PASS: dot b1 identical (got %d)\n", dot); tests_passed++; }
    else { printf("  FAIL: dot b1 identical (got %d, exp 128)\n", dot); tests_failed++; }

    if (ham == 0) { printf("  PASS: ham b1 identical (got %d)\n", ham); tests_passed++; }
    else { printf("  FAIL: ham b1 identical (got %d, exp 0)\n", ham); tests_failed++; }

    if (float_eq_rel(cos, 1.0f, REL_EPS, ABS_EPS)) { printf("  PASS: cosine b1 identical (got %f)\n", cos); tests_passed++; }
    else { printf("  FAIL: cosine b1 identical (got %f, exp 1.0)\n", cos); tests_failed++; }

    uint64_t a2[] = {0xAAAAAAAAAAAAAAAAULL};
    uint64_t b2[] = {0x5555555555555555ULL};
    if (vek_dot_b1(a2, b2, 64) == 0) { printf("  PASS: dot b1 orthogonal\n"); tests_passed++; }
    else { printf("  FAIL: dot b1 orthogonal\n"); tests_failed++; }

    if (vek_hamming_b1(a2, b2, 64) == 64) { printf("  PASS: ham b1 orthogonal\n"); tests_passed++; }
    else { printf("  FAIL: ham b1 orthogonal\n"); tests_failed++; }
}

static void print_backend(void)
{
    printf("\n=== Active backend: %s ===\n", vek_backend_name());
}

int main(void)
{
    print_backend();

    if (vek_init() != 0) {
        printf("Failed to initialize vek\n");
        return 1;
    }

    print_backend();

    run_basic_tests();
    run_random_tests();
    run_edge_cases();
    test_deterministic();
    test_symmetry();
    test_cosine_properties();
    run_i8_tests();
    run_u8_tests();
    run_b1_tests();

    printf("\n=== SUMMARY ===\n");
    printf("Total: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
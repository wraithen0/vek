/**
 * vek - SSE2 intrinsics implementation for x86_64
 * SSE2 (128-bit) vectorized kernels for f32 dot, L2, cosine
 * Baseline for all x86_64 CPUs
 */

#include <emmintrin.h>
#include <stddef.h>
#include <math.h>
#include "vek.h"

/* SSE2 dot product: sum(a[i] * b[i]) */
float vek_dot_f32_sse2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4; /* 128-bit = 4 floats */
    size_t i = 0;

    __m128 sum_vec = _mm_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m128 a_vec = _mm_loadu_ps(a + i);
        __m128 b_vec = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(a_vec, b_vec);
        sum_vec = _mm_add_ps(sum_vec, prod);
    }

    /* Horizontal sum - SSE2 compatible version */
    /* sum_vec = [s0, s1, s2, s3] */
    __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1)); /* [s2, s3, s0, s1] */
    __m128 sum = _mm_add_ps(sum_vec, shuf);                                    /* [s0+s2, s1+s3, s2+s0, s3+s1] */
    shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));                  /* [s1+s3, s0+s2, s3+s1, s2+s0] */
    sum = _mm_add_ss(sum, shuf);                                               /* [s0+s1+s2+s3, ...] */
    float sum_scalar = _mm_cvtss_f32(sum);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += a[i] * b[i];
    }

    return sum_scalar;
}

/* SSE2 squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32_sse2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4;
    size_t i = 0;

    __m128 sum_vec = _mm_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m128 a_vec = _mm_loadu_ps(a + i);
        __m128 b_vec = _mm_loadu_ps(b + i);
        __m128 diff = _mm_sub_ps(a_vec, b_vec);
        __m128 sq = _mm_mul_ps(diff, diff);
        sum_vec = _mm_add_ps(sum_vec, sq);
    }

    /* Horizontal sum */
    __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sum = _mm_add_ps(sum_vec, shuf);
    shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_ss(sum, shuf);
    float sum_scalar = _mm_cvtss_f32(sum);

    /* Tail */
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* SSE2 cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32_sse2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4;
    size_t i = 0;

    __m128 dot_vec = _mm_setzero_ps();
    __m128 norm_a_vec = _mm_setzero_ps();
    __m128 norm_b_vec = _mm_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m128 a_vec = _mm_loadu_ps(a + i);
        __m128 b_vec = _mm_loadu_ps(b + i);

        dot_vec = _mm_add_ps(dot_vec, _mm_mul_ps(a_vec, b_vec));
        norm_a_vec = _mm_add_ps(norm_a_vec, _mm_mul_ps(a_vec, a_vec));
        norm_b_vec = _mm_add_ps(norm_b_vec, _mm_mul_ps(b_vec, b_vec));
    }

    /* Horizontal sum for dot */
    __m128 dot_shuf = _mm_shuffle_ps(dot_vec, dot_vec, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 dot = _mm_add_ps(dot_vec, dot_shuf);
    dot_shuf = _mm_shuffle_ps(dot, dot, _MM_SHUFFLE(1, 0, 3, 2));
    dot = _mm_add_ss(dot, dot_shuf);
    float dot_scalar = _mm_cvtss_f32(dot);

    /* Horizontal sum for norm_a */
    __m128 na_shuf = _mm_shuffle_ps(norm_a_vec, norm_a_vec, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 na = _mm_add_ps(norm_a_vec, na_shuf);
    na_shuf = _mm_shuffle_ps(na, na, _MM_SHUFFLE(1, 0, 3, 2));
    na = _mm_add_ss(na, na_shuf);
    float norm_a_scalar = _mm_cvtss_f32(na);

    /* Horizontal sum for norm_b */
    __m128 nb_shuf = _mm_shuffle_ps(norm_b_vec, norm_b_vec, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 nb = _mm_add_ps(norm_b_vec, nb_shuf);
    nb_shuf = _mm_shuffle_ps(nb, nb, _MM_SHUFFLE(1, 0, 3, 2));
    nb = _mm_add_ss(nb, nb_shuf);
    float norm_b_scalar = _mm_cvtss_f32(nb);

    /* Tail */
    for (; i < n; i++) {
        float ai = a[i];
        float bi = b[i];
        dot_scalar += ai * bi;
        norm_a_scalar += ai * ai;
        norm_b_scalar += bi * bi;
    }

    float norm_a_sqrt = sqrtf(norm_a_scalar);
    float norm_b_sqrt = sqrtf(norm_b_scalar);

    if (norm_a_sqrt == 0.0f || norm_b_sqrt == 0.0f) {
        return 0.0f;
    }

    return dot_scalar / (norm_a_sqrt * norm_b_sqrt);
}
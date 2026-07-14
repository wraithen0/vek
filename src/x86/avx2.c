/**
 * vek - AVX2 intrinsics implementation for x86_64
 * AVX2 (256-bit) vectorized kernels for f32 dot, L2, cosine
 */

#include <immintrin.h>
#include <stddef.h>
#include <math.h>
#include "vek.h"

/* AVX2 dot product: sum(a[i] * b[i]) */
float vek_dot_f32_avx2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 8; /* 256-bit = 8 floats */
    size_t i = 0;

    __m256 sum_vec = _mm256_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(a_vec, b_vec);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }

    /* Horizontal sum of the 8 lanes */
    __m128 sum_hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum = _mm_add_ps(sum_hi, sum_lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float sum_scalar = _mm_cvtss_f32(sum);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += a[i] * b[i];
    }

    return sum_scalar;
}

/* AVX2 squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32_avx2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 8;
    size_t i = 0;

    __m256 sum_vec = _mm256_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(a_vec, b_vec);
        __m256 sq = _mm256_mul_ps(diff, diff);
        sum_vec = _mm256_add_ps(sum_vec, sq);
    }

    /* Horizontal sum */
    __m128 sum_hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum = _mm_add_ps(sum_hi, sum_lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float sum_scalar = _mm_cvtss_f32(sum);

    /* Tail */
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX2 cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32_avx2(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 8;
    size_t i = 0;

    __m256 dot_vec = _mm256_setzero_ps();
    __m256 norm_a_vec = _mm256_setzero_ps();
    __m256 norm_b_vec = _mm256_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);

        dot_vec = _mm256_fmadd_ps(a_vec, b_vec, dot_vec);
        norm_a_vec = _mm256_fmadd_ps(a_vec, a_vec, norm_a_vec);
        norm_b_vec = _mm256_fmadd_ps(b_vec, b_vec, norm_b_vec);
    }

    /* Horizontal sum for dot */
    __m128 dot_hi = _mm256_extractf128_ps(dot_vec, 1);
    __m128 dot_lo = _mm256_castps256_ps128(dot_vec);
    __m128 dot = _mm_add_ps(dot_hi, dot_lo);
    dot = _mm_hadd_ps(dot, dot);
    dot = _mm_hadd_ps(dot, dot);
    float dot_scalar = _mm_cvtss_f32(dot);

    /* Horizontal sum for norm_a */
    __m128 na_hi = _mm256_extractf128_ps(norm_a_vec, 1);
    __m128 na_lo = _mm256_castps256_ps128(norm_a_vec);
    __m128 na = _mm_add_ps(na_hi, na_lo);
    na = _mm_hadd_ps(na, na);
    na = _mm_hadd_ps(na, na);
    float norm_a_scalar = _mm_cvtss_f32(na);

    /* Horizontal sum for norm_b */
    __m128 nb_hi = _mm256_extractf128_ps(norm_b_vec, 1);
    __m128 nb_lo = _mm256_castps256_ps128(norm_b_vec);
    __m128 nb = _mm_add_ps(nb_hi, nb_lo);
    nb = _mm_hadd_ps(nb, nb);
    nb = _mm_hadd_ps(nb, nb);
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
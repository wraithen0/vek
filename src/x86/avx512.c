/**
 * vek - AVX-512 intrinsics implementation for x86_64
 * AVX-512F (512-bit) vectorized kernels for f32 dot, L2, cosine
 *
 * Note: Requires AVX-512F. Falls back to AVX2 if not available at runtime.
 */

#include <immintrin.h>
#include <stddef.h>
#include <math.h>
#include "vek.h"

/* AVX-512 dot product: sum(a[i] * b[i]) */
float vek_dot_f32_avx512(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 16; /* 512-bit = 16 floats */
    size_t i = 0;

    __m512 sum_vec = _mm512_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m512 a_vec = _mm512_loadu_ps(a + i);
        __m512 b_vec = _mm512_loadu_ps(b + i);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    /* Horizontal sum of 16 lanes */
    float sum_scalar = _mm512_reduce_add_ps(sum_vec);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += a[i] * b[i];
    }

    return sum_scalar;
}

/* AVX-512 squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32_avx512(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m512 sum_vec = _mm512_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m512 a_vec = _mm512_loadu_ps(a + i);
        __m512 b_vec = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(a_vec, b_vec);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }

    /* Horizontal sum */
    float sum_scalar = _mm512_reduce_add_ps(sum_vec);

    /* Tail */
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX-512 cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32_avx512(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m512 dot_vec = _mm512_setzero_ps();
    __m512 norm_a_vec = _mm512_setzero_ps();
    __m512 norm_b_vec = _mm512_setzero_ps();

    for (; i + simd_width <= n; i += simd_width) {
        __m512 a_vec = _mm512_loadu_ps(a + i);
        __m512 b_vec = _mm512_loadu_ps(b + i);

        dot_vec = _mm512_fmadd_ps(a_vec, b_vec, dot_vec);
        norm_a_vec = _mm512_fmadd_ps(a_vec, a_vec, norm_a_vec);
        norm_b_vec = _mm512_fmadd_ps(b_vec, b_vec, norm_b_vec);
    }

    /* Horizontal sums */
    float dot_scalar = _mm512_reduce_add_ps(dot_vec);
    float norm_a_scalar = _mm512_reduce_add_ps(norm_a_vec);
    float norm_b_scalar = _mm512_reduce_add_ps(norm_b_vec);

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
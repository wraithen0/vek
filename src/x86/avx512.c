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

/* ===== Quantized int8/uint8 variants (AVX-512 with VNNI) ===== */

/* AVX-512 VNNI int8 dot product using _mm512_dpbusd_epi32 */
int32_t vek_dot_i8_avx512(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 64; /* 512-bit = 64 int8 */
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* VNNI: Multiply pairs of signed 8-bit integers, sum pairs to 16-bit, then accumulate to 32-bit */
        sum_vec = _mm512_dpbusd_epi32(sum_vec, a_vec, b_vec);
    }

    /* Horizontal sum */
    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum_scalar;
}

/* AVX-512 VNNI uint8 dot product */
uint32_t vek_dot_u8_avx512(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 64;
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        sum_vec = _mm512_dpbusd_epi32(sum_vec, a_vec, b_vec);
    }

    uint32_t sum_scalar = (uint32_t)_mm512_reduce_add_epi32(sum_vec);

    for (; i < n; i++) {
        sum_scalar += (uint32_t)a[i] * (uint32_t)b[i];
    }

    return sum_scalar;
}

/* AVX-512 int8 squared L2 distance */
int32_t vek_l2sq_i8_avx512(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 64;
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* Convert to 16-bit for subtraction */
        __m512i a_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_vec, 0));
        __m512i a_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_vec, 1));
        __m512i b_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_vec, 0));
        __m512i b_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_vec, 1));

        __m512i diff_lo = _mm512_sub_epi16(a_lo, b_lo);
        __m512i diff_hi = _mm512_sub_epi16(a_hi, b_hi);

        __m512i sq_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        __m512i sq_hi = _mm512_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm512_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi);
    }

    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);

    for (; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX-512 uint8 squared L2 distance */
uint32_t vek_l2sq_u8_avx512(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 64;
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        __m512i a_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a_vec, 0));
        __m512i a_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a_vec, 1));
        __m512i b_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_vec, 0));
        __m512i b_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_vec, 1));

        __m512i diff_lo = _mm512_sub_epi16(a_lo, b_lo);
        __m512i diff_hi = _mm512_sub_epi16(a_hi, b_hi);

        __m512i sq_lo = _mm512_madd_epi16(diff_lo, diff_lo);
        __m512i sq_hi = _mm512_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm512_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi);
    }

    uint32_t sum_scalar = (uint32_t)_mm512_reduce_add_epi32(sum_vec);

    for (; i < n; i++) {
        uint32_t diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX-512 VNNI int8 cosine similarity */
float vek_cosine_i8_avx512(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 64;
    size_t i = 0;

    __m512i dot_vec = _mm512_setzero_si512();
    __m512i norm_a_vec = _mm512_setzero_si512();
    __m512i norm_b_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* dot product using VNNI */
        dot_vec = _mm512_dpbusd_epi32(dot_vec, a_vec, b_vec);

        /* norm a - convert to 16-bit, square with madd */
        __m512i a_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_vec, 0));
        __m512i a_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a_vec, 1));
        __m512i sq_a_lo = _mm512_madd_epi16(a_lo, a_lo);
        __m512i sq_a_hi = _mm512_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b */
        __m512i b_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_vec, 0));
        __m512i b_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(b_vec, 1));
        __m512i sq_b_lo = _mm512_madd_epi16(b_lo, b_lo);
        __m512i sq_b_hi = _mm512_madd_epi16(b_hi, b_hi);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi);
    }

    int32_t dot_scalar = _mm512_reduce_add_epi32(dot_vec);
    int32_t norm_a_scalar = _mm512_reduce_add_epi32(norm_a_vec);
    int32_t norm_b_scalar = _mm512_reduce_add_epi32(norm_b_vec);

    for (; i < n; i++) {
        int32_t ai = a[i];
        int32_t bi = b[i];
        dot_scalar += ai * bi;
        norm_a_scalar += ai * ai;
        norm_b_scalar += bi * bi;
    }

    float norm_a_sqrt = sqrtf((float)norm_a_scalar);
    float norm_b_sqrt = sqrtf((float)norm_b_scalar);

    if (norm_a_sqrt == 0.0f || norm_b_sqrt == 0.0f) {
        return 0.0f;
    }

    return (float)dot_scalar / (norm_a_sqrt * norm_b_sqrt);
}

/* AVX-512 VNNI uint8 cosine similarity */
float vek_cosine_u8_avx512(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 64;
    size_t i = 0;

    __m512i dot_vec = _mm512_setzero_si512();
    __m512i norm_a_vec = _mm512_setzero_si512();
    __m512i norm_b_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        dot_vec = _mm512_dpbusd_epi32(dot_vec, a_vec, b_vec);

        /* norm a */
        __m512i a_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a_vec, 0));
        __m512i a_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a_vec, 1));
        __m512i sq_a_lo = _mm512_madd_epi16(a_lo, a_lo);
        __m512i sq_a_hi = _mm512_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b */
        __m512i b_lo = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_vec, 0));
        __m512i b_hi = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_vec, 1));
        __m512i sq_b_lo = _mm512_madd_epi16(b_lo, b_lo);
        __m512i sq_b_hi = _mm512_madd_epi16(b_hi, b_hi);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi);
    }

    int32_t dot_scalar = _mm512_reduce_add_epi32(dot_vec);
    int32_t norm_a_scalar = _mm512_reduce_add_epi32(norm_a_vec);
    int32_t norm_b_scalar = _mm512_reduce_add_epi32(norm_b_vec);

    for (; i < n; i++) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        dot_scalar += ai * bi;
        norm_a_scalar += ai * ai;
        norm_b_scalar += bi * bi;
    }

    float norm_a_sqrt = sqrtf((float)norm_a_scalar);
    float norm_b_sqrt = sqrtf((float)norm_b_scalar);

    if (norm_a_sqrt == 0.0f || norm_b_sqrt == 0.0f) {
        return 0.0f;
    }

    return (float)dot_scalar / (norm_a_sqrt * norm_b_sqrt);
}
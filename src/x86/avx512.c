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

/* AVX-512 VNNI int8 dot product using _mm512_dpbusd_epi32 with bias correction */
/* VNNI: _mm512_dpbusd_epi32(dst, a_u8, b_s8) where a is unsigned, b is signed */
/* For int8: dot = sum(a[i]*b[i]) = sum((a[i]+128)*b[i]) - 128*sum(b[i]) */
int32_t vek_dot_i8_avx512(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 64; /* 512-bit = 64 int8 */
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();
    __m512i b_sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* Bias a by +128 to convert to unsigned range for VNNI */
        __m512i a_biased = _mm512_add_epi8(a_vec, _mm512_set1_epi8(128));
        sum_vec = _mm512_dpbusd_epi32(sum_vec, a_biased, b_vec);

        /* Accumulate sum(b) for correction term - use 128-bit extracts */
        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m512i b_lo = _mm512_cvtepi8_epi32(b_lo128);
        __m512i b_hi = _mm512_cvtepi8_epi32(b_hi128);
        __m512i b_lo2 = _mm512_cvtepi8_epi32(b_lo2_128);
        __m512i b_hi2 = _mm512_cvtepi8_epi32(b_hi2_128);

        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_lo);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_hi);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_lo2);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_hi2);
    }

    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);
    int32_t b_sum_scalar = _mm512_reduce_add_epi32(b_sum_vec);
    sum_scalar -= 128 * b_sum_scalar; /* Correction term */

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum_scalar;
}

/* AVX-512 VNNI uint8 dot product using widen-then-multiply for correctness */
uint32_t vek_dot_u8_avx512(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 64; /* 512-bit = 64 uint8 */
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* Zero-extend uint8 to uint16, then multiply uint16×uint16 → uint32, accumulate */
        __m128i a_lo128 = _mm512_castsi512_si128(a_vec);
        __m128i a_hi128 = _mm512_extracti32x4_epi32(a_vec, 1);
        __m128i a_lo2_128 = _mm512_extracti32x4_epi32(a_vec, 2);
        __m128i a_hi2_128 = _mm512_extracti32x4_epi32(a_vec, 3);

        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m256i a_lo256 = _mm256_cvtepu8_epi16(a_lo128);
        __m256i a_hi256 = _mm256_cvtepu8_epi16(a_hi128);
        __m256i a_lo2_256 = _mm256_cvtepu8_epi16(a_lo2_128);
        __m256i a_hi2_256 = _mm256_cvtepu8_epi16(a_hi2_128);

        __m256i b_lo256 = _mm256_cvtepu8_epi16(b_lo128);
        __m256i b_hi256 = _mm256_cvtepu8_epi16(b_hi128);
        __m256i b_lo2_256 = _mm256_cvtepu8_epi16(b_lo2_128);
        __m256i b_hi2_256 = _mm256_cvtepu8_epi16(b_hi2_128);

        __m512i a_lo = _mm512_cvtepu16_epi32(a_lo256);
        __m512i a_hi = _mm512_cvtepu16_epi32(a_hi256);
        __m512i a_lo2 = _mm512_cvtepu16_epi32(a_lo2_256);
        __m512i a_hi2 = _mm512_cvtepu16_epi32(a_hi2_256);

        __m512i b_lo = _mm512_cvtepu16_epi32(b_lo256);
        __m512i b_hi = _mm512_cvtepu16_epi32(b_hi256);
        __m512i b_lo2 = _mm512_cvtepu16_epi32(b_lo2_256);
        __m512i b_hi2 = _mm512_cvtepu16_epi32(b_hi2_256);

        __m512i sum_lo = _mm512_madd_epi16(a_lo, b_lo);
        __m512i sum_hi = _mm512_madd_epi16(a_hi, b_hi);
        __m512i sum_lo2 = _mm512_madd_epi16(a_lo2, b_lo2);
        __m512i sum_hi2 = _mm512_madd_epi16(a_hi2, b_hi2);

        sum_vec = _mm512_add_epi32(sum_vec, sum_lo);
        sum_vec = _mm512_add_epi32(sum_vec, sum_hi);
        sum_vec = _mm512_add_epi32(sum_vec, sum_lo2);
        sum_vec = _mm512_add_epi32(sum_vec, sum_hi2);
    }

    uint32_t sum_scalar = (uint32_t)_mm512_reduce_add_epi32(sum_vec);

    /* Tail */
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

        /* Convert to 16-bit for subtraction - use 128-bit extracts */
        __m128i a_lo128 = _mm512_castsi512_si128(a_vec);
        __m128i a_hi128 = _mm512_extracti32x4_epi32(a_vec, 1);
        __m128i a_lo2_128 = _mm512_extracti32x4_epi32(a_vec, 2);
        __m128i a_hi2_128 = _mm512_extracti32x4_epi32(a_vec, 3);

        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m256i a_lo256 = _mm256_cvtepi8_epi16(a_lo128);
        __m256i a_hi256 = _mm256_cvtepi8_epi16(a_hi128);
        __m256i a_lo2_256 = _mm256_cvtepi8_epi16(a_lo2_128);
        __m256i a_hi2_256 = _mm256_cvtepi8_epi16(a_hi2_128);

        __m256i b_lo256 = _mm256_cvtepi8_epi16(b_lo128);
        __m256i b_hi256 = _mm256_cvtepi8_epi16(b_hi128);
        __m256i b_lo2_256 = _mm256_cvtepi8_epi16(b_lo2_128);
        __m256i b_hi2_256 = _mm256_cvtepi8_epi16(b_hi2_128);

        __m512i a_lo = _mm512_cvtepi16_epi32(a_lo256);
        __m512i a_hi = _mm512_cvtepi16_epi32(a_hi256);
        __m512i a_lo2 = _mm512_cvtepi16_epi32(a_lo2_256);
        __m512i a_hi2 = _mm512_cvtepi16_epi32(a_hi2_256);

        __m512i b_lo = _mm512_cvtepi16_epi32(b_lo256);
        __m512i b_hi = _mm512_cvtepi16_epi32(b_hi256);
        __m512i b_lo2 = _mm512_cvtepi16_epi32(b_lo2_256);
        __m512i b_hi2 = _mm512_cvtepi16_epi32(b_hi2_256);

        __m512i diff_lo = _mm512_sub_epi32(a_lo, b_lo);
        __m512i diff_hi = _mm512_sub_epi32(a_hi, b_hi);
        __m512i diff_lo2 = _mm512_sub_epi32(a_lo2, b_lo2);
        __m512i diff_hi2 = _mm512_sub_epi32(a_hi2, b_hi2);

        /* Square using mullo + add (no madd_epi32 instruction) */
        __m512i sq_lo = _mm512_mullo_epi32(diff_lo, diff_lo);
        __m512i sq_hi = _mm512_mullo_epi32(diff_hi, diff_hi);
        __m512i sq_lo2 = _mm512_mullo_epi32(diff_lo2, diff_lo2);
        __m512i sq_hi2 = _mm512_mullo_epi32(diff_hi2, diff_hi2);

        sum_vec = _mm512_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi);
        sum_vec = _mm512_add_epi32(sum_vec, sq_lo2);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi2);
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

        __m128i a_lo128 = _mm512_castsi512_si128(a_vec);
        __m128i a_hi128 = _mm512_extracti32x4_epi32(a_vec, 1);
        __m128i a_lo2_128 = _mm512_extracti32x4_epi32(a_vec, 2);
        __m128i a_hi2_128 = _mm512_extracti32x4_epi32(a_vec, 3);

        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m256i a_lo256 = _mm256_cvtepu8_epi16(a_lo128);
        __m256i a_hi256 = _mm256_cvtepu8_epi16(a_hi128);
        __m256i a_lo2_256 = _mm256_cvtepu8_epi16(a_lo2_128);
        __m256i a_hi2_256 = _mm256_cvtepu8_epi16(a_hi2_128);

        __m256i b_lo256 = _mm256_cvtepu8_epi16(b_lo128);
        __m256i b_hi256 = _mm256_cvtepu8_epi16(b_hi128);
        __m256i b_lo2_256 = _mm256_cvtepu8_epi16(b_lo2_128);
        __m256i b_hi2_256 = _mm256_cvtepu8_epi16(b_hi2_128);

        __m512i a_lo = _mm512_cvtepu16_epi32(a_lo256);
        __m512i a_hi = _mm512_cvtepu16_epi32(a_hi256);
        __m512i a_lo2 = _mm512_cvtepu16_epi32(a_lo2_256);
        __m512i a_hi2 = _mm512_cvtepu16_epi32(a_hi2_256);

        __m512i b_lo = _mm512_cvtepu16_epi32(b_lo256);
        __m512i b_hi = _mm512_cvtepu16_epi32(b_hi256);
        __m512i b_lo2 = _mm512_cvtepu16_epi32(b_lo2_256);
        __m512i b_hi2 = _mm512_cvtepu16_epi32(b_hi2_256);

        __m512i diff_lo = _mm512_sub_epi32(a_lo, b_lo);
        __m512i diff_hi = _mm512_sub_epi32(a_hi, b_hi);
        __m512i diff_lo2 = _mm512_sub_epi32(a_lo2, b_lo2);
        __m512i diff_hi2 = _mm512_sub_epi32(a_hi2, b_hi2);

        /* Square using mullo + add (no madd_epi32 instruction) */
        __m512i sq_lo = _mm512_mullo_epi32(diff_lo, diff_lo);
        __m512i sq_hi = _mm512_mullo_epi32(diff_hi, diff_hi);
        __m512i sq_lo2 = _mm512_mullo_epi32(diff_lo2, diff_lo2);
        __m512i sq_hi2 = _mm512_mullo_epi32(diff_hi2, diff_hi2);

        sum_vec = _mm512_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi);
        sum_vec = _mm512_add_epi32(sum_vec, sq_lo2);
        sum_vec = _mm512_add_epi32(sum_vec, sq_hi2);
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
    __m512i b_sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* dot product using VNNI with bias correction */
        __m512i a_biased = _mm512_add_epi8(a_vec, _mm512_set1_epi8(-128));
        dot_vec = _mm512_dpbusd_epi32(dot_vec, a_biased, b_vec);

        /* Accumulate sum(b) for dot correction - use 128-bit extracts */
        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m512i b_lo = _mm512_cvtepi8_epi32(b_lo128);
        __m512i b_hi = _mm512_cvtepi8_epi32(b_hi128);
        __m512i b_lo2 = _mm512_cvtepi8_epi32(b_lo2_128);
        __m512i b_hi2 = _mm512_cvtepi8_epi32(b_hi2_128);

        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_lo);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_hi);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_lo2);
        b_sum_vec = _mm512_add_epi32(b_sum_vec, b_hi2);

        /* norm a - convert to 16-bit, square with madd */
        __m128i a_lo128 = _mm512_castsi512_si128(a_vec);
        __m128i a_hi128 = _mm512_extracti32x4_epi32(a_vec, 1);
        __m128i a_lo2_128 = _mm512_extracti32x4_epi32(a_vec, 2);
        __m128i a_hi2_128 = _mm512_extracti32x4_epi32(a_vec, 3);

        __m256i a_lo256 = _mm256_cvtepi8_epi16(a_lo128);
        __m256i a_hi256 = _mm256_cvtepi8_epi16(a_hi128);
        __m256i a_lo2_256 = _mm256_cvtepi8_epi16(a_lo2_128);
        __m256i a_hi2_256 = _mm256_cvtepi8_epi16(a_hi2_128);

        __m512i a_lo16 = _mm512_cvtepi16_epi32(a_lo256);
        __m512i a_hi16 = _mm512_cvtepi16_epi32(a_hi256);
        __m512i a_lo16_2 = _mm512_cvtepi16_epi32(a_lo2_256);
        __m512i a_hi16_2 = _mm512_cvtepi16_epi32(a_hi2_256);

        __m512i sq_a_lo = _mm512_madd_epi16(a_lo16, a_lo16);
        __m512i sq_a_hi = _mm512_madd_epi16(a_hi16, a_hi16);
        __m512i sq_a_lo2 = _mm512_madd_epi16(a_lo16_2, a_lo16_2);
        __m512i sq_a_hi2 = _mm512_madd_epi16(a_hi16_2, a_hi16_2);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo2);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi2);

        /* norm b */
        __m256i b_lo256 = _mm256_cvtepi8_epi16(b_lo128);
        __m256i b_hi256 = _mm256_cvtepi8_epi16(b_hi128);
        __m256i b_lo2_256 = _mm256_cvtepi8_epi16(b_lo2_128);
        __m256i b_hi2_256 = _mm256_cvtepi8_epi16(b_hi2_128);

        __m512i b_lo16 = _mm512_cvtepi16_epi32(b_lo256);
        __m512i b_hi16 = _mm512_cvtepi16_epi32(b_hi256);
        __m512i b_lo16_2 = _mm512_cvtepi16_epi32(b_lo2_256);
        __m512i b_hi16_2 = _mm512_cvtepi16_epi32(b_hi2_256);

        __m512i sq_b_lo = _mm512_madd_epi16(b_lo16, b_lo16);
        __m512i sq_b_hi = _mm512_madd_epi16(b_hi16, b_hi16);
        __m512i sq_b_lo2 = _mm512_madd_epi16(b_lo16_2, b_lo16_2);
        __m512i sq_b_hi2 = _mm512_madd_epi16(b_hi16_2, b_hi16_2);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo2);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi2);
    }

    int32_t dot_scalar = _mm512_reduce_add_epi32(dot_vec);
    int32_t b_sum_scalar = _mm512_reduce_add_epi32(b_sum_vec);
    dot_scalar -= (-128) * b_sum_scalar; /* VNNI bias correction: a was biased by -128 */

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
    __m512i a_sum_vec = _mm512_setzero_si512();

    for (; i + simd_width <= n; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* dot product using widen-then-multiply */
        __m128i a_lo128 = _mm512_castsi512_si128(a_vec);
        __m128i a_hi128 = _mm512_extracti32x4_epi32(a_vec, 1);
        __m128i a_lo2_128 = _mm512_extracti32x4_epi32(a_vec, 2);
        __m128i a_hi2_128 = _mm512_extracti32x4_epi32(a_vec, 3);

        __m128i b_lo128 = _mm512_castsi512_si128(b_vec);
        __m128i b_hi128 = _mm512_extracti32x4_epi32(b_vec, 1);
        __m128i b_lo2_128 = _mm512_extracti32x4_epi32(b_vec, 2);
        __m128i b_hi2_128 = _mm512_extracti32x4_epi32(b_vec, 3);

        __m256i a_lo256 = _mm256_cvtepu8_epi16(a_lo128);
        __m256i a_hi256 = _mm256_cvtepu8_epi16(a_hi128);
        __m256i a_lo2_256 = _mm256_cvtepu8_epi16(a_lo2_128);
        __m256i a_hi2_256 = _mm256_cvtepu8_epi16(a_hi2_128);

        __m256i b_lo256 = _mm256_cvtepu8_epi16(b_lo128);
        __m256i b_hi256 = _mm256_cvtepu8_epi16(b_hi128);
        __m256i b_lo2_256 = _mm256_cvtepu8_epi16(b_lo2_128);
        __m256i b_hi2_256 = _mm256_cvtepu8_epi16(b_hi2_128);

        __m512i a_lo = _mm512_cvtepu16_epi32(a_lo256);
        __m512i a_hi = _mm512_cvtepu16_epi32(a_hi256);
        __m512i a_lo2 = _mm512_cvtepu16_epi32(a_lo2_256);
        __m512i a_hi2 = _mm512_cvtepu16_epi32(a_hi2_256);

        __m512i b_lo = _mm512_cvtepu16_epi32(b_lo256);
        __m512i b_hi = _mm512_cvtepu16_epi32(b_hi256);
        __m512i b_lo2 = _mm512_cvtepu16_epi32(b_lo2_256);
        __m512i b_hi2 = _mm512_cvtepu16_epi32(b_hi2_256);

        __m512i dot_lo = _mm512_madd_epi16(a_lo, b_lo);
        __m512i dot_hi = _mm512_madd_epi16(a_hi, b_hi);
        __m512i dot_lo2 = _mm512_madd_epi16(a_lo2, b_lo2);
        __m512i dot_hi2 = _mm512_madd_epi16(a_hi2, b_hi2);
        dot_vec = _mm512_add_epi32(dot_vec, dot_lo);
        dot_vec = _mm512_add_epi32(dot_vec, dot_hi);
        dot_vec = _mm512_add_epi32(dot_vec, dot_lo2);
        dot_vec = _mm512_add_epi32(dot_vec, dot_hi2);

        /* norm a */
        __m512i sq_a_lo = _mm512_madd_epi16(a_lo, a_lo);
        __m512i sq_a_hi = _mm512_madd_epi16(a_hi, a_hi);
        __m512i sq_a_lo2 = _mm512_madd_epi16(a_lo2, a_lo2);
        __m512i sq_a_hi2 = _mm512_madd_epi16(a_hi2, a_hi2);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_lo2);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, sq_a_hi2);

        /* Accumulate sum(a) for VNNI correction term - use 128-bit extracts */
        __m512i a_lo32 = _mm512_cvtepu8_epi32(a_lo128);
        __m512i a_hi32 = _mm512_cvtepu8_epi32(a_hi128);
        __m512i a_lo32_2 = _mm512_cvtepu8_epi32(a_lo2_128);
        __m512i a_hi32_2 = _mm512_cvtepu8_epi32(a_hi2_128);
        a_sum_vec = _mm512_add_epi32(a_sum_vec, a_lo32);
        a_sum_vec = _mm512_add_epi32(a_sum_vec, a_hi32);
        a_sum_vec = _mm512_add_epi32(a_sum_vec, a_lo32_2);
        a_sum_vec = _mm512_add_epi32(a_sum_vec, a_hi32_2);

        /* norm b */
        __m512i sq_b_lo = _mm512_madd_epi16(b_lo, b_lo);
        __m512i sq_b_hi = _mm512_madd_epi16(b_hi, b_hi);
        __m512i sq_b_lo2 = _mm512_madd_epi16(b_lo2, b_lo2);
        __m512i sq_b_hi2 = _mm512_madd_epi16(b_hi2, b_hi2);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_lo2);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, sq_b_hi2);
    }

    int32_t dot_scalar = _mm512_reduce_add_epi32(dot_vec);
    int32_t a_sum_scalar = _mm512_reduce_add_epi32(a_sum_vec);
    dot_scalar -= 128 * a_sum_scalar; /* VNNI correction: sum(a*b) = sum((a+128)*b) - 128*sum(a) */

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

/* ===== Binary (1-bit) variants (AVX-512) ===== */

int32_t vek_dot_b1_avx512(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 8; /* 512-bit = 8 x 64-bit words */
    size_t words = (n + 63) / 64;
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_epi32();

    for (; i + simd_width <= words; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* XNOR = ~(a ^ b), then popcnt */
        __m512i xnor = _mm512_xor_epi64(a_vec, b_vec);
        xnor = _mm512_ternarylogic_epi64(xnor, _mm512_set1_epi64(0), _mm512_set1_epi64(0), 0xFF); // NOT
        __m512i popcnt = _mm512_popcnt_epi64(xnor);

        /* Horizontal sum across 8 lanes of 64-bit popcnts */
        __m256i popcnt32 = _mm512_cvtepi64_epi32(popcnt);
        sum_vec = _mm512_add_epi32(sum_vec, _mm512_inserti32x8(_mm512_castsi256_si512(popcnt32), _mm256_setzero_si256(), 1));
    }

    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);

    for (; i < words; i++) {
        sum_scalar += __builtin_popcountll(~(a[i] ^ b[i]));
    }

    return sum_scalar;
}

int32_t vek_hamming_b1_avx512(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 8;
    size_t words = (n + 63) / 64;
    size_t i = 0;

    __m512i sum_vec = _mm512_setzero_epi32();

    for (; i + simd_width <= words; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* XOR + popcnt */
        __m512i xor_vec = _mm512_xor_epi64(a_vec, b_vec);
        __m512i popcnt = _mm512_popcnt_epi64(xor_vec);

        __m256i popcnt32 = _mm512_cvtepi64_epi32(popcnt);
        sum_vec = _mm512_add_epi32(sum_vec, _mm512_inserti32x8(_mm512_castsi256_si512(popcnt32), _mm256_setzero_si256(), 1));
    }

    int32_t sum_scalar = _mm512_reduce_add_epi32(sum_vec);

    for (; i < words; i++) {
        sum_scalar += __builtin_popcountll(a[i] ^ b[i]);
    }

    return sum_scalar;
}

float vek_cosine_b1_avx512(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 8;
    size_t words = (n + 63) / 64;
    size_t i = 0;

    __m512i dot_vec = _mm512_setzero_epi32();
    __m512i norm_a_vec = _mm512_setzero_epi32();
    __m512i norm_b_vec = _mm512_setzero_epi32();

    for (; i + simd_width <= words; i += simd_width) {
        __m512i a_vec = _mm512_loadu_si512((const __m512i*)(a + i));
        __m512i b_vec = _mm512_loadu_si512((const __m512i*)(b + i));

        /* dot product via AND + popcnt */
        __m512i and_vec = _mm512_and_epi64(a_vec, b_vec);
        __m512i popcnt = _mm512_popcnt_epi64(and_vec);
        __m256i popcnt32 = _mm512_cvtepi64_epi32(popcnt);
        dot_vec = _mm512_add_epi32(dot_vec, _mm512_inserti32x8(_mm512_castsi256_si512(popcnt32), _mm256_setzero_si256(), 1));

        /* norm a */
        __m512i pop_a = _mm512_popcnt_epi64(a_vec);
        __m256i pop_a32 = _mm512_cvtepi64_epi32(pop_a);
        norm_a_vec = _mm512_add_epi32(norm_a_vec, _mm512_inserti32x8(_mm512_castsi256_si512(pop_a32), _mm256_setzero_si256(), 1));

        /* norm b */
        __m512i pop_b = _mm512_popcnt_epi64(b_vec);
        __m256i pop_b32 = _mm512_cvtepi64_epi32(pop_b);
        norm_b_vec = _mm512_add_epi32(norm_b_vec, _mm512_inserti32x8(_mm512_castsi256_si512(pop_b32), _mm256_setzero_si256(), 1));
    }

    int32_t dot_scalar = _mm512_reduce_add_epi32(dot_vec);
    int32_t norm_a_scalar = _mm512_reduce_add_epi32(norm_a_vec);
    int32_t norm_b_scalar = _mm512_reduce_add_epi32(norm_b_vec);

    for (; i < words; i++) {
        uint64_t and_bits = a[i] & b[i];
        dot_scalar += __builtin_popcountll(and_bits);
        norm_a_scalar += __builtin_popcountll(a[i]);
        norm_b_scalar += __builtin_popcountll(b[i]);
    }

    if (norm_a_scalar == 0 || norm_b_scalar == 0) return 0.0f;
    return (float)dot_scalar / (sqrtf((float)norm_a_scalar) * sqrtf((float)norm_b_scalar));
}
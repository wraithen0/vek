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

/* ===== Quantized int8/uint8 variants (AVX2) ===== */

/* AVX2 int8 dot product: sign-extend int8→int16, multiply int16×int16→int32, sum */
int32_t vek_dot_i8_avx2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 32; /* 256-bit = 32 int8 */
    size_t i = 0;

    __m256i sum_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        /* Sign-extend int8 to int16 (lower 128 bits) */
        __m128i a_lo128 = _mm256_castsi256_si128(a_vec);
        __m128i a_hi128 = _mm256_extracti128_si256(a_vec, 1);
        __m256i a_lo = _mm256_cvtepi8_epi16(a_lo128);
        __m256i a_hi = _mm256_cvtepi8_epi16(a_hi128);

        __m128i b_lo128 = _mm256_castsi256_si128(b_vec);
        __m128i b_hi128 = _mm256_extracti128_si256(b_vec, 1);
        __m256i b_lo = _mm256_cvtepi8_epi16(b_lo128);
        __m256i b_hi = _mm256_cvtepi8_epi16(b_hi128);

        /* Multiply int16 × int16 → int32, accumulate */
        __m256i sum_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i sum_hi = _mm256_madd_epi16(a_hi, b_hi);
        sum_vec = _mm256_add_epi32(sum_vec, sum_lo);
        sum_vec = _mm256_add_epi32(sum_vec, sum_hi);
    }

    /* Horizontal sum of 8x 32-bit integers */
    __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
    __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
    __m128i sum = _mm_add_epi32(sum_hi, sum_lo);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t sum_scalar = _mm_cvtsi128_si32(sum);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum_scalar;
}

/* AVX2 uint8 dot product: zero-extend uint8→uint16, multiply uint16×uint16→uint32, sum */
uint32_t vek_dot_u8_avx2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 32;
    size_t i = 0;

    __m256i sum_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        /* Zero-extend uint8 to uint16 (lower 128 bits) */
        __m128i a_lo128 = _mm256_castsi256_si128(a_vec);
        __m128i a_hi128 = _mm256_extracti128_si256(a_vec, 1);
        __m256i a_lo = _mm256_cvtepu8_epi16(a_lo128);
        __m256i a_hi = _mm256_cvtepu8_epi16(a_hi128);

        __m128i b_lo128 = _mm256_castsi256_si128(b_vec);
        __m128i b_hi128 = _mm256_extracti128_si256(b_vec, 1);
        __m256i b_lo = _mm256_cvtepu8_epi16(b_lo128);
        __m256i b_hi = _mm256_cvtepu8_epi16(b_hi128);

        /* Multiply uint16 × uint16 → uint32, accumulate */
        __m256i sum_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i sum_hi = _mm256_madd_epi16(a_hi, b_hi);
        sum_vec = _mm256_add_epi32(sum_vec, sum_lo);
        sum_vec = _mm256_add_epi32(sum_vec, sum_hi);
    }

    /* Horizontal sum of 8x 32-bit integers */
    __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
    __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
    __m128i sum = _mm_add_epi32(sum_hi, sum_lo);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    uint32_t sum_scalar = (uint32_t)_mm_cvtsi128_si32(sum);

    for (; i < n; i++) {
        sum_scalar += (uint32_t)a[i] * (uint32_t)b[i];
    }

    return sum_scalar;
}

/* AVX2 int8 squared L2 distance */
int32_t vek_l2sq_i8_avx2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 32;
    size_t i = 0;

    __m256i sum_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        /* Convert to 16-bit for subtraction */
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a_vec));
        __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_vec));
        __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_vec, 1));

        __m256i diff_lo = _mm256_sub_epi16(a_lo, b_lo);
        __m256i diff_hi = _mm256_sub_epi16(a_hi, b_hi);

        __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm256_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm256_add_epi32(sum_vec, sq_hi);
    }

    /* Horizontal sum */
    __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
    __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
    __m128i sum = _mm_add_epi32(sum_hi, sum_lo);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t sum_scalar = _mm_cvtsi128_si32(sum);

    for (; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX2 uint8 squared L2 distance */
uint32_t vek_l2sq_u8_avx2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 32;
    size_t i = 0;

    __m256i sum_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        __m256i a_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_vec));
        __m256i a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a_vec, 1));
        __m256i b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_vec));
        __m256i b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b_vec, 1));

        __m256i diff_lo = _mm256_sub_epi16(a_lo, b_lo);
        __m256i diff_hi = _mm256_sub_epi16(a_hi, b_hi);

        __m256i sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        __m256i sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm256_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm256_add_epi32(sum_vec, sq_hi);
    }

    __m128i sum_hi = _mm256_extracti128_si256(sum_vec, 1);
    __m128i sum_lo = _mm256_castsi256_si128(sum_vec);
    __m128i sum = _mm_add_epi32(sum_hi, sum_lo);
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    uint32_t sum_scalar = (uint32_t)_mm_cvtsi128_si32(sum);

    for (; i < n; i++) {
        uint32_t diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* AVX2 int8 cosine similarity */
float vek_cosine_i8_avx2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 32;
    size_t i = 0;

    __m256i dot_vec = _mm256_setzero_si256();
    __m256i norm_a_vec = _mm256_setzero_si256();
    __m256i norm_b_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        /* dot product: sign-extend int8→int16, madd_epi16 multiply+pairwise-add */
        __m128i a_lo128 = _mm256_castsi256_si128(a_vec);
        __m128i a_hi128 = _mm256_extracti128_si256(a_vec, 1);
        __m256i a_lo = _mm256_cvtepi8_epi16(a_lo128);
        __m256i a_hi = _mm256_cvtepi8_epi16(a_hi128);

        __m128i b_lo128 = _mm256_castsi256_si128(b_vec);
        __m128i b_hi128 = _mm256_extracti128_si256(b_vec, 1);
        __m256i b_lo = _mm256_cvtepi8_epi16(b_lo128);
        __m256i b_hi = _mm256_cvtepi8_epi16(b_hi128);

        __m256i dot_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i dot_hi = _mm256_madd_epi16(a_hi, b_hi);
        dot_vec = _mm256_add_epi32(dot_vec, dot_lo);
        dot_vec = _mm256_add_epi32(dot_vec, dot_hi);

        /* norm a — reuse sign-extended a_lo, a_hi from dot product */
        __m256i sq_a_lo = _mm256_madd_epi16(a_lo, a_lo);
        __m256i sq_a_hi = _mm256_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm256_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm256_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b — reuse sign-extended b_lo, b_hi from dot product */
        __m256i sq_b_lo = _mm256_madd_epi16(b_lo, b_lo);
        __m256i sq_b_hi = _mm256_madd_epi16(b_hi, b_hi);
        norm_b_vec = _mm256_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm256_add_epi32(norm_b_vec, sq_b_hi);
    }

    /* Horizontal sums */
    __m128i dot_hi = _mm256_extracti128_si256(dot_vec, 1);
    __m128i dot_lo = _mm256_castsi256_si128(dot_vec);
    __m128i dot = _mm_add_epi32(dot_hi, dot_lo);
    dot = _mm_add_epi32(dot, _mm_shuffle_epi32(dot, _MM_SHUFFLE(2, 3, 0, 1)));
    dot = _mm_add_epi32(dot, _mm_shuffle_epi32(dot, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t dot_scalar = _mm_cvtsi128_si32(dot);

    __m128i na_hi = _mm256_extracti128_si256(norm_a_vec, 1);
    __m128i na_lo = _mm256_castsi256_si128(norm_a_vec);
    __m128i na = _mm_add_epi32(na_hi, na_lo);
    na = _mm_add_epi32(na, _mm_shuffle_epi32(na, _MM_SHUFFLE(2, 3, 0, 1)));
    na = _mm_add_epi32(na, _mm_shuffle_epi32(na, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_a_scalar = _mm_cvtsi128_si32(na);

    __m128i nb_hi = _mm256_extracti128_si256(norm_b_vec, 1);
    __m128i nb_lo = _mm256_castsi256_si128(norm_b_vec);
    __m128i nb = _mm_add_epi32(nb_hi, nb_lo);
    nb = _mm_add_epi32(nb, _mm_shuffle_epi32(nb, _MM_SHUFFLE(2, 3, 0, 1)));
    nb = _mm_add_epi32(nb, _mm_shuffle_epi32(nb, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_b_scalar = _mm_cvtsi128_si32(nb);

    /* Tail */
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

/* AVX2 uint8 cosine similarity */
float vek_cosine_u8_avx2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 32;
    size_t i = 0;

    __m256i dot_vec = _mm256_setzero_si256();
    __m256i norm_a_vec = _mm256_setzero_si256();
    __m256i norm_b_vec = _mm256_setzero_si256();

    for (; i + simd_width <= n; i += simd_width) {
        __m256i a_vec = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((const __m256i*)(b + i));

        /* dot product - use cvtepu8_epi16 then madd_epi16 on u16 data */
        __m128i a_lo128 = _mm256_castsi256_si128(a_vec);
        __m128i a_hi128 = _mm256_extracti128_si256(a_vec, 1);
        __m256i a_lo = _mm256_cvtepu8_epi16(a_lo128);
        __m256i a_hi = _mm256_cvtepu8_epi16(a_hi128);

        __m128i b_lo128 = _mm256_castsi256_si128(b_vec);
        __m128i b_hi128 = _mm256_extracti128_si256(b_vec, 1);
        __m256i b_lo = _mm256_cvtepu8_epi16(b_lo128);
        __m256i b_hi = _mm256_cvtepu8_epi16(b_hi128);

        __m256i dot_lo = _mm256_madd_epi16(a_lo, b_lo);
        __m256i dot_hi = _mm256_madd_epi16(a_hi, b_hi);
        dot_vec = _mm256_add_epi32(dot_vec, dot_lo);
        dot_vec = _mm256_add_epi32(dot_vec, dot_hi);

        /* norm a */
        __m256i sq_a_lo = _mm256_madd_epi16(a_lo, a_lo);
        __m256i sq_a_hi = _mm256_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm256_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm256_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b */
        __m256i sq_b_lo = _mm256_madd_epi16(b_lo, b_lo);
        __m256i sq_b_hi = _mm256_madd_epi16(b_hi, b_hi);
        norm_b_vec = _mm256_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm256_add_epi32(norm_b_vec, sq_b_hi);
    }

    /* Horizontal sums */
    __m128i dot_hi = _mm256_extracti128_si256(dot_vec, 1);
    __m128i dot_lo = _mm256_castsi256_si128(dot_vec);
    __m128i dot = _mm_add_epi32(dot_hi, dot_lo);
    dot = _mm_add_epi32(dot, _mm_shuffle_epi32(dot, _MM_SHUFFLE(2, 3, 0, 1)));
    dot = _mm_add_epi32(dot, _mm_shuffle_epi32(dot, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t dot_scalar = _mm_cvtsi128_si32(dot);

    __m128i na_hi = _mm256_extracti128_si256(norm_a_vec, 1);
    __m128i na_lo = _mm256_castsi256_si128(norm_a_vec);
    __m128i na = _mm_add_epi32(na_hi, na_lo);
    na = _mm_add_epi32(na, _mm_shuffle_epi32(na, _MM_SHUFFLE(2, 3, 0, 1)));
    na = _mm_add_epi32(na, _mm_shuffle_epi32(na, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_a_scalar = _mm_cvtsi128_si32(na);

    __m128i nb_hi = _mm256_extracti128_si256(norm_b_vec, 1);
    __m128i nb_lo = _mm256_castsi256_si128(norm_b_vec);
    __m128i nb = _mm_add_epi32(nb_hi, nb_lo);
    nb = _mm_add_epi32(nb, _mm_shuffle_epi32(nb, _MM_SHUFFLE(2, 3, 0, 1)));
    nb = _mm_add_epi32(nb, _mm_shuffle_epi32(nb, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_b_scalar = _mm_cvtsi128_si32(nb);

    /* Tail */
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
/* ===== Binary (1-bit) variants - fall back to scalar for AVX2 ===== */
/* AVX2 lacks 64-bit popcnt; use scalar fallback */

int32_t vek_dot_b1_avx2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_dot_b1_scalar(a, b, n);
}

int32_t vek_hamming_b1_avx2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_hamming_b1_scalar(a, b, n);
}

float vek_cosine_b1_avx2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_cosine_b1_scalar(a, b, n);
}
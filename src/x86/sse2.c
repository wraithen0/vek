/**
 * vek - SSE2 intrinsics implementation for x86_64
 * SSE2 (128-bit) vectorized kernels for f32 dot, L2, cosine
 * Baseline for all x86_64 CPUs
 * 
 * Quantized int8/uint8 kernels use PURE SSE2 only (no SSSE3)
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

/* ===== Quantized int8/uint8 variants (PURE SSE2 - no SSSE3) ===== */

/* SSE2 int8 dot product: sum(a[i] * b[i]) using pure SSE2 */
int32_t vek_dot_i8_sse2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16; /* 128-bit = 16 int8 */
    size_t i = 0;

    __m128i sum_vec = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi8(-1); /* for sign-extension */

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        /* SSE2: sign-extend int8 to int16 */
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, ones);  /* 8x 16-bit (sign-extended) */
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, ones);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, ones);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, ones);

        /* PMADDWD: multiply 16-bit pairs, add adjacent pairs -> 32-bit */
        __m128i sum_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i sum_hi = _mm_madd_epi16(a_hi, b_hi);

        sum_vec = _mm_add_epi32(sum_vec, sum_lo);
        sum_vec = _mm_add_epi32(sum_vec, sum_hi);
    }

    /* Horizontal sum of 4x 32-bit integers */
    __m128i sum2 = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i sum4 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t sum_scalar = _mm_cvtsi128_si32(sum4);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum_scalar;
}

/* SSE2 uint8 dot product */
uint32_t vek_dot_u8_sse2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m128i sum_vec = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        /* Unpack unsigned 8-bit to 16-bit */
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, zero);
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, zero);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, zero);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, zero);

        /* PMADDWD: multiply 16-bit pairs, sum adjacent pairs to 32-bit */
        __m128i sum_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i sum_hi = _mm_madd_epi16(a_hi, b_hi);

        sum_vec = _mm_add_epi32(sum_vec, sum_lo);
        sum_vec = _mm_add_epi32(sum_vec, sum_hi);
    }

    /* Horizontal sum */
    __m128i sum2 = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i sum4 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    uint32_t sum_scalar = (uint32_t)_mm_cvtsi128_si32(sum4);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (uint32_t)a[i] * (uint32_t)b[i];
    }

    return sum_scalar;
}

/* SSE2 int8 squared L2 distance */
int32_t vek_l2sq_i8_sse2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m128i sum_vec = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        /* Unpack to 16-bit for subtraction */
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, zero);
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, zero);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, zero);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, zero);

        __m128i diff_lo = _mm_sub_epi16(a_lo, b_lo);
        __m128i diff_hi = _mm_sub_epi16(a_hi, b_hi);

        /* Square via PMADDWD (multiply 16-bit pairs, sum to 32-bit) */
        __m128i sq_lo = _mm_madd_epi16(diff_lo, diff_lo);
        __m128i sq_hi = _mm_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm_add_epi32(sum_vec, sq_hi);
    }

    /* Horizontal sum */
    __m128i sum2 = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i sum4 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t sum_scalar = _mm_cvtsi128_si32(sum4);

    /* Tail */
    for (; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* SSE2 uint8 squared L2 distance */
uint32_t vek_l2sq_u8_sse2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m128i sum_vec = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        __m128i a_lo = _mm_unpacklo_epi8(a_vec, zero);
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, zero);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, zero);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, zero);

        __m128i diff_lo = _mm_sub_epi16(a_lo, b_lo);
        __m128i diff_hi = _mm_sub_epi16(a_hi, b_hi);

        __m128i sq_lo = _mm_madd_epi16(diff_lo, diff_lo);
        __m128i sq_hi = _mm_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm_add_epi32(sum_vec, sq_hi);
    }

    /* Horizontal sum */
    __m128i sum2 = _mm_add_epi32(sum_vec, _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i sum4 = _mm_add_epi32(sum2, _mm_shuffle_epi32(sum2, _MM_SHUFFLE(1, 0, 3, 2)));
    uint32_t sum_scalar = (uint32_t)_mm_cvtsi128_si32(sum4);

    /* Tail */
    for (; i < n; i++) {
        uint32_t diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* SSE2 int8 cosine similarity */
float vek_cosine_i8_sse2(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m128i dot_vec = _mm_setzero_si128();
    __m128i norm_a_vec = _mm_setzero_si128();
    __m128i norm_b_vec = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi8(-1); /* for sign-extension */

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        /* dot product - sign-extend int8 to int16 */
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, ones);
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, ones);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, ones);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, ones);

        __m128i dot_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i dot_hi = _mm_madd_epi16(a_hi, b_hi);
        dot_vec = _mm_add_epi32(dot_vec, dot_lo);
        dot_vec = _mm_add_epi32(dot_vec, dot_hi);

        /* norm a */
        __m128i sq_a_lo = _mm_madd_epi16(a_lo, a_lo);
        __m128i sq_a_hi = _mm_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b */
        __m128i b_lo2 = _mm_unpacklo_epi8(b_vec, ones);
        __m128i b_hi2 = _mm_unpackhi_epi8(b_vec, ones);
        __m128i sq_b_lo = _mm_madd_epi16(b_lo2, b_lo2);
        __m128i sq_b_hi = _mm_madd_epi16(b_hi2, b_hi2);
        norm_b_vec = _mm_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm_add_epi32(norm_b_vec, sq_b_hi);
    }

    /* Horizontal sums */
    __m128i dot2 = _mm_add_epi32(dot_vec, _mm_shuffle_epi32(dot_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i dot4 = _mm_add_epi32(dot2, _mm_shuffle_epi32(dot2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t dot_scalar = _mm_cvtsi128_si32(dot4);

    __m128i na2 = _mm_add_epi32(norm_a_vec, _mm_shuffle_epi32(norm_a_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i na4 = _mm_add_epi32(na2, _mm_shuffle_epi32(na2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_a_scalar = _mm_cvtsi128_si32(na4);

    __m128i nb2 = _mm_add_epi32(norm_b_vec, _mm_shuffle_epi32(norm_b_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i nb4 = _mm_add_epi32(nb2, _mm_shuffle_epi32(nb2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_b_scalar = _mm_cvtsi128_si32(nb4);

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

/* SSE2 uint8 cosine similarity */
float vek_cosine_u8_sse2(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    __m128i dot_vec = _mm_setzero_si128();
    __m128i norm_a_vec = _mm_setzero_si128();
    __m128i norm_b_vec = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    for (; i + simd_width <= n; i += simd_width) {
        __m128i a_vec = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((const __m128i*)(b + i));

        /* dot product */
        __m128i a_lo = _mm_unpacklo_epi8(a_vec, zero);
        __m128i a_hi = _mm_unpackhi_epi8(a_vec, zero);
        __m128i b_lo = _mm_unpacklo_epi8(b_vec, zero);
        __m128i b_hi = _mm_unpackhi_epi8(b_vec, zero);

        __m128i dot_lo = _mm_madd_epi16(a_lo, b_lo);
        __m128i dot_hi = _mm_madd_epi16(a_hi, b_hi);
        dot_vec = _mm_add_epi32(dot_vec, dot_lo);
        dot_vec = _mm_add_epi32(dot_vec, dot_hi);

        /* norm a */
        __m128i sq_a_lo = _mm_madd_epi16(a_lo, a_lo);
        __m128i sq_a_hi = _mm_madd_epi16(a_hi, a_hi);
        norm_a_vec = _mm_add_epi32(norm_a_vec, sq_a_lo);
        norm_a_vec = _mm_add_epi32(norm_a_vec, sq_a_hi);

        /* norm b */
        __m128i b_lo2 = _mm_unpacklo_epi8(b_vec, zero);
        __m128i b_hi2 = _mm_unpackhi_epi8(b_vec, zero);
        __m128i sq_b_lo = _mm_madd_epi16(b_lo2, b_lo2);
        __m128i sq_b_hi = _mm_madd_epi16(b_hi2, b_hi2);
        norm_b_vec = _mm_add_epi32(norm_b_vec, sq_b_lo);
        norm_b_vec = _mm_add_epi32(norm_b_vec, sq_b_hi);
    }

    /* Horizontal sums */
    __m128i dot2 = _mm_add_epi32(dot_vec, _mm_shuffle_epi32(dot_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i dot4 = _mm_add_epi32(dot2, _mm_shuffle_epi32(dot2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t dot_scalar = _mm_cvtsi128_si32(dot4);

    __m128i na2 = _mm_add_epi32(norm_a_vec, _mm_shuffle_epi32(norm_a_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i na4 = _mm_add_epi32(na2, _mm_shuffle_epi32(na2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_a_scalar = _mm_cvtsi128_si32(na4);

    __m128i nb2 = _mm_add_epi32(norm_b_vec, _mm_shuffle_epi32(norm_b_vec, _MM_SHUFFLE(2, 3, 0, 1)));
    __m128i nb4 = _mm_add_epi32(nb2, _mm_shuffle_epi32(nb2, _MM_SHUFFLE(1, 0, 3, 2)));
    int32_t norm_b_scalar = _mm_cvtsi128_si32(nb4);

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

/* ===== Binary (1-bit) variants - fall back to scalar for SSE2 ===== */

int32_t vek_dot_b1_sse2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_dot_b1_scalar(a, b, n);
}

int32_t vek_hamming_b1_sse2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_hamming_b1_scalar(a, b, n);
}

float vek_cosine_b1_sse2(const uint64_t *a, const uint64_t *b, size_t n)
{
    return vek_cosine_b1_scalar(a, b, n);
}
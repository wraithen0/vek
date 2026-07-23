/**
 * vek - NEON intrinsics implementation for ARM64
 * NEON (128-bit) vectorized kernels for f32 dot, L2, cosine
 */

#include <arm_neon.h>
#include <stddef.h>
#include <math.h>
#include "vek.h"
#include "../internal.h"

/* NEON dot product: sum(a[i] * b[i]) */
float vek_dot_f32_neon(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4; /* 128-bit = 4 floats */
    size_t i = 0;

    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (; i + simd_width <= n; i += simd_width) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t prod = vmulq_f32(a_vec, b_vec);
        sum_vec = vaddq_f32(sum_vec, prod);
    }

    /* Horizontal sum */
    float32x2_t sum_lo = vget_low_f32(sum_vec);
    float32x2_t sum_hi = vget_high_f32(sum_vec);
    float32x2_t sum = vpadd_f32(sum_lo, sum_hi);
    float sum_scalar = vget_lane_f32(vpadd_f32(sum, sum), 0);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += a[i] * b[i];
    }

    return sum_scalar;
}

/* NEON squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32_neon(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4;
    size_t i = 0;

    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (; i + simd_width <= n; i += simd_width) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(a_vec, b_vec);
        float32x4_t sq = vmulq_f32(diff, diff);
        sum_vec = vaddq_f32(sum_vec, sq);
    }

    /* Horizontal sum */
    float32x2_t sum_lo = vget_low_f32(sum_vec);
    float32x2_t sum_hi = vget_high_f32(sum_vec);
    float32x2_t sum = vpadd_f32(sum_lo, sum_hi);
    float sum_scalar = vget_lane_f32(vpadd_f32(sum, sum), 0);

    /* Tail */
    for (; i < n; i++) {
        float diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* NEON cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32_neon(const float *a, const float *b, size_t n)
{
    const size_t simd_width = 4;
    size_t i = 0;

    float32x4_t dot_vec = vdupq_n_f32(0.0f);
    float32x4_t norm_a_vec = vdupq_n_f32(0.0f);
    float32x4_t norm_b_vec = vdupq_n_f32(0.0f);

    for (; i + simd_width <= n; i += simd_width) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);

        dot_vec = vfmaq_f32(dot_vec, a_vec, b_vec);
        norm_a_vec = vfmaq_f32(norm_a_vec, a_vec, a_vec);
        norm_b_vec = vfmaq_f32(norm_b_vec, b_vec, b_vec);
    }

    /* Horizontal sum for dot */
    float32x2_t dot_lo = vget_low_f32(dot_vec);
    float32x2_t dot_hi = vget_high_f32(dot_vec);
    float32x2_t dot = vpadd_f32(dot_lo, dot_hi);
    float dot_scalar = vget_lane_f32(vpadd_f32(dot, dot), 0);

    /* Horizontal sum for norm_a */
    float32x2_t na_lo = vget_low_f32(norm_a_vec);
    float32x2_t na_hi = vget_high_f32(norm_a_vec);
    float32x2_t na = vpadd_f32(na_lo, na_hi);
    float norm_a_scalar = vget_lane_f32(vpadd_f32(na, na), 0);

    /* Horizontal sum for norm_b */
    float32x2_t nb_lo = vget_low_f32(norm_b_vec);
    float32x2_t nb_hi = vget_high_f32(norm_b_vec);
    float32x2_t nb = vpadd_f32(nb_lo, nb_hi);
    float norm_b_scalar = vget_lane_f32(vpadd_f32(nb, nb), 0);

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

/* ===== Quantized int8/uint8 variants (NEON) ===== */

/* NEON int8 dot product */
int32_t vek_dot_i8_neon(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16; /* 128-bit = 16 int8 */
    size_t i = 0;

    int32x4_t sum_vec = vdupq_n_s32(0);

    for (; i + simd_width <= n; i += simd_width) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);

        /* Convert to 16-bit, multiply, accumulate to 32-bit */
        int16x8_t a_lo = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_hi = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_lo = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t b_hi = vmovl_s8(vget_high_s8(b_vec));

        int32x4_t prod_lo = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
        int32x4_t prod_hi = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
        int32x4_t prod_lo2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
        int32x4_t prod_hi2 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));

        sum_vec = vaddq_s32(sum_vec, prod_lo);
        sum_vec = vaddq_s32(sum_vec, prod_hi);
        sum_vec = vaddq_s32(sum_vec, prod_lo2);
        sum_vec = vaddq_s32(sum_vec, prod_hi2);
    }

    /* Horizontal sum */
    int32x2_t sum_lo = vget_low_s32(sum_vec);
    int32x2_t sum_hi = vget_high_s32(sum_vec);
    int32x2_t sum = vpadd_s32(sum_lo, sum_hi);
    int32_t sum_scalar = vget_lane_s32(vpadd_s32(sum, sum), 0);

    /* Tail */
    for (; i < n; i++) {
        sum_scalar += (int32_t)a[i] * (int32_t)b[i];
    }

    return sum_scalar;
}

/* NEON uint8 dot product */
uint32_t vek_dot_u8_neon(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    uint32x4_t sum_vec = vdupq_n_u32(0);

    for (; i + simd_width <= n; i += simd_width) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);

        uint16x8_t a_lo = vmovl_u8(vget_low_u8(a_vec));
        uint16x8_t a_hi = vmovl_u8(vget_high_u8(a_vec));
        uint16x8_t b_lo = vmovl_u8(vget_low_u8(b_vec));
        uint16x8_t b_hi = vmovl_u8(vget_high_u8(b_vec));

        uint32x4_t prod_lo = vmull_u16(vget_low_u16(a_lo), vget_low_u16(b_lo));
        uint32x4_t prod_hi = vmull_u16(vget_high_u16(a_lo), vget_high_u16(b_lo));
        uint32x4_t prod_lo2 = vmull_u16(vget_low_u16(a_hi), vget_low_u16(b_hi));
        uint32x4_t prod_hi2 = vmull_u16(vget_high_u16(a_hi), vget_high_u16(b_hi));

        sum_vec = vaddq_u32(sum_vec, prod_lo);
        sum_vec = vaddq_u32(sum_vec, prod_hi);
        sum_vec = vaddq_u32(sum_vec, prod_lo2);
        sum_vec = vaddq_u32(sum_vec, prod_hi2);
    }

    /* Horizontal sum */
    uint32x2_t sum_lo = vget_low_u32(sum_vec);
    uint32x2_t sum_hi = vget_high_u32(sum_vec);
    uint32x2_t sum = vpadd_u32(sum_lo, sum_hi);
    uint32_t sum_scalar = vget_lane_u32(vpadd_u32(sum, sum), 0);

    for (; i < n; i++) {
        sum_scalar += (uint32_t)a[i] * (uint32_t)b[i];
    }

    return sum_scalar;
}

/* NEON int8 squared L2 distance */
int32_t vek_l2sq_i8_neon(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    int32x4_t sum_vec = vdupq_n_s32(0);

    for (; i + simd_width <= n; i += simd_width) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);

        int16x8_t a_lo = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_hi = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_lo = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t b_hi = vmovl_s8(vget_high_s8(b_vec));

        int16x8_t diff_lo = vsubq_s16(a_lo, b_lo);
        int16x8_t diff_hi = vsubq_s16(a_hi, b_hi);

        int32x4_t sq_lo = vmull_s16(vget_low_s16(diff_lo), vget_low_s16(diff_lo));
        int32x4_t sq_hi = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        int32x4_t sq_lo2 = vmull_s16(vget_low_s16(diff_hi), vget_low_s16(diff_hi));
        int32x4_t sq_hi2 = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(diff_hi));

        sum_vec = vaddq_s32(sum_vec, sq_lo);
        sum_vec = vaddq_s32(sum_vec, sq_hi);
        sum_vec = vaddq_s32(sum_vec, sq_lo2);
        sum_vec = vaddq_s32(sum_vec, sq_hi2);
    }

    /* Horizontal sum */
    int32x2_t sum_lo = vget_low_s32(sum_vec);
    int32x2_t sum_hi = vget_high_s32(sum_vec);
    int32x2_t sum = vpadd_s32(sum_lo, sum_hi);
    int32_t sum_scalar = vget_lane_s32(vpadd_s32(sum, sum), 0);

    for (; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum_scalar += diff * diff;
    }

    return sum_scalar;
}

/* NEON uint8 squared L2 distance */
uint32_t vek_l2sq_u8_neon(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    uint32x4_t sum_vec = vdupq_n_u32(0);

    for (; i + simd_width <= n; i += simd_width) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);

        /* Widen to u16, then reinterpret as s16 for signed subtraction.
         * u8 values [0,255] fit in both u16 and s16, so reinterpret is safe. */
        int16x8_t a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a_vec)));
        int16x8_t a_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a_vec)));
        int16x8_t b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_vec)));
        int16x8_t b_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_vec)));

        /* Signed subtraction: wraps correctly as signed, giving [-255, 255] */
        int16x8_t diff_lo = vsubq_s16(a_lo, b_lo);
        int16x8_t diff_hi = vsubq_s16(a_hi, b_hi);

        /* Signed widening multiply: (-255)^2 = 65025 fits in int32 */
        int32x4_t sq_lo  = vmull_s16(vget_low_s16(diff_lo),  vget_low_s16(diff_lo));
        int32x4_t sq_hi  = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(diff_lo));
        int32x4_t sq_lo2 = vmull_s16(vget_low_s16(diff_hi),  vget_low_s16(diff_hi));
        int32x4_t sq_hi2 = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(diff_hi));

        sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_lo));
        sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_hi));
        sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_lo2));
        sum_vec = vaddq_u32(sum_vec, vreinterpretq_u32_s32(sq_hi2));
    }

    uint32x2_t sum_lo = vget_low_u32(sum_vec);
    uint32x2_t sum_hi = vget_high_u32(sum_vec);
    uint32x2_t sum = vpadd_u32(sum_lo, sum_hi);
    uint32_t sum_scalar = vget_lane_u32(vpadd_u32(sum, sum), 0);

    for (; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum_scalar += (uint32_t)(diff * diff);
    }

    return sum_scalar;
}

/* NEON int8 cosine similarity */
float vek_cosine_i8_neon(const int8_t *a, const int8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    int32x4_t dot_vec = vdupq_n_s32(0);
    int32x4_t norm_a_vec = vdupq_n_s32(0);
    int32x4_t norm_b_vec = vdupq_n_s32(0);

    for (; i + simd_width <= n; i += simd_width) {
        int8x16_t a_vec = vld1q_s8(a + i);
        int8x16_t b_vec = vld1q_s8(b + i);

        /* dot product */
        int16x8_t a_lo = vmovl_s8(vget_low_s8(a_vec));
        int16x8_t a_hi = vmovl_s8(vget_high_s8(a_vec));
        int16x8_t b_lo = vmovl_s8(vget_low_s8(b_vec));
        int16x8_t b_hi = vmovl_s8(vget_high_s8(b_vec));

        int32x4_t dot_lo = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
        int32x4_t dot_hi = vmull_s16(vget_high_s16(a_lo), vget_high_s16(b_lo));
        int32x4_t dot_lo2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
        int32x4_t dot_hi2 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(b_hi));

        dot_vec = vaddq_s32(dot_vec, dot_lo);
        dot_vec = vaddq_s32(dot_vec, dot_hi);
        dot_vec = vaddq_s32(dot_vec, dot_lo2);
        dot_vec = vaddq_s32(dot_vec, dot_hi2);

        /* norm a */
        int32x4_t na_lo = vmull_s16(vget_low_s16(a_lo), vget_low_s16(a_lo));
        int32x4_t na_hi = vmull_s16(vget_high_s16(a_lo), vget_high_s16(a_lo));
        int32x4_t na_lo2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(a_hi));
        int32x4_t na_hi2 = vmull_s16(vget_high_s16(a_hi), vget_high_s16(a_hi));

        norm_a_vec = vaddq_s32(norm_a_vec, na_lo);
        norm_a_vec = vaddq_s32(norm_a_vec, na_hi);
        norm_a_vec = vaddq_s32(norm_a_vec, na_lo2);
        norm_a_vec = vaddq_s32(norm_a_vec, na_hi2);

        /* norm b */
        int32x4_t nb_lo = vmull_s16(vget_low_s16(b_lo), vget_low_s16(b_lo));
        int32x4_t nb_hi = vmull_s16(vget_high_s16(b_lo), vget_high_s16(b_lo));
        int32x4_t nb_lo2 = vmull_s16(vget_low_s16(b_hi), vget_low_s16(b_hi));
        int32x4_t nb_hi2 = vmull_s16(vget_high_s16(b_hi), vget_high_s16(b_hi));

        norm_b_vec = vaddq_s32(norm_b_vec, nb_lo);
        norm_b_vec = vaddq_s32(norm_b_vec, nb_hi);
        norm_b_vec = vaddq_s32(norm_b_vec, nb_lo2);
        norm_b_vec = vaddq_s32(norm_b_vec, nb_hi2);
    }

    /* Horizontal sums */
    int32x2_t dot_lo = vget_low_s32(dot_vec);
    int32x2_t dot_hi = vget_high_s32(dot_vec);
    int32x2_t dot = vpadd_s32(dot_lo, dot_hi);
    int32_t dot_scalar = vget_lane_s32(vpadd_s32(dot, dot), 0);

    int32x2_t na_lo = vget_low_s32(norm_a_vec);
    int32x2_t na_hi = vget_high_s32(norm_a_vec);
    int32x2_t na = vpadd_s32(na_lo, na_hi);
    int32_t norm_a_scalar = vget_lane_s32(vpadd_s32(na, na), 0);

    int32x2_t nb_lo = vget_low_s32(norm_b_vec);
    int32x2_t nb_hi = vget_high_s32(norm_b_vec);
    int32x2_t nb = vpadd_s32(nb_lo, nb_hi);
    int32_t norm_b_scalar = vget_lane_s32(vpadd_s32(nb, nb), 0);

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

/* NEON uint8 cosine similarity */
float vek_cosine_u8_neon(const uint8_t *a, const uint8_t *b, size_t n)
{
    const size_t simd_width = 16;
    size_t i = 0;

    uint32x4_t dot_vec = vdupq_n_u32(0);
    uint32x4_t norm_a_vec = vdupq_n_u32(0);
    uint32x4_t norm_b_vec = vdupq_n_u32(0);

    for (; i + simd_width <= n; i += simd_width) {
        uint8x16_t a_vec = vld1q_u8(a + i);
        uint8x16_t b_vec = vld1q_u8(b + i);

        /* dot product */
        uint16x8_t a_lo = vmovl_u8(vget_low_u8(a_vec));
        uint16x8_t a_hi = vmovl_u8(vget_high_u8(a_vec));
        uint16x8_t b_lo = vmovl_u8(vget_low_u8(b_vec));
        uint16x8_t b_hi = vmovl_u8(vget_high_u8(b_vec));

        uint32x4_t dot_lo = vmull_u16(vget_low_u16(a_lo), vget_low_u16(b_lo));
        uint32x4_t dot_hi = vmull_u16(vget_high_u16(a_lo), vget_high_u16(b_lo));
        uint32x4_t dot_lo2 = vmull_u16(vget_low_u16(a_hi), vget_low_u16(b_hi));
        uint32x4_t dot_hi2 = vmull_u16(vget_high_u16(a_hi), vget_high_u16(b_hi));

        dot_vec = vaddq_u32(dot_vec, dot_lo);
        dot_vec = vaddq_u32(dot_vec, dot_hi);
        dot_vec = vaddq_u32(dot_vec, dot_lo2);
        dot_vec = vaddq_u32(dot_vec, dot_hi2);

        /* norm a */
        uint32x4_t na_lo = vmull_u16(vget_low_u16(a_lo), vget_low_u16(a_lo));
        uint32x4_t na_hi = vmull_u16(vget_high_u16(a_lo), vget_high_u16(a_lo));
        uint32x4_t na_lo2 = vmull_u16(vget_low_u16(a_hi), vget_low_u16(a_hi));
        uint32x4_t na_hi2 = vmull_u16(vget_high_u16(a_hi), vget_high_u16(a_hi));

        norm_a_vec = vaddq_u32(norm_a_vec, na_lo);
        norm_a_vec = vaddq_u32(norm_a_vec, na_hi);
        norm_a_vec = vaddq_u32(norm_a_vec, na_lo2);
        norm_a_vec = vaddq_u32(norm_a_vec, na_hi2);

        /* norm b */
        uint32x4_t nb_lo = vmull_u16(vget_low_u16(b_lo), vget_low_u16(b_lo));
        uint32x4_t nb_hi = vmull_u16(vget_high_u16(b_lo), vget_high_u16(b_lo));
        uint32x4_t nb_lo2 = vmull_u16(vget_low_u16(b_hi), vget_low_u16(b_hi));
        uint32x4_t nb_hi2 = vmull_u16(vget_high_u16(b_hi), vget_high_u16(b_hi));

        norm_b_vec = vaddq_u32(norm_b_vec, nb_lo);
        norm_b_vec = vaddq_u32(norm_b_vec, nb_hi);
        norm_b_vec = vaddq_u32(norm_b_vec, nb_lo2);
        norm_b_vec = vaddq_u32(norm_b_vec, nb_hi2);
    }

    /* Horizontal sums */
    uint32x2_t dot_lo = vget_low_u32(dot_vec);
    uint32x2_t dot_hi = vget_high_u32(dot_vec);
    uint32x2_t dot = vpadd_u32(dot_lo, dot_hi);
    uint32_t dot_scalar = vget_lane_u32(vpadd_u32(dot, dot), 0);

    uint32x2_t na_lo = vget_low_u32(norm_a_vec);
    uint32x2_t na_hi = vget_high_u32(norm_a_vec);
    uint32x2_t na = vpadd_u32(na_lo, na_hi);
    uint32_t norm_a_scalar = vget_lane_u32(vpadd_u32(na, na), 0);

    uint32x2_t nb_lo = vget_low_u32(norm_b_vec);
    uint32x2_t nb_hi = vget_high_u32(norm_b_vec);
    uint32x2_t nb = vpadd_u32(nb_lo, nb_hi);
    uint32_t norm_b_scalar = vget_lane_u32(vpadd_u32(nb, nb), 0);

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
/* Binary (1-bit) variants (NEON) ===== */

int32_t vek_dot_b1_neon(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 2;
    size_t words = (n + 63) / 64;
    size_t i = 0;

    int32x4_t sum_vec = vdupq_n_s32(0);

    for (; i + simd_width < words; i += simd_width) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);

        uint64x2_t and_vec = vandq_u64(a_vec, b_vec);
        uint8x16_t popcnt = vcntq_u8(vreinterpretq_u8_u64(and_vec));
        uint16x8_t popcnt16 = vpaddlq_u8(popcnt);
        uint32x4_t popcnt32 = vpaddlq_u16(popcnt16);
        int32x4_t popcnt32s = vreinterpretq_s32_u32(popcnt32);

        sum_vec = vaddq_s32(sum_vec, popcnt32s);
    }

    /* Horizontal sum */
    int32x2_t sum_lo = vget_low_s32(sum_vec);
    int32x2_t sum_hi = vget_high_s32(sum_vec);
    int32x2_t sum = vpadd_s32(sum_lo, sum_hi);
    int32_t sum_scalar = vget_lane_s32(vpadd_s32(sum, sum), 0);

    /* Tail */
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (; i < words; i++) {
        uint64_t and_bits = a[i] & b[i];
        if (i == words - 1) and_bits &= mask; /* ignore padding bits past n */
        sum_scalar += vek_popcount64(and_bits);
    }

    return sum_scalar;
}

int32_t vek_hamming_b1_neon(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 2;
    size_t words = (n + 63) / 64;
    size_t i = 0;

    int32x4_t sum_vec = vdupq_n_s32(0);

    for (; i + simd_width < words; i += simd_width) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);

        uint64x2_t xor_vec = veorq_u64(a_vec, b_vec);
        uint8x16_t popcnt = vcntq_u8(vreinterpretq_u8_u64(xor_vec));
        uint16x8_t popcnt16 = vpaddlq_u8(popcnt);
        uint32x4_t popcnt32 = vpaddlq_u16(popcnt16);
        int32x4_t popcnt32s = vreinterpretq_s32_u32(popcnt32);

        sum_vec = vaddq_s32(sum_vec, popcnt32s);
    }

    /* Horizontal sum */
    int32x2_t sum_lo = vget_low_s32(sum_vec);
    int32x2_t sum_hi = vget_high_s32(sum_vec);
    int32x2_t sum = vpadd_s32(sum_lo, sum_hi);
    int32_t sum_scalar = vget_lane_s32(vpadd_s32(sum, sum), 0);

    /* Tail - XOR + popcount for hamming distance */
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (; i < words; i++) {
        uint64_t xor_bits = a[i] ^ b[i];
        if (i == words - 1) xor_bits &= mask; /* ignore padding bits past n */
        sum_scalar += vek_popcount64(xor_bits);
    }

    return sum_scalar;
}

float vek_cosine_b1_neon(const uint64_t *a, const uint64_t *b, size_t n)
{
    const size_t simd_width = 2;
    size_t words = (n + 63) / 64;
    size_t i = 0;

    int32x4_t dot_vec = vdupq_n_s32(0);
    int32x4_t norm_a_vec = vdupq_n_s32(0);
    int32x4_t norm_b_vec = vdupq_n_s32(0);

    for (; i + simd_width < words; i += simd_width) {
        uint64x2_t a_vec = vld1q_u64(a + i);
        uint64x2_t b_vec = vld1q_u64(b + i);

        uint64x2_t and_vec = vandq_u64(a_vec, b_vec);
        uint8x16_t dot_popcnt = vcntq_u8(vreinterpretq_u8_u64(and_vec));
        uint16x8_t dot_popcnt16 = vpaddlq_u8(dot_popcnt);
        uint32x4_t dot_popcnt32 = vpaddlq_u16(dot_popcnt16);
        int32x4_t dot_popcnt32s = vreinterpretq_s32_u32(dot_popcnt32);

        dot_vec = vaddq_s32(dot_vec, dot_popcnt32s);

        uint8x16_t a_popcnt = vcntq_u8(vreinterpretq_u8_u64(a_vec));
        uint16x8_t a_popcnt16 = vpaddlq_u8(a_popcnt);
        uint32x4_t a_popcnt32 = vpaddlq_u16(a_popcnt16);
        int32x4_t a_popcnt32s = vreinterpretq_s32_u32(a_popcnt32);

        norm_a_vec = vaddq_s32(norm_a_vec, a_popcnt32s);

        uint8x16_t b_popcnt = vcntq_u8(vreinterpretq_u8_u64(b_vec));
        uint16x8_t b_popcnt16 = vpaddlq_u8(b_popcnt);
        uint32x4_t b_popcnt32 = vpaddlq_u16(b_popcnt16);
        int32x4_t b_popcnt32s = vreinterpretq_s32_u32(b_popcnt32);

        norm_b_vec = vaddq_s32(norm_b_vec, b_popcnt32s);
    }

    /* Horizontal sums */
    int32x2_t dot_lo = vget_low_s32(dot_vec);
    int32x2_t dot_hi = vget_high_s32(dot_vec);
    int32x2_t dot = vpadd_s32(dot_lo, dot_hi);
    int32_t dot_scalar = vget_lane_s32(vpadd_s32(dot, dot), 0);

    int32x2_t na_lo = vget_low_s32(norm_a_vec);
    int32x2_t na_hi = vget_high_s32(norm_a_vec);
    int32x2_t na = vpadd_s32(na_lo, na_hi);
    int32_t norm_a_scalar = vget_lane_s32(vpadd_s32(na, na), 0);

    int32x2_t nb_lo = vget_low_s32(norm_b_vec);
    int32x2_t nb_hi = vget_high_s32(norm_b_vec);
    int32x2_t nb = vpadd_s32(nb_lo, nb_hi);
    int32_t norm_b_scalar = vget_lane_s32(vpadd_s32(nb, nb), 0);

    /* Tail - AND + popcount for dot, popcount for norms */
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (; i < words; i++) {
        uint64_t av = a[i], bv = b[i];
        if (i == words - 1) { av &= mask; bv &= mask; } /* ignore padding bits past n */
        uint64_t and_bits = av & bv;
        dot_scalar += vek_popcount64(and_bits);
        norm_a_scalar += vek_popcount64(av);
        norm_b_scalar += vek_popcount64(bv);
    }

    float norm_a_sqrt = sqrtf((float)norm_a_scalar);
    float norm_b_sqrt = sqrtf((float)norm_b_scalar);

    if (norm_a_sqrt == 0.0f || norm_b_sqrt == 0.0f) {
        return 0.0f;
    }

    return (float)dot_scalar / (norm_a_sqrt * norm_b_sqrt);
}

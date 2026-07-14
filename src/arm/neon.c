/**
 * vek - NEON intrinsics implementation for ARM64
 * NEON (128-bit) vectorized kernels for f32 dot, L2, cosine
 */

#include <arm_neon.h>
#include <stddef.h>
#include <math.h>
#include "vek.h"

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
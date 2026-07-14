/**
 * vek - Scalar reference implementations
 * Portable C11 reference implementations for correctness testing
 */

#include <stddef.h>
#include <math.h>
#include "vek.h"

/* Scalar dot product: sum(a[i] * b[i]) */
float vek_dot_f32_scalar(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

/* Scalar squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32_scalar(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

/* Scalar cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32_scalar(const float *a, const float *b, size_t n)
{
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < n; i++) {
        float ai = a[i];
        float bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    float norm_a_sqrt = sqrtf(norm_a);
    float norm_b_sqrt = sqrtf(norm_b);

    if (norm_a_sqrt == 0.0f || norm_b_sqrt == 0.0f) {
        return 0.0f;
    }

    return dot / (norm_a_sqrt * norm_b_sqrt);
}
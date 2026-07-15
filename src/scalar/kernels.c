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

/* ===== Quantized int8/uint8 scalar reference ===== */

int32_t vek_dot_i8_scalar(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int32_t)a[i] * (int32_t)b[i];
    }
    return sum;
}

uint32_t vek_dot_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (uint32_t)a[i] * (uint32_t)b[i];
    }
    return sum;
}

int32_t vek_l2sq_i8_scalar(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += diff * diff;
    }
    return sum;
}

uint32_t vek_l2sq_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint32_t)(diff * diff);
    }
    return sum;
}

float vek_cosine_i8_scalar(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t dot = 0;
    int32_t norm_a = 0;
    int32_t norm_b = 0;

    for (size_t i = 0; i < n; i++) {
        int32_t ai = a[i];
        int32_t bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if (norm_a == 0 || norm_b == 0) {
        return 0.0f;
    }

    return (float)dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b));
}

float vek_cosine_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint32_t dot = 0;
    uint32_t norm_a = 0;
    uint32_t norm_b = 0;

    for (size_t i = 0; i < n; i++) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    if (norm_a == 0 || norm_b == 0) {
        return 0.0f;
    }

    return (float)dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b));
}

/* Binary (1-bit) scalar reference */
int32_t vek_dot_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n)
{
    int32_t sum = 0;
    size_t words = (n + 63) / 64;
    for (size_t i = 0; i < words; i++) {
        uint64_t and_bits = a[i] & b[i];
        sum += __builtin_popcountll(and_bits);
    }
    return sum;
}

int32_t vek_hamming_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n)
{
    int32_t sum = 0;
    size_t words = (n + 63) / 64;
    for (size_t i = 0; i < words; i++) {
        sum += __builtin_popcountll(a[i] ^ b[i]);
    }
    return sum;
}

float vek_cosine_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n)
{
    int32_t dot = 0;
    int32_t norm_a = 0;
    int32_t norm_b = 0;
    size_t words = (n + 63) / 64;
    for (size_t i = 0; i < words; i++) {
        uint64_t and_bits = a[i] & b[i];
        dot += __builtin_popcountll(and_bits);
        norm_a += __builtin_popcountll(a[i]);
        norm_b += __builtin_popcountll(b[i]);
    }
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return (float)dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b));
}
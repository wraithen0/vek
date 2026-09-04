/**
 * vek - Scalar reference implementations
 * Portable C11 reference implementations for correctness testing
 */

#include <stddef.h>
#include <math.h>
#include <string.h>
#include "vek.h"
#include "../internal.h"

/* ===== f16/bf16 Conversion Helpers ===== */

/* f16: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits */
vek_f16 vek_f16_from_float(float val)
{
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  /* f32 exponent */
    uint32_t mantissa = bits & 0x7FFFFF;

    vek_f16 result;
    if (exp > 15) {
        /* Overflow -> INF */
        result = (vek_f16)((sign << 15) | 0x7C00);
    } else if (exp < -14) {
        /* Subnormal or zero */
        if (exp < -24) {
            result = (vek_f16)(sign << 15);  /* too small -> zero */
        } else {
            mantissa |= 0x800000;  /* add implicit 1 */
            result = (vek_f16)((sign << 15) | (mantissa >> (1 - exp - 15)));
        }
    } else {
        /* Normal range */
        uint32_t f16exp = (uint32_t)((exp + 15) & 0x1F);
        result = (vek_f16)((sign << 15) | (f16exp << 10) | (mantissa >> 13));
    }
    return result;
}

float vek_f16_to_float(vek_f16 val)
{
    uint32_t sign = (val >> 15) & 1;
    uint32_t exp = (val >> 10) & 0x1F;
    uint32_t mantissa = val & 0x3FF;
    uint32_t f32bits;

    if (exp == 0) {
        if (mantissa == 0) {
            f32bits = sign << 31;  /* zero */
        } else {
            /* Subnormal */
            exp = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            f32bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 0x1F) {
        f32bits = (sign << 31) | 0x7F800000 | (mantissa << 13);  /* INF/NaN */
    } else {
        f32bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32bits, sizeof(result));
    return result;
}

/* bf16: 1 sign bit, 8 exponent bits, 7 mantissa bits (truncated f32) */
vek_bf16 vek_bf16_from_float(float val)
{
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    return (vek_bf16)(bits >> 16);  /* truncate lower 16 bits */
}

float vek_bf16_to_float(vek_bf16 val)
{
    uint32_t f32bits = (uint32_t)val << 16;
    float result;
    memcpy(&result, &f32bits, sizeof(result));
    return result;
}

/* Scalar dot product: sum(a[i] * b[i]) with Neumaier compensated summation.
 * Falls back to naive summation for INF/NaN inputs to preserve IEEE 754 semantics. */
float vek_dot_f32_scalar(const float *a, const float *b, size_t n)
{
    float sum = 0.0f;
    float comp = 0.0f;  /* compensation for lost low-order bits */
    for (size_t i = 0; i < n; i++) {
        float prod = a[i] * b[i];
        /* Once sum becomes INF/NaN, compensation arithmetic breaks down.
         * Fall back to naive addition to preserve IEEE 754 semantics. */
        if (!isfinite(sum) || !isfinite(prod)) {
            sum += prod;
            continue;
        }
        float t = sum + prod;
        if (fabsf(sum) >= fabsf(prod))
            comp += (sum - t) + prod;
        else
            comp += (prod - t) + sum;
        sum = t;
    }
    return sum + comp;
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
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (int64_t)a[i] * (int64_t)b[i];
    }
    return (int32_t)sum;
}

uint32_t vek_dot_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        sum += (uint64_t)a[i] * (uint64_t)b[i];
    }
    return (uint32_t)sum;
}

int32_t vek_l2sq_i8_scalar(const int8_t *a, const int8_t *b, size_t n)
{
    int64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (int64_t)(diff * diff);
    }
    return (int32_t)sum;
}

uint32_t vek_l2sq_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < n; i++) {
        int32_t diff = (int32_t)a[i] - (int32_t)b[i];
        sum += (uint64_t)(diff * diff);
    }
    return (uint32_t)sum;
}

float vek_cosine_i8_scalar(const int8_t *a, const int8_t *b, size_t n)
{
    int64_t dot = 0;
    int64_t norm_a = 0;
    int64_t norm_b = 0;

    for (size_t i = 0; i < n; i++) {
        int32_t ai = a[i];
        int32_t bi = b[i];
        dot += (int64_t)ai * bi;
        norm_a += (int64_t)ai * ai;
        norm_b += (int64_t)bi * bi;
    }

    if (norm_a == 0 || norm_b == 0) {
        return 0.0f;
    }

    return (float)dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b));
}

float vek_cosine_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint64_t dot = 0;
    uint64_t norm_a = 0;
    uint64_t norm_b = 0;

    for (size_t i = 0; i < n; i++) {
        uint32_t ai = a[i];
        uint32_t bi = b[i];
        dot += (uint64_t)ai * bi;
        norm_a += (uint64_t)ai * ai;
        norm_b += (uint64_t)bi * bi;
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
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (size_t i = 0; i < words; i++) {
        uint64_t and_bits = a[i] & b[i];
        if (i == words - 1) and_bits &= mask; /* ignore padding bits past n */
        sum += vek_popcount64(and_bits);
    }
    return sum;
}

int32_t vek_hamming_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n)
{
    int32_t sum = 0;
    size_t words = (n + 63) / 64;
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (size_t i = 0; i < words; i++) {
        uint64_t xor_bits = a[i] ^ b[i];
        if (i == words - 1) xor_bits &= mask; /* ignore padding bits past n */
        sum += vek_popcount64(xor_bits);
    }
    return sum;
}

float vek_cosine_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n)
{
    int32_t dot = 0;
    int32_t norm_a = 0;
    int32_t norm_b = 0;
    size_t words = (n + 63) / 64;
    uint64_t rem = n & 63;
    uint64_t mask = (rem == 0) ? ~0ULL : ((1ULL << rem) - 1ULL);
    for (size_t i = 0; i < words; i++) {
        uint64_t av = a[i], bv = b[i];
        if (i == words - 1) { av &= mask; bv &= mask; } /* ignore padding bits past n */
        uint64_t and_bits = av & bv;
        dot += vek_popcount64(and_bits);
        norm_a += vek_popcount64(av);
        norm_b += vek_popcount64(bv);
    }
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return (float)dot / (sqrtf((float)norm_a) * sqrtf((float)norm_b));
}

/* ===== f16 Scalar Kernels (convert to f32, then use f32 kernels) ===== */

float vek_dot_f16_scalar(const vek_f16 *a, const vek_f16 *b, size_t n)
{
    float sum = 0.0f;
    float comp = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float prod = vek_f16_to_float(a[i]) * vek_f16_to_float(b[i]);
        if (!isfinite(sum) || !isfinite(prod)) {
            sum += prod;
            continue;
        }
        float t = sum + prod;
        if (fabsf(sum) >= fabsf(prod))
            comp += (sum - t) + prod;
        else
            comp += (prod - t) + sum;
        sum = t;
    }
    return sum + comp;
}

float vek_l2sq_f16_scalar(const vek_f16 *a, const vek_f16 *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = vek_f16_to_float(a[i]) - vek_f16_to_float(b[i]);
        sum += diff * diff;
    }
    return sum;
}

float vek_cosine_f16_scalar(const vek_f16 *a, const vek_f16 *b, size_t n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ai = vek_f16_to_float(a[i]);
        float bi = vek_f16_to_float(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

/* ===== bf16 Scalar Kernels (convert to f32, then use f32 kernels) ===== */

float vek_dot_bf16_scalar(const vek_bf16 *a, const vek_bf16 *b, size_t n)
{
    float sum = 0.0f;
    float comp = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float prod = vek_bf16_to_float(a[i]) * vek_bf16_to_float(b[i]);
        if (!isfinite(sum) || !isfinite(prod)) {
            sum += prod;
            continue;
        }
        float t = sum + prod;
        if (fabsf(sum) >= fabsf(prod))
            comp += (sum - t) + prod;
        else
            comp += (prod - t) + sum;
        sum = t;
    }
    return sum + comp;
}

float vek_l2sq_bf16_scalar(const vek_bf16 *a, const vek_bf16 *b, size_t n)
{
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = vek_bf16_to_float(a[i]) - vek_bf16_to_float(b[i]);
        sum += diff * diff;
    }
    return sum;
}

float vek_cosine_bf16_scalar(const vek_bf16 *a, const vek_bf16 *b, size_t n)
{
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float ai = vek_bf16_to_float(a[i]);
        float bi = vek_bf16_to_float(b[i]);
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}
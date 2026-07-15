/**
 * vek - Vector operations library
 * Public C API header
 *
 * Provides high-performance vector operations (dot product, L2 distance,
 * cosine similarity) with runtime CPU feature detection and dispatch.
 *
 * Supported backends:
 *   - scalar (C11 reference)
 *   - sse2   (x86_64 baseline)
 *   - avx2   (x86_64 AVX2)
 *   - avx512 (x86_64 AVX-512F)
 *   - neon   (ARM64 NEON)
 *
 * Usage:
 *   vek_init();  // Optional: auto-called on first API call
 *   float dot = vek_dot_f32(a, b, n);
 *   float l2 = vek_l2sq_f32(a, b, n);
 *   float cos = vek_cosine_f32(a, b, n);
 *   const char *backend = vek_backend_name();
 */

#ifndef VEK_H
#define VEK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the library (optional - called automatically on first use)
 * Returns 0 on success, non-zero on error */
int vek_init(void);

/* Get name of the active backend: "scalar", "sse2", "avx2", "avx512", "neon" */
const char* vek_backend_name(void);

/* Dot product: sum(a[i] * b[i]) */
float vek_dot_f32(const float *a, const float *b, size_t n);

/* Squared L2 distance: sum((a[i] - b[i])^2) */
float vek_l2sq_f32(const float *a, const float *b, size_t n);

/* Cosine similarity: (a·b) / (||a|| * ||b||) */
float vek_cosine_f32(const float *a, const float *b, size_t n);

/* ===== Quantized int8/uint8 variants (v0.5) ===== */

/* int8 dot product: sum(a[i] * b[i]) - accumulator is int32 */
int32_t vek_dot_i8(const int8_t *a, const int8_t *b, size_t n);

/* uint8 dot product: sum(a[i] * b[i]) - accumulator is uint32 */
uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* int8 squared L2 distance: sum((a[i] - b[i])^2) - accumulator is int32 */
int32_t vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n);

/* uint8 squared L2 distance: sum((a[i] - b[i])^2) - accumulator is uint32 */
uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* int8 cosine similarity: (a·b) / (||a|| * ||b||) - returns float */
float vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n);

/* uint8 cosine similarity: (a·b) / (||a|| * ||b||) - returns float */
float vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n);

/* ===== Scalar reference declarations (for testing) ===== */
float vek_dot_f32_scalar(const float *a, const float *b, size_t n);
float vek_l2sq_f32_scalar(const float *a, const float *b, size_t n);
float vek_cosine_f32_scalar(const float *a, const float *b, size_t n);

int32_t vek_dot_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_dot_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);
int32_t vek_l2sq_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_l2sq_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);
float vek_cosine_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
float vek_cosine_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* VEK_H */
/**
 * @file vek.h
 * @brief Vector Engine Kernels — zero-dependency SIMD vector similarity library.
 *
 * vek provides high-performance dot product, L2 distance, and cosine similarity
 * for f32, int8, uint8, and 1-bit (binary) vectors. Features runtime CPU
 * dispatch to select the optimal backend (scalar, SSE2, AVX2, AVX-512, NEON)
 * at first call.
 *
 * @section quick_start Quick Start
 *
 * @code
 * #include "vek.h"
 *
 * int main() {
 *     float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
 *     float b[] = {0.5f, 1.5f, 2.5f, 3.5f};
 *     size_t n = 4;
 *
 *     vek_init();  // Optional — auto-called on first use (thread-safe)
 *
 *     float dot = vek_dot_f32(a, b, n);      // 28.0
 *     float l2  = vek_l2sq_f32(a, b, n);     // 1.0
 *     float cos = vek_cosine_f32(a, b, n);   // ≈ 0.997
 *
 *     printf("Backend: %s\\n", vek_backend_name());
 *     return 0;
 * }
 * @endcode
 *
 * @section backends Supported Backends
 *
 * | Architecture | Baseline | SIMD |
 * |---|---|---|
 * | x86_64 | SSE2 (always) | AVX2 (FMA), AVX-512F + VNNI + VPOPCNTDQ |
 * | ARM64 | scalar | NEON |
 *
 * @section threads Thread Safety
 *
 * vek uses `pthread_once` for lazy initialization. All functions are safe to
 * call from any thread without explicit @c vek_init(). The dispatch table is
 * read-only after initialization.
 *
 * @section overflow Overflow Limits
 *
 * | Type | Max n (all 127/255) | Max n (typical embeddings) |
 * |------|---------------------|---------------------------|
 * | int8 dot | ~133K elements | unlimited (128–4096 dims) |
 * | uint8 dot | ~66K elements | unlimited |
 * | int8 l2sq | ~266K elements | unlimited |
 * | uint8 l2sq | ~133K elements | unlimited |
 *
 * @section license License
 *
 * Dual-licensed under MIT OR Apache-2.0.
 */

#ifndef VEK_H
#define VEK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================== */
/* Initialization and Backend Info                                           */
/* ======================================================================== */

/**
 * @brief Initialize the library and detect CPU features.
 *
 * Optional — called automatically on first API call. Thread-safe via
 * `pthread_once`. Subsequent calls are no-ops.
 *
 * @return 0 on success, non-zero on error.
 *
 * @note Safe to call from multiple threads concurrently.
 * @note After this call, @c vek_backend_name() returns the optimal backend.
 */
int vek_init(void);

/**
 * @brief Get the name of the active SIMD backend.
 *
 * Returns one of: `"scalar"`, `"sse2"`, `"avx2"`, `"avx512"`, `"neon"`.
 * The backend is selected once at first API call based on CPU features.
 *
 * @return Pointer to a static string. Do not free.
 *
 * @code
 * printf("Using: %s\\n", vek_backend_name());  // e.g. "avx512"
 * @endcode
 */
const char* vek_backend_name(void);

/* ======================================================================== */
/* f32 Operations                                                           */
/* ======================================================================== */

/**
 * @name f32 Vector Operations
 * @{
 */

/**
 * @brief Dot product of two f32 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} a[i] \cdot b[i]\f$
 *
 * @param[in] a First input vector. Must have at least @p n elements.
 * @param[in] b Second input vector. Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The dot product as a single-precision float.
 *
 * @pre @p a and @p b must be valid, non-null pointers.
 * @pre @p n must be greater than 0.
 * @post No memory is allocated. Result is computed on the stack.
 *
 * @performance O(n) with SIMD vectorization. On AVX-512, processes 16 floats
 * per iteration with 4 accumulators for instruction-level parallelism.
 *
 * @code
 * float a[] = {1.0f, 2.0f, 3.0f};
 * float b[] = {4.0f, 5.0f, 6.0f};
 * float result = vek_dot_f32(a, b, 3);  // 32.0
 * @endcode
 */
float vek_dot_f32(const float *a, const float *b, size_t n);

/**
 * @brief Squared L2 (Euclidean) distance between two f32 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} (a[i] - b[i])^2\f$
 *
 * This is faster than L2 distance because it avoids the square root.
 * For ranking/sorting, squared L2 preserves order.
 *
 * @param[in] a First input vector. Must have at least @p n elements.
 * @param[in] b Second input vector. Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The squared L2 distance as a single-precision float.
 *
 * @pre @p a and @p b must be valid, non-null pointers.
 * @pre @p n must be greater than 0.
 *
 * @performance O(n) with SIMD vectorization.
 *
 * @code
 * float a[] = {0.0f, 0.0f};
 * float b[] = {3.0f, 4.0f};
 * float dist = vek_l2sq_f32(a, b, 2);  // 25.0
 * float l2 = sqrtf(dist);              // 5.0
 * @endcode
 */
float vek_l2sq_f32(const float *a, const float *b, size_t n);

/**
 * @brief Cosine similarity between two f32 vectors.
 *
 * Computes: \f$\frac{a \cdot b}{\|a\| \cdot \|b\|}\f$
 *
 * Returns a value in [-1.0, 1.0] where:
 * - 1.0 means identical direction
 * - 0.0 means orthogonal
 * - -1.0 means opposite direction
 *
 * @param[in] a First input vector. Must have at least @p n elements.
 * @param[in] b Second input vector. Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return Cosine similarity as a single-precision float in [-1.0, 1.0].
 *
 * @pre @p a and @p b must be valid, non-null pointers.
 * @pre @p n must be greater than 0.
 * @pre Neither vector should be zero-length (would cause division by zero).
 *
 * @performance O(n) with SIMD vectorization. Computes dot product, L2 norms,
 * then divides.
 *
 * @code
 * float a[] = {1.0f, 0.0f};
 * float b[] = {0.0f, 1.0f};
 * float cos = vek_cosine_f32(a, b, 2);  // 0.0 (orthogonal)
 * @endcode
 */
float vek_cosine_f32(const float *a, const float *b, size_t n);

/** @} */ /* end of f32 group */

/* ======================================================================== */
/* Quantized int8/uint8 Operations                                          */
/* ======================================================================== */

/**
 * @name Quantized Integer Vector Operations
 * @{
 */

/**
 * @brief Dot product of two int8 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} a[i] \cdot b[i]\f$
 *
 * Accumulator is int32. For max-magnitude inputs (all 127 or all -128),
 * overflow occurs at ~133K elements. Typical embedding dimensions (128–4096)
 * never approach this limit.
 *
 * @param[in] a First input vector (int8). Must have at least @p n elements.
 * @param[in] b Second input vector (int8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The dot product as int32.
 *
 * @warning For n > 133K with max-magnitude inputs, overflow may occur.
 *
 * @code
 * int8_t a[] = {1, 2, 3};
 * int8_t b[] = {4, 5, 6};
 * int32_t result = vek_dot_i8(a, b, 3);  // 32
 * @endcode
 */
int32_t vek_dot_i8(const int8_t *a, const int8_t *b, size_t n);

/**
 * @brief Dot product of two uint8 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} a[i] \cdot b[i]\f$
 *
 * Accumulator is uint32. For max-magnitude inputs (all 255), overflow
 * occurs at ~66K elements.
 *
 * @param[in] a First input vector (uint8). Must have at least @p n elements.
 * @param[in] b Second input vector (uint8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The dot product as uint32.
 *
 * @warning For n > 66K with all-255 inputs, overflow may occur.
 */
uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n);

/**
 * @brief Squared L2 distance between two int8 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} (a[i] - b[i])^2\f$
 *
 * Accumulator is int32. For max-magnitude inputs (all 127 vs all -128),
 * overflow occurs at ~266K elements.
 *
 * @param[in] a First input vector (int8). Must have at least @p n elements.
 * @param[in] b Second input vector (int8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The squared L2 distance as int32.
 */
int32_t vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n);

/**
 * @brief Squared L2 distance between two uint8 vectors.
 *
 * Computes: \f$\sum_{i=0}^{n-1} (a[i] - b[i])^2\f$
 *
 * Accumulator is uint32. For max-magnitude inputs (all 255 vs all 0),
 * overflow occurs at ~133K elements.
 *
 * @param[in] a First input vector (uint8). Must have at least @p n elements.
 * @param[in] b Second input vector (uint8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return The squared L2 distance as uint32.
 */
uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n);

/**
 * @brief Cosine similarity between two int8 vectors.
 *
 * Computes: \f$\frac{a \cdot b}{\|a\| \cdot \|b\|}\f$
 *
 * Returns a value in [-1.0, 1.0]. Overflow in the dot accumulator affects
 * the result above ~133K elements with max-magnitude inputs.
 *
 * @param[in] a First input vector (int8). Must have at least @p n elements.
 * @param[in] b Second input vector (int8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return Cosine similarity as float in [-1.0, 1.0].
 */
float vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n);

/**
 * @brief Cosine similarity between two uint8 vectors.
 *
 * Computes: \f$\frac{a \cdot b}{\|a\| \cdot \|b\|}\f$
 *
 * Returns a value in [0.0, 1.0] (unsigned inputs). Cosine is correct for
 * uint8 at any n — no overflow in the cosine formula itself.
 *
 * @param[in] a First input vector (uint8). Must have at least @p n elements.
 * @param[in] b Second input vector (uint8). Must have at least @p n elements.
 * @param[in] n Number of elements. Must be > 0.
 *
 * @return Cosine similarity as float in [0.0, 1.0].
 */
float vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n);

/** @} */ /* end of quantized group */

/* ======================================================================== */
/* Binary (1-bit) Operations                                                */
/* ======================================================================== */

/**
 * @name Binary (1-bit) Vector Operations
 * @{
 *
 * Binary vectors are stored as arrays of @c uint64_t words. Each bit
 * represents a binary feature. The @p n parameter is the number of
 * @c uint64_t words (not bits).
 *
 * For a vector of @c B bits, use @c n = (B + 63) / 64 words.
 * Bits beyond @c B * 64 are ignored (masked to zero).
 */

/**
 * @brief Binary dot product (popcount of AND).
 *
 * Computes: \f$\text{popcount}(a \mathbin{\&} b)\f$
 *
 * Equivalent to counting the number of co-occurring set bits.
 *
 * @param[in] a First binary vector (array of uint64_t words).
 * @param[in] b Second binary vector (array of uint64_t words).
 * @param[in] n Number of uint64_t words. Must be > 0.
 *
 * @return Number of set bits where both vectors have 1.
 *
 * @performance Uses AVX-512 VPOPCNTDQ when available, otherwise scalar
 * popcount fallback.
 *
 * @code
 * uint64_t a[] = {0xFF};  // 8 bits set
 * uint64_t b[] = {0x0F};  // 4 bits set
 * int32_t dot = vek_dot_b1(a, b, 1);  // 4
 * @endcode
 */
int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n);

/**
 * @brief Hamming distance between two binary vectors.
 *
 * Computes: \f$\text{popcat}(a \oplus b)\f$
 *
 * Counts the number of positions where the two vectors differ.
 *
 * @param[in] a First binary vector (array of uint64_t words).
 * @param[in] b Second binary vector (array of uint64_t words).
 * @param[in] n Number of uint64_t words. Must be > 0.
 *
 * @return Number of differing bit positions.
 */
int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n);

/**
 * @brief Cosine similarity for binary vectors.
 *
 * Computes: \f$\frac{\text{popcount}(a \& b)}{\sqrt{\text{popcount}(a) \cdot \text{popcount}(b)}}\f$
 *
 * Returns a value in [0.0, 1.0]. Returns 0.0 if either vector is all-zeros.
 *
 * @param[in] a First binary vector (array of uint64_t words).
 * @param[in] b Second binary vector (array of uint64_t words).
 * @param[in] n Number of uint64_t words. Must be > 0.
 *
 * @return Cosine similarity as float in [0.0, 1.0].
 *
 * @code
 * uint64_t a[] = {0xFF};  // 8 bits set
 * uint64_t b[] = {0xFF};  // 8 bits set
 * float cos = vek_cosine_b1(a, b, 1);  // 1.0
 * @endcode
 */
float vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n);

/** @} */ /* end of binary group */

/* ======================================================================== */
/* Scalar Reference Implementations (for testing)                            */
/* ======================================================================== */

/**
 * @name Scalar Reference Implementations
 * @{
 *
 * These are pure-C reference implementations. They are always compiled
 * and serve as the correctness baseline. They are slower than SIMD backends
 * and should not be used in production.
 */

float vek_dot_f32_scalar(const float *a, const float *b, size_t n);
float vek_l2sq_f32_scalar(const float *a, const float *b, size_t n);
float vek_cosine_f32_scalar(const float *a, const float *b, size_t n);

int32_t vek_dot_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_dot_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);
int32_t vek_l2sq_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
uint32_t vek_l2sq_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);
float vek_cosine_i8_scalar(const int8_t *a, const int8_t *b, size_t n);
float vek_cosine_u8_scalar(const uint8_t *a, const uint8_t *b, size_t n);

int32_t vek_dot_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n);
int32_t vek_hamming_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n);
float vek_cosine_b1_scalar(const uint64_t *a, const uint64_t *b, size_t n);

/** @} */ /* end of scalar group */

#ifdef __cplusplus
}
#endif

#endif /* VEK_H */

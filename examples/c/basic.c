/**
 * vek - C usage example
 * Basic vector similarity operations
 */

#include <vek.h>
#include <stdio.h>
#include <stdlib.h>

void print_float_array(const float *a, size_t n, const char *name) {
    printf("%s: [", name);
    for (size_t i = 0; i < n; i++) {
        printf("%s%.4f", i == 0 ? "" : ", ", a[i]);
    }
    printf("]\n");
}

void print_int8_array(const int8_t *a, size_t n, const char *name) {
    printf("%s: [", name);
    for (size_t i = 0; i < n; i++) {
        printf("%s%d", i == 0 ? "" : ", ", a[i]);
    }
    printf("]\n");
}

void print_uint8_array(const uint8_t *a, size_t n, const char *name) {
    printf("%s: [", name);
    for (size_t i = 0; i < n; i++) {
        printf("%s%u", i == 0 ? "" : ", ", a[i]);
    }
    printf("]\n");
}

void print_binary_array(const uint64_t *a, size_t n_bits, const char *name) {
    size_t n_words = (n_bits + 63) / 64;
    printf("%s: ", name);
    for (size_t w = 0; w < n_words; w++) {
        for (int b = 63; b >= 0; b--) {
            if (w * 64 + b < n_bits) {
                printf("%d", (a[w] >> b) & 1);
            }
        }
        if (w + 1 < n_words) printf(" ");
    }
    printf(" (%zu bits)\n", n_bits);
}

int main() {
    /* Initialize dispatch (optional; happens lazily on first API call) */
    vek_init();
    printf("Active backend: %s\n\n", vek_backend_name());

    /* ===== f32 vectors ===== */
    size_t n = 4;
    float a_f32[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_f32[] = {0.5f, 1.5f, 2.5f, 3.5f};

    print_float_array(a_f32, n, "a_f32");
    print_float_array(b_f32, n, "b_f32");

    float dot_f32 = vek_dot_f32(a_f32, b_f32, n);
    float l2sq_f32 = vek_l2sq_f32(a_f32, b_f32, n);
    float cos_f32 = vek_cosine_f32(a_f32, b_f32, n);

    printf("dot_f32   = %.6f\n", dot_f32);       // 1*0.5 + 2*1.5 + 3*2.5 + 4*3.5 = 30
    printf("l2sq_f32  = %.6f\n", l2sq_f32);     // sum((a-b)^2)
    printf("cos_f32   = %.6f\n\n", cos_f32);

    /* ===== int8 vectors ===== */
    int8_t a_i8[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int8_t b_i8[] = {2, 3, 4, 5, 6, 7, 8, 9};
    size_t n_i8 = 8;

    print_int8_array(a_i8, n_i8, "a_i8");
    print_int8_array(b_i8, n_i8, "b_i8");

    int32_t dot_i8 = vek_dot_i8(a_i8, b_i8, n_i8);
    int32_t l2sq_i8 = vek_l2sq_i8(a_i8, b_i8, n_i8);
    float cos_i8 = vek_cosine_i8(a_i8, b_i8, n_i8);

    printf("dot_i8   = %d\n", dot_i8);
    printf("l2sq_i8  = %d\n", l2sq_i8);
    printf("cos_i8   = %.6f\n\n", cos_i8);

    /* ===== uint8 vectors ===== */
    uint8_t a_u8[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint8_t b_u8[] = {15, 25, 35, 45, 55, 65, 75, 85};
    size_t n_u8 = 8;

    print_uint8_array(a_u8, n_u8, "a_u8");
    print_uint8_array(b_u8, n_u8, "b_u8");

    uint32_t dot_u8 = vek_dot_u8(a_u8, b_u8, n_u8);
    uint32_t l2sq_u8 = vek_l2sq_u8(a_u8, b_u8, n_u8);
    float cos_u8 = vek_cosine_u8(a_u8, b_u8, n_u8);

    printf("dot_u8   = %u\n", dot_u8);
    printf("l2sq_u8  = %u\n", l2sq_u8);
    printf("cos_u8   = %.6f\n\n", cos_u8);

    /* ===== Binary (1-bit) vectors ===== */
    /* Pack 128 bits into two uint64_t */
    uint64_t a_b1[2] = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL};  // 1010... 0101...
    uint64_t b_b1[2] = {0xAAAAAAAAAAAAAAAAULL, 0x5555555555555555ULL};
    size_t n_b1 = 128;

    print_binary_array(a_b1, n_b1, "a_b1");
    print_binary_array(b_b1, n_b1, "b_b1");

    int32_t dot_b1 = vek_dot_b1(a_b1, b_b1, n_b1);
    int32_t ham_b1 = vek_hamming_b1(a_b1, b_b1, n_b1);
    float cos_b1 = vek_cosine_b1(a_b1, b_b1, n_b1);

    printf("dot_b1   = %d (matching 1-bits)\n", dot_b1);
    printf("ham_b1   = %d (differing bits)\n", ham_b1);
    printf("cos_b1   = %.6f\n\n", cos_b1);

    /* Test opposite vectors */
    uint64_t c_b1[2] = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    uint64_t d_b1[2] = {0x0000000000000000ULL, 0x0000000000000000ULL};
    print_binary_array(c_b1, 128, "c_b1 (all 1s)");
    print_binary_array(d_b1, 128, "d_b1 (all 0s)");

    int32_t dot_b1_opp = vek_dot_b1(c_b1, d_b1, 128);
    int32_t ham_b1_opp = vek_hamming_b1(c_b1, d_b1, 128);
    float cos_b1_opp = vek_cosine_b1(c_b1, d_b1, 128);

    printf("dot_b1   = %d\n", dot_b1_opp);  // 0 (no matching 1s)
    printf("ham_b1   = %d\n", ham_b1_opp);  // 128 (all differ)
    printf("cos_b1   = %.6f (zero norm -> 0)\n\n", cos_b1_opp);

    printf("Done.\n");
    return 0;
}
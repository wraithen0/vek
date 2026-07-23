/**
 * vek - Internal portable helpers
 * Not part of the public API.
 */

#ifndef VEK_INTERNAL_H
#define VEK_INTERNAL_H

#include <stdint.h>

static inline int vek_popcount64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER)
    return (int)__popcnt64(x);
#else
    /* Standard bit-manipulation fallback */
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (int)((x * 0x0101010101010101ULL) >> 56);
#endif
}

#endif /* VEK_INTERNAL_H */

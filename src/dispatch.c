/**
 * vek - CPU feature detection and dispatch
 * Runtime CPU feature detection with function pointer dispatch table
 */

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include "vek.h"

/* Function pointer types */
typedef float (*vek_dot_f32_fn)(const float*, const float*, size_t);
typedef float (*vek_l2sq_f32_fn)(const float*, const float*, size_t);
typedef float (*vek_cosine_f32_fn)(const float*, const float*, size_t);
typedef int32_t (*vek_dot_i8_fn)(const int8_t*, const int8_t*, size_t);
typedef uint32_t (*vek_dot_u8_fn)(const uint8_t*, const uint8_t*, size_t);
typedef int32_t (*vek_l2sq_i8_fn)(const int8_t*, const int8_t*, size_t);
typedef uint32_t (*vek_l2sq_u8_fn)(const uint8_t*, const uint8_t*, size_t);
typedef float (*vek_cosine_i8_fn)(const int8_t*, const int8_t*, size_t);
typedef float (*vek_cosine_u8_fn)(const uint8_t*, const uint8_t*, size_t);
typedef int32_t (*vek_dot_b1_fn)(const uint64_t*, const uint64_t*, size_t);
typedef int32_t (*vek_hamming_b1_fn)(const uint64_t*, const uint64_t*, size_t);
typedef float (*vek_cosine_b1_fn)(const uint64_t*, const uint64_t*, size_t);

/* Dispatch table */
static struct {
    vek_dot_f32_fn    dot_f32;
    vek_l2sq_f32_fn   l2sq_f32;
    vek_cosine_f32_fn cosine_f32;
    vek_dot_i8_fn     dot_i8;
    vek_dot_u8_fn     dot_u8;
    vek_l2sq_i8_fn    l2sq_i8;
    vek_l2sq_u8_fn    l2sq_u8;
    vek_cosine_i8_fn  cosine_i8;
    vek_cosine_u8_fn  cosine_u8;
    vek_dot_b1_fn     dot_b1;
    vek_hamming_b1_fn hamming_b1;
    vek_cosine_b1_fn  cosine_b1;
    const char*       name;
    int               initialized;
} g_dispatch = {0};

/* Thread-safe initialization */
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

/* Forward declarations for static functions */
static void dispatch_init_scalar(void);
static void dispatch_init_sse2(void);
static void dispatch_init_avx2(void);
static void dispatch_init_avx512(void);
#if defined(__aarch64__) || defined(_M_ARM64)
static void dispatch_init_neon(void);
#endif

static int cpu_has_avx2_runtime(void);
static int cpu_has_avx512f_runtime(void);

static void dispatch_init(void)
{
    /* Default to scalar */
    dispatch_init_scalar();

#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_avx512f_runtime()) {
#ifdef VEK_HAVE_AVX512
        dispatch_init_avx512();
#else
        dispatch_init_avx2();
#endif
    }
    else if (cpu_has_avx2_runtime()) {
        dispatch_init_avx2();
    }
    else {
        dispatch_init_sse2();
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (cpu_has_neon()) {
        dispatch_init_neon();
    }
#endif

    g_dispatch.initialized = 1;
}

/* Forward declarations for backend functions */
float vek_dot_f32_scalar(const float*, const float*, size_t);
float vek_l2sq_f32_scalar(const float*, const float*, size_t);
float vek_cosine_f32_scalar(const float*, const float*, size_t);

/* Quantized scalar reference */
int32_t vek_dot_i8_scalar(const int8_t*, const int8_t*, size_t);
uint32_t vek_dot_u8_scalar(const uint8_t*, const uint8_t*, size_t);
int32_t vek_l2sq_i8_scalar(const int8_t*, const int8_t*, size_t);
uint32_t vek_l2sq_u8_scalar(const uint8_t*, const uint8_t*, size_t);
float vek_cosine_i8_scalar(const int8_t*, const int8_t*, size_t);
float vek_cosine_u8_scalar(const uint8_t*, const uint8_t*, size_t);

/* Binary scalar reference */
int32_t vek_dot_b1_scalar(const uint64_t*, const uint64_t*, size_t);
int32_t vek_hamming_b1_scalar(const uint64_t*, const uint64_t*, size_t);
float vek_cosine_b1_scalar(const uint64_t*, const uint64_t*, size_t);

/* SSE2 backend (x86_64 baseline) */
#if defined(__x86_64__) || defined(_M_X64)
float vek_dot_f32_sse2(const float*, const float*, size_t);
float vek_l2sq_f32_sse2(const float*, const float*, size_t);
float vek_cosine_f32_sse2(const float*, const float*, size_t);

/* SSE2 quantized */
int32_t vek_dot_i8_sse2(const int8_t*, const int8_t*, size_t);
uint32_t vek_dot_u8_sse2(const uint8_t*, const uint8_t*, size_t);
int32_t vek_l2sq_i8_sse2(const int8_t*, const int8_t*, size_t);
uint32_t vek_l2sq_u8_sse2(const uint8_t*, const uint8_t*, size_t);
float vek_cosine_i8_sse2(const int8_t*, const int8_t*, size_t);
float vek_cosine_u8_sse2(const uint8_t*, const uint8_t*, size_t);

/* AVX2 backend */
float vek_dot_f32_avx2(const float*, const float*, size_t);
float vek_l2sq_f32_avx2(const float*, const float*, size_t);
float vek_cosine_f32_avx2(const float*, const float*, size_t);

/* AVX2 quantized */
int32_t vek_dot_i8_avx2(const int8_t*, const int8_t*, size_t);
uint32_t vek_dot_u8_avx2(const uint8_t*, const uint8_t*, size_t);
int32_t vek_l2sq_i8_avx2(const int8_t*, const int8_t*, size_t);
uint32_t vek_l2sq_u8_avx2(const uint8_t*, const uint8_t*, size_t);
float vek_cosine_i8_avx2(const int8_t*, const int8_t*, size_t);
float vek_cosine_u8_avx2(const uint8_t*, const uint8_t*, size_t);

/* AVX-512 backend - only if compiled in */
#ifdef VEK_HAVE_AVX512
float vek_dot_f32_avx512(const float*, const float*, size_t);
float vek_l2sq_f32_avx512(const float*, const float*, size_t);
float vek_cosine_f32_avx512(const float*, const float*, size_t);

/* AVX-512 quantized (VNNI) */
int32_t vek_dot_i8_avx512(const int8_t*, const int8_t*, size_t);
uint32_t vek_dot_u8_avx512(const uint8_t*, const uint8_t*, size_t);
int32_t vek_l2sq_i8_avx512(const int8_t*, const int8_t*, size_t);
uint32_t vek_l2sq_u8_avx512(const uint8_t*, const uint8_t*, size_t);
float vek_cosine_i8_avx512(const int8_t*, const int8_t*, size_t);
float vek_cosine_u8_avx512(const uint8_t*, const uint8_t*, size_t);

/* Binary (1-bit) AVX-512 */
int32_t vek_dot_b1_avx512(const uint64_t*, const uint64_t*, size_t);
int32_t vek_hamming_b1_avx512(const uint64_t*, const uint64_t*, size_t);
float vek_cosine_b1_avx512(const uint64_t*, const uint64_t*, size_t);
#endif
#endif

/* NEON backend (ARM64) */
#if defined(__aarch64__) || defined(_M_ARM64)
float vek_dot_f32_neon(const float*, const float*, size_t);
float vek_l2sq_f32_neon(const float*, const float*, size_t);
float vek_cosine_f32_neon(const float*, const float*, size_t);

/* NEON quantized */
int32_t vek_dot_i8_neon(const int8_t*, const int8_t*, size_t);
uint32_t vek_dot_u8_neon(const uint8_t*, const uint8_t*, size_t);
int32_t vek_l2sq_i8_neon(const int8_t*, const int8_t*, size_t);
uint32_t vek_l2sq_u8_neon(const uint8_t*, const uint8_t*, size_t);
float vek_cosine_i8_neon(const int8_t*, const int8_t*, size_t);
float vek_cosine_u8_neon(const uint8_t*, const uint8_t*, size_t);
#endif

/* CPU feature detection */
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>

static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t *eax, uint32_t *ebx, uint32_t *ecx, uint32_t *edx)
{
#if defined(_MSC_VER)
    int cpuInfo[4];
    __cpuidex(cpuInfo, leaf, subleaf);
    *eax = cpuInfo[0];
    *ebx = cpuInfo[1];
    *ecx = cpuInfo[2];
    *edx = cpuInfo[3];
#else
    __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
#endif
}

static int cpu_has_avx2(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1u << 5)) != 0; /* AVX2 bit */
}

static int cpu_has_avx512f(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ebx & (1u << 16)) != 0; /* AVX-512F bit */
}

static int cpu_has_avx512vnni(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 11)) != 0; /* AVX-512 VNNI bit */
}

static int cpu_has_avx512vpopcntdq(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(7, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 14)) != 0; /* AVX-512 VPOPCNTDQ bit */
}

static int cpu_has_avx(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 28)) != 0; /* AVX bit */
}

static int cpu_has_osxsave(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 27)) != 0; /* OSXSAVE bit */
}

static int xgetbv_supported(void)
{
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    return (ecx & (1u << 27)) != 0; /* OSXSAVE */
}

static uint64_t xgetbv(void)
{
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
#endif
}

static int cpu_has_avx2_runtime(void)
{
    if (!cpu_has_osxsave() || !xgetbv_supported()) return 0;
    if (!cpu_has_avx() || !cpu_has_avx2()) return 0;
    /* Check FMA3 support (required by avx2.c's _mm256_fmadd_ps) */
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, &eax, &ebx, &ecx, &edx);
    if (!(ecx & (1u << 12))) return 0; /* FMA3 bit */
    uint64_t xcr0 = xgetbv();
    return (xcr0 & 0x6) == 0x6; /* XMM and YMM state saved */
}

static int cpu_has_avx512f_runtime(void)
{
    if (!cpu_has_osxsave() || !xgetbv_supported()) return 0;
    if (!cpu_has_avx() || !cpu_has_avx2() || !cpu_has_avx512f()) return 0;
    /* AVX-512 kernels use VNNI (dpbusd) and VPOPCNTDQ (vpopcntq) — require both */
    if (!cpu_has_avx512vnni() || !cpu_has_avx512vpopcntdq()) return 0;
    uint64_t xcr0 = xgetbv();
    return (xcr0 & 0xe6) == 0xe6; /* XMM, YMM, ZMM, opmask state saved */
}
#endif /* x86_64 */

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
static int cpu_has_neon(void)
{
    unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & HWCAP_ASIMD) != 0;
}
#elif defined(__APPLE__)
/* Apple Silicon always has NEON */
static int cpu_has_neon(void) { return 1; }
#elif defined(_MSC_VER)
/* Windows ARM64 - assume NEON present */
static int cpu_has_neon(void) { return 1; }
#else
static int cpu_has_neon(void) { return 0; }
#endif
#endif

/* Initialize dispatch table */
static void dispatch_init_scalar(void)
{
    g_dispatch.dot_f32    = vek_dot_f32_scalar;
    g_dispatch.l2sq_f32   = vek_l2sq_f32_scalar;
    g_dispatch.cosine_f32 = vek_cosine_f32_scalar;
    g_dispatch.dot_i8     = vek_dot_i8_scalar;
    g_dispatch.dot_u8     = vek_dot_u8_scalar;
    g_dispatch.l2sq_i8    = vek_l2sq_i8_scalar;
    g_dispatch.l2sq_u8    = vek_l2sq_u8_scalar;
    g_dispatch.cosine_i8  = vek_cosine_i8_scalar;
    g_dispatch.cosine_u8  = vek_cosine_u8_scalar;
    g_dispatch.dot_b1     = vek_dot_b1_scalar;
    g_dispatch.hamming_b1 = vek_hamming_b1_scalar;
    g_dispatch.cosine_b1  = vek_cosine_b1_scalar;
    g_dispatch.name       = "scalar";
}

#if defined(__x86_64__) || defined(_M_X64)
static void dispatch_init_sse2(void)
{
    g_dispatch.dot_f32    = vek_dot_f32_sse2;
    g_dispatch.l2sq_f32   = vek_l2sq_f32_sse2;
    g_dispatch.cosine_f32 = vek_cosine_f32_sse2;
    g_dispatch.dot_i8     = vek_dot_i8_sse2;
    g_dispatch.dot_u8     = vek_dot_u8_sse2;
    g_dispatch.l2sq_i8    = vek_l2sq_i8_sse2;
    g_dispatch.l2sq_u8    = vek_l2sq_u8_sse2;
    g_dispatch.cosine_i8  = vek_cosine_i8_sse2;
    g_dispatch.cosine_u8  = vek_cosine_u8_sse2;
    g_dispatch.dot_b1     = vek_dot_b1_scalar;
    g_dispatch.hamming_b1 = vek_hamming_b1_scalar;
    g_dispatch.cosine_b1  = vek_cosine_b1_scalar;
    g_dispatch.name       = "sse2";
}

static void dispatch_init_avx2(void)
{
    g_dispatch.dot_f32    = vek_dot_f32_avx2;
    g_dispatch.l2sq_f32   = vek_l2sq_f32_avx2;
    g_dispatch.cosine_f32 = vek_cosine_f32_avx2;
    g_dispatch.dot_i8     = vek_dot_i8_avx2;
    g_dispatch.dot_u8     = vek_dot_u8_avx2;
    g_dispatch.l2sq_i8    = vek_l2sq_i8_avx2;
    g_dispatch.l2sq_u8    = vek_l2sq_u8_avx2;
    g_dispatch.cosine_i8  = vek_cosine_i8_avx2;
    g_dispatch.cosine_u8  = vek_cosine_u8_avx2;
    g_dispatch.dot_b1     = vek_dot_b1_scalar;
    g_dispatch.hamming_b1 = vek_hamming_b1_scalar;
    g_dispatch.cosine_b1  = vek_cosine_b1_scalar;
    g_dispatch.name       = "avx2";
}

#ifdef VEK_HAVE_AVX512
static void dispatch_init_avx512(void)
{
    g_dispatch.dot_f32    = vek_dot_f32_avx512;
    g_dispatch.l2sq_f32   = vek_l2sq_f32_avx512;
    g_dispatch.cosine_f32 = vek_cosine_f32_avx512;
    g_dispatch.dot_i8     = vek_dot_i8_avx512;
    g_dispatch.dot_u8     = vek_dot_u8_avx512;
    g_dispatch.l2sq_i8    = vek_l2sq_i8_avx512;
    g_dispatch.l2sq_u8    = vek_l2sq_u8_avx512;
    g_dispatch.cosine_i8  = vek_cosine_i8_avx512;
    g_dispatch.cosine_u8  = vek_cosine_u8_avx512;
    g_dispatch.dot_b1     = vek_dot_b1_avx512;
    g_dispatch.hamming_b1 = vek_hamming_b1_avx512;
    g_dispatch.cosine_b1  = vek_cosine_b1_avx512;
    g_dispatch.name       = "avx512";
}
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
static void dispatch_init_neon(void)
{
    g_dispatch.dot_f32    = vek_dot_f32_neon;
    g_dispatch.l2sq_f32   = vek_l2sq_f32_neon;
    g_dispatch.cosine_f32 = vek_cosine_f32_neon;
    g_dispatch.dot_i8     = vek_dot_i8_neon;
    g_dispatch.dot_u8     = vek_dot_u8_neon;
    g_dispatch.l2sq_i8    = vek_l2sq_i8_neon;
    g_dispatch.l2sq_u8    = vek_l2sq_u8_neon;
    g_dispatch.cosine_i8  = vek_cosine_i8_neon;
    g_dispatch.cosine_u8  = vek_cosine_u8_neon;
    g_dispatch.name       = "neon";
}
#endif

int vek_init(void)
{
    return pthread_once(&g_init_once, dispatch_init);
}

const char* vek_backend_name(void)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.name;
}

float vek_dot_f32(const float *a, const float *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.dot_f32(a, b, n);
}

float vek_l2sq_f32(const float *a, const float *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.l2sq_f32(a, b, n);
}

float vek_cosine_f32(const float *a, const float *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.cosine_f32(a, b, n);
}

int32_t vek_dot_i8(const int8_t *a, const int8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.dot_i8(a, b, n);
}

uint32_t vek_dot_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.dot_u8(a, b, n);
}

int32_t vek_l2sq_i8(const int8_t *a, const int8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.l2sq_i8(a, b, n);
}

uint32_t vek_l2sq_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.l2sq_u8(a, b, n);
}

float vek_cosine_i8(const int8_t *a, const int8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.cosine_i8(a, b, n);
}

float vek_cosine_u8(const uint8_t *a, const uint8_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.cosine_u8(a, b, n);
}

int32_t vek_dot_b1(const uint64_t *a, const uint64_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.dot_b1(a, b, n);
}

int32_t vek_hamming_b1(const uint64_t *a, const uint64_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.hamming_b1(a, b, n);
}

float vek_cosine_b1(const uint64_t *a, const uint64_t *b, size_t n)
{
    if (!g_dispatch.initialized) {
        vek_init();
    }
    return g_dispatch.cosine_b1(a, b, n);
}
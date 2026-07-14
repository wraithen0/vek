# vek - Vector Engine Kernels
# Makefile for building the library and tests

CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra -pedantic -Iinclude -fPIC
LDFLAGS = -lm

# Architecture-specific flags
UNAME_M := $(shell uname -m)
UNAME_S := $(shell uname -s)

# Detect x86_64
ifeq ($(UNAME_M),x86_64)
    CFLAGS_X86 = -msse2 -mavx -mavx2 -mfma
    ifeq ($(UNAME_S),Linux)
        CFLAGS_AVX512 = -mavx512f -mavx512vl -mavx512bw -mavx512dq
    endif
endif

# Detect ARM64
ifeq ($(UNAME_M),aarch64)
    CFLAGS_ARM = -march=armv8-a+simd
endif

# Source files
SRC_SCALAR = src/scalar/kernels.c
SRC_DISPATCH = src/dispatch.c
SRC_X86_SSE2 = src/x86/sse2.c
SRC_X86_AVX2 = src/x86/avx2.c
SRC_X86_AVX512 = src/x86/avx512.c
SRC_ARM_NEON = src/arm/neon.c

# Test and bench sources
TEST_SRC = tests/test_kernels.c
BENCH_SRC = bench/bench_kernels.c

# Object files
OBJ_SCALAR = $(SRC_SCALAR:.c=.o)
OBJ_DISPATCH = $(SRC_DISPATCH:.c=.o)
OBJ_X86_SSE2 = $(SRC_X86_SSE2:.c=.o)
OBJ_X86_AVX2 = $(SRC_X86_AVX2:.c=.o)
OBJ_X86_AVX512 = $(SRC_X86_AVX512:.c=.o)
OBJ_ARM_NEON = $(SRC_ARM_NEON:.c=.o)

OBJ_ALL = $(OBJ_SCALAR) $(OBJ_DISPATCH)

ifeq ($(UNAME_M),x86_64)
    OBJ_ALL += $(OBJ_X86_SSE2) $(OBJ_X86_AVX2)
    ifneq ($(CFLAGS_AVX512),)
        OBJ_ALL += $(OBJ_X86_AVX512)
    endif
endif

ifeq ($(UNAME_M),aarch64)
    OBJ_ALL += $(OBJ_ARM_NEON)
endif

# Targets
TARGET_LIB = libvek.a
TARGET_TEST = test_kernels
TARGET_BENCH = bench_kernels

.PHONY: all clean test bench install

all: $(TARGET_LIB) $(TARGET_TEST) $(TARGET_BENCH)

$(TARGET_LIB): $(OBJ_ALL)
	ar rcs $@ $^

# Compile scalar (always)
$(OBJ_SCALAR): $(SRC_SCALAR) include/vek.h
	$(CC) $(CFLAGS) -c $< -o $@

# Compile dispatch (always)
$(OBJ_DISPATCH): $(SRC_DISPATCH) include/vek.h
	$(CC) $(CFLAGS) -c $< -o $@

# x86_64 targets
ifeq ($(UNAME_M),x86_64)
$(OBJ_X86_SSE2): $(SRC_X86_SSE2) include/vek.h
	$(CC) $(CFLAGS) $(CFLAGS_X86) -c $< -o $@

$(OBJ_X86_AVX2): $(SRC_X86_AVX2) include/vek.h
	$(CC) $(CFLAGS) $(CFLAGS_X86) -c $< -o $@

ifneq ($(CFLAGS_AVX512),)
$(OBJ_X86_AVX512): $(SRC_X86_AVX512) include/vek.h
	$(CC) $(CFLAGS) $(CFLAGS_X86) $(CFLAGS_AVX512) -c $< -o $@
endif
endif

# ARM64 targets
ifeq ($(UNAME_M),aarch64)
$(OBJ_ARM_NEON): $(SRC_ARM_NEON) include/vek.h
	$(CC) $(CFLAGS) $(CFLAGS_ARM) -c $< -o $@
endif

# Test executable
$(TARGET_TEST): $(TARGET_LIB) $(TEST_SRC)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) -L. -lvek $(LDFLAGS)

# Benchmark executable
$(TARGET_BENCH): $(TARGET_LIB) $(BENCH_SRC)
	$(CC) $(CFLAGS) -o $@ $(BENCH_SRC) -L. -lvek $(LDFLAGS)

test: $(TARGET_TEST)
	./$(TARGET_TEST)

bench: $(TARGET_BENCH)
	./$(TARGET_BENCH) 10000

clean:
	rm -f $(OBJ_ALL) $(TARGET_LIB) $(TARGET_TEST) $(TARGET_BENCH)

install: $(TARGET_LIB)
	install -d $(DESTDIR)/usr/local/include
	install -d $(DESTDIR)/usr/local/lib
	install include/vek.h $(DESTDIR)/usr/local/include/
	install $(TARGET_LIB) $(DESTDIR)/usr/local/lib/

# Build with clang instead
clang: CC=clang
clang: clean all

# Debug build
debug: CFLAGS += -g -O0 -fsanitize=address -fno-omit-frame-pointer
debug: LDFLAGS += -fsanitize=address
debug: clean all

# Check for warnings (treat as errors)
warn: CFLAGS += -Werror
warn: clean all
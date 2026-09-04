# vek - Vector Engine Kernels
# Single build pipeline: library, tests, benchmarks

CC = gcc
CFLAGS = -std=c11 -O3 -Wall -Wextra -pedantic -Iinclude -fPIC
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null)
UNAME_M := $(shell uname -m 2>/dev/null)

# Shared library extension per platform
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
    SHLIB_FLAGS = -dynamiclib
else ifeq ($(OS),Windows_NT)
    LIB_EXT = .dll
    SHLIB_FLAGS = -shared
    EXE_EXT = .exe
else
    LIB_EXT = .so
    SHLIB_FLAGS = -shared
endif

# SIMD flags per backend (x86_64)
ifneq (,$(filter $(UNAME_M),x86_64 AMD64))
    FLAGS_SSE2 = -msse2
    FLAGS_AVX2 = -msse2 -mavx -mavx2 -mfma
    ifneq (,$(filter $(UNAME_S),Linux))
        # AVX-512 only on Linux (most reliable support)
        FLAGS_AVX512 = -msse2 -mavx -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vpopcntdq
        CFLAGS += -DVEK_HAVE_AVX512=1
    endif
endif

# ARM64 NEON
ifneq (,$(filter $(UNAME_M),aarch64 arm64))
    FLAGS_NEON = -march=armv8-a+simd
endif

# Source files
SRC = src/scalar/kernels.c src/dispatch.c
ifneq (,$(filter $(UNAME_M),x86_64 AMD64))
    SRC += src/x86/sse2.c src/x86/avx2.c
    ifneq ($(FLAGS_AVX512),)
        SRC += src/x86/avx512.c
    endif
endif
ifneq (,$(filter $(UNAME_M),aarch64 arm64))
    SRC += src/arm/neon.c
endif

# Objects
OBJ = $(SRC:.c=.o)

# Targets
LIB_STATIC = libvek.a
LIB_SHARED = libvek$(LIB_EXT)
TEST = test_kernels$(EXE_EXT)
TEST_BACKENDS = test_backends$(EXE_EXT)
BENCH = bench_kernels$(EXE_EXT)

.PHONY: all lib test bench clean install

# Default: build library + tests
all: lib test

# --- Library (static + shared) ---
lib: $(LIB_STATIC) $(LIB_SHARED)

$(LIB_STATIC): $(OBJ)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJ)
	$(CC) $(SHLIB_FLAGS) -o $@ $^ $(LDFLAGS)

# --- Object files with per-file SIMD flags ---
src/scalar/kernels.o: src/scalar/kernels.c include/vek.h
	$(CC) $(CFLAGS) -c $< -o $@

src/dispatch.o: src/dispatch.c include/vek.h
	$(CC) $(CFLAGS) -c $< -o $@

src/x86/sse2.o: src/x86/sse2.c include/vek.h
	$(CC) $(CFLAGS) $(FLAGS_SSE2) -c $< -o $@

src/x86/avx2.o: src/x86/avx2.c include/vek.h
	$(CC) $(CFLAGS) $(FLAGS_AVX2) -c $< -o $@

src/x86/avx512.o: src/x86/avx512.c include/vek.h
	$(CC) $(CFLAGS) $(FLAGS_AVX512) -c $< -o $@

src/arm/neon.o: src/arm/neon.c include/vek.h
	$(CC) $(CFLAGS) $(FLAGS_NEON) -c $< -o $@

# --- Tests ---
# Use rpath so tests find libvek without LD_LIBRARY_PATH
RPATH = -Wl,-rpath,'$$ORIGIN'

test: $(TEST) $(TEST_BACKENDS) $(LIB_SHARED)
	./$(TEST)
	./$(TEST_BACKENDS)

$(TEST): tests/test_kernels.c $(LIB_STATIC)
	$(CC) $(CFLAGS) -o $@ $< -L. -lvek $(LDFLAGS) $(RPATH)

$(TEST_BACKENDS): tests/test_backends.c $(LIB_STATIC)
	$(CC) $(CFLAGS) -o $@ $< -L. -lvek $(LDFLAGS) $(RPATH)

# --- Benchmarks ---
bench: $(BENCH) $(LIB_SHARED)
	./$(BENCH) 10000

$(BENCH): bench/bench_kernels.c $(LIB_STATIC)
	$(CC) $(CFLAGS) -o $@ $< -L. -lvek $(LDFLAGS) $(RPATH)

# --- Install ---
PREFIX ?= /usr/local
install: $(LIB_STATIC) $(LIB_SHARED)
	install -d $(DESTDIR)$(PREFIX)/include
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install include/vek.h $(DESTDIR)$(PREFIX)/include/
	install $(LIB_STATIC) $(DESTDIR)$(PREFIX)/lib/
	install $(LIB_SHARED) $(DESTDIR)$(PREFIX)/lib/
	sed 's|@PREFIX@|$(PREFIX)|g' pkgconfig/vek.pc.in > $(DESTDIR)$(PREFIX)/lib/pkgconfig/vek.pc

# --- Clean ---
clean:
	rm -f $(OBJ) $(LIB_STATIC) $(LIB_SHARED) $(TEST) $(TEST_BACKENDS) $(BENCH)

# --- Debug build ---
debug: CFLAGS += -g -O0 -fsanitize=address -fno-omit-frame-pointer
debug: LDFLAGS += -fsanitize=address
debug: clean all

# --- Warnings as errors ---
warn: CFLAGS += -Werror
warn: clean all

# --- Clang build ---
clang: CC=clang
clang: clean all

cmake_minimum_required(VERSION 3.16)
project(vek VERSION 0.8.0 LANGUAGES C)

# Project metadata
set(VEK_VERSION_MAJOR 0)
set(VEK_VERSION_MINOR 8)
set(VEK_VERSION_PATCH 0)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Library target ---
add_library(vek_static STATIC
    src/scalar/kernels.c
    src/dispatch.c
)
target_include_directories(vek_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(vek_shared SHARED
    src/scalar/kernels.c
    src/dispatch.c
)
target_include_directories(vek_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Base compile options - MSVC needs different flags
if(MSVC)
    set(VEK_BASE_COMPILE_OPTIONS /W4)
else()
    set(VEK_BASE_COMPILE_OPTIONS
        -std=c11 -O3 -Wall -Wextra -pedantic -fPIC
    )
endif()

# Apply base options to both targets
target_compile_options(vek_static PRIVATE ${VEK_BASE_COMPILE_OPTIONS} -DVEK_BUILD_STATIC)
target_compile_options(vek_shared PRIVATE ${VEK_BASE_COMPILE_OPTIONS} -DVEK_BUILD_SHARED)

# RPATH for finding libvek.so at runtime
if(UNIX AND NOT APPLE)
    target_link_options(vek_shared PRIVATE -Wl,-rpath,'$ORIGIN/../lib')
elseif(APPLE)
    target_link_options(vek_shared PRIVATE -Wl,-rpath,'@loader_path/../lib')
endif()

# Alias for convenience
add_library(vek::vek ALIAS vek_shared)
add_library(vek::static ALIAS vek_static)

# --- Architecture-specific source files ---
# x86_64 backends
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_sources(vek_static PRIVATE
        src/x86/sse2.c
        src/x86/avx2.c
    )
    target_sources(vek_shared PRIVATE
        src/x86/sse2.c
        src/x86/avx2.c
    )

    # Per-file SIMD flags
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # SSE2 - baseline for x86_64
        set_source_files_properties(src/x86/sse2.c PROPERTIES COMPILE_FLAGS "-msse2")
        
        # AVX2 + FMA
        set_source_files_properties(src/x86/avx2.c PROPERTIES COMPILE_FLAGS "-msse2 -mavx -mavx2 -mfma")
        
        # AVX-512 (Linux only). Enabled by default: extra ISA object code is
        # harmless, and the runtime CPUID check in dispatch.c still gates
        # whether the AVX-512 path is *used*. Set -DVEK_BUILD_AVX512=OFF to
        # exclude it (e.g. on hosts without AVX-512 toolchain support).
        if(UNIX AND NOT APPLE)
            option(VEK_BUILD_AVX512 "Build AVX-512 kernels (requires host support)" ON)
            if(VEK_BUILD_AVX512)
                set_source_files_properties(src/x86/avx512.c PROPERTIES COMPILE_FLAGS "-msse2 -mavx -mavx2 -mfma -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vpopcntdq")
                target_sources(vek_static PRIVATE src/x86/avx512.c)
                target_sources(vek_shared PRIVATE src/x86/avx512.c)
                # Define macro for dispatch + tests to know AVX-512 is available
                target_compile_definitions(vek_static PRIVATE VEK_HAVE_AVX512=1)
                target_compile_definitions(vek_shared PRIVATE VEK_HAVE_AVX512=1)
            endif()
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set_source_files_properties(src/x86/sse2.c PROPERTIES COMPILE_FLAGS "/arch:SSE2")
        set_source_files_properties(src/x86/avx2.c PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_sources(vek_static PRIVATE src/arm/neon.c)
    target_sources(vek_shared PRIVATE src/arm/neon.c)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(src/arm/neon.c PROPERTIES COMPILE_FLAGS "-march=armv8-a+simd")
    elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set_source_files_properties(src/arm/neon.c PROPERTIES COMPILE_FLAGS "/arch:ARM64")
    endif()
endif()

# --- Tests ---
enable_testing()

add_executable(test_kernels tests/test_kernels.c)
target_link_libraries(test_kernels PRIVATE vek_shared m)
if(UNIX AND NOT APPLE)
    target_link_options(test_kernels PRIVATE -Wl,-rpath,'$ORIGIN/../lib')
elseif(APPLE)
    target_link_options(test_kernels PRIVATE -Wl,-rpath,'@loader_path/../lib')
endif()
add_test(NAME test_kernels COMMAND test_kernels)

add_executable(test_backends tests/test_backends.c)
target_link_libraries(test_backends PRIVATE vek_shared m)
if(UNIX AND NOT APPLE)
    target_link_options(test_backends PRIVATE -Wl,-rpath,'$ORIGIN/../lib')
elseif(APPLE)
    target_link_options(test_backends PRIVATE -Wl,-rpath,'@loader_path/../lib')
endif()
add_test(NAME test_backends COMMAND test_backends)

# --- Benchmarks ---
add_executable(bench_kernels bench/bench_kernels.c)
target_link_libraries(bench_kernels PRIVATE vek_shared m)
if(UNIX AND NOT APPLE)
    target_link_options(bench_kernels PRIVATE -Wl,-rpath,'$ORIGIN/../lib')
elseif(APPLE)
    target_link_options(bench_kernels PRIVATE -Wl,-rpath,'@loader_path/../lib')
endif()

# --- Install ---
install(TARGETS vek_static vek_shared
    EXPORT vek-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES include/vek.h DESTINATION include)
install(FILES LICENSE LICENSE-MIT LICENSE-APACHE DESTINATION share/licenses/vek)
install(FILES docs/man/vek.3 DESTINATION share/man/man3)

# Generate and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vek-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vek-config.cmake
    INSTALL_DESTINATION lib/cmake/vek
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vek-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT vek-targets
    FILE vek-targets.cmake
    NAMESPACE vek::
    DESTINATION lib/cmake/vek
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vek-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vek-config-version.cmake
    DESTINATION lib/cmake/vek
)

# --- pkg-config ---
set(VEK_VERSION ${PROJECT_VERSION})
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vek.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/vek.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vek.pc DESTINATION lib/pkgconfig)

# --- CPack ---
set(CPACK_PACKAGE_NAME "vek")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Vector Engine Kernels - SIMD vector similarity")
set(CPACK_PACKAGE_VENDOR "vek contributors")
set(CPACK_PACKAGE_LICENSE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)
cmake_minimum_required(VERSION 3.16)
project(vek VERSION 0.7.0 LANGUAGES C)

# Project metadata
set(VEK_VERSION_MAJOR 0)
set(VEK_VERSION_MINOR 7)
set(VEK_VERSION_PATCH 0)

# C standard
set(C_STANDARD 11)
set(C_STANDARD_REQUIRED ON)
set(C_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Platform-specific compile options
function(vek_add_simd_flags target)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # SSE2 baseline - always available on x86_64
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:C>: -msse2>)

        # AVX2 + FMA
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:C>: -mavx -mavx2 -mfma>
            )
        elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE /arch:AVX2)
        endif()

        # AVX-512 (Linux only for now, check at configure time)
        if(UNIX AND NOT APPLE)
            target_compile_options(${target} PRIVATE
                $<$<COMPILE_LANGUAGE:C>: -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vnni -mavx512vpopcntdq>
            )
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # NEON on ARM64
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C>: -march=armv8-a+simd>
        )
    endif()
endfunction()

# --- Library target ---
add_library(vek_static STATIC
    src/scalar/kernels.c
    src/dispatch.c
)
target_include_directories(vek_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(vek_static PRIVATE
    -std=c11 -O3 -Wall -Wextra -pedantic -fPIC
    $<$<COMPILE_LANGUAGE:C>: -DVEK_BUILD_STATIC>
)
vek_add_simd_flags(vek_static)

add_library(vek_shared SHARED
    src/scalar/kernels.c
    src/dispatch.c
)
target_include_directories(vek_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(vek_shared PRIVATE
    -std=c11 -O3 -Wall -Wextra -pedantic -fPIC
    $<$<COMPILE_LANGUAGE:C>: -DVEK_BUILD_SHARED>
)
vek_add_simd_flags(vek_shared)

# Alias for convenience
add_library(vek::vek ALIAS vek_shared)
add_library(vek::static ALIAS vek_static)

# --- Architecture-specific source files ---
# These are compiled into both static and shared libs automatically
# because we list them in both targets

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

    # AVX-512 (Linux only)
    if(UNIX AND NOT APPLE)
        target_sources(vek_static PRIVATE src/x86/avx512.c)
        target_sources(vek_shared PRIVATE src/x86/avx512.c)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_sources(vek_static PRIVATE src/arm/neon.c)
    target_sources(vek_shared PRIVATE src/arm/neon.c)
endif()

# --- Tests ---
enable_testing()

add_executable(test_kernels tests/test_kernels.c)
target_link_libraries(test_kernels PRIVATE vek_shared m)
add_test(NAME test_kernels COMMAND test_kernels)

add_executable(test_backends tests/test_backends.c)
target_link_libraries(test_backends PRIVATE vek_shared m)
add_test(NAME test_backends COMMAND test_backends)

# --- Benchmarks ---
add_executable(bench_kernels bench/bench_kernels.c)
target_link_libraries(bench_kernels PRIVATE vek_shared m)

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
cmake_minimum_required(VERSION 3.16)
project(resnet101_int8 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Compiler flags ────────────────────────────────────────────
# Detect architecture and enable appropriate SIMD
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "ARM64 detected — enabling NEON + dotprod + native tuning")
    set(ARCH_FLAGS "-march=native -O3 -ffast-math -funroll-loops")
    set(SIMD_SRC "src/ops_neon.cpp")
else()
    message(STATUS "x86_64 detected — enabling AVX-512 VNNI")
    set(ARCH_FLAGS "-march=native -O3 -ffast-math -mavx512f -mavx512bw -mavx512vl -mavx512vnni")
    set(SIMD_SRC "src/ops_avx512.cpp" "src/ops_avx2.cpp")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")

# ── Sources ───────────────────────────────────────────────────
add_executable(resnet101_int8
    src/benchmark_model.cpp
    ${SIMD_SRC}
    src/model.cpp
)

target_include_directories(resnet101_int8 PRIVATE src)

# ── OpenMP ────────────────────────────────────────────────────
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} found — multi-threaded GEMM enabled")
    target_link_libraries(resnet101_int8 PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found — single-threaded mode")
endif()

# ── Link ──────────────────────────────────────────────────────
# stdc++fs for std::filesystem on older compilers
# target_link_libraries(resnet101_int8 PRIVATE stdc++fs)

# ── Install ───────────────────────────────────────────────────
install(TARGETS resnet101_int8 DESTINATION bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")

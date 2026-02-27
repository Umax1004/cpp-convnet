#pragma once
#include <cstdint>

// ──────────────────────────────────────────────────────────────────────────────
// AVX2 (256-bit, no VNNI) GEMM + Conv kernels
//
// Weight packing layout: Co8 × K4 × 32 bytes
//   For each block of 8 output channels and block of 4 K elements:
//     packed[co_blk*K4*32 + k_blk*32 + ki*8 + oc] = w[(co_blk*8+oc)*K + k_blk*4+ki]
//   This lets the kernels load 8 consecutive B values per (k_blk, ki) with one
//   _mm_loadl_epi64, then convert to 8 int32 with _mm256_cvtepi8_epi32.
//
// Int8  kernel: _mm256_mullo_epi32 (full multiply)
// Ternary kernel: _mm256_sign_epi32 (conditional add/sub, no multiply)
//
// Both kernels use an 8×8 output tile (8 M-rows × 8 N-cols) accumulating
// 8 __m256i registers. No horizontal reduction required — each lane of
// acc[row] corresponds directly to one output channel.
// ──────────────────────────────────────────────────────────────────────────────

int8_t* pack_weights_avx2(const int8_t* w, int C_out, int K);
void    free_packed_avx2(int8_t* p);

void gemm_int8_avx2(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out    = false,
    bool           in_parallel = false
);

void gemm_ternary_avx2(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out    = false,
    bool           in_parallel = false
);

void conv2d_int8_avx2(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups = 1,
    int8_t* scratch_col = nullptr
);

void conv2d_ternary_avx2(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups = 1,
    int8_t* scratch_col = nullptr
);

void ops_avx2_profile_reset();
void ops_avx2_profile_print(int runs);

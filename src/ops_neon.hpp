#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

// ──────────────────────────────────────────────
// Requantize: int64 accumulator → int8
// ──────────────────────────────────────────────
inline int8_t requant(int64_t acc, float scale, int out_zp) {
    float val = static_cast<float>(acc) * scale;
    int32_t q = static_cast<int32_t>(std::roundf(val)) + out_zp;
    return static_cast<int8_t>(std::clamp(q, -128, 127));
}

// ──────────────────────────────────────────────
// Repack weights for SDOT: [C_out, K] → [C_out/4, K/4, 4, 4]
// Handles padding when C_out or K are not multiples of 4.
// Caller must free returned pointer (allocated with new[]).
// ──────────────────────────────────────────────
int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K);
void    free_packed(int8_t* p);

// ──────────────────────────────────────────────
// INT8 GEMM with NEON SDOT (for 1×1 conv / FC)
//   A: [M, K]  (int8, row-major)
//   B: packed  [N/4, K/4, 4, 4]   where N = C_out
//   C: [M, N]  (float32 output or int8 output depending on is_float)
// For int8 output:  out[m,n] = clamp(round(acc * req_scale[n]) + out_zp)
// For float output: out[m,n] = acc * req_scale[n]   (used for final FC)
// ──────────────────────────────────────────────
void gemm_int8_neon(
    const int8_t*  A,          // [M, K]
    const int8_t*  B_packed,   // packed [N/4, K/4, 4, 4]
    const int64_t* eff_bias,   // [N]
    const float*   req_scale,  // [N]
    int8_t         out_zp,
    void*          C,          // [M, N] row-major, or [N, M] if nchw_out
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out    = false, // write C[n*M+m] — avoids caller transpose
    bool           in_parallel = false  // true when already inside an omp parallel region
);

// ──────────────────────────────────────────────
// Direct int8 conv2d (NCHW layout).
// Handles any kernel size; uses im2col + GEMM internally.
// w_pre_packed: optional pre-packed weights (groups=1 only); pass nullptr to auto-pack.
// in_zp: input zero point used for padding (MUST match layer's in_zp).
// ──────────────────────────────────────────────
void conv2d_int8(
    const int8_t*  input,         // [1, C_in, H, W]
    const int8_t*  weight,        // [C_out, C_in, kH, kW]
    const int8_t*  w_pre_packed,  // pre-packed (may be nullptr)
    const int64_t* eff_bias,      // [C_out]
    const float*   req_scale,     // [C_out]
    int8_t         in_zp,         // input zero point (padding value)
    int8_t         out_zp,
    int8_t*        output,        // [1, C_out, oH, oW]  NCHW
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups = 1,
    int8_t* scratch_col = nullptr  // [oHW * K] pre-allocated (nullptr → internal alloc)
);

// Final FC layer: output is float32
void linear_int8_to_float(
    const int8_t*  input,      // [C_in]
    const int8_t*  weight,     // [C_out, C_in]
    const int64_t* eff_bias,   // [C_out]
    const float*   req_scale,  // [C_out]
    float*         output,     // [C_out]
    int C_in, int C_out
);

// ──────────────────────────────────────────────
// MaxPool int8 (NCHW)
// ──────────────────────────────────────────────
void maxpool_int8(
    const int8_t* input,   // [1, C, H, W]
    int8_t*       output,  // [1, C, oH, oW]
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
);

// ──────────────────────────────────────────────
// Global average pool int8 → int8 (NCHW)
// ──────────────────────────────────────────────
void avgpool_global_int8(
    const int8_t* input,       // [1, C, H, W]
    float         in_scale,
    int           in_zp,
    float         out_scale,
    int           out_zp,
    int8_t*       output,      // [C]
    int C, int H, int W
);

// ──────────────────────────────────────────────
// Op profiling (call reset before benchmark, print after)
// ──────────────────────────────────────────────
void ops_profile_reset();
void ops_profile_print(int runs);

// ──────────────────────────────────────────────
// Residual Add with requantization (NCHW)
// ──────────────────────────────────────────────
void add_requant_int8(
    const int8_t* in1,      // [C, H, W]
    const int8_t* in2,      // [C, H, W]
    float in1_scale, int in1_zp,
    float in2_scale, int in2_zp,
    float out_scale, int out_zp,
    int8_t*       output,   // [C, H, W]
    int N
);

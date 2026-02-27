#pragma once
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

inline int8_t requant(int64_t acc, float scale, int out_zp) {
    float val = static_cast<float>(acc) * scale;
    int32_t q = static_cast<int32_t>(std::roundf(val)) + out_zp;
    return static_cast<int8_t>(std::clamp(q, -128, 127));
}

int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K);
void    free_packed(int8_t* p);

void gemm_int8_avx512(
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

void conv2d_int8(
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,
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

void linear_int8_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out
);

void maxpool_int8(
    const int8_t* input,
    int8_t*       output,
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w
);

void avgpool_global_int8(
    const int8_t* input,
    float         in_scale,
    int           in_zp,
    float         out_scale,
    int           out_zp,
    int8_t*       output,
    int C, int H, int W
);

void ops_profile_reset();
void ops_profile_print(int runs);

void add_requant_int8(
    const int8_t* in1,
    const int8_t* in2,
    float in1_scale, int in1_zp,
    float in2_scale, int in2_zp,
    float out_scale, int out_zp,
    int8_t*       output,
    int N
);

int8_t* pack_weights_ternary(const int8_t* w, int C_out, int K);

void gemm_ternary_avx512(
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

void conv2d_ternary(
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,
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

void linear_ternary_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out
);

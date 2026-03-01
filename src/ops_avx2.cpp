#include "ops_avx2.hpp"
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

// ── Profiling ─────────────────────────────────────────────────────────────────
static double g_avx2_im2col_ms = 0.0;
static double g_avx2_gemm_ms   = 0.0;

static inline double now_ms_avx2() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

void ops_avx2_profile_reset() {
    g_avx2_im2col_ms = g_avx2_gemm_ms = 0.0;
}

void ops_avx2_profile_print(int runs) {
    double d = runs > 0 ? (double)runs : 1.0;
    printf("\n=== AVX2 Op Profiling (avg per image, %d runs) ===\n", runs);
    printf("  im2col  : %6.3f ms\n", g_avx2_im2col_ms / d);
    printf("  gemm    : %6.3f ms\n", g_avx2_gemm_ms   / d);
    printf("=================================================\n");
}

// ── Weight packing ────────────────────────────────────────────────────────────
// Layout: Co8 × K4 × 32 bytes
//   packed[co_blk*K4*32 + k_blk*32 + ki*8 + oc] = w[(co_blk*8+oc)*K + k_blk*4+ki]
//
// For each (k_blk, ki): 8 consecutive bytes hold one weight value per output
// channel — loaded in a single _mm_loadl_epi64 then extended with
// _mm256_cvtepi8_epi32 to produce the 8 int32 values for the output tile.
int8_t* pack_weights_avx2(const int8_t* w, int C_out, int K)
{
    int Co8 = (C_out + 7) / 8;
    int K4  = (K     + 3) / 4;
    int sz  = Co8 * K4 * 32;
    int8_t* packed = new int8_t[sz]();

    for (int co_blk = 0; co_blk < Co8; ++co_blk) {
        for (int k_blk = 0; k_blk < K4; ++k_blk) {
            int8_t* dst = packed + (co_blk * K4 + k_blk) * 32;
            for (int ki = 0; ki < 4; ++ki) {
                for (int oc = 0; oc < 8; ++oc) {
                    int co = co_blk * 8 + oc;
                    int k  = k_blk  * 4 + ki;
                    dst[ki * 8 + oc] = (co < C_out && k < K) ? w[co * K + k] : 0;
                }
            }
        }
    }
    return packed;
}

void free_packed_avx2(int8_t* p) { delete[] p; }

// Ternary packing: same Co8 × K4 × 32 layout as pack_weights_avx2.
// Values {-1, 0, +1} stored as raw int8 — no 2-bit compression.
// The kernel uses sign_epi32 instead of mullo_epi32, so there is zero
// decode overhead and the instruction count matches the int8 kernel exactly.
int8_t* pack_weights_ternary_avx2(const int8_t* w, int C_out, int K)
{
    return pack_weights_avx2(w, C_out, K);
}

// ── Im2col (identical logic to ops_avx512.cpp; static to this TU) ─────────────

static void im2col_1x1_gen_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int oW, int stride_h, int stride_w, int W)
{
    const int ih = oh * stride_h;
    for (int ow = 0; ow < oW; ++ow) {
        const int iw = ow * stride_w;
        int8_t* dst = col + (oh * oW + ow) * C_in;
        for (int c = 0; c < C_in; ++c)
            dst[c] = input[c * HW + ih * W + iw];
    }
}

static void im2col_1x1(
    const int8_t* input,
    int C_in, int H, int W,
    int stride_h, int stride_w,
    int oH, int oW,
    int8_t* col,
    bool in_parallel = false)
{
    const int HW = H * W;
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_1x1_gen_oh(input, col, oh, C_in, HW, oW, stride_h, stride_w, W);
#endif
}

static void im2col_3x3s1p1_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W)
{
    const int row0_off = (oh - 1) * W;
    for (int ow = 1; ow < W - 1; ++ow) {
        int8_t* dst = col + (oh * W + ow) * K;
        const int base_col = row0_off + (ow - 1);
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + base_col;
            uint32_t v0, v1, v2;
            memcpy(&v0, base,         4);
            memcpy(&v1, base + W,     4);
            memcpy(&v2, base + 2 * W, 4);
            uint64_t q = ((uint64_t)(v0 & 0xFFFFFFu))
                       | ((uint64_t)(v1 & 0xFFFFFFu) << 24)
                       | ((uint64_t)(v2 & 0xFFFFu)   << 48);
            memcpy(dst, &q, 8);
            dst[8] = (uint8_t)(v2 >> 16);
            dst += 9;
        }
    }
}

static void im2col_3x3s1p1_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int8_t pad_val)
{
    for (int ow = 0; ow < W; ++ow) {
        if (oh > 0 && oh < H - 1 && ow > 0 && ow < W - 1) continue;
        int8_t* dst = col + (oh * W + ow) * K;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh - 1 + kh;
                if (ih < 0 || ih >= H) {
                    dst[0] = dst[1] = dst[2] = pad_val;
                } else {
                    const int8_t* row = in_c + ih * W;
                    dst[0] = (ow > 0)     ? row[ow - 1] : pad_val;
                    dst[1] =                row[ow];
                    dst[2] = (ow < W - 1) ? row[ow + 1] : pad_val;
                }
                dst += 3;
            }
        }
    }
}

static void im2col_3x3s1p1(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int8_t* col,
    bool in_parallel = false)
{
    const int K  = 9 * C_in;
    const int HW = H * W;
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    } else {
        for (int oh = 1; oh < H - 1; ++oh)
            im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
    }
#else
    for (int oh = 1; oh < H - 1; ++oh)
        im2col_3x3s1p1_interior_oh(input, col, oh, C_in, HW, K, W);
#endif
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    } else {
        for (int oh = 0; oh < H; ++oh)
            im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
    }
#else
    for (int oh = 0; oh < H; ++oh)
        im2col_3x3s1p1_border_oh(input, col, oh, C_in, HW, K, H, W, pad_val);
#endif
}

static void im2col_3x3s2p1_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W, int oW, int ow_lo, int ow_hi)
{
    const int row0_off = (oh * 2 - 1) * W;
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        const int base_col = row0_off + (ow * 2 - 1);
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + base_col;
            uint32_t v0, v1, v2;
            memcpy(&v0, base,         4);
            memcpy(&v1, base + W,     4);
            memcpy(&v2, base + 2 * W, 4);
            uint64_t q = ((uint64_t)(v0 & 0xFFFFFFu))
                       | ((uint64_t)(v1 & 0xFFFFFFu) << 24)
                       | ((uint64_t)(v2 & 0xFFFFu)   << 48);
            memcpy(dst, &q, 8);
            dst[8] = (uint8_t)(v2 >> 16);
            dst += 9;
        }
    }
}

static void im2col_3x3s2p1_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int oW,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 3; ++kh) {
                int ih = oh * 2 - 1 + kh;
                if (ih < 0 || ih >= H) {
                    dst[0] = dst[1] = dst[2] = pad_val;
                } else {
                    const int8_t* row = in_c + ih * W;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow * 2 - 1 + kw;
                        dst[kw] = (iw >= 0 && iw < W) ? row[iw] : pad_val;
                    }
                }
                dst += 3;
            }
        }
    }
}

static void im2col_3x3s2p1(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,
    bool in_parallel = false)
{
    const int K  = 9 * C_in;
    const int HW = H * W;
    const int oh_lo = 1, oh_hi = (H - 2) / 2;
    const int ow_lo = 1, ow_hi = (W - 2) / 2;
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_3x3s2p1_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
#endif
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_3x3s2p1_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                 oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}

static void im2col_7x7s2p3_interior_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int W, int oW, int ow_lo, int ow_hi)
{
    for (int ow = ow_lo; ow <= ow_hi; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * K;
        const int ih0 = oh * 2 - 3;
        const int iw0 = ow * 2 - 3;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* base = input + (size_t)c * HW + ih0 * W + iw0;
            memcpy(dst,    base,         7);
            memcpy(dst+ 7, base +     W, 7);
            memcpy(dst+14, base + 2 * W, 7);
            memcpy(dst+21, base + 3 * W, 7);
            memcpy(dst+28, base + 4 * W, 7);
            memcpy(dst+35, base + 5 * W, 7);
            memcpy(dst+42, base + 6 * W, 7);
            dst += 49;
        }
    }
}

static void im2col_7x7s2p3_border_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int K, int H, int W, int oW,
    int oh_lo, int oh_hi, int ow_lo, int ow_hi, int8_t pad_val)
{
    for (int ow = 0; ow < oW; ++ow) {
        if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
        int8_t* dst = col + (oh * oW + ow) * K;
        int idx = 0;
        for (int c = 0; c < C_in; ++c) {
            const int8_t* in_c = input + (size_t)c * HW;
            for (int kh = 0; kh < 7; ++kh) {
                int ih = oh * 2 - 3 + kh;
                if (ih < 0 || ih >= H) {
                    memset(dst + idx, pad_val, 7); idx += 7;
                } else {
                    const int8_t* row = in_c + ih * W;
                    for (int kw = 0; kw < 7; ++kw) {
                        int iw = ow * 2 - 3 + kw;
                        dst[idx++] = (iw >= 0 && iw < W) ? row[iw] : pad_val;
                    }
                }
            }
        }
    }
}

static void im2col_7x7s2p3(
    const int8_t* input,
    int C_in, int H, int W,
    int8_t pad_val,
    int oH, int oW,
    int8_t* col,
    bool in_parallel = false)
{
    const int K  = 49 * C_in;
    const int HW = H * W;
    const int oh_lo = 2, oh_hi = (H - 4) / 2;
    const int ow_lo = 2, ow_hi = (W - 4) / 2;
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static) nowait
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    } else {
        for (int oh = oh_lo; oh <= oh_hi; ++oh)
            im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
    }
#else
    for (int oh = oh_lo; oh <= oh_hi; ++oh)
        im2col_7x7s2p3_interior_oh(input, col, oh, C_in, HW, K, W, oW, ow_lo, ow_hi);
#endif
#ifdef _OPENMP
    if (in_parallel) {
#pragma omp for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    } else {
        for (int oh = 0; oh < oH; ++oh)
            im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                     oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
    }
#else
    for (int oh = 0; oh < oH; ++oh)
        im2col_7x7s2p3_border_oh(input, col, oh, C_in, HW, K, H, W, oW,
                                 oh_lo, oh_hi, ow_lo, ow_hi, pad_val);
#endif
}

static void im2col(
    const int8_t* input,
    int C_in, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int oH, int oW,
    int8_t* col,
    int8_t  pad_val)
{
    const int K  = kH * kW * C_in;
    const int HW = H * W;

    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    for (int oh = oh_lo; oh <= oh_hi; ++oh) {
        for (int ow = ow_lo; ow <= ow_hi; ++ow) {
            int8_t* dst = col + (oh * oW + ow) * K;
            int ih0 = oh * stride_h - pad_h;
            int iw0 = ow * stride_w - pad_w;
            for (int c = 0; c < C_in; ++c) {
                const int8_t* in_c = input + (size_t)c * HW;
                for (int kh = 0; kh < kH; ++kh) {
                    memcpy(dst, in_c + (ih0 + kh) * W + iw0, kW);
                    dst += kW;
                }
            }
        }
    }

    for (int oh = 0; oh < oH; ++oh) {
        for (int ow = 0; ow < oW; ++ow) {
            if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
            int8_t* dst = col + (oh * oW + ow) * K;
            int idx = 0;
            for (int c = 0; c < C_in; ++c) {
                const int8_t* in_c = input + (size_t)c * HW;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh * stride_h - pad_h + kh;
                    if (ih < 0 || ih >= H) {
                        memset(dst + idx, pad_val, kW); idx += kW;
                    } else {
                        const int8_t* in_row = in_c + ih * W;
                        for (int kw = 0; kw < kW; ++kw) {
                            int iw = ow * stride_w - pad_w + kw;
                            dst[idx++] = (iw >= 0 && iw < W) ? in_row[iw] : pad_val;
                        }
                    }
                }
            }
        }
    }
}

// ── Helper: store one output tile ─────────────────────────────────────────────
// r[row][oc] holds the raw int32 accumulator for that (row, oc) pair.
// Applies bias + requant/float conversion and writes to C.
static inline void store_tile(
    const int32_t r[8][8], int n_valid, int n0, int m, int M, int N,
    const int64_t* eff_bias, const float* req_scale, int8_t out_zp,
    void* C, bool is_float, bool nchw_out)
{
    for (int oc = 0; oc < n_valid; ++oc) {
        const int n      = n0 + oc;
        const int64_t bias = eff_bias[n];
        const float   s    = req_scale[n];
        for (int row = 0; row < 8 && m + row < M; ++row) {
            int64_t total = (int64_t)r[row][oc] + bias;
            if (is_float) {
                float val = (float)total * s;
                if (nchw_out) ((float*)C)[n * M + m + row] = val;
                else          ((float*)C)[(m + row) * N + n] = val;
            } else {
                int32_t q = (int32_t)std::roundf((float)total * s) + out_zp;
                int8_t  val = (int8_t)std::clamp(q, -128, 127);
                if (nchw_out) ((int8_t*)C)[n * M + m + row] = val;
                else          ((int8_t*)C)[(m + row) * N + n] = val;
            }
        }
    }
}

// ── AVX2 inner kernels ────────────────────────────────────────────────────────
// Both kernels use an 8-row × 8-col output tile.
// Each acc[row] is a __m256i holding 8 int32 partial sums (one per output channel).
// No horizontal reduction needed: each lane accumulates one (row, oc) dot product.
//
//   Int8    B layout: Co8 × K4 × 32 bytes (one int8 per weight)
//   Ternary B layout: Co8 × K4 × 8 bytes  (4 ternary weights packed per byte)

#ifdef __AVX2__

static void gemm_int8_avx2_inner(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    int K4, int Co8,
    int mi_s, int mi_e, int co_s, int co_e)
{
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 8;
        // Pre-compute per-row A base pointers for this M-tile
        const int8_t* a[8];
        for (int r = 0; r < 8; ++r) a[r] = A + (size_t)(m + r) * K;

        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co    = B_packed + (size_t)co_blk * K4 * 32;
            const int     n0      = co_blk * 8;
            const int     n_valid = std::min(8, N - n0);

            __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                const int8_t* b_kb = b_co + k_blk * 32;
                const int     k0   = k_blk * 4;

                for (int ki = 0; ki < 4 && k0 + ki < K; ++ki) {
                    // Load 8 B values for this (k_blk, ki): one weight per output channel
                    __m128i b_ki = _mm_loadl_epi64((__m128i*)(b_kb + ki * 8));
                    __m256i b_32 = _mm256_cvtepi8_epi32(b_ki); // 8 int8 → 8 int32

                    // For each M-row: broadcast A scalar, multiply, accumulate
                    // Int8: full multiply (no VNNI)
#define INT8_ACC(row, areg) do { \
    __m256i a_32 = _mm256_set1_epi32((int32_t)a[row][k0 + ki]); \
    areg = _mm256_add_epi32(areg, _mm256_mullo_epi32(a_32, b_32)); \
} while(0)
                    INT8_ACC(0, acc0); INT8_ACC(1, acc1);
                    INT8_ACC(2, acc2); INT8_ACC(3, acc3);
                    INT8_ACC(4, acc4); INT8_ACC(5, acc5);
                    INT8_ACC(6, acc6); INT8_ACC(7, acc7);
#undef INT8_ACC
                }
            }

            // Extract 8×8 int32 results and write output
            int32_t r[8][8];
            _mm256_storeu_si256((__m256i*)r[0], acc0);
            _mm256_storeu_si256((__m256i*)r[1], acc1);
            _mm256_storeu_si256((__m256i*)r[2], acc2);
            _mm256_storeu_si256((__m256i*)r[3], acc3);
            _mm256_storeu_si256((__m256i*)r[4], acc4);
            _mm256_storeu_si256((__m256i*)r[5], acc5);
            _mm256_storeu_si256((__m256i*)r[6], acc6);
            _mm256_storeu_si256((__m256i*)r[7], acc7);
            store_tile(r, n_valid, n0, m, M, N, eff_bias, req_scale, out_zp, C, is_float, nchw_out);
        }
    }
}

// Ternary inner kernel: 8-row × 8-col tile.
// B_packed uses the same Co8 × K4 × 32 layout as the int8 kernel.
// Values are raw {-1, 0, +1} int8 — no 2-bit decode needed.
// sign_epi32 replaces mullo_epi32: same throughput (1 cycle), but only
// 1-cycle latency vs 10-cycle for mullo, giving headroom on latency-bound paths.
static void gemm_ternary_avx2_inner(
    const int8_t*  A,
    const int8_t*  B_packed,   // Co8 × K4 × 32 bytes, values ∈ {-1, 0, +1}
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    int K4, int Co8,
    int mi_s, int mi_e, int co_s, int co_e)
{
    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 8;
        const int8_t* a[8];
        for (int r = 0; r < 8; ++r) a[r] = A + (size_t)(m + r) * K;

        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co    = B_packed + (size_t)co_blk * K4 * 32;
            const int     n0      = co_blk * 8;
            const int     n_valid = std::min(8, N - n0);

            __m256i acc0 = _mm256_setzero_si256(), acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256(), acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256(), acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256(), acc7 = _mm256_setzero_si256();

            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                const int8_t* b_kb = b_co + k_blk * 32;
                const int     k0   = k_blk * 4;

                for (int ki = 0; ki < 4 && k0 + ki < K; ++ki) {
                    // Load 8 ternary weights as int8, extend to int32 ({-1,0,+1})
                    __m128i b_ki = _mm_loadl_epi64((__m128i*)(b_kb + ki * 8));
                    __m256i b_32 = _mm256_cvtepi8_epi32(b_ki);

                    // sign_epi32(a, b): keeps a if b>0, zeroes if b=0, negates if b<0.
                    // With b∈{-1,0,+1} this is identical to mullo(a,b) but faster.
#define TERN_ACC(row, areg) do { \
    __m256i a_32 = _mm256_set1_epi32((int32_t)a[row][k0 + ki]); \
    areg = _mm256_add_epi32(areg, _mm256_sign_epi32(a_32, b_32)); \
} while(0)
                    TERN_ACC(0, acc0); TERN_ACC(1, acc1);
                    TERN_ACC(2, acc2); TERN_ACC(3, acc3);
                    TERN_ACC(4, acc4); TERN_ACC(5, acc5);
                    TERN_ACC(6, acc6); TERN_ACC(7, acc7);
#undef TERN_ACC
                }
            }

            int32_t r[8][8];
            _mm256_storeu_si256((__m256i*)r[0], acc0);
            _mm256_storeu_si256((__m256i*)r[1], acc1);
            _mm256_storeu_si256((__m256i*)r[2], acc2);
            _mm256_storeu_si256((__m256i*)r[3], acc3);
            _mm256_storeu_si256((__m256i*)r[4], acc4);
            _mm256_storeu_si256((__m256i*)r[5], acc5);
            _mm256_storeu_si256((__m256i*)r[6], acc6);
            _mm256_storeu_si256((__m256i*)r[7], acc7);
            store_tile(r, n_valid, n0, m, M, N, eff_bias, req_scale, out_zp, C, is_float, nchw_out);
        }
    }
}

#endif // __AVX2__

// ── Scalar tail for ternary (M % 8 remaining rows) ───────────────────────────
// B_packed is in the same Co8 × K4 × 32 layout as int8; values are {-1,0,+1}.
static void gemm_scalar_tail_ternary(
    const int8_t*  A,
    const int8_t*  B_packed,   // Co8 × K4 × 32 bytes, values ∈ {-1, 0, +1}
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int m_start, int M, int K, int N,
    bool           nchw_out,
    int K4, int Co8, int co_s, int co_e)
{
    for (int m = m_start; m < M; ++m) {
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co    = B_packed + (size_t)co_blk * K4 * 32;
            const int     n0      = co_blk * 8;
            const int     n_valid = std::min(8, N - n0);
            int64_t acc_arr[8]    = {};

            for (int k = 0; k < K; ++k) {
                int    k_blk = k / 4, ki = k % 4;
                int8_t a_val = A[m * K + k];
                for (int oc = 0; oc < n_valid; ++oc)
                    acc_arr[oc] += (int64_t)a_val * (int8_t)b_co[k_blk * 32 + ki * 8 + oc];
            }
            for (int oc = 0; oc < n_valid; ++oc) {
                const int n     = n0 + oc;
                int64_t   total = acc_arr[oc] + eff_bias[n];
                if (is_float) {
                    float val = (float)total * req_scale[n];
                    if (nchw_out) ((float*)C)[n * M + m] = val;
                    else          ((float*)C)[m * N + n] = val;
                } else {
                    int32_t q = (int32_t)std::roundf((float)total * req_scale[n]) + out_zp;
                    int8_t val = (int8_t)std::clamp(q, -128, 127);
                    if (nchw_out) ((int8_t*)C)[n * M + m] = val;
                    else          ((int8_t*)C)[m * N + n] = val;
                }
            }
        }
    }
}

// ── Scalar tail: handles M % 8 remaining rows (int8 only) ────────────────────
static void gemm_scalar_tail(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int m_start, int M, int K, int N,
    bool           nchw_out,
    int K4, int Co8, int co_s, int co_e)
{
    for (int m = m_start; m < M; ++m) {
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co    = B_packed + (size_t)co_blk * K4 * 32;
            const int     n0      = co_blk * 8;
            const int     n_valid = std::min(8, N - n0);
            int64_t acc_arr[8]    = {};

            for (int k = 0; k < K; ++k) {
                int k_blk = k / 4, ki = k % 4;
                int8_t a_val = A[m * K + k];
                for (int oc = 0; oc < n_valid; ++oc)
                    acc_arr[oc] += (int64_t)a_val * (int8_t)b_co[k_blk * 32 + ki * 8 + oc];
            }
            for (int oc = 0; oc < n_valid; ++oc) {
                const int n      = n0 + oc;
                int64_t   total  = acc_arr[oc] + eff_bias[n];
                if (is_float) {
                    float val = (float)total * req_scale[n];
                    if (nchw_out) ((float*)C)[n * M + m] = val;
                    else          ((float*)C)[m * N + n] = val;
                } else {
                    int32_t q = (int32_t)std::roundf((float)total * req_scale[n]) + out_zp;
                    int8_t val = (int8_t)std::clamp(q, -128, 127);
                    if (nchw_out) ((int8_t*)C)[n * M + m] = val;
                    else          ((int8_t*)C)[m * N + n] = val;
                }
            }
        }
    }
}

// ── Public GEMM wrappers ──────────────────────────────────────────────────────

void gemm_int8_avx2(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    bool           in_parallel)
{
    if (!in_parallel) {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
#pragma omp parallel num_threads(nthreads)
            gemm_int8_avx2(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                           M, K, N, nchw_out, /*in_parallel=*/true);
            return;
        }
#endif
    }

    const int K4  = (K + 3) / 4;
    const int Co8 = (N + 7) / 8;
    const int m8_count = M / 8;

#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0, nT = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    const int mi_s = n_part ? 0           : (tid * m8_count) / nT;
    const int mi_e = n_part ? m8_count    : ((tid + 1) * m8_count) / nT;
    const int co_s = n_part ? (tid * Co8) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co8) / nT : Co8;

#ifdef __AVX2__
    gemm_int8_avx2_inner(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                         M, K, N, nchw_out, K4, Co8, mi_s, mi_e, co_s, co_e);
#else
    // Scalar fallback covers all rows (treat as one big tail)
    gemm_scalar_tail(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                     mi_s * 8, std::min(mi_e * 8, M), K, N, nchw_out, K4, Co8, co_s, co_e);
#endif

    // Scalar tail for M % 8 remaining rows
    if (n_part || tid == 0) {
        gemm_scalar_tail(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                         m8_count * 8, M, K, N, nchw_out, K4, Co8, co_s, co_e);
    }
}

void gemm_ternary_avx2(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    bool           in_parallel)
{
    if (!in_parallel) {
#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
#pragma omp parallel num_threads(nthreads)
            gemm_ternary_avx2(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                              M, K, N, nchw_out, /*in_parallel=*/true);
            return;
        }
#endif
    }

    // B_packed is in Co8 × K4 × 32 layout (same as int8), values ∈ {-1, 0, +1}
    const int K4      = (K + 3) / 4;
    const int Co8     = (N + 7) / 8;
    const int m8_count = M / 8;

#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0, nT = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    const int mi_s = n_part ? 0           : (tid * m8_count) / nT;
    const int mi_e = n_part ? m8_count    : ((tid + 1) * m8_count) / nT;
    const int co_s = n_part ? (tid * Co8) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co8) / nT : Co8;

#ifdef __AVX2__
    gemm_ternary_avx2_inner(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                            M, K, N, nchw_out, K4, Co8, mi_s, mi_e, co_s, co_e);
#else
    gemm_scalar_tail_ternary(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                             mi_s * 8, std::min(mi_e * 8, M), K, N, nchw_out, K4, Co8, co_s, co_e);
#endif

    if (n_part || tid == 0) {
        gemm_scalar_tail_ternary(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                                 m8_count * 8, M, K, N, nchw_out, K4, Co8, co_s, co_e);
    }
}

// ── Conv2d boilerplate ────────────────────────────────────────────────────────
// Mirrors conv2d_int8 / conv2d_ternary from ops_avx512.cpp but:
//  - Uses pack_weights_avx2 (always re-packs; w_pre_packed is in sdot format,
//    incompatible with the AVX2 Co8×K4×32 layout)
//  - Calls gemm_int8_avx2 / gemm_ternary_avx2

static void conv2d_avx2_impl(
    bool           is_ternary,
    const int8_t*  input,
    const int8_t*  weight,
    const int8_t*  w_pre_packed,   // if non-null and groups==1, skip packing
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         in_zp,
    int8_t         out_zp,
    int8_t*        output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int groups,
    int8_t* scratch_col)
{
    const int oH  = (H + 2 * pad_h - kH) / stride_h + 1;
    const int oW  = (W + 2 * pad_w - kW) / stride_w + 1;
    const int C_in_g  = C_in  / groups;
    const int C_out_g = C_out / groups;
    const int K   = kH * kW * C_in_g;
    const int oHW = oH * oW;

    std::vector<int8_t> _col_local;
    int8_t* col = scratch_col;
    if (!col) {
        _col_local.resize((size_t)oHW * K);
        col = _col_local.data();
    }

    for (int g = 0; g < groups; ++g) {
        const int8_t* in_g  = input  + g * C_in_g  * H * W;
        const int8_t* w_g   = weight + g * C_out_g * K;
        int8_t*       out_g = output + g * C_out_g * oHW;

        // Use pre-packed weights when available (groups==1); otherwise pack on the fly.
        // Ternary: Co8×K4×8 (compressed 2-bit, 4× smaller → fits in L2 for large layers)
        // Int8:    Co8×K4×32 (one int8 per weight)
        bool own_packed = false;
        int8_t* w_packed;
        if (w_pre_packed && groups == 1) {
            w_packed = const_cast<int8_t*>(w_pre_packed);
        } else {
            w_packed = is_ternary
                ? pack_weights_ternary_avx2(w_g, C_out_g, K)
                : pack_weights_avx2(w_g, C_out_g, K);
            own_packed = true;
        }

#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            double t0 = now_ms_avx2();
#pragma omp parallel num_threads(nthreads)
            {
                if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0)
                    im2col_1x1(in_g, C_in_g, H, W, stride_h, stride_w, oH, oW,
                               col, /*in_parallel=*/true);
                else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                         && pad_h == 1 && pad_w == 1)
                    im2col_3x3s1p1(in_g, C_in_g, H, W, in_zp, col,
                                   /*in_parallel=*/true);
                else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                         && pad_h == 1 && pad_w == 1)
                    im2col_3x3s2p1(in_g, C_in_g, H, W, in_zp, oH, oW,
                                   col, /*in_parallel=*/true);
                else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                         && pad_h == 3 && pad_w == 3)
                    im2col_7x7s2p3(in_g, C_in_g, H, W, in_zp, oH, oW,
                                   col, /*in_parallel=*/true);
                else {
#pragma omp single
                    im2col(in_g, C_in_g, H, W, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, oH, oW,
                           col, in_zp);
                }
                if (is_ternary)
                    gemm_ternary_avx2(col, w_packed,
                                      eff_bias + g * C_out_g,
                                      req_scale + g * C_out_g,
                                      out_zp, out_g, false,
                                      oHW, K, C_out_g,
                                      /*nchw_out=*/true, /*in_parallel=*/true);
                else
                    gemm_int8_avx2(col, w_packed,
                                   eff_bias + g * C_out_g,
                                   req_scale + g * C_out_g,
                                   out_zp, out_g, false,
                                   oHW, K, C_out_g,
                                   /*nchw_out=*/true, /*in_parallel=*/true);
            }
            g_avx2_gemm_ms += now_ms_avx2() - t0;
        } else
#endif
        {
            {
                double t0 = now_ms_avx2();
                if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0) {
                    im2col_1x1(in_g, C_in_g, H, W, stride_h, stride_w, oH, oW, col);
                } else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s1p1(in_g, C_in_g, H, W, in_zp, col);
                } else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s2p1(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                } else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                           && pad_h == 3 && pad_w == 3) {
                    im2col_7x7s2p3(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                } else {
                    im2col(in_g, C_in_g, H, W, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, oH, oW,
                           col, in_zp);
                }
                g_avx2_im2col_ms += now_ms_avx2() - t0;
            }
            {
                double t0 = now_ms_avx2();
                if (is_ternary)
                    gemm_ternary_avx2(col, w_packed,
                                      eff_bias + g * C_out_g,
                                      req_scale + g * C_out_g,
                                      out_zp, out_g, false,
                                      oHW, K, C_out_g,
                                      /*nchw_out=*/true);
                else
                    gemm_int8_avx2(col, w_packed,
                                   eff_bias + g * C_out_g,
                                   req_scale + g * C_out_g,
                                   out_zp, out_g, false,
                                   oHW, K, C_out_g,
                                   /*nchw_out=*/true);
                g_avx2_gemm_ms += now_ms_avx2() - t0;
            }
        }

        if (own_packed) free_packed_avx2(w_packed);
    }
}

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
    int groups,
    int8_t* scratch_col,
    const int8_t* w_pre_packed)
{
    conv2d_avx2_impl(false, input, weight, w_pre_packed, eff_bias, req_scale, in_zp, out_zp, output,
                     C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
                     groups, scratch_col);
}

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
    int groups,
    int8_t* scratch_col,
    const int8_t* w_pre_packed)
{
    conv2d_avx2_impl(true, input, weight, w_pre_packed, eff_bias, req_scale, in_zp, out_zp, output,
                     C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
                     groups, scratch_col);
}

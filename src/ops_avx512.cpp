#include "ops_avx512.hpp"
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX512F__
#include <immintrin.h>
#endif

static double g_im2col_1x1_ms  = 0.0;
static double g_im2col_3x3_ms  = 0.0;
static double g_im2col_gen_ms  = 0.0;
static double g_gemm_ms        = 0.0;
static double g_add_req_ms     = 0.0;
static double g_maxpool_ms      = 0.0;

static inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

void ops_profile_reset() {
    g_im2col_1x1_ms = g_im2col_3x3_ms = g_im2col_gen_ms = g_gemm_ms = g_add_req_ms = g_maxpool_ms = 0.0;
}

void ops_profile_print(int runs) {
    double d = (runs > 0) ? (double)runs : 1.0;
    double im2col_total = g_im2col_1x1_ms + g_im2col_3x3_ms + g_im2col_gen_ms;
    printf("\n=== Op Profiling (avg per image, %d runs) ===\n", runs);
    printf("  im2col 1x1   : %6.3f ms\n", g_im2col_1x1_ms / d);
    printf("  im2col 3x3   : %6.3f ms\n", g_im2col_3x3_ms / d);
    printf("  im2col gen   : %6.3f ms\n", g_im2col_gen_ms / d);
    printf("  im2col total : %6.3f ms\n", im2col_total     / d);
    printf("  gemm         : %6.3f ms\n", g_gemm_ms        / d);
    printf("  add_requant  : %6.3f ms\n", g_add_req_ms     / d);
    printf("  maxpool      : %6.3f ms\n", g_maxpool_ms     / d);
    printf("  other (FC,...): [total - above]\n");
    printf("=============================================\n");
}

int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K)
{
    int Co4 = (C_out + 3) / 4;
    int K4  = (K    + 3) / 4;
    int sz  = Co4 * K4 * 16;
    int8_t* packed = new int8_t[sz]();

    for (int co_blk = 0; co_blk < Co4; ++co_blk) {
        for (int k_blk = 0; k_blk < K4; ++k_blk) {
            int8_t* dst = packed + (co_blk * K4 + k_blk) * 16;
            for (int oc = 0; oc < 4; ++oc) {
                int co = co_blk * 4 + oc;
                for (int ki = 0; ki < 4; ++ki) {
                    int k = k_blk * 4 + ki;
                    int8_t v = (co < C_out && k < K) ? w[co * K + k] : 0;
                    dst[oc * 4 + ki] = v;
                }
            }
        }
    }
    return packed;
}

void free_packed(int8_t* p) { delete[] p; }

#ifdef __AVX512F__
void gemm_int8_avx512_inner(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    int K4, int Co16, int m16_count,
    int mi_s, int mi_e, int co_s, int co_e);
#endif


#ifdef __AVX512F__
void gemm_ternary_avx512(
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
    // START BODY REPLACEMENT HERE (after signature params)
    double t0_gemm = now_ms();
    int K4 = (K + 3) / 4;
    int Co16 = (N + 15) / 16;
    int m16_count = M / 16;

#ifdef _OPENMP
    int tid = omp_get_thread_num();
    int nT = omp_get_num_threads();
#else
    int tid = 0, nT = 1;
#endif
    bool n_part = (nT > 1) && (M < N);
    int mi_s = n_part ? 0 : (tid * m16_count) / nT;
    int mi_e = n_part ? m16_count : ((tid + 1) * m16_count) / nT;
    int co_s = n_part ? (tid * Co16) / nT : 0;
    int co_e = n_part ? ((tid + 1) * Co16) / nT : Co16;

    // Main AVX512 kernel
    for (int mi = mi_s; mi < mi_e; ++mi) {
        int m_base = mi * 16;
        __m256i a_load;
        __m512i a_rows[16];
        for (int row = 0; row < 16; ++row) {
            a_load = _mm256_loadu_si256((__m256i*)(A + (m_base + row) * K));
            a_rows[row] = _mm512_cvtepi8_epi16(a_load);
        }
        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
            int n0 = co_blk * 16;
            int n_valid = std::min(16, N - n0);
            __m512i acc[16] = {};
            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                for (int oc = 0; oc < n_valid; ++oc) {
                    __m128i w_vec = _mm_loadu_si128((__m128i*)(b_co + k_blk * 16 + oc * 4));
                    __mmask16 pos_mask = _mm_cmpge_epi8_mask(w_vec, _mm_set1_epi8(1));
                    __mmask16 neg_mask = _mm_cmplt_epi8_mask(w_vec, _mm_set1_epi8(0));
                    for (int row = 0; row < 16; ++row) {
                        __m512i pos_delta = _mm512_mask_add_epi32(acc[row], pos_mask, acc[row], a_rows[row]);
                        acc[row] = _mm512_mask_sub_epi32(pos_delta, neg_mask, pos_delta, a_rows[row]);
                    }
                }
            }
            // Store main block
            int8_t* C_i8 = static_cast<int8_t*>(C);
            float* C_f32 = static_cast<float*>(C);
            for (int oc = 0; oc < n_valid; ++oc) {
                int n = n0 + oc;
                int64_t bias = eff_bias[n];
                float scale = req_scale[n];
                for (int row = 0; row < 16; ++row) {
                    int64_t total = _mm512_reduce_add_epi32(acc[row]) + bias;
                    if (is_float) {
                        if (nchw_out) C_f32[n * M + m_base + row] = static_cast<float>(total) * scale;
                        else C_f32[(m_base + row) * N + n] = static_cast<float>(total) * scale;
                    } else {
                        int8_t val = static_cast<int8_t>(std::clamp<int32_t>(std::round(total * scale), -128, 127) + out_zp);
                        if (nchw_out) C_i8[n * M + m_base + row] = val;
                        else C_i8[(m_base + row) * N + n] = val;
                    }
                }
            }
        }
    }

    // Tail M % 16 - scoped pointers inside
    int m_tail = m16_count * 16;
    if (n_part || tid == 0) {
        int8_t* C_i8 = static_cast<int8_t*>(C);
        float* C_f32 = static_cast<float*>(C);
        for (int m = m_tail; m < M; ++m) {
            for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
                int n0 = co_blk * 16;
                int n_valid = std::min(16, N - n0);
                for (int oc = 0; oc < n_valid; ++oc) {
                    int n = n0 + oc;
                    int64_t acc = eff_bias[n];
                    for (int k = 0; k < K; ++k) {
                        int k_blk = k / 4, ki = k % 4;
                        int8_t w = b_co[k_blk * 16 + oc * 4 + ki];
                        int8_t aval = A[m * K + k];
                        if (w == 1) acc += aval;
                        else if (w == -1) acc -= aval;
                    }
                    if (is_float) {
                        float val = static_cast<float>(acc) * req_scale[n];
                        if (nchw_out) C_f32[n * M + m] = val;
                        else C_f32[m * N + n] = val;
                    } else {
                        int8_t val = static_cast<int8_t>(std::clamp<int32_t>(std::round(acc * req_scale[n]), -128, 127) + out_zp);
                        if (nchw_out) C_i8[n * M + m] = val;
                        else C_i8[m * N + n] = val;
                    }
                }
            }
        }
    }
    g_gemm_ms += now_ms() - t0_gemm;  // Your profiler integration
}
#endif

void gemm_int8_avx512(
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
            gemm_int8_avx512(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                           M, K, N, nchw_out, /*in_parallel=*/true);
            return;
        }
#endif
    }

    int K4  = (K + 3) / 4;
    int Co16 = (N + 15) / 16;
    const int m16_count = M / 16;

#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0;
    const int nT  = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    const int mi_s = n_part ? 0           : (tid * m16_count) / nT;
    const int mi_e = n_part ? m16_count   : ((tid + 1) * m16_count) / nT;
    const int co_s = n_part ? (tid * Co16) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co16) / nT : Co16;

#ifdef __AVX512F__
    gemm_int8_avx512_inner(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                           M, K, N, nchw_out, K4, Co16, m16_count, mi_s, mi_e, co_s, co_e);
#else
    for (int m = mi_s; m < mi_e; ++m) {
        for (int n = co_s; n < co_e; ++n) {
            int64_t acc = eff_bias[n];
            const int8_t* a_row = A + m * K;
            int co_blk = n / 4, oc = n % 4;
            for (int k = 0; k < K; ++k) {
                int k_blk = k / 4, ki = k % 4;
                const int8_t* w_ptr = B_packed + (co_blk * K4 + k_blk) * 16 + oc * 4 + ki;
                acc += (int32_t)a_row[k] * (int32_t)*w_ptr;
            }
            if (nchw_out)
                static_cast<int8_t*>(C)[n * M + m] = requant(acc, req_scale[n], out_zp);
            else if (is_float)
                static_cast<float*>(C)[m * N + n] = (float)acc * req_scale[n];
            else
                static_cast<int8_t*>(C)[m * N + n] = requant(acc, req_scale[n], out_zp);
        }
    }
#endif

    if (n_part || tid == 0) {
        const int co_s2 = n_part ? co_s : 0;
        const int co_e2 = n_part ? co_e : Co16;
        int m = m16_count * 16;
        for (; m + 7 < M; m += 8) {
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
                const int n0      = co_blk * 16;
                const int n_valid = std::min(16, N - n0);
                int64_t acc_arr[8][16] = {};
                for (int k_blk = 0; k_blk < K4; ++k_blk) {
                    for (int row = 0; row < 8; ++row) {
                        const int8_t* a_row = A + (size_t)(m+row) * K + k_blk * 4;
                        for (int ki = 0; ki < 4 && k_blk * 4 + ki < K; ++ki) {
                            for (int oc = 0; oc < n_valid; ++oc) {
                                acc_arr[row][oc] += (int64_t)a_row[ki] * (int8_t)b_co[k_blk*16 + oc*4 + ki];
                            }
                        }
                    }
                }
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    const float   s   = req_scale[n];
                    if (is_float) {
                        float* out_f = static_cast<float*>(C);
                        for (int row = 0; row < 8; ++row)
                            out_f[(m+row)*N+n] = (float)(acc_arr[row][oc]+b64)*s;
                    } else {
                        int8_t* out_i = static_cast<int8_t*>(C);
                        for (int row = 0; row < 8; ++row)
                            out_i[(m+row)*N+n] = requant(acc_arr[row][oc]+b64, s, out_zp);
                    }
                }
            }
        }
        for (; m + 3 < M; m += 4) {
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
                const int n0      = co_blk * 16;
                const int n_valid = std::min(16, N - n0);
                int64_t acc_arr[4][16] = {};
                for (int k_blk = 0; k_blk < K4; ++k_blk) {
                    for (int row = 0; row < 4; ++row) {
                        const int8_t* a_row = A + (size_t)(m+row) * K + k_blk * 4;
                        for (int ki = 0; ki < 4 && k_blk * 4 + ki < K; ++ki) {
                            for (int oc = 0; oc < n_valid; ++oc) {
                                acc_arr[row][oc] += (int64_t)a_row[ki] * (int8_t)b_co[k_blk*16 + oc*4 + ki];
                            }
                        }
                    }
                }
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    const float   s   = req_scale[n];
                    if (is_float) {
                        float* out_f = static_cast<float*>(C);
                        for (int row = 0; row < 4; ++row)
                            out_f[(m+row)*N+n] = (float)(acc_arr[row][oc]+b64)*s;
                    } else {
                        int8_t* out_i = static_cast<int8_t*>(C);
                        for (int row = 0; row < 4; ++row)
                            out_i[(m+row)*N+n] = requant(acc_arr[row][oc]+b64, s, out_zp);
                    }
                }
            }
        }
        for (; m < M; ++m) {
            for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
                const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
                const int n0      = co_blk * 16;
                const int n_valid = std::min(16, N - n0);
                int64_t acc_arr[16] = {};
                for (int k = 0; k < K; ++k) {
                    int8_t a_val = A[m * K + k];
                    int k_blk = k / 4, ki = k % 4;
                    for (int oc = 0; oc < n_valid; ++oc) {
                        acc_arr[oc] += (int64_t)a_val * (int8_t)b_co[k_blk*16 + oc*4 + ki];
                    }
                }
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n = n0 + oc;
                    int64_t total = acc_arr[oc] + eff_bias[n];
                    if (nchw_out)
                        static_cast<int8_t*>(C)[n * M + m] = requant(total, req_scale[n], out_zp);
                    else if (is_float)
                        static_cast<float*>(C)[m * N + n] = (float)total * req_scale[n];
                    else
                        static_cast<int8_t*>(C)[m * N + n] = requant(total, req_scale[n], out_zp);
                }
            }
        }
    }
}

#ifdef __AVX512F__
void gemm_int8_avx512_inner(
    const int8_t*  A,
    const int8_t*  B_packed,
    const int64_t* eff_bias,
    const float*   req_scale,
    int8_t         out_zp,
    void*          C,
    bool           is_float,
    int M, int K, int N,
    bool           nchw_out,
    int K4, int Co16, int m16_count,
    int mi_s, int mi_e, int co_s, int co_e)
{
    const int k_full = K / 4;

    for (int mi = mi_s; mi < mi_e; ++mi) {
        const int m = mi * 16;
        const int8_t* a0  = A + (size_t)(m+ 0) * K;
        const int8_t* a1  = A + (size_t)(m+ 1) * K;
        const int8_t* a2  = A + (size_t)(m+ 2) * K;
        const int8_t* a3  = A + (size_t)(m+ 3) * K;
        const int8_t* a4  = A + (size_t)(m+ 4) * K;
        const int8_t* a5  = A + (size_t)(m+ 5) * K;
        const int8_t* a6  = A + (size_t)(m+ 6) * K;
        const int8_t* a7  = A + (size_t)(m+ 7) * K;
        const int8_t* a8  = A + (size_t)(m+ 8) * K;
        const int8_t* a9  = A + (size_t)(m+ 9) * K;
        const int8_t* a10 = A + (size_t)(m+10) * K;
        const int8_t* a11 = A + (size_t)(m+11) * K;
        const int8_t* a12 = A + (size_t)(m+12) * K;
        const int8_t* a13 = A + (size_t)(m+13) * K;
        const int8_t* a14 = A + (size_t)(m+14) * K;
        const int8_t* a15 = A + (size_t)(m+15) * K;

        for (int co_blk = co_s; co_blk < co_e; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
            const int n0      = co_blk * 16;
            const int n_valid = std::min(16, N - n0);

            __m512i acc0  = _mm512_setzero_epi32(), acc1  = _mm512_setzero_epi32();
            __m512i acc2  = _mm512_setzero_epi32(), acc3  = _mm512_setzero_epi32();
            __m512i acc4  = _mm512_setzero_epi32(), acc5  = _mm512_setzero_epi32();
            __m512i acc6  = _mm512_setzero_epi32(), acc7  = _mm512_setzero_epi32();
            __m512i acc8  = _mm512_setzero_epi32(), acc9  = _mm512_setzero_epi32();
            __m512i acc10 = _mm512_setzero_epi32(), acc11 = _mm512_setzero_epi32();
            __m512i acc12 = _mm512_setzero_epi32(), acc13 = _mm512_setzero_epi32();
            __m512i acc14 = _mm512_setzero_epi32(), acc15 = _mm512_setzero_epi32();

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                __m512i bv0 = _mm512_loadu_si512(b_co + (k_blk+0) * 16);
                __m512i bv1 = _mm512_loadu_si512(b_co + (k_blk+1) * 16);
                __m512i bv2 = _mm512_loadu_si512(b_co + (k_blk+2) * 16);
                __m512i bv3 = _mm512_loadu_si512(b_co + (k_blk+3) * 16);

                __m512i av0=_mm512_loadu_si512(a0+k_blk*4), av1=_mm512_loadu_si512(a1+k_blk*4);
                __m512i av2=_mm512_loadu_si512(a2+k_blk*4), av3=_mm512_loadu_si512(a3+k_blk*4);
                __m512i av4=_mm512_loadu_si512(a4+k_blk*4), av5=_mm512_loadu_si512(a5+k_blk*4);
                __m512i av6=_mm512_loadu_si512(a6+k_blk*4), av7=_mm512_loadu_si512(a7+k_blk*4);

                acc0=_mm512_dpbusd_epi32(acc0,bv0,av0); acc0=_mm512_dpbusd_epi32(acc0,bv1,av0);
                acc0=_mm512_dpbusd_epi32(acc0,bv2,av0); acc0=_mm512_dpbusd_epi32(acc0,bv3,av0);
                acc1=_mm512_dpbusd_epi32(acc1,bv0,av1); acc1=_mm512_dpbusd_epi32(acc1,bv1,av1);
                acc1=_mm512_dpbusd_epi32(acc1,bv2,av1); acc1=_mm512_dpbusd_epi32(acc1,bv3,av1);
                acc2=_mm512_dpbusd_epi32(acc2,bv0,av2); acc2=_mm512_dpbusd_epi32(acc2,bv1,av2);
                acc2=_mm512_dpbusd_epi32(acc2,bv2,av2); acc2=_mm512_dpbusd_epi32(acc2,bv3,av2);
                acc3=_mm512_dpbusd_epi32(acc3,bv0,av3); acc3=_mm512_dpbusd_epi32(acc3,bv1,av3);
                acc3=_mm512_dpbusd_epi32(acc3,bv2,av3); acc3=_mm512_dpbusd_epi32(acc3,bv3,av3);
                acc4=_mm512_dpbusd_epi32(acc4,bv0,av4); acc4=_mm512_dpbusd_epi32(acc4,bv1,av4);
                acc4=_mm512_dpbusd_epi32(acc4,bv2,av4); acc4=_mm512_dpbusd_epi32(acc4,bv3,av4);
                acc5=_mm512_dpbusd_epi32(acc5,bv0,av5); acc5=_mm512_dpbusd_epi32(acc5,bv1,av5);
                acc5=_mm512_dpbusd_epi32(acc5,bv2,av5); acc5=_mm512_dpbusd_epi32(acc5,bv3,av5);
                acc6=_mm512_dpbusd_epi32(acc6,bv0,av6); acc6=_mm512_dpbusd_epi32(acc6,bv1,av6);
                acc6=_mm512_dpbusd_epi32(acc6,bv2,av6); acc6=_mm512_dpbusd_epi32(acc6,bv3,av6);
                acc7=_mm512_dpbusd_epi32(acc7,bv0,av7); acc7=_mm512_dpbusd_epi32(acc7,bv1,av7);
                acc7=_mm512_dpbusd_epi32(acc7,bv2,av7); acc7=_mm512_dpbusd_epi32(acc7,bv3,av7);

                __m512i av8 =_mm512_loadu_si512(a8 +k_blk*4), av9 =_mm512_loadu_si512(a9 +k_blk*4);
                __m512i av10=_mm512_loadu_si512(a10+k_blk*4), av11=_mm512_loadu_si512(a11+k_blk*4);
                __m512i av12=_mm512_loadu_si512(a12+k_blk*4), av13=_mm512_loadu_si512(a13+k_blk*4);
                __m512i av14=_mm512_loadu_si512(a14+k_blk*4), av15=_mm512_loadu_si512(a15+k_blk*4);

                acc8 =_mm512_dpbusd_epi32(acc8 ,bv0,av8 ); acc8 =_mm512_dpbusd_epi32(acc8 ,bv1,av8 );
                acc8 =_mm512_dpbusd_epi32(acc8 ,bv2,av8 ); acc8 =_mm512_dpbusd_epi32(acc8 ,bv3,av8 );
                acc9 =_mm512_dpbusd_epi32(acc9 ,bv0,av9 ); acc9 =_mm512_dpbusd_epi32(acc9 ,bv1,av9 );
                acc9 =_mm512_dpbusd_epi32(acc9 ,bv2,av9 ); acc9 =_mm512_dpbusd_epi32(acc9 ,bv3,av9 );
                acc10=_mm512_dpbusd_epi32(acc10,bv0,av10); acc10=_mm512_dpbusd_epi32(acc10,bv1,av10);
                acc10=_mm512_dpbusd_epi32(acc10,bv2,av10); acc10=_mm512_dpbusd_epi32(acc10,bv3,av10);
                acc11=_mm512_dpbusd_epi32(acc11,bv0,av11); acc11=_mm512_dpbusd_epi32(acc11,bv1,av11);
                acc11=_mm512_dpbusd_epi32(acc11,bv2,av11); acc11=_mm512_dpbusd_epi32(acc11,bv3,av11);
                acc12=_mm512_dpbusd_epi32(acc12,bv0,av12); acc12=_mm512_dpbusd_epi32(acc12,bv1,av12);
                acc12=_mm512_dpbusd_epi32(acc12,bv2,av12); acc12=_mm512_dpbusd_epi32(acc12,bv3,av12);
                acc13=_mm512_dpbusd_epi32(acc13,bv0,av13); acc13=_mm512_dpbusd_epi32(acc13,bv1,av13);
                acc13=_mm512_dpbusd_epi32(acc13,bv2,av13); acc13=_mm512_dpbusd_epi32(acc13,bv3,av13);
                acc14=_mm512_dpbusd_epi32(acc14,bv0,av14); acc14=_mm512_dpbusd_epi32(acc14,bv1,av14);
                acc14=_mm512_dpbusd_epi32(acc14,bv2,av14); acc14=_mm512_dpbusd_epi32(acc14,bv3,av14);
                acc15=_mm512_dpbusd_epi32(acc15,bv0,av15); acc15=_mm512_dpbusd_epi32(acc15,bv1,av15);
                acc15=_mm512_dpbusd_epi32(acc15,bv2,av15); acc15=_mm512_dpbusd_epi32(acc15,bv3,av15);
            }

            for (; k_blk < k_full; ++k_blk) {
                __m512i wv = _mm512_loadu_si512(b_co + k_blk * 16);
                __m512i i0=_mm512_loadu_si512(a0+k_blk*4), i1=_mm512_loadu_si512(a1+k_blk*4);
                __m512i i2=_mm512_loadu_si512(a2+k_blk*4), i3=_mm512_loadu_si512(a3+k_blk*4);
                __m512i i4=_mm512_loadu_si512(a4+k_blk*4), i5=_mm512_loadu_si512(a5+k_blk*4);
                __m512i i6=_mm512_loadu_si512(a6+k_blk*4), i7=_mm512_loadu_si512(a7+k_blk*4);
                __m512i i8 =_mm512_loadu_si512(a8+k_blk*4), i9 =_mm512_loadu_si512(a9+k_blk*4);
                __m512i i10=_mm512_loadu_si512(a10+k_blk*4), i11=_mm512_loadu_si512(a11+k_blk*4);
                __m512i i12=_mm512_loadu_si512(a12+k_blk*4), i13=_mm512_loadu_si512(a13+k_blk*4);
                __m512i i14=_mm512_loadu_si512(a14+k_blk*4), i15=_mm512_loadu_si512(a15+k_blk*4);
                acc0=_mm512_dpbusd_epi32(acc0,wv,i0); acc1=_mm512_dpbusd_epi32(acc1,wv,i1);
                acc2=_mm512_dpbusd_epi32(acc2,wv,i2); acc3=_mm512_dpbusd_epi32(acc3,wv,i3);
                acc4=_mm512_dpbusd_epi32(acc4,wv,i4); acc5=_mm512_dpbusd_epi32(acc5,wv,i5);
                acc6=_mm512_dpbusd_epi32(acc6,wv,i6); acc7=_mm512_dpbusd_epi32(acc7,wv,i7);
                acc8 =_mm512_dpbusd_epi32(acc8 ,wv,i8); acc9 =_mm512_dpbusd_epi32(acc9 ,wv,i9);
                acc10=_mm512_dpbusd_epi32(acc10,wv,i10); acc11=_mm512_dpbusd_epi32(acc11,wv,i11);
                acc12=_mm512_dpbusd_epi32(acc12,wv,i12); acc13=_mm512_dpbusd_epi32(acc13,wv,i13);
                acc14=_mm512_dpbusd_epi32(acc14,wv,i14); acc15=_mm512_dpbusd_epi32(acc15,wv,i15);
            }

            for (int oc = 0; oc < n_valid; ++oc) {
                const int n   = n0 + oc;
                const int64_t b64 = eff_bias[n];
                const float   s   = req_scale[n];
                if (is_float) {
                    float* out_f = static_cast<float*>(C);
                    int32_t r0[16],r1[16],r2[16],r3[16],r4[16],r5[16],r6[16],r7[16];
                    int32_t r8[16],r9[16],r10[16],r11[16],r12[16],r13[16],r14[16],r15[16];
                    _mm512_storeu_si512(r0,acc0); _mm512_storeu_si512(r1,acc1);
                    _mm512_storeu_si512(r2,acc2); _mm512_storeu_si512(r3,acc3);
                    _mm512_storeu_si512(r4,acc4); _mm512_storeu_si512(r5,acc5);
                    _mm512_storeu_si512(r6,acc6); _mm512_storeu_si512(r7,acc7);
                    _mm512_storeu_si512(r8,acc8); _mm512_storeu_si512(r9,acc9);
                    _mm512_storeu_si512(r10,acc10); _mm512_storeu_si512(r11,acc11);
                    _mm512_storeu_si512(r12,acc12); _mm512_storeu_si512(r13,acc13);
                    _mm512_storeu_si512(r14,acc14); _mm512_storeu_si512(r15,acc15);
                    out_f[(m+0)*N+n]=(float)((int64_t)r0[oc]+b64)*s;
                    out_f[(m+1)*N+n]=(float)((int64_t)r1[oc]+b64)*s;
                    out_f[(m+2)*N+n]=(float)((int64_t)r2[oc]+b64)*s;
                    out_f[(m+3)*N+n]=(float)((int64_t)r3[oc]+b64)*s;
                    out_f[(m+4)*N+n]=(float)((int64_t)r4[oc]+b64)*s;
                    out_f[(m+5)*N+n]=(float)((int64_t)r5[oc]+b64)*s;
                    out_f[(m+6)*N+n]=(float)((int64_t)r6[oc]+b64)*s;
                    out_f[(m+7)*N+n]=(float)((int64_t)r7[oc]+b64)*s;
                    out_f[(m+8)*N+n]=(float)((int64_t)r8[oc]+b64)*s;
                    out_f[(m+9)*N+n]=(float)((int64_t)r9[oc]+b64)*s;
                    out_f[(m+10)*N+n]=(float)((int64_t)r10[oc]+b64)*s;
                    out_f[(m+11)*N+n]=(float)((int64_t)r11[oc]+b64)*s;
                    out_f[(m+12)*N+n]=(float)((int64_t)r12[oc]+b64)*s;
                    out_f[(m+13)*N+n]=(float)((int64_t)r13[oc]+b64)*s;
                    out_f[(m+14)*N+n]=(float)((int64_t)r14[oc]+b64)*s;
                    out_f[(m+15)*N+n]=(float)((int64_t)r15[oc]+b64)*s;
                } else {
                    int8_t* out_i = static_cast<int8_t*>(C);
                    int32_t r0[16],r1[16],r2[16],r3[16],r4[16],r5[16],r6[16],r7[16];
                    int32_t r8[16],r9[16],r10[16],r11[16],r12[16],r13[16],r14[16],r15[16];
                    _mm512_storeu_si512(r0,acc0); _mm512_storeu_si512(r1,acc1);
                    _mm512_storeu_si512(r2,acc2); _mm512_storeu_si512(r3,acc3);
                    _mm512_storeu_si512(r4,acc4); _mm512_storeu_si512(r5,acc5);
                    _mm512_storeu_si512(r6,acc6); _mm512_storeu_si512(r7,acc7);
                    _mm512_storeu_si512(r8,acc8); _mm512_storeu_si512(r9,acc9);
                    _mm512_storeu_si512(r10,acc10); _mm512_storeu_si512(r11,acc11);
                    _mm512_storeu_si512(r12,acc12); _mm512_storeu_si512(r13,acc13);
                    _mm512_storeu_si512(r14,acc14); _mm512_storeu_si512(r15,acc15);
                    int32_t b32 = (int32_t)b64;
                    out_i[(m+0)*N+n]=requant((int64_t)r0[oc]+b32,s,out_zp);
                    out_i[(m+1)*N+n]=requant((int64_t)r1[oc]+b32,s,out_zp);
                    out_i[(m+2)*N+n]=requant((int64_t)r2[oc]+b32,s,out_zp);
                    out_i[(m+3)*N+n]=requant((int64_t)r3[oc]+b32,s,out_zp);
                    out_i[(m+4)*N+n]=requant((int64_t)r4[oc]+b32,s,out_zp);
                    out_i[(m+5)*N+n]=requant((int64_t)r5[oc]+b32,s,out_zp);
                    out_i[(m+6)*N+n]=requant((int64_t)r6[oc]+b32,s,out_zp);
                    out_i[(m+7)*N+n]=requant((int64_t)r7[oc]+b32,s,out_zp);
                    out_i[(m+8)*N+n]=requant((int64_t)r8[oc]+b32,s,out_zp);
                    out_i[(m+9)*N+n]=requant((int64_t)r9[oc]+b32,s,out_zp);
                    out_i[(m+10)*N+n]=requant((int64_t)r10[oc]+b32,s,out_zp);
                    out_i[(m+11)*N+n]=requant((int64_t)r11[oc]+b32,s,out_zp);
                    out_i[(m+12)*N+n]=requant((int64_t)r12[oc]+b32,s,out_zp);
                    out_i[(m+13)*N+n]=requant((int64_t)r13[oc]+b32,s,out_zp);
                    out_i[(m+14)*N+n]=requant((int64_t)r14[oc]+b32,s,out_zp);
                    out_i[(m+15)*N+n]=requant((int64_t)r15[oc]+b32,s,out_zp);
                }
            }
        }
    }
}
#endif

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
            memcpy(dst,    base,           7);
            memcpy(dst+ 7, base + W,       7);
            memcpy(dst+14, base + 2 * W,   7);
            memcpy(dst+21, base + 3 * W,   7);
            memcpy(dst+28, base + 4 * W,   7);
            memcpy(dst+35, base + 5 * W,   7);
            memcpy(dst+42, base + 6 * W,   7);
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
                        memset(dst + idx, pad_val, kW);
                        idx += kW;
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
    int groups,
    int8_t* scratch_col)
{
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    int C_in_g  = C_in  / groups;
    int C_out_g = C_out / groups;
    int K = kH * kW * C_in_g;
    int oHW = oH * oW;

    std::vector<int8_t> _col_local;
    int8_t* col = scratch_col;
    if (!col) {
        _col_local.resize((size_t)oHW * K);
        col = _col_local.data();
    }

    for (int g = 0; g < groups; ++g) {
        const int8_t* in_g  = input  + g * C_in_g * H * W;
        const int8_t* w_g   = weight + g * C_out_g * K;
        int8_t*       out_g = output + g * C_out_g * oHW;

        int8_t* w_packed_local = nullptr;
        const int8_t* w_pack_ptr;
        if (w_pre_packed && groups == 1) {
            w_pack_ptr = w_pre_packed;
        } else {
            w_packed_local = pack_weights_sdot(w_g, C_out_g, K);
            w_pack_ptr = w_packed_local;
        }

#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            double t0 = now_ms();
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
                gemm_int8_avx512(col, w_pack_ptr,
                               eff_bias + g * C_out_g,
                               req_scale + g * C_out_g,
                               out_zp, out_g, false,
                               oHW, K, C_out_g,
                               /*nchw_out=*/true, /*in_parallel=*/true);
            }
            g_gemm_ms += now_ms() - t0;
        } else
#endif
        {
            {
                double t0 = now_ms();
                if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0) {
                    im2col_1x1(in_g, C_in_g, H, W, stride_h, stride_w, oH, oW, col);
                    g_im2col_1x1_ms += now_ms() - t0;
                } else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s1p1(in_g, C_in_g, H, W, in_zp, col);
                    g_im2col_3x3_ms += now_ms() - t0;
                } else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s2p1(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                    g_im2col_3x3_ms += now_ms() - t0;
                } else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                           && pad_h == 3 && pad_w == 3) {
                    im2col_7x7s2p3(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                    g_im2col_gen_ms += now_ms() - t0;
                } else {
                    im2col(in_g, C_in_g, H, W, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, oH, oW,
                           col, in_zp);
                    g_im2col_gen_ms += now_ms() - t0;
                }
            }
            {
                double t0 = now_ms();
                gemm_int8_avx512(col, w_pack_ptr,
                               eff_bias + g * C_out_g,
                               req_scale + g * C_out_g,
                               out_zp, out_g, false,
                               oHW, K, C_out_g,
                               /*nchw_out=*/true);
                g_gemm_ms += now_ms() - t0;
            }
        }

        if (w_packed_local) free_packed(w_packed_local);
    }
}

void linear_int8_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out)
{
    int8_t* w_packed = pack_weights_sdot(weight, C_out, C_in);
    gemm_int8_avx512(
        input, w_packed,
        eff_bias, req_scale, 0,
        output, /*is_float=*/true,
        1, C_in, C_out,
        /*nchw_out=*/false);
    free_packed(w_packed);
}

void maxpool_int8(
    const int8_t* input,
    int8_t*       output,
    int C, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    double _t0_mp = now_ms();
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    for (int c = 0; c < C; ++c) {
        const int8_t* in_c = input + c * H * W;
        int8_t*       out_c = output + c * oH * oW;

        for (int oh = oh_lo; oh <= oh_hi; ++oh) {
            const int ih0 = oh * stride_h - pad_h;
            for (int ow = ow_lo; ow <= ow_hi; ++ow) {
                const int iw0 = ow * stride_w - pad_w;
                int8_t mx = -128;
                for (int kh = 0; kh < kH; ++kh) {
                    const int8_t* row = in_c + (ih0 + kh) * W + iw0;
                    for (int kw = 0; kw < kW; ++kw) {
                        if (row[kw] > mx) mx = row[kw];
                    }
                }
                out_c[oh * oW + ow] = mx;
            }
        }

        for (int oh = 0; oh < oH; ++oh) {
            for (int ow = 0; ow < oW; ++ow) {
                if (oh >= oh_lo && oh <= oh_hi && ow >= ow_lo && ow <= ow_hi) continue;
                int8_t mx = -128;
                for (int kh = 0; kh < kH; ++kh) {
                    int ih = oh * stride_h - pad_h + kh;
                    if (ih < 0 || ih >= H) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int iw = ow * stride_w - pad_w + kw;
                        if (iw < 0 || iw >= W) continue;
                        int8_t v = in_c[ih * W + iw];
                        if (v > mx) mx = v;
                    }
                }
                out_c[oh * oW + ow] = mx;
            }
        }
    }
    g_maxpool_ms += now_ms() - _t0_mp;
}

void avgpool_global_int8(
    const int8_t* input,
    float  in_scale,
    int    in_zp,
    float  out_scale,
    int    out_zp,
    int8_t* output,
    int C, int H, int W)
{
    int HW = H * W;
    float m = in_scale / out_scale;

    for (int c = 0; c < C; ++c) {
        int64_t sum = 0;
        const int8_t* in_c = input + c * HW;
        for (int i = 0; i < HW; ++i)
            sum += static_cast<int64_t>(in_c[i]) - in_zp;
        float avg = static_cast<float>(sum) / static_cast<float>(HW);
        int32_t q = static_cast<int32_t>(std::roundf(avg * m)) + out_zp;
        output[c] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
}

void add_requant_int8(
    const int8_t* in1,
    const int8_t* in2,
    float in1_scale, int in1_zp,
    float in2_scale, int in2_zp,
    float out_scale, int out_zp,
    int8_t* output,
    int N)
{
    double _t0_add = now_ms();
    float m1 = in1_scale / out_scale;
    float m2 = in2_scale / out_scale;

    for (int i = 0; i < N; ++i) {
        float v = (static_cast<float>(in1[i]) - in1_zp) * m1
                + (static_cast<float>(in2[i]) - in2_zp) * m2;
        int32_t q = static_cast<int32_t>(std::roundf(v)) + out_zp;
        output[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
    g_add_req_ms += now_ms() - _t0_add;
}

int8_t* pack_weights_ternary(const int8_t* w, int C_out, int K)
{
    return pack_weights_sdot(w, C_out, K);
}

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
    int groups,
    int8_t* scratch_col)
{
    int oH = (H + 2 * pad_h - kH) / stride_h + 1;
    int oW = (W + 2 * pad_w - kW) / stride_w + 1;

    int C_in_g  = C_in  / groups;
    int C_out_g = C_out / groups;
    int K = kH * kW * C_in_g;
    int oHW = oH * oW;

    std::vector<int8_t> _col_local;
    int8_t* col = scratch_col;
    if (!col) {
        _col_local.resize((size_t)oHW * K);
        col = _col_local.data();
    }

    for (int g = 0; g < groups; ++g) {
        const int8_t* in_g  = input  + g * C_in_g * H * W;
        const int8_t* w_g   = weight + g * C_out_g * K;
        int8_t*       out_g = output + g * C_out_g * oHW;

        int8_t* w_packed_local = nullptr;
        const int8_t* w_pack_ptr;
        if (w_pre_packed && groups == 1) {
            w_pack_ptr = w_pre_packed;
        } else {
            w_packed_local = pack_weights_ternary(w_g, C_out_g, K);
            w_pack_ptr = w_packed_local;
        }

#ifdef _OPENMP
        const int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            double t0 = now_ms();
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
                gemm_ternary_avx512(col, w_pack_ptr,
                               eff_bias + g * C_out_g,
                               req_scale + g * C_out_g,
                               out_zp, out_g, false,
                               oHW, K, C_out_g,
                               /*nchw_out=*/true, /*in_parallel=*/true);
            }
            g_gemm_ms += now_ms() - t0;
        } else
#endif
        {
            {
                double t0 = now_ms();
                if (kH == 1 && kW == 1 && pad_h == 0 && pad_w == 0) {
                    im2col_1x1(in_g, C_in_g, H, W, stride_h, stride_w, oH, oW, col);
                    g_im2col_1x1_ms += now_ms() - t0;
                } else if (kH == 3 && kW == 3 && stride_h == 1 && stride_w == 1
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s1p1(in_g, C_in_g, H, W, in_zp, col);
                    g_im2col_3x3_ms += now_ms() - t0;
                } else if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2
                           && pad_h == 1 && pad_w == 1) {
                    im2col_3x3s2p1(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                    g_im2col_3x3_ms += now_ms() - t0;
                } else if (kH == 7 && kW == 7 && stride_h == 2 && stride_w == 2
                           && pad_h == 3 && pad_w == 3) {
                    im2col_7x7s2p3(in_g, C_in_g, H, W, in_zp, oH, oW, col);
                    g_im2col_gen_ms += now_ms() - t0;
                } else {
                    im2col(in_g, C_in_g, H, W, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, oH, oW,
                           col, in_zp);
                    g_im2col_gen_ms += now_ms() - t0;
                }
            }
            {
                double t0 = now_ms();
                gemm_ternary_avx512(col, w_pack_ptr,
                               eff_bias + g * C_out_g,
                               req_scale + g * C_out_g,
                               out_zp, out_g, false,
                               oHW, K, C_out_g,
                               /*nchw_out=*/true, /*in_parallel=*/false);
                g_gemm_ms += now_ms() - t0;
            }
        }

        if (w_packed_local) free_packed(w_packed_local);
    }
}

void linear_ternary_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out)
{
    linear_int8_to_float(input, weight, eff_bias, req_scale, output, C_in, C_out);
}

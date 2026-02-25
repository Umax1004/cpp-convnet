#include "ops_neon.hpp"
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

// ── Op profiling accumulators (accumulated across all conv calls) ──────────
static double g_im2col_1x1_ms  = 0.0;
static double g_im2col_3x3_ms  = 0.0;
static double g_im2col_gen_ms  = 0.0;
static double g_gemm_ms        = 0.0;
static double g_add_req_ms     = 0.0;
static double g_maxpool_ms     = 0.0;

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

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// ──────────────────────────────────────────────────────────────
// Weight packing for SDOT: [C_out, K] → [C_out/4, K/4, 4, 4]
// Each 4×4 block: 4 consecutive out-channels × 4 consecutive K elements.
// Both dims are padded to multiples of 4 with zeros.
// ──────────────────────────────────────────────────────────────
int8_t* pack_weights_sdot(const int8_t* w, int C_out, int K)
{
    int Co4 = (C_out + 3) / 4;
    int K4  = (K    + 3) / 4;
    int sz  = Co4 * K4 * 16;
    int8_t* packed = new int8_t[sz]();   // zero-init for padding

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


// ──────────────────────────────────────────────────────────────
// INT8 GEMM  A[M,K] × B_packed → C
//
// Loop order: co_blk (N/4) outer → B slice stays in L1 across all M rows.
// Inner m-loop unrolled by 4 → 4× compute-to-load ratio improvement.
// nchw_out=true: writes C[n*M + m] (NCHW layout) instead of C[m*N + n].
//   Eliminates the caller's post-GEMM [oHW,N]→[N,oHW] transpose.
//
// Recursive dispatch: when in_parallel=false, forks an OMP team and calls
// itself with in_parallel=true; each thread runs the body directly using
// omp_get_thread_num()/omp_get_num_threads() from the enclosing team.
// ──────────────────────────────────────────────────────────────
void gemm_int8_neon(
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
            gemm_int8_neon(A, B_packed, eff_bias, req_scale, out_zp, C, is_float,
                           M, K, N, nchw_out, /*in_parallel=*/true);
            return;
        }
#endif
    }
    int K4  = (K + 3) / 4;
    int Co4 = (N + 3) / 4;

#ifdef __ARM_NEON
    // Hybrid M/N partition — thread-local range computation.
    //  M >= N  →  M-partition: thread tid owns m-tiles [mi_s, mi_e).
    //  M < N   →  N-partition: all threads own co_blks [co_s, co_e).
    const int k_full = K / 4;
    const int m16_count = M / 16;

    // ── Thread-local range computation ────────────────────────────────────────
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
    const int nT  = omp_get_num_threads();
#else
    const int tid = 0;
    const int nT  = 1;
#endif
    const bool n_part = (nT > 1) && (M < N);
    // M-partition ranges (used when M >= N)
    const int mi_s = n_part ? 0           : (tid * m16_count) / nT;
    const int mi_e = n_part ? m16_count   : ((tid + 1) * m16_count) / nT;
    // N-partition ranges (used when M < N)
    const int co_s = n_part ? (tid * Co4) / nT       : 0;
    const int co_e = n_part ? ((tid + 1) * Co4) / nT : Co4;

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
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0  = vdupq_n_s32(0), acc1  = vdupq_n_s32(0);
            int32x4_t acc2  = vdupq_n_s32(0), acc3  = vdupq_n_s32(0);
            int32x4_t acc4  = vdupq_n_s32(0), acc5  = vdupq_n_s32(0);
            int32x4_t acc6  = vdupq_n_s32(0), acc7  = vdupq_n_s32(0);
            int32x4_t acc8  = vdupq_n_s32(0), acc9  = vdupq_n_s32(0);
            int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
            int32x4_t acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
            int32x4_t acc14 = vdupq_n_s32(0), acc15 = vdupq_n_s32(0);

            // 4-k_blk-at-a-time: load B once (4 vectors), load A in two halves
            // of 8 rows each, reusing B across both halves.
            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t bv0 = vld1q_s8(b_co + (k_blk+0) * 16);
                int8x16_t bv1 = vld1q_s8(b_co + (k_blk+1) * 16);
                int8x16_t bv2 = vld1q_s8(b_co + (k_blk+2) * 16);
                int8x16_t bv3 = vld1q_s8(b_co + (k_blk+3) * 16);
                // Rows 0–7
                int8x16_t av0=vld1q_s8(a0+k_blk*4), av1=vld1q_s8(a1+k_blk*4);
                int8x16_t av2=vld1q_s8(a2+k_blk*4), av3=vld1q_s8(a3+k_blk*4);
                int8x16_t av4=vld1q_s8(a4+k_blk*4), av5=vld1q_s8(a5+k_blk*4);
                int8x16_t av6=vld1q_s8(a6+k_blk*4), av7=vld1q_s8(a7+k_blk*4);
                acc0=vdotq_laneq_s32(acc0,bv0,av0,0); acc0=vdotq_laneq_s32(acc0,bv1,av0,1);
                acc0=vdotq_laneq_s32(acc0,bv2,av0,2); acc0=vdotq_laneq_s32(acc0,bv3,av0,3);
                acc1=vdotq_laneq_s32(acc1,bv0,av1,0); acc1=vdotq_laneq_s32(acc1,bv1,av1,1);
                acc1=vdotq_laneq_s32(acc1,bv2,av1,2); acc1=vdotq_laneq_s32(acc1,bv3,av1,3);
                acc2=vdotq_laneq_s32(acc2,bv0,av2,0); acc2=vdotq_laneq_s32(acc2,bv1,av2,1);
                acc2=vdotq_laneq_s32(acc2,bv2,av2,2); acc2=vdotq_laneq_s32(acc2,bv3,av2,3);
                acc3=vdotq_laneq_s32(acc3,bv0,av3,0); acc3=vdotq_laneq_s32(acc3,bv1,av3,1);
                acc3=vdotq_laneq_s32(acc3,bv2,av3,2); acc3=vdotq_laneq_s32(acc3,bv3,av3,3);
                acc4=vdotq_laneq_s32(acc4,bv0,av4,0); acc4=vdotq_laneq_s32(acc4,bv1,av4,1);
                acc4=vdotq_laneq_s32(acc4,bv2,av4,2); acc4=vdotq_laneq_s32(acc4,bv3,av4,3);
                acc5=vdotq_laneq_s32(acc5,bv0,av5,0); acc5=vdotq_laneq_s32(acc5,bv1,av5,1);
                acc5=vdotq_laneq_s32(acc5,bv2,av5,2); acc5=vdotq_laneq_s32(acc5,bv3,av5,3);
                acc6=vdotq_laneq_s32(acc6,bv0,av6,0); acc6=vdotq_laneq_s32(acc6,bv1,av6,1);
                acc6=vdotq_laneq_s32(acc6,bv2,av6,2); acc6=vdotq_laneq_s32(acc6,bv3,av6,3);
                acc7=vdotq_laneq_s32(acc7,bv0,av7,0); acc7=vdotq_laneq_s32(acc7,bv1,av7,1);
                acc7=vdotq_laneq_s32(acc7,bv2,av7,2); acc7=vdotq_laneq_s32(acc7,bv3,av7,3);
                // Rows 8–15 (reuse bv0..bv3)
                int8x16_t av8 =vld1q_s8(a8 +k_blk*4), av9 =vld1q_s8(a9 +k_blk*4);
                int8x16_t av10=vld1q_s8(a10+k_blk*4), av11=vld1q_s8(a11+k_blk*4);
                int8x16_t av12=vld1q_s8(a12+k_blk*4), av13=vld1q_s8(a13+k_blk*4);
                int8x16_t av14=vld1q_s8(a14+k_blk*4), av15=vld1q_s8(a15+k_blk*4);
                acc8 =vdotq_laneq_s32(acc8 ,bv0,av8 ,0); acc8 =vdotq_laneq_s32(acc8 ,bv1,av8 ,1);
                acc8 =vdotq_laneq_s32(acc8 ,bv2,av8 ,2); acc8 =vdotq_laneq_s32(acc8 ,bv3,av8 ,3);
                acc9 =vdotq_laneq_s32(acc9 ,bv0,av9 ,0); acc9 =vdotq_laneq_s32(acc9 ,bv1,av9 ,1);
                acc9 =vdotq_laneq_s32(acc9 ,bv2,av9 ,2); acc9 =vdotq_laneq_s32(acc9 ,bv3,av9 ,3);
                acc10=vdotq_laneq_s32(acc10,bv0,av10,0); acc10=vdotq_laneq_s32(acc10,bv1,av10,1);
                acc10=vdotq_laneq_s32(acc10,bv2,av10,2); acc10=vdotq_laneq_s32(acc10,bv3,av10,3);
                acc11=vdotq_laneq_s32(acc11,bv0,av11,0); acc11=vdotq_laneq_s32(acc11,bv1,av11,1);
                acc11=vdotq_laneq_s32(acc11,bv2,av11,2); acc11=vdotq_laneq_s32(acc11,bv3,av11,3);
                acc12=vdotq_laneq_s32(acc12,bv0,av12,0); acc12=vdotq_laneq_s32(acc12,bv1,av12,1);
                acc12=vdotq_laneq_s32(acc12,bv2,av12,2); acc12=vdotq_laneq_s32(acc12,bv3,av12,3);
                acc13=vdotq_laneq_s32(acc13,bv0,av13,0); acc13=vdotq_laneq_s32(acc13,bv1,av13,1);
                acc13=vdotq_laneq_s32(acc13,bv2,av13,2); acc13=vdotq_laneq_s32(acc13,bv3,av13,3);
                acc14=vdotq_laneq_s32(acc14,bv0,av14,0); acc14=vdotq_laneq_s32(acc14,bv1,av14,1);
                acc14=vdotq_laneq_s32(acc14,bv2,av14,2); acc14=vdotq_laneq_s32(acc14,bv3,av14,3);
                acc15=vdotq_laneq_s32(acc15,bv0,av15,0); acc15=vdotq_laneq_s32(acc15,bv1,av15,1);
                acc15=vdotq_laneq_s32(acc15,bv2,av15,2); acc15=vdotq_laneq_s32(acc15,bv3,av15,3);
            }
            // Scalar tail for remaining k_blks (0–3)
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 16);
                int32_t i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15;
                memcpy(&i0, a0+k_blk*4,4); memcpy(&i1, a1+k_blk*4,4);
                memcpy(&i2, a2+k_blk*4,4); memcpy(&i3, a3+k_blk*4,4);
                memcpy(&i4, a4+k_blk*4,4); memcpy(&i5, a5+k_blk*4,4);
                memcpy(&i6, a6+k_blk*4,4); memcpy(&i7, a7+k_blk*4,4);
                memcpy(&i8, a8+k_blk*4,4); memcpy(&i9, a9+k_blk*4,4);
                memcpy(&i10,a10+k_blk*4,4); memcpy(&i11,a11+k_blk*4,4);
                memcpy(&i12,a12+k_blk*4,4); memcpy(&i13,a13+k_blk*4,4);
                memcpy(&i14,a14+k_blk*4,4); memcpy(&i15,a15+k_blk*4,4);
                #define SDOT1(acc,iv) acc=vdotq_s32(acc,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wv)
                SDOT1(acc0,i0); SDOT1(acc1,i1); SDOT1(acc2,i2); SDOT1(acc3,i3);
                SDOT1(acc4,i4); SDOT1(acc5,i5); SDOT1(acc6,i6); SDOT1(acc7,i7);
                SDOT1(acc8,i8); SDOT1(acc9,i9); SDOT1(acc10,i10); SDOT1(acc11,i11);
                SDOT1(acc12,i12); SDOT1(acc13,i13); SDOT1(acc14,i14); SDOT1(acc15,i15);
                #undef SDOT1
            }
            if (K & 3) {
                int k_base = k_full * 4;
                auto tail8 = [&](int32x4_t& acc, const int8_t* ar) {
                    uint8_t t[4]={};
                    for (int i=0;i<(K&3);++i) t[i]=(uint8_t)ar[k_base+i];
                    int32_t iv; memcpy(&iv,t,4);
                    int8x16_t wv=vld1q_s8(b_co+k_full*16);
                    acc=vdotq_s32(acc,vreinterpretq_s8_s32(vdupq_n_s32(iv)),wv);
                };
                tail8(acc0,a0); tail8(acc1,a1); tail8(acc2,a2); tail8(acc3,a3);
                tail8(acc4,a4); tail8(acc5,a5); tail8(acc6,a6); tail8(acc7,a7);
                tail8(acc8,a8); tail8(acc9,a9); tail8(acc10,a10); tail8(acc11,a11);
                tail8(acc12,a12); tail8(acc13,a13); tail8(acc14,a14); tail8(acc15,a15);
            }

            // Write 16 rows of output
            if (nchw_out) {
                // Use vtrnq_s32 to transpose 4 groups of [4rows×4chans] → [4chans×4rows].
                // Eliminates element extraction from stack arrays.
                auto do4x4 = [](int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d,
                                int32x4_t& c0, int32x4_t& c1, int32x4_t& c2, int32x4_t& c3) {
                    int32x4x2_t tr0 = vtrnq_s32(a, b);
                    int32x4x2_t tr1 = vtrnq_s32(c, d);
                    c0 = vcombine_s32(vget_low_s32(tr0.val[0]), vget_low_s32(tr1.val[0]));
                    c1 = vcombine_s32(vget_low_s32(tr0.val[1]), vget_low_s32(tr1.val[1]));
                    c2 = vcombine_s32(vget_high_s32(tr0.val[0]), vget_high_s32(tr1.val[0]));
                    c3 = vcombine_s32(vget_high_s32(tr0.val[1]), vget_high_s32(tr1.val[1]));
                };
                // g{k}c{j} = 4 row-values (rows k*4..k*4+3) for output channel j
                int32x4_t g0c0,g0c1,g0c2,g0c3; do4x4(acc0, acc1, acc2, acc3,   g0c0,g0c1,g0c2,g0c3);
                int32x4_t g1c0,g1c1,g1c2,g1c3; do4x4(acc4, acc5, acc6, acc7,   g1c0,g1c1,g1c2,g1c3);
                int32x4_t g2c0,g2c1,g2c2,g2c3; do4x4(acc8, acc9, acc10,acc11,  g2c0,g2c1,g2c2,g2c3);
                int32x4_t g3c0,g3c1,g3c2,g3c3; do4x4(acc12,acc13,acc14,acc15,  g3c0,g3c1,g3c2,g3c3);
                // Write 16 int8 bytes per channel: rows 0-3, 4-7, 8-11, 12-15
                auto write16 = [&](int oc,
                                   int32x4_t a, int32x4_t b,
                                   int32x4_t c, int32x4_t d) {
                    const int n   = n0 + oc;
                    const float s = req_scale[n];
                    const float bias_f = (float)eff_bias[n] * s + (float)out_zp;
                    const float32x4_t vbias = vdupq_n_f32(bias_f);
                    auto rq4 = [&](int32x4_t v) {
                        return vcvtnq_s32_f32(vfmaq_n_f32(vbias, vcvtq_f32_s32(v), s));
                    };
                    int8_t* dst = static_cast<int8_t*>(C) + n * M + m;
                    vst1q_s8(dst, vcombine_s8(
                        vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(a)), vqmovn_s32(rq4(b)))),
                        vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(c)), vqmovn_s32(rq4(d))))));
                };
                if (n_valid > 0) write16(0, g0c0,g1c0,g2c0,g3c0);
                if (n_valid > 1) write16(1, g0c1,g1c1,g2c1,g3c1);
                if (n_valid > 2) write16(2, g0c2,g1c2,g2c2,g3c2);
                if (n_valid > 3) write16(3, g0c3,g1c3,g2c3,g3c3);
            } else {
                int32_t r[16][4];
                vst1q_s32(r[0],acc0);  vst1q_s32(r[1],acc1);  vst1q_s32(r[2],acc2);  vst1q_s32(r[3],acc3);
                vst1q_s32(r[4],acc4);  vst1q_s32(r[5],acc5);  vst1q_s32(r[6],acc6);  vst1q_s32(r[7],acc7);
                vst1q_s32(r[8],acc8);  vst1q_s32(r[9],acc9);  vst1q_s32(r[10],acc10); vst1q_s32(r[11],acc11);
                vst1q_s32(r[12],acc12); vst1q_s32(r[13],acc13); vst1q_s32(r[14],acc14); vst1q_s32(r[15],acc15);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    const float   s   = req_scale[n];
                    if (is_float) {
                        float* out_f = static_cast<float*>(C);
                        for (int row = 0; row < 16; ++row)
                            out_f[(m+row)*N+n] = (float)((int64_t)r[row][oc]+b64)*s;
                    } else {
                        int8_t* out_i = static_cast<int8_t*>(C);
                        for (int row = 0; row < 16; ++row)
                            out_i[(m+row)*N+n] = requant((int64_t)r[row][oc]+(int32_t)b64, s, out_zp);
                    }
                }
            }
        }   // end co_blk loop for 16-row tile
    }   // end 16-row mi loop

    // ── 8/4/scalar row tail loops (remainder < 16 rows) ────────────────────────
    // M-partition: only thread 0 runs the tail (≤15 rows, cost is negligible).
    // N-partition: all threads run the tail with their co_blk range.
    if (n_part || tid == 0) {
    const int co_s2 = n_part ? co_s : 0;
    const int co_e2 = n_part ? co_e : Co4;
    int m = m16_count * 16;
    for (; m + 7 < M; m += 8) {
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;
        const int8_t* a2 = A + (size_t)(m+2) * K;
        const int8_t* a3 = A + (size_t)(m+3) * K;
        const int8_t* a4 = A + (size_t)(m+4) * K;
        const int8_t* a5 = A + (size_t)(m+5) * K;
        const int8_t* a6 = A + (size_t)(m+6) * K;
        const int8_t* a7 = A + (size_t)(m+7) * K;

        for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);
            int32x4_t acc4 = vdupq_n_s32(0), acc5 = vdupq_n_s32(0);
            int32x4_t acc6 = vdupq_n_s32(0), acc7 = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t av0 = vld1q_s8(a0 + k_blk * 4);
                int8x16_t av1 = vld1q_s8(a1 + k_blk * 4);
                int8x16_t av2 = vld1q_s8(a2 + k_blk * 4);
                int8x16_t av3 = vld1q_s8(a3 + k_blk * 4);
                int8x16_t av4 = vld1q_s8(a4 + k_blk * 4);
                int8x16_t av5 = vld1q_s8(a5 + k_blk * 4);
                int8x16_t av6 = vld1q_s8(a6 + k_blk * 4);
                int8x16_t av7 = vld1q_s8(a7 + k_blk * 4);
                int8x16_t bv0 = vld1q_s8(b_co + (k_blk+0) * 16);
                int8x16_t bv1 = vld1q_s8(b_co + (k_blk+1) * 16);
                int8x16_t bv2 = vld1q_s8(b_co + (k_blk+2) * 16);
                int8x16_t bv3 = vld1q_s8(b_co + (k_blk+3) * 16);
                acc0 = vdotq_laneq_s32(acc0, bv0, av0, 0); acc0 = vdotq_laneq_s32(acc0, bv1, av0, 1);
                acc0 = vdotq_laneq_s32(acc0, bv2, av0, 2); acc0 = vdotq_laneq_s32(acc0, bv3, av0, 3);
                acc1 = vdotq_laneq_s32(acc1, bv0, av1, 0); acc1 = vdotq_laneq_s32(acc1, bv1, av1, 1);
                acc1 = vdotq_laneq_s32(acc1, bv2, av1, 2); acc1 = vdotq_laneq_s32(acc1, bv3, av1, 3);
                acc2 = vdotq_laneq_s32(acc2, bv0, av2, 0); acc2 = vdotq_laneq_s32(acc2, bv1, av2, 1);
                acc2 = vdotq_laneq_s32(acc2, bv2, av2, 2); acc2 = vdotq_laneq_s32(acc2, bv3, av2, 3);
                acc3 = vdotq_laneq_s32(acc3, bv0, av3, 0); acc3 = vdotq_laneq_s32(acc3, bv1, av3, 1);
                acc3 = vdotq_laneq_s32(acc3, bv2, av3, 2); acc3 = vdotq_laneq_s32(acc3, bv3, av3, 3);
                acc4 = vdotq_laneq_s32(acc4, bv0, av4, 0); acc4 = vdotq_laneq_s32(acc4, bv1, av4, 1);
                acc4 = vdotq_laneq_s32(acc4, bv2, av4, 2); acc4 = vdotq_laneq_s32(acc4, bv3, av4, 3);
                acc5 = vdotq_laneq_s32(acc5, bv0, av5, 0); acc5 = vdotq_laneq_s32(acc5, bv1, av5, 1);
                acc5 = vdotq_laneq_s32(acc5, bv2, av5, 2); acc5 = vdotq_laneq_s32(acc5, bv3, av5, 3);
                acc6 = vdotq_laneq_s32(acc6, bv0, av6, 0); acc6 = vdotq_laneq_s32(acc6, bv1, av6, 1);
                acc6 = vdotq_laneq_s32(acc6, bv2, av6, 2); acc6 = vdotq_laneq_s32(acc6, bv3, av6, 3);
                acc7 = vdotq_laneq_s32(acc7, bv0, av7, 0); acc7 = vdotq_laneq_s32(acc7, bv1, av7, 1);
                acc7 = vdotq_laneq_s32(acc7, bv2, av7, 2); acc7 = vdotq_laneq_s32(acc7, bv3, av7, 3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 16);
                int32_t i0, i1, i2, i3, i4, i5, i6, i7;
                memcpy(&i0, a0+k_blk*4, 4); memcpy(&i1, a1+k_blk*4, 4);
                memcpy(&i2, a2+k_blk*4, 4); memcpy(&i3, a3+k_blk*4, 4);
                memcpy(&i4, a4+k_blk*4, 4); memcpy(&i5, a5+k_blk*4, 4);
                memcpy(&i6, a6+k_blk*4, 4); memcpy(&i7, a7+k_blk*4, 4);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(i0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(i1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(i2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(i3)), wv);
                acc4 = vdotq_s32(acc4, vreinterpretq_s8_s32(vdupq_n_s32(i4)), wv);
                acc5 = vdotq_s32(acc5, vreinterpretq_s8_s32(vdupq_n_s32(i5)), wv);
                acc6 = vdotq_s32(acc6, vreinterpretq_s8_s32(vdupq_n_s32(i6)), wv);
                acc7 = vdotq_s32(acc7, vreinterpretq_s8_s32(vdupq_n_s32(i7)), wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={},t1[4]={},t2[4]={},t3[4]={},t4[4]={},t5[4]={},t6[4]={},t7[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i]=(uint8_t)a0[k_base+i]; t1[i]=(uint8_t)a1[k_base+i];
                    t2[i]=(uint8_t)a2[k_base+i]; t3[i]=(uint8_t)a3[k_base+i];
                    t4[i]=(uint8_t)a4[k_base+i]; t5[i]=(uint8_t)a5[k_base+i];
                    t6[i]=(uint8_t)a6[k_base+i]; t7[i]=(uint8_t)a7[k_base+i];
                }
                int32_t i0,i1,i2,i3,i4,i5,i6,i7;
                memcpy(&i0,t0,4); memcpy(&i1,t1,4); memcpy(&i2,t2,4); memcpy(&i3,t3,4);
                memcpy(&i4,t4,4); memcpy(&i5,t5,4); memcpy(&i6,t6,4); memcpy(&i7,t7,4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 16);
                acc0=vdotq_s32(acc0,vreinterpretq_s8_s32(vdupq_n_s32(i0)),wv);
                acc1=vdotq_s32(acc1,vreinterpretq_s8_s32(vdupq_n_s32(i1)),wv);
                acc2=vdotq_s32(acc2,vreinterpretq_s8_s32(vdupq_n_s32(i2)),wv);
                acc3=vdotq_s32(acc3,vreinterpretq_s8_s32(vdupq_n_s32(i3)),wv);
                acc4=vdotq_s32(acc4,vreinterpretq_s8_s32(vdupq_n_s32(i4)),wv);
                acc5=vdotq_s32(acc5,vreinterpretq_s8_s32(vdupq_n_s32(i5)),wv);
                acc6=vdotq_s32(acc6,vreinterpretq_s8_s32(vdupq_n_s32(i6)),wv);
                acc7=vdotq_s32(acc7,vreinterpretq_s8_s32(vdupq_n_s32(i7)),wv);
            }

            if (nchw_out) {
                // vtrnq_s32 transpose: [8rows×4chans] → [4chans×8rows]
                int32x4x2_t tr0 = vtrnq_s32(acc0, acc1);
                int32x4x2_t tr1 = vtrnq_s32(acc2, acc3);
                int32x4x2_t tr2 = vtrnq_s32(acc4, acc5);
                int32x4x2_t tr3 = vtrnq_s32(acc6, acc7);
                int32x4_t g0c0 = vcombine_s32(vget_low_s32(tr0.val[0]), vget_low_s32(tr1.val[0]));
                int32x4_t g0c1 = vcombine_s32(vget_low_s32(tr0.val[1]), vget_low_s32(tr1.val[1]));
                int32x4_t g0c2 = vcombine_s32(vget_high_s32(tr0.val[0]), vget_high_s32(tr1.val[0]));
                int32x4_t g0c3 = vcombine_s32(vget_high_s32(tr0.val[1]), vget_high_s32(tr1.val[1]));
                int32x4_t g1c0 = vcombine_s32(vget_low_s32(tr2.val[0]), vget_low_s32(tr3.val[0]));
                int32x4_t g1c1 = vcombine_s32(vget_low_s32(tr2.val[1]), vget_low_s32(tr3.val[1]));
                int32x4_t g1c2 = vcombine_s32(vget_high_s32(tr2.val[0]), vget_high_s32(tr3.val[0]));
                int32x4_t g1c3 = vcombine_s32(vget_high_s32(tr2.val[1]), vget_high_s32(tr3.val[1]));
                auto write8 = [&](int oc, int32x4_t a, int32x4_t b) {
                    const int n   = n0 + oc;
                    const float s = req_scale[n];
                    const float bias_f = (float)eff_bias[n] * s + (float)out_zp;
                    const float32x4_t vbias = vdupq_n_f32(bias_f);
                    auto rq4 = [&](int32x4_t v) {
                        return vcvtnq_s32_f32(vfmaq_n_f32(vbias, vcvtq_f32_s32(v), s));
                    };
                    vst1_s8(static_cast<int8_t*>(C)+n*M+m,
                            vqmovn_s16(vcombine_s16(vqmovn_s32(rq4(a)), vqmovn_s32(rq4(b)))));
                };
                if (n_valid > 0) write8(0, g0c0, g1c0);
                if (n_valid > 1) write8(1, g0c1, g1c1);
                if (n_valid > 2) write8(2, g0c2, g1c2);
                if (n_valid > 3) write8(3, g0c3, g1c3);
            } else {
                int32_t r0[4],r1[4],r2[4],r3[4],r4[4],r5[4],r6[4],r7[4];
                vst1q_s32(r0,acc0); vst1q_s32(r1,acc1); vst1q_s32(r2,acc2); vst1q_s32(r3,acc3);
                vst1q_s32(r4,acc4); vst1q_s32(r5,acc5); vst1q_s32(r6,acc6); vst1q_s32(r7,acc7);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    const float   s   = req_scale[n];
                    if (is_float) {
                        float* out_f = static_cast<float*>(C);
                        out_f[(m+0)*N+n]=(float)((int64_t)r0[oc]+b64)*s; out_f[(m+1)*N+n]=(float)((int64_t)r1[oc]+b64)*s;
                        out_f[(m+2)*N+n]=(float)((int64_t)r2[oc]+b64)*s; out_f[(m+3)*N+n]=(float)((int64_t)r3[oc]+b64)*s;
                        out_f[(m+4)*N+n]=(float)((int64_t)r4[oc]+b64)*s; out_f[(m+5)*N+n]=(float)((int64_t)r5[oc]+b64)*s;
                        out_f[(m+6)*N+n]=(float)((int64_t)r6[oc]+b64)*s; out_f[(m+7)*N+n]=(float)((int64_t)r7[oc]+b64)*s;
                    } else {
                        int8_t* out_i = static_cast<int8_t*>(C);
                        int32_t b32 = (int32_t)b64;
                        out_i[(m+0)*N+n]=requant((int64_t)r0[oc]+b32,s,out_zp); out_i[(m+1)*N+n]=requant((int64_t)r1[oc]+b32,s,out_zp);
                        out_i[(m+2)*N+n]=requant((int64_t)r2[oc]+b32,s,out_zp); out_i[(m+3)*N+n]=requant((int64_t)r3[oc]+b32,s,out_zp);
                        out_i[(m+4)*N+n]=requant((int64_t)r4[oc]+b32,s,out_zp); out_i[(m+5)*N+n]=requant((int64_t)r5[oc]+b32,s,out_zp);
                        out_i[(m+6)*N+n]=requant((int64_t)r6[oc]+b32,s,out_zp); out_i[(m+7)*N+n]=requant((int64_t)r7[oc]+b32,s,out_zp);
                    }
                }
            }
        }   // end co_blk loop for 8-row tile
    }

    for (; m + 3 < M; m += 4) {
        const int8_t* a0 = A + (size_t)(m+0) * K;
        const int8_t* a1 = A + (size_t)(m+1) * K;
        const int8_t* a2 = A + (size_t)(m+2) * K;
        const int8_t* a3 = A + (size_t)(m+3) * K;

        for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            int k_blk = 0;
            for (; k_blk + 3 < k_full; k_blk += 4) {
                int8x16_t av0 = vld1q_s8(a0 + k_blk * 4);
                int8x16_t av1 = vld1q_s8(a1 + k_blk * 4);
                int8x16_t av2 = vld1q_s8(a2 + k_blk * 4);
                int8x16_t av3 = vld1q_s8(a3 + k_blk * 4);
                int8x16_t bv0 = vld1q_s8(b_co + (k_blk+0) * 16);
                int8x16_t bv1 = vld1q_s8(b_co + (k_blk+1) * 16);
                int8x16_t bv2 = vld1q_s8(b_co + (k_blk+2) * 16);
                int8x16_t bv3 = vld1q_s8(b_co + (k_blk+3) * 16);
                acc0 = vdotq_laneq_s32(acc0, bv0, av0, 0); acc0 = vdotq_laneq_s32(acc0, bv1, av0, 1);
                acc0 = vdotq_laneq_s32(acc0, bv2, av0, 2); acc0 = vdotq_laneq_s32(acc0, bv3, av0, 3);
                acc1 = vdotq_laneq_s32(acc1, bv0, av1, 0); acc1 = vdotq_laneq_s32(acc1, bv1, av1, 1);
                acc1 = vdotq_laneq_s32(acc1, bv2, av1, 2); acc1 = vdotq_laneq_s32(acc1, bv3, av1, 3);
                acc2 = vdotq_laneq_s32(acc2, bv0, av2, 0); acc2 = vdotq_laneq_s32(acc2, bv1, av2, 1);
                acc2 = vdotq_laneq_s32(acc2, bv2, av2, 2); acc2 = vdotq_laneq_s32(acc2, bv3, av2, 3);
                acc3 = vdotq_laneq_s32(acc3, bv0, av3, 0); acc3 = vdotq_laneq_s32(acc3, bv1, av3, 1);
                acc3 = vdotq_laneq_s32(acc3, bv2, av3, 2); acc3 = vdotq_laneq_s32(acc3, bv3, av3, 3);
            }
            for (; k_blk < k_full; ++k_blk) {
                int8x16_t wv = vld1q_s8(b_co + k_blk * 16);
                int32_t in0, in1, in2, in3;
                memcpy(&in0, a0+k_blk*4, 4); memcpy(&in1, a1+k_blk*4, 4);
                memcpy(&in2, a2+k_blk*4, 4); memcpy(&in3, a3+k_blk*4, 4);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(in0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(in1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(in2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(in3)), wv);
            }
            if (K & 3) {
                int k_base = k_full * 4;
                uint8_t t0[4]={}, t1[4]={}, t2[4]={}, t3[4]={};
                for (int i = 0; i < (K & 3); ++i) {
                    t0[i] = (uint8_t)a0[k_base+i];
                    t1[i] = (uint8_t)a1[k_base+i];
                    t2[i] = (uint8_t)a2[k_base+i];
                    t3[i] = (uint8_t)a3[k_base+i];
                }
                int32_t in0, in1, in2, in3;
                memcpy(&in0, t0, 4); memcpy(&in1, t1, 4);
                memcpy(&in2, t2, 4); memcpy(&in3, t3, 4);
                int8x16_t wv = vld1q_s8(b_co + k_full * 16);
                acc0 = vdotq_s32(acc0, vreinterpretq_s8_s32(vdupq_n_s32(in0)), wv);
                acc1 = vdotq_s32(acc1, vreinterpretq_s8_s32(vdupq_n_s32(in1)), wv);
                acc2 = vdotq_s32(acc2, vreinterpretq_s8_s32(vdupq_n_s32(in2)), wv);
                acc3 = vdotq_s32(acc3, vreinterpretq_s8_s32(vdupq_n_s32(in3)), wv);
            }

            if (nchw_out) {
                // vtrnq_s32 transpose: [4rows×4chans] → [4chans×4rows]
                int32x4x2_t tr0 = vtrnq_s32(acc0, acc1);
                int32x4x2_t tr1 = vtrnq_s32(acc2, acc3);
                int32x4_t cha0 = vcombine_s32(vget_low_s32(tr0.val[0]), vget_low_s32(tr1.val[0]));
                int32x4_t cha1 = vcombine_s32(vget_low_s32(tr0.val[1]), vget_low_s32(tr1.val[1]));
                int32x4_t cha2 = vcombine_s32(vget_high_s32(tr0.val[0]), vget_high_s32(tr1.val[0]));
                int32x4_t cha3 = vcombine_s32(vget_high_s32(tr0.val[1]), vget_high_s32(tr1.val[1]));
                auto write4 = [&](int oc, int32x4_t cv) {
                    const int n   = n0 + oc;
                    const float s = req_scale[n];
                    const float bias_f = (float)eff_bias[n] * s + (float)out_zp;
                    int32x4_t qv = vcvtnq_s32_f32(vfmaq_n_f32(vdupq_n_f32(bias_f), vcvtq_f32_s32(cv), s));
                    int8_t* dst = static_cast<int8_t*>(C) + n * M + m;
                    vst1_lane_s32((int32_t*)dst,
                        vreinterpret_s32_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(qv), vqmovn_s32(qv)))), 0);
                };
                if (n_valid > 0) write4(0, cha0);
                if (n_valid > 1) write4(1, cha1);
                if (n_valid > 2) write4(2, cha2);
                if (n_valid > 3) write4(3, cha3);
            } else {
                int32_t r0[4], r1[4], r2[4], r3[4];
                vst1q_s32(r0, acc0); vst1q_s32(r1, acc1);
                vst1q_s32(r2, acc2); vst1q_s32(r3, acc3);
                for (int oc = 0; oc < n_valid; ++oc) {
                    const int n   = n0 + oc;
                    const int64_t b64 = eff_bias[n];
                    const float   s   = req_scale[n];
                    if (is_float) {
                        float* out_f = static_cast<float*>(C);
                        out_f[(m+0)*N+n] = (float)((int64_t)r0[oc]+b64) * s;
                        out_f[(m+1)*N+n] = (float)((int64_t)r1[oc]+b64) * s;
                        out_f[(m+2)*N+n] = (float)((int64_t)r2[oc]+b64) * s;
                        out_f[(m+3)*N+n] = (float)((int64_t)r3[oc]+b64) * s;
                    } else {
                        int8_t* out_i = static_cast<int8_t*>(C);
                        int32_t b32 = (int32_t)b64;
                        out_i[(m+0)*N+n] = requant((int64_t)r0[oc]+b32, s, out_zp);
                        out_i[(m+1)*N+n] = requant((int64_t)r1[oc]+b32, s, out_zp);
                        out_i[(m+2)*N+n] = requant((int64_t)r2[oc]+b32, s, out_zp);
                        out_i[(m+3)*N+n] = requant((int64_t)r3[oc]+b32, s, out_zp);
                    }
                }
            }
        }   // end co_blk loop for 4-row tile
    }

    // ── Scalar tail: remaining 0–3 rows ─────────────────────
    for (; m < M; ++m) {
        const int8_t* a_row = A + (size_t)m * K;
        for (int co_blk = co_s2; co_blk < co_e2; ++co_blk) {
            const int8_t* b_co = B_packed + (size_t)co_blk * K4 * 16;
            const int n0      = co_blk * 4;
            const int n_valid = std::min(4, N - n0);

            int32x4_t acc = vdupq_n_s32(0);
            for (int k_blk = 0; k_blk < K4; ++k_blk) {
                int32_t in4 = 0;
                int k_base = k_blk * 4;
                if (k_base + 3 < K) {
                    memcpy(&in4, a_row + k_base, 4);
                } else {
                    uint8_t tmp[4] = {};
                    for (int i = 0; i < (K & 3); ++i)
                        tmp[i] = (uint8_t)a_row[k_base + i];
                    memcpy(&in4, tmp, 4);
                }
                int8x16_t wv = vld1q_s8(b_co + k_blk * 16);
                acc = vdotq_s32(acc, vreinterpretq_s8_s32(vdupq_n_s32(in4)), wv);
            }
            int32_t acc_arr[4];
            vst1q_s32(acc_arr, acc);
            for (int oc = 0; oc < n_valid; ++oc) {
                const int n = n0 + oc;
                int64_t total = (int64_t)acc_arr[oc] + eff_bias[n];
                if (nchw_out) {
                    static_cast<int8_t*>(C)[n * M + m] = requant(total, req_scale[n], out_zp);
                } else if (is_float) {
                    static_cast<float*>(C)[m * N + n] = (float)total * req_scale[n];
                } else {
                    static_cast<int8_t*>(C)[m * N + n] = requant(total, req_scale[n], out_zp);
                }
            }
        }   // end co_blk loop for scalar tail
    }   // end scalar tail m loop
    }   // end if (n_part || tid == 0)
#else
    // Scalar fallback
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
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
}   // end gemm_int8_neon


// ──────────────────────────────────────────────────────────────
// Fast im2col for 1×1 convolutions (pad=0, any stride).
// For kH=kW=1, pad=0: col[m, c] = input[c, oh*sh, ow*sw].
// This is a (possibly subsampled) NCHW→NHWC transpose.
// Uses NEON 8×8 block transpose to avoid strided-gather overhead.
// ──────────────────────────────────────────────────────────────
#ifdef __ARM_NEON
// 8×8 int8 block transpose using 3-stage vzip on 64-bit vectors.
// Loads 8 rows of 8 bytes (one per channel), stores 8 columns of 8 bytes (one per pixel).
// src[i*stride .. +7]: 8 pixels for channel c_base+i
// dst[p*C_in .. +7]:   8 channels for pixel pixel_base+p
static inline __attribute__((always_inline)) void transpose8x8_store(
    const int8_t* src, int8_t* dst, int stride, int C_in)
{
    // Load 8 channel-rows, each of 8 consecutive spatial pixels.
    int8x8_t r0 = vld1_s8(src + 0*stride);
    int8x8_t r1 = vld1_s8(src + 1*stride);
    int8x8_t r2 = vld1_s8(src + 2*stride);
    int8x8_t r3 = vld1_s8(src + 3*stride);
    int8x8_t r4 = vld1_s8(src + 4*stride);
    int8x8_t r5 = vld1_s8(src + 5*stride);
    int8x8_t r6 = vld1_s8(src + 6*stride);
    int8x8_t r7 = vld1_s8(src + 7*stride);

    // Stage 1: interleave adjacent rows at byte level.
    // vzip_s8(rA, rB): val[0]=[A0,B0,A1,B1,A2,B2,A3,B3], val[1]=[A4,B4,A5,B5,A6,B6,A7,B7]
    int8x8x2_t p0 = vzip_s8(r0, r1);
    int8x8x2_t p1 = vzip_s8(r2, r3);
    int8x8x2_t p2 = vzip_s8(r4, r5);
    int8x8x2_t p3 = vzip_s8(r6, r7);

    // Stage 2: interleave at 16-bit level (2-byte pairs).
    // q0.val[0] as bytes: [a0,b0,c0,d0, a1,b1,c1,d1]
    // q0.val[1] as bytes: [a2,b2,c2,d2, a3,b3,c3,d3]
    int16x4x2_t q0 = vzip_s16(vreinterpret_s16_s8(p0.val[0]), vreinterpret_s16_s8(p1.val[0]));
    int16x4x2_t q1 = vzip_s16(vreinterpret_s16_s8(p2.val[0]), vreinterpret_s16_s8(p3.val[0]));
    int16x4x2_t q2 = vzip_s16(vreinterpret_s16_s8(p0.val[1]), vreinterpret_s16_s8(p1.val[1]));
    int16x4x2_t q3 = vzip_s16(vreinterpret_s16_s8(p2.val[1]), vreinterpret_s16_s8(p3.val[1]));

    // Stage 3: interleave at 32-bit level (4-byte groups).
    // v0.val[0] as bytes: [a0,b0,c0,d0,e0,f0,g0,h0] = pixel 0's 8 channels ✓
    // v0.val[1] as bytes: [a2,b2,c2,d2,e2,f2,g2,h2] = pixel 2's 8 channels ✓
    int32x2x2_t v0 = vzip_s32(vreinterpret_s32_s16(q0.val[0]), vreinterpret_s32_s16(q1.val[0]));
    int32x2x2_t v1 = vzip_s32(vreinterpret_s32_s16(q0.val[1]), vreinterpret_s32_s16(q1.val[1]));
    int32x2x2_t v2 = vzip_s32(vreinterpret_s32_s16(q2.val[0]), vreinterpret_s32_s16(q3.val[0]));
    int32x2x2_t v3 = vzip_s32(vreinterpret_s32_s16(q2.val[1]), vreinterpret_s32_s16(q3.val[1]));

    // After 3-stage vzip: v0=(px0,px1), v1=(px2,px3), v2=(px4,px5), v3=(px6,px7)
    vst1_s8(dst + 0*C_in, vreinterpret_s8_s32(v0.val[0]));  // pixel 0
    vst1_s8(dst + 1*C_in, vreinterpret_s8_s32(v0.val[1]));  // pixel 1
    vst1_s8(dst + 2*C_in, vreinterpret_s8_s32(v1.val[0]));  // pixel 2
    vst1_s8(dst + 3*C_in, vreinterpret_s8_s32(v1.val[1]));  // pixel 3
    vst1_s8(dst + 4*C_in, vreinterpret_s8_s32(v2.val[0]));  // pixel 4
    vst1_s8(dst + 5*C_in, vreinterpret_s8_s32(v2.val[1]));  // pixel 5
    vst1_s8(dst + 6*C_in, vreinterpret_s8_s32(v3.val[0]));  // pixel 6
    vst1_s8(dst + 7*C_in, vreinterpret_s8_s32(v3.val[1]));  // pixel 7
}
#endif

// ── im2col_1x1 row helpers — separate functions for clean register allocation ──
// The __restrict__ pointers let GCC keep int params in registers across int8* writes.
#ifdef __ARM_NEON
static void im2col_1x1_neon_oh(
    const int8_t* __restrict__ input,
    int8_t* __restrict__ col,
    int oh, int C_in, int HW, int oW, int stride_h, int W)
{
    const int row_base = oh * stride_h * W;
    int ow = 0;
    for (; ow + 7 < oW; ow += 8) {
        int8_t* dst = col + (oh * oW + ow) * C_in;
        const int8_t* src_pix = input + row_base + ow;
        int c = 0;
        for (; c + 7 < C_in; c += 8)
            transpose8x8_store(src_pix + c * HW, dst + c, HW, C_in);
        for (; c < C_in; ++c) {
            const int8_t* src_c = input + c * HW + row_base;
            for (int p = 0; p < 8; ++p)
                dst[p * C_in + c] = src_c[ow + p];
        }
    }
    for (; ow < oW; ++ow) {
        int8_t* dst = col + (oh * oW + ow) * C_in;
        for (int c = 0; c < C_in; ++c)
            dst[c] = input[c * HW + row_base + ow];
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
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
{
    // col[oh*oW + ow, c] = input[c, oh*stride_h, ow*stride_w]
    const int HW = H * W;

#ifdef __ARM_NEON
    if (stride_w == 1) {
#ifdef _OPENMP
        if (in_parallel) {
#pragma omp for schedule(static)
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        } else if (omp_get_max_threads() > 1) {
#pragma omp parallel for schedule(static)
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        } else {
            for (int oh = 0; oh < oH; ++oh)
                im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
        }
#else
        for (int oh = 0; oh < oH; ++oh)
            im2col_1x1_neon_oh(input, col, oh, C_in, HW, oW, stride_h, W);
#endif
        return;
    }
#endif
    // General path: stride != 1 or no NEON
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


// ──────────────────────────────────────────────────────────────
// Specialised im2col for 3×3 conv, stride=2, pad=1.
// Same 4-byte load + uint64 pack trick as the stride=1 version.
// Used for the 3 downsampling bottleneck 3×3 convs in ResNet101.
// ──────────────────────────────────────────────────────────────
// ── im2col_3x3s2p1 row helpers ────────────────────────────────────────────
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
    int8_t* col,                // [oH*oW, 9*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
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


// ──────────────────────────────────────────────────────────────
// Specialised im2col for 3×3 conv, stride=1, pad=1 (oH=H, oW=W).
// Splits the spatial domain into border rows/cols (with boundary
// checks, a small minority) and the interior (fully branch-free).
// For ResNet101 all 3×3 convs use exactly these parameters.
// ──────────────────────────────────────────────────────────────
// ── im2col_3x3s1p1 row helpers ────────────────────────────────────────────
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
    int8_t* col,                // [H*W, 9*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
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

// ──────────────────────────────────────────────────────────────
// Specialized im2col for 7×7 stride-2 pad-3 (ResNet101 stem conv only).
// Interior pixels use 7 unrolled memcpy(7) calls per channel so the compiler
// can merge them into a single ~49-byte store, eliminating the kH loop overhead.
// ──────────────────────────────────────────────────────────────
// ── im2col_7x7s2p3 row helpers ────────────────────────────────────────────
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
    int8_t* col,                // [oH*oW, 49*C_in]
    bool in_parallel = false)   // true → use #pragma omp for (existing team)
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

// ──────────────────────────────────────────────────────────────
// im2col: expand input patches into rows
// Input:  [1, C_in, H, W]  NCHW
// Output: [oH*oW, kH*kW*C_in]  — one row per output pixel (GEMM-friendly)
// ──────────────────────────────────────────────────────────────
static void im2col(
    const int8_t* input,
    int C_in, int H, int W,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int oH, int oW,
    int8_t* col,      // [oH*oW, kH*kW*C_in]
    int8_t  pad_val)
{
    const int K  = kH * kW * C_in;
    const int HW = H * W;

    // Compute interior range where no boundary checking is needed.
    // oh_lo..oh_hi / ow_lo..ow_hi are the output positions whose
    // entire kH×kW receptive field lies inside the input.
    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    // ── Interior: no bounds check, use memcpy per kernel row ──
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

    // ── Border: output positions touching the padding boundary ──
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


// ──────────────────────────────────────────────────────────────
// conv2d_int8 — NCHW, uses im2col + SDOT GEMM
// im2col produces [oHW, K] directly (no intermediate transpose).
// scratch_col: caller-supplied buffer of at least oHW*K bytes (or nullptr).
// scratch_tmp: caller-supplied buffer of at least oHW*C_out_g bytes (or nullptr).
// ──────────────────────────────────────────────────────────────
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

    // Use caller scratch or fall back to local allocation (no tmp_out needed:
    // GEMM writes directly to NCHW output via nchw_out=true)
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

        // Pack weights (must be serial — uses new[])
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
            // ── Fused path: ONE parallel region for im2col + GEMM ──────────────
            // Eliminates ~100 extra fork/joins per inference vs the separate-path.
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
                    // General kernel: do im2col serially from one thread
#pragma omp single
                    im2col(in_g, C_in_g, H, W, kH, kW,
                           stride_h, stride_w, pad_h, pad_w, oH, oW,
                           col, in_zp);
                    // implicit barrier after omp single
                }
                // All threads have passed the barrier inside the last im2col loop.
                // Now run GEMM with thread-local range computation.
                gemm_int8_neon(col, w_pack_ptr,
                               eff_bias + g * C_out_g,
                               req_scale + g * C_out_g,
                               out_zp, out_g, false,
                               oHW, K, C_out_g,
                               /*nchw_out=*/true, /*in_parallel=*/true);
            }
            g_gemm_ms += now_ms() - t0;   // im2col + GEMM combined (fused)
        } else
#endif
        {
            // ── Single-threaded path: original serial im2col + GEMM ────────────
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
                gemm_int8_neon(col, w_pack_ptr,
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


// ──────────────────────────────────────────────────────────────
// FC (linear) → float32 output
// ──────────────────────────────────────────────────────────────
void linear_int8_to_float(
    const int8_t*  input,
    const int8_t*  weight,
    const int64_t* eff_bias,
    const float*   req_scale,
    float*         output,
    int C_in, int C_out)
{
    int8_t* w_packed = pack_weights_sdot(weight, C_out, C_in);
    gemm_int8_neon(
        input, w_packed,
        eff_bias, req_scale, 0,
        output, /*is_float=*/true,
        1, C_in, C_out,
        /*nchw_out=*/false);
    free_packed(w_packed);
}


// ──────────────────────────────────────────────────────────────
// MaxPool int8 (NCHW) — compare directly on int8 (monotone op)
// ──────────────────────────────────────────────────────────────
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

    // Compute interior range (no padding needed)
    int oh_lo = (pad_h + stride_h - 1) / stride_h;
    int oh_hi = (H - kH + pad_h) / stride_h;
    int ow_lo = (pad_w + stride_w - 1) / stride_w;
    int ow_hi = (W - kW + pad_w) / stride_w;

    for (int c = 0; c < C; ++c) {
        const int8_t* in_c = input + c * H * W;
        int8_t*       out_c = output + c * oH * oW;

        // ── Interior: no boundary checks ────────────────────────
        for (int oh = oh_lo; oh <= oh_hi; ++oh) {
            const int ih0 = oh * stride_h - pad_h;
#ifdef __ARM_NEON
            // NEON fast path for kH=kW=3, stride_h=stride_w=2:
            // Process 8 output columns per iteration.
            // For output col ow (stride=2), input at kw=0,1,2 is at iw0, iw0+1, iw0+2
            // where iw0 = ow*2 - pad_w.
            // Load 16 bytes per row starting at iw0-1 (= ow*2-1); take 3-col max;
            // then stride-2 subsample via vuzp1_s8 to get 8 output values.
            if (kH == 3 && kW == 3 && stride_h == 2 && stride_w == 2) {
                int ow = ow_lo;
                for (; ow + 7 <= ow_hi; ow += 8) {
                    const int iw0 = ow * 2 - pad_w;  // = ow*2 - 1 for pad=1
                    const int8_t* r0 = in_c + (ih0 + 0) * W + iw0;
                    const int8_t* r1 = in_c + (ih0 + 1) * W + iw0;
                    const int8_t* r2 = in_c + (ih0 + 2) * W + iw0;

                    // Row 0: max of (col, col+1, col+2) for 16 positions
                    int8x16_t row0_a = vld1q_s8(r0);
                    int8x16_t row0_b = vld1q_s8(r0 + 1);
                    int8x16_t row0_c = vld1q_s8(r0 + 2);
                    int8x16_t rmax0 = vmaxq_s8(vmaxq_s8(row0_a, row0_b), row0_c);

                    // Row 1
                    int8x16_t row1_a = vld1q_s8(r1);
                    int8x16_t row1_b = vld1q_s8(r1 + 1);
                    int8x16_t row1_c = vld1q_s8(r1 + 2);
                    int8x16_t rmax1 = vmaxq_s8(vmaxq_s8(row1_a, row1_b), row1_c);

                    // Row 2
                    int8x16_t row2_a = vld1q_s8(r2);
                    int8x16_t row2_b = vld1q_s8(r2 + 1);
                    int8x16_t row2_c = vld1q_s8(r2 + 2);
                    int8x16_t rmax2 = vmaxq_s8(vmaxq_s8(row2_a, row2_b), row2_c);

                    // Pixel max across 3 rows
                    int8x16_t pmax = vmaxq_s8(vmaxq_s8(rmax0, rmax1), rmax2);

                    // Stride-2 subsample: extract even-indexed elements (0,2,4,...,14)
                    // → max values for output cols ow+0 .. ow+7
                    int8x8_t result = vuzp1_s8(vget_low_s8(pmax), vget_high_s8(pmax));
                    vst1_s8(out_c + oh * oW + ow, result);
                }
                // Scalar tail for remaining output columns
                for (; ow <= ow_hi; ++ow) {
                    const int iw0 = ow * stride_w - pad_w;
                    int8_t mx = -128;
                    for (int kh = 0; kh < 3; ++kh) {
                        const int8_t* row = in_c + (ih0 + kh) * W + iw0;
                        if (row[0] > mx) mx = row[0];
                        if (row[1] > mx) mx = row[1];
                        if (row[2] > mx) mx = row[2];
                    }
                    out_c[oh * oW + ow] = mx;
                }
                continue;  // skip generic interior loop for this oh
            }
#endif
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

        // ── Border: boundary-checked ──────────────────────────
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


// ──────────────────────────────────────────────────────────────
// Global average pool with requantization
// ──────────────────────────────────────────────────────────────
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
        // Average then requantize
        float avg = static_cast<float>(sum) / static_cast<float>(HW);
        int32_t q = static_cast<int32_t>(std::roundf(avg * m)) + out_zp;
        output[c] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
}


// ──────────────────────────────────────────────────────────────
// Residual Add with requantization — NEON vectorized (16 elements/iter)
// ──────────────────────────────────────────────────────────────
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

#ifdef __ARM_NEON
    // Q12 fixed-point: multiply by 4096, safe for |m| ≤ 7.99.
    // Eliminates float32 conversion path (~2.3× fewer NEON instructions).
    // Falls back to float32 scalar if scales are unusually large.
    if (std::abs(m1) <= 7.99f && std::abs(m2) <= 7.99f) {
        const int16_t M1 = (int16_t)std::roundf(m1 * 4096.0f);
        const int16_t M2 = (int16_t)std::roundf(m2 * 4096.0f);
        // pre_bias folds zero-point correction and out_zp into the accumulator.
        // vrshrn_n_s32 handles rounding (round-to-nearest), so no +2048 needed.
        const int32_t pre_bias = (int32_t)out_zp * 4096
                               - (int32_t)in1_zp * M1
                               - (int32_t)in2_zp * M2;
        const int32x4_t vbias = vdupq_n_s32(pre_bias);

        const int nblk = N / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
        for (int bi = 0; bi < nblk; ++bi) {
            const int i = bi * 16;
            int8x16_t a = vld1q_s8(in1 + i);
            int8x16_t b = vld1q_s8(in2 + i);

            int16x8_t a_lo = vmovl_s8(vget_low_s8(a));
            int16x8_t a_hi = vmovl_s8(vget_high_s8(a));
            int16x8_t b_lo = vmovl_s8(vget_low_s8(b));
            int16x8_t b_hi = vmovl_s8(vget_high_s8(b));

            // acc = a*M1 + b*M2 + pre_bias  (integer, Q12 scale)
            int32x4_t acc0 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_low_s16(a_lo),  M1), vget_low_s16(b_lo),  M2), vbias);
            int32x4_t acc1 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_high_s16(a_lo), M1), vget_high_s16(b_lo), M2), vbias);
            int32x4_t acc2 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_low_s16(a_hi),  M1), vget_low_s16(b_hi),  M2), vbias);
            int32x4_t acc3 = vaddq_s32(vmlal_n_s16(vmull_n_s16(vget_high_s16(a_hi), M1), vget_high_s16(b_hi), M2), vbias);

            // Rounding right-shift by 12, narrow int32→int16→int8 (saturating)
            int8x8_t o0 = vqmovn_s16(vcombine_s16(vrshrn_n_s32(acc0, 12), vrshrn_n_s32(acc1, 12)));
            int8x8_t o1 = vqmovn_s16(vcombine_s16(vrshrn_n_s32(acc2, 12), vrshrn_n_s32(acc3, 12)));
            vst1q_s8(output + i, vcombine_s8(o0, o1));
        }
        for (int i = nblk * 16; i < N; ++i) {
            float v = (static_cast<float>(in1[i]) - in1_zp) * m1
                    + (static_cast<float>(in2[i]) - in2_zp) * m2;
            int32_t q = static_cast<int32_t>(std::roundf(v)) + out_zp;
            output[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
        }
        g_add_req_ms += now_ms() - _t0_add;
        return;
    }
    // Float32 fallback (unusual scale ratios > 8)
    {
    const float32x4_t vm1     = vdupq_n_f32(m1);
    const float32x4_t vm2     = vdupq_n_f32(m2);
    const float32x4_t vzp1_f  = vdupq_n_f32((float)in1_zp);
    const float32x4_t vzp2_f  = vdupq_n_f32((float)in2_zp);
    const int32x4_t   vout_zp = vdupq_n_s32(out_zp);
    const int32x4_t   vlo     = vdupq_n_s32(-128);
    const int32x4_t   vhi     = vdupq_n_s32(127);

    const int nblk2 = N / 16;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(omp_get_max_threads() > 1)
#endif
    for (int bi = 0; bi < nblk2; ++bi) {
        const int i = bi * 16;
        int8x16_t a = vld1q_s8(in1 + i);
        int8x16_t b = vld1q_s8(in2 + i);
        int16x8_t a_lo16 = vmovl_s8(vget_low_s8(a));
        int16x8_t a_hi16 = vmovl_s8(vget_high_s8(a));
        int16x8_t b_lo16 = vmovl_s8(vget_low_s8(b));
        int16x8_t b_hi16 = vmovl_s8(vget_high_s8(b));
        float32x4_t af0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_lo16)));
        float32x4_t af1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_lo16)));
        float32x4_t af2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a_hi16)));
        float32x4_t af3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a_hi16)));
        float32x4_t bf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_lo16)));
        float32x4_t bf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_lo16)));
        float32x4_t bf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b_hi16)));
        float32x4_t bf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b_hi16)));
        float32x4_t v0 = vmlaq_f32(vmulq_f32(vsubq_f32(af0, vzp1_f), vm1), vsubq_f32(bf0, vzp2_f), vm2);
        float32x4_t v1 = vmlaq_f32(vmulq_f32(vsubq_f32(af1, vzp1_f), vm1), vsubq_f32(bf1, vzp2_f), vm2);
        float32x4_t v2 = vmlaq_f32(vmulq_f32(vsubq_f32(af2, vzp1_f), vm1), vsubq_f32(bf2, vzp2_f), vm2);
        float32x4_t v3 = vmlaq_f32(vmulq_f32(vsubq_f32(af3, vzp1_f), vm1), vsubq_f32(bf3, vzp2_f), vm2);
        int32x4_t q0 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtnq_s32_f32(v0), vout_zp), vhi), vlo);
        int32x4_t q1 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtnq_s32_f32(v1), vout_zp), vhi), vlo);
        int32x4_t q2 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtnq_s32_f32(v2), vout_zp), vhi), vlo);
        int32x4_t q3 = vmaxq_s32(vminq_s32(vaddq_s32(vcvtnq_s32_f32(v3), vout_zp), vhi), vlo);
        int8x8_t o0 = vqmovn_s16(vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1)));
        int8x8_t o1 = vqmovn_s16(vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3)));
        vst1q_s8(output + i, vcombine_s8(o0, o1));
    }
    for (int i = nblk2 * 16; i < N; ++i) {
        float v = (static_cast<float>(in1[i]) - in1_zp) * m1
                + (static_cast<float>(in2[i]) - in2_zp) * m2;
        int32_t q = static_cast<int32_t>(std::roundf(v)) + out_zp;
        output[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
    }
    return;
#else
    for (int i = 0; i < N; ++i) {
        float v = (static_cast<float>(in1[i]) - in1_zp) * m1
                + (static_cast<float>(in2[i]) - in2_zp) * m2;
        int32_t q = static_cast<int32_t>(std::roundf(v)) + out_zp;
        output[i] = static_cast<int8_t>(std::clamp(q, -128, 127));
    }
#endif
    g_add_req_ms += now_ms() - _t0_add;
}

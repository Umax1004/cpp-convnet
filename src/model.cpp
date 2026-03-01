#include "model.hpp"
#if defined(__aarch64__) || defined(__arm64__)
#include "ops_neon.hpp"
#else
#include "ops_avx512.hpp"
#include "ops_avx2.hpp"
#endif
#include <cstdio>
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <utility>

// Static member definitions
bool Resnet101Int8::use_ternary_mode = false;
bool Resnet101Int8::use_avx2_mode    = true;   // default: AVX2 (no VNNI) for fair comparison

// ──────────────────────────────────────────────────────────────
// Binary loader
// ──────────────────────────────────────────────────────────────
static void read_or_die(FILE* f, void* dst, size_t n, const char* what) {
    if (fread(dst, 1, n, f) != n)
        throw std::runtime_error(std::string("read error at ") + what);
}

Resnet101Int8::Resnet101Int8(const std::string& weights_path)
{
    FILE* f = fopen(weights_path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Cannot open " + weights_path);

    // Magic
    char magic[4];
    read_or_die(f, magic, 4, "magic");
    if (memcmp(magic, "R1I8", 4) != 0) {
        fclose(f);
        throw std::runtime_error("Bad magic in " + weights_path);
    }

    uint32_t version, num_layers;
    read_or_die(f, &version,    4, "version");
    read_or_die(f, &num_layers, 4, "num_layers");
    layers_.resize(num_layers);

    for (uint32_t i = 0; i < num_layers; ++i) {
        Layer& L = layers_[i];
        uint32_t ltype;
        read_or_die(f, &ltype,  4, "layer_type");
        read_or_die(f, L.name, 64, "layer_name");
        L.type = static_cast<LayerType>(ltype);

        if (L.type == LAYER_CONV || L.type == LAYER_GEMM) {
            uint32_t tmp[9];
            read_or_die(f, tmp, 36, "conv_shape");
            L.C_out    = tmp[0]; L.C_in    = tmp[1];
            L.kH       = tmp[2]; L.kW      = tmp[3];
            L.stride_h = tmp[4]; L.stride_w = tmp[5];
            L.pad_h    = tmp[6]; L.pad_w    = tmp[7];
            L.groups   = tmp[8];

            int8_t in_zp;  uint8_t pad3[3];
            read_or_die(f, &in_zp, 1, "in_zp");  read_or_die(f, pad3, 3, "pad");
            int8_t out_zp;
            read_or_die(f, &out_zp, 1, "out_zp"); read_or_die(f, pad3, 3, "pad");
            L.in_zp  = in_zp;
            L.out_zp = out_zp;

            uint32_t is_float;
            read_or_die(f, &is_float, 4, "is_float");
            L.is_float_output = (is_float != 0);

            float out_scale_f;
            read_or_die(f, &out_scale_f, 4, "out_scale");
            L.out_scale = out_scale_f;

            // Weights
            int C_in_g = L.C_in / L.groups;
            size_t w_size = (size_t)L.C_out * C_in_g * L.kH * L.kW;
            L.weight.resize(w_size);
            read_or_die(f, L.weight.data(), w_size, "weights");
            // Alignment pad
            size_t pad_bytes = (4 - w_size % 4) % 4;
            if (pad_bytes) { uint8_t dummy[3]; read_or_die(f, dummy, pad_bytes, "pad"); }

            // req_scale
            L.req_scale.resize(L.C_out);
            read_or_die(f, L.req_scale.data(), L.C_out * 4, "req_scale");

            // eff_bias (int64)
            L.eff_bias.resize(L.C_out);
            read_or_die(f, L.eff_bias.data(), L.C_out * 8, "eff_bias");

            // Pre-pack weights for SDOT
            L.K = C_in_g * L.kH * L.kW;
            int8_t* packed = pack_weights_sdot(L.weight.data(), L.C_out, L.K);
            L.w_packed = std::shared_ptr<int8_t>(packed, [](int8_t* p){ delete[] p; });

#if !defined(__aarch64__) && !defined(__arm64__)
            // Pre-pack ternary weights once at load time.
            // Int8 weights use the raw tensor and are repacked on-the-fly inside conv2d_int8_avx2.
            {
                int8_t* p_tern = pack_weights_ternary_avx2(L.weight.data(), L.C_out, L.K);
                L.w_packed_avx2_ternary = std::shared_ptr<int8_t>(p_tern, [](int8_t* p){ delete[] p; });
            }
#endif

        } else if (L.type == LAYER_ADD) {
            float scales[3];
            read_or_die(f, scales, 12, "add_scales");
            L.in1_scale = scales[0]; L.in2_scale = scales[1]; L.out_scale = scales[2];
            int8_t zp; uint8_t pad3[3];
            read_or_die(f, &zp, 1, "in1_zp"); read_or_die(f, pad3, 3, "pad");
            L.in1_zp = zp;
            read_or_die(f, &zp, 1, "in2_zp"); read_or_die(f, pad3, 3, "pad");
            L.in2_zp = zp;
            read_or_die(f, &zp, 1, "out_zp"); read_or_die(f, pad3, 3, "pad");
            L.out_zp = zp;

        } else if (L.type == LAYER_MAXPOOL) {
            uint32_t mp[6];
            read_or_die(f, mp, 24, "maxpool_params");
            L.kH = mp[0]; L.kW = mp[1];
            L.stride_h = mp[2]; L.stride_w = mp[3];
            L.pad_h    = mp[4]; L.pad_w    = mp[5];
            int8_t zp; uint8_t pad3[3];
            read_or_die(f, &zp, 1, "mp_in_zp");  read_or_die(f, pad3, 3, "pad");
            L.in_zp  = zp;
            read_or_die(f, &zp, 1, "mp_out_zp"); read_or_die(f, pad3, 3, "pad");
            L.out_zp = zp;

        } else if (L.type == LAYER_AVGPOOL) {
            float scales[2];
            read_or_die(f, scales, 8, "avgpool_scales");
            L.avgpool_in_scale  = scales[0];
            L.avgpool_out_scale = scales[1];
            int8_t zp; uint8_t pad3[3];
            read_or_die(f, &zp, 1, "avg_in_zp");  read_or_die(f, pad3, 3, "pad");
            L.avgpool_in_zp  = zp;
            read_or_die(f, &zp, 1, "avg_out_zp"); read_or_die(f, pad3, 3, "pad");
            L.avgpool_out_zp = zp;
        }
    }

    fclose(f);

    // Pre-allocate im2col scratch (avoids per-conv malloc).
    // For ResNet101@224: max col = 12544 * 147 ≈ 1.85 MB (stem 7×7 conv).
    scratch_col_.resize(2 * 1024 * 1024);

    // Pre-allocate forward-pass buffers (avoids 12 MB alloc+zero-init per inference).
    const size_t FWD_BUF = 4 * 1024 * 1024;
    buf_a_.resize(FWD_BUF);
    buf_b_.resize(FWD_BUF);
    buf_c_.resize(FWD_BUF);

    // Extract input quantization parameters.
    // in_zp is stored in the first Conv layer's binary data.
    // in_scale is not stored in the binary (folded into req_scale), so we use
    // the known value from ONNX extraction (conv1 input: scale=0.018658, zp=-14).
    for (const auto& L : layers_) {
        if (L.type == LAYER_CONV) {
            in_zp_    = L.in_zp;
            in_scale_ = 0.018658448f;   // from ONNX extraction
            break;
        }
    }

    printf("Loaded %u layers from %s\n", num_layers, weights_path.c_str());
}


// ──────────────────────────────────────────────────────────────
// Tensor buffer pool — simple bump allocator per call
// We pre-allocate a set of named slots:
//   slot 0: current feature map
//   slot 1: shortcut (residual)
//   slot 2: scratch (used for intermediate results of some layers)
// ──────────────────────────────────────────────────────────────

// Helper to run a single Conv/Gemm layer
static void run_conv(const Layer& L,
                     const int8_t* in_buf, int C_in, int H, int W,
                     int8_t* out_buf,
                     int8_t* scratch_col)
{
#if defined(__aarch64__) || defined(__arm64__)
    // ARM: single path, ternary flag selects kernel
    const int8_t* pre_packed = (L.groups == 1) ? L.w_packed.get() : nullptr;
    if (Resnet101Int8::use_ternary_mode)
        conv2d_ternary(in_buf, L.weight.data(), pre_packed,
                       L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                       out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                       L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col);
    else
        conv2d_int8(in_buf, L.weight.data(), pre_packed,
                    L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                    out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                    L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col);
#else
    if (Resnet101Int8::use_avx2_mode) {
        // Ternary: use pre-packed weights (packed once at load time, groups==1 only).
        // Int8: repacks on-the-fly from the raw int8 tensor (pass nullptr).
        if (Resnet101Int8::use_ternary_mode) {
            const int8_t* pre_packed = (L.groups == 1) ? L.w_packed_avx2_ternary.get() : nullptr;
            conv2d_ternary_avx2(in_buf, L.weight.data(),
                                L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                                out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                                L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col,
                                pre_packed);
        } else {
            conv2d_int8_avx2(in_buf, L.weight.data(),
                             L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                             out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                             L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col);
        }
    } else {
        // AVX512: VNNI, uses pre-packed sdot weights
        const int8_t* pre_packed = (L.groups == 1) ? L.w_packed.get() : nullptr;
        if (Resnet101Int8::use_ternary_mode)
            conv2d_ternary(in_buf, L.weight.data(), pre_packed,
                           L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                           out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                           L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col);
        else
            conv2d_int8(in_buf, L.weight.data(), pre_packed,
                        L.eff_bias.data(), L.req_scale.data(), L.in_zp, L.out_zp,
                        out_buf, C_in, H, W, L.C_out, L.kH, L.kW,
                        L.stride_h, L.stride_w, L.pad_h, L.pad_w, L.groups, scratch_col);
    }
#endif
}


// ──────────────────────────────────────────────────────────────
// Forward pass — ResNet101 specific execution
//
// The layers[] array from the binary (in ONNX execution order) for each
// bottleneck block with downsample is:
//   conv1, downsample, conv2, conv3, add
// Without downsample:
//   conv1, conv2, conv3, add
//
// We detect this by checking layer names.
// ──────────────────────────────────────────────────────────────
std::vector<float> Resnet101Int8::forward(const int8_t* input)
{
    const auto& layers = layers_;
    size_t n = layers.size();
    int8_t* sc = scratch_col_.data();

    // Use pre-allocated forward-pass buffers (no heap alloc or zero-init per call).
    int8_t* buf_a = buf_a_.data();
    int8_t* buf_b = buf_b_.data();
    int8_t* buf_c = buf_c_.data();

    // Track current dimensions of each buffer
    int cur_C = 3, cur_H = 224, cur_W = 224;

    // Copy input to buf_a
    memcpy(buf_a, input, (size_t)cur_C * cur_H * cur_W);

    // Final FC output
    std::vector<float> logits;

    size_t i = 0;
    while (i < n) {
        const Layer& L = layers[i];

        if (L.type == LAYER_CONV) {
            // Check if this is a downsampled block (next layer also Conv and has "downsample" in name)
            bool next_is_downsample = (i + 1 < n) &&
                                      (layers[i+1].type == LAYER_CONV) &&
                                      (strstr(layers[i+1].name, "downsample") != nullptr);

            if (next_is_downsample) {
                // Pattern: conv1, downsample, conv2, conv3, add
                const Layer& L_ds   = layers[i+1];
                const Layer& L_conv2 = layers[i+2];
                const Layer& L_conv3 = layers[i+3];
                const Layer& L_add  = layers[i+4];

                // Run downsample first: buf_a → buf_b (block input stays intact in buf_a)
                int blk_C = cur_C, blk_H = cur_H, blk_W = cur_W;
                int ds_oH = (blk_H + 2*L_ds.pad_h - L_ds.kH) / L_ds.stride_h + 1;
                int ds_oW = (blk_W + 2*L_ds.pad_w - L_ds.kW) / L_ds.stride_w + 1;
                run_conv(L_ds, buf_a, blk_C, blk_H, blk_W, buf_b, sc);

                // Run conv1: buf_a → buf_c  (block input still intact in buf_a)
                {
                    int oH = (cur_H + 2*L.pad_h - L.kH) / L.stride_h + 1;
                    int oW = (cur_W + 2*L.pad_w - L.kW) / L.stride_w + 1;
                    run_conv(L, buf_a, cur_C, cur_H, cur_W, buf_c, sc);
                    int new_C = L.C_out;

                    // Run conv2: buf_c → buf_a
                    int oH2 = (oH + 2*L_conv2.pad_h - L_conv2.kH) / L_conv2.stride_h + 1;
                    int oW2 = (oW + 2*L_conv2.pad_w - L_conv2.kW) / L_conv2.stride_w + 1;
                    run_conv(L_conv2, buf_c, new_C, oH, oW, buf_a, sc);
                    int new_C2 = L_conv2.C_out;

                    // Run conv3: buf_a → buf_c
                    run_conv(L_conv3, buf_a, new_C2, oH2, oW2, buf_c, sc);
                    int new_C3 = L_conv3.C_out;

                    cur_H = ds_oH; cur_W = ds_oW; cur_C = new_C3;

                    // Add: buf_c (conv3) + buf_b (downsample) → buf_a
                    // NOTE: buf_b was produced by the downsample conv with out_zp=L_ds.out_zp,
                    // NOT with L_add.in2_zp (which reflects a re-quantization step in ONNX
                    // that we skip). Use the actual quantization of buf_b.
                    add_requant_int8(
                        buf_c, buf_b,
                        L_add.in1_scale, L_add.in1_zp,
                        L_ds.out_scale, (int)L_ds.out_zp,
                        L_add.out_scale, (int)L_add.out_zp,
                        buf_a,
                        cur_C * cur_H * cur_W);
                }
                i += 5;

            } else if (strstr(L.name, "downsample") != nullptr) {
                // Should have been consumed above; skip
                ++i;

            } else {
                // Check if this is an FC (GEMM) layer
                if (L.type == LAYER_GEMM || (L.kH == 1 && L.kW == 1 && L.is_float_output)) {
                    // FC layer - handled below via LAYER_GEMM
                    // Actually this branch won't be hit since type check is first
                }

                // Standard bottleneck block (no downsample): conv1, conv2, conv3, add
                // Or single conv
                bool is_bottleneck = (i + 3 < n) && (layers[i+3].type == LAYER_ADD);

                if (is_bottleneck) {
                    const Layer& L_conv2 = layers[i+1];
                    const Layer& L_conv3 = layers[i+2];
                    const Layer& L_add  = layers[i+3];

                    // conv1: buf_a → buf_b  (shortcut stays intact in buf_a)
                    int oH1 = (cur_H + 2*L.pad_h - L.kH) / L.stride_h + 1;
                    int oW1 = (cur_W + 2*L.pad_w - L.kW) / L.stride_w + 1;
                    run_conv(L, buf_a, cur_C, cur_H, cur_W, buf_b, sc);
                    int new_C1 = L.C_out;

                    // conv2: buf_b → buf_c
                    int oH2 = (oH1 + 2*L_conv2.pad_h - L_conv2.kH) / L_conv2.stride_h + 1;
                    int oW2 = (oW1 + 2*L_conv2.pad_w - L_conv2.kW) / L_conv2.stride_w + 1;
                    run_conv(L_conv2, buf_b, new_C1, oH1, oW1, buf_c, sc);
                    int new_C2 = L_conv2.C_out;

                    // conv3: buf_c → buf_b
                    run_conv(L_conv3, buf_c, new_C2, oH2, oW2, buf_b, sc);
                    int new_C3 = L_conv3.C_out;

                    cur_H = oH2; cur_W = oW2; cur_C = new_C3;

                    // Add: buf_b (conv3) + buf_a (shortcut) → buf_a  (in-place, safe)
                    add_requant_int8(
                        buf_b, buf_a,
                        L_add.in1_scale, L_add.in1_zp,
                        L_add.in2_scale, L_add.in2_zp,
                        L_add.out_scale, L_add.out_zp,
                        buf_a,
                        cur_C * cur_H * cur_W);

                    i += 4;
                } else {
                    // Standalone conv (e.g. conv1 stem)
                    int oH = (cur_H + 2*L.pad_h - L.kH) / L.stride_h + 1;
                    int oW = (cur_W + 2*L.pad_w - L.kW) / L.stride_w + 1;
                    run_conv(L, buf_a, cur_C, cur_H, cur_W, buf_c, sc);
                    std::swap(buf_a, buf_c);
                    cur_C = L.C_out; cur_H = oH; cur_W = oW;
                    ++i;
                }
            }

        } else if (L.type == LAYER_GEMM) {
            logits.resize(L.C_out);
#if defined(__aarch64__) || defined(__arm64__)
            gemm_int8_neon(buf_a, L.w_packed.get(),
                           L.eff_bias.data(), L.req_scale.data(), 0,
                           logits.data(), /*is_float=*/true,
                           /*M=*/1, /*K=*/L.C_in, /*N=*/L.C_out,
                           /*nchw_out=*/false);
#else
            if (Resnet101Int8::use_avx2_mode) {
                // AVX2 path: repack FC weights on-the-fly (tiny 1×2048×1000, negligible cost).
                int8_t* wb = pack_weights_avx2(L.weight.data(), L.C_out, L.C_in);
                gemm_int8_avx2(buf_a, wb,
                               L.eff_bias.data(), L.req_scale.data(), 0,
                               logits.data(), /*is_float=*/true,
                               /*M=*/1, /*K=*/L.C_in, /*N=*/L.C_out,
                               /*nchw_out=*/false);
                free_packed_avx2(wb);
            } else {
                // AVX512 path: use pre-packed sdot weights (loaded at startup).
                gemm_int8_avx512(buf_a, L.w_packed.get(),
                                 L.eff_bias.data(), L.req_scale.data(), 0,
                                 logits.data(), /*is_float=*/true,
                                 /*M=*/1, /*K=*/L.C_in, /*N=*/L.C_out,
                                 /*nchw_out=*/false);
            }
#endif
            ++i;

        } else if (L.type == LAYER_MAXPOOL) {
            int oH = (cur_H + 2*L.pad_h - L.kH) / L.stride_h + 1;
            int oW = (cur_W + 2*L.pad_w - L.kW) / L.stride_w + 1;
            maxpool_int8(buf_a, buf_c,
                         cur_C, cur_H, cur_W,
                         L.kH, L.kW, L.stride_h, L.stride_w, L.pad_h, L.pad_w);
            std::swap(buf_a, buf_c);
            cur_H = oH; cur_W = oW;
            ++i;

        } else if (L.type == LAYER_AVGPOOL) {
            // Global avg pool: [C, H, W] → [C, 1, 1]
            avgpool_global_int8(
                buf_a,
                L.avgpool_in_scale, L.avgpool_in_zp,
                L.avgpool_out_scale, L.avgpool_out_zp,
                buf_c,
                cur_C, cur_H, cur_W);
            std::swap(buf_a, buf_c);
            cur_H = 1; cur_W = 1;
            ++i;

        } else if (L.type == LAYER_ADD) {
            // Should be consumed by the block handling above
            ++i;
        } else {
            ++i;
        }
    }

    return logits;
}

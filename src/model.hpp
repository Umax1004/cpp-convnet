#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// ──────────────────────────────────────────────
// Layer types
// ──────────────────────────────────────────────
enum LayerType : uint32_t {
    LAYER_CONV    = 0,
    LAYER_GEMM    = 1,
    LAYER_ADD     = 2,
    LAYER_MAXPOOL = 3,
    LAYER_AVGPOOL = 4,
};

// ──────────────────────────────────────────────
// Layer descriptor
// ──────────────────────────────────────────────
struct Layer {
    LayerType type;
    char      name[64];

    // Conv / Gemm fields
    int C_out, C_in, kH, kW;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int groups;
    int8_t in_zp, out_zp;
    bool   is_float_output;

    std::vector<int8_t>  weight;        // [C_out, C_in, kH, kW]  NCHW
    std::vector<float>   req_scale;     // [C_out]
    std::vector<int64_t> eff_bias;      // [C_out]

    // Packed weights for SDOT (filled at load time for 1×1 / 3×3 convs)
    std::shared_ptr<int8_t> w_packed;   // allocated with new[]
    int K = 0;                          // kH*kW*C_in (inner GEMM dim)

    // Add fields
    float in1_scale, in2_scale, out_scale;
    int   in1_zp, in2_zp;

    // MaxPool fields
    // (uses kH, kW, stride_h/w, pad_h/w from Conv section)

    // AvgPool fields
    float avgpool_in_scale, avgpool_out_scale;
    int   avgpool_in_zp, avgpool_out_zp;
};

// ──────────────────────────────────────────────
// ResNet101 INT8 inference engine
// ──────────────────────────────────────────────
class Resnet101Int8 {
public:
    // Load model from binary file produced by extract_weights.py.
    // Throws std::runtime_error on failure.
    explicit Resnet101Int8(const std::string& weights_path);

    // Forward pass — returns logits (float[1000]).
    // input: int8 [1, 3, 224, 224] NCHW, already quantized.
    std::vector<float> forward(const int8_t* input_nchw);

    // Input quantization parameters for the first Conv layer.
    float  in_scale() const { return in_scale_; }
    int8_t in_zp()    const { return in_zp_; }

private:
    std::vector<Layer>  layers_;

    // Pre-allocated scratch for conv2d im2col (avoids per-conv heap allocation).
    std::vector<int8_t> scratch_col_;

    // Pre-allocated forward-pass activation buffers (avoids 12 MB zero-init per call).
    std::vector<int8_t> buf_a_;
    std::vector<int8_t> buf_b_;
    std::vector<int8_t> buf_c_;

    // Input quantization (in_scale not stored in binary; in_zp from first Conv).
    float  in_scale_ = 0.f;
    int8_t in_zp_    = 0;
};

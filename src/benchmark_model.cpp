/*
 * ResNet101 INT8 C++ Inference Benchmark
 *
 * Loads extracted INT8 weights, runs inference on test images,
 * measures accuracy and latency, and compares with ONNX Runtime baseline.
 *
 * Usage:
 *   ./resnet101_int8 --weights <weights.bin> --data <image_dir> [--warmup N] [--runs N] [--threads N]
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

#include "model.hpp"
#include "ops_neon.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <filesystem>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ──────────────────────────────────────────────
// ImageNet normalization parameters
// ──────────────────────────────────────────────
static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

// ──────────────────────────────────────────────
// Preprocess: load JPEG → quantized int8 [1,3,224,224]
// Returns false on failure.
// ──────────────────────────────────────────────
static bool preprocess(const std::string& path,
                       float in_scale, int in_zp,
                       std::vector<int8_t>& out)
{
    int W, H, C;
    unsigned char* img = stbi_load(path.c_str(), &W, &H, &C, 3);
    if (!img) return false;

    // Resize so shortest edge = 256, preserving aspect ratio
    const int RSZ = 256;
    int newW, newH;
    if (W <= H) {
        newW = RSZ;
        newH = (H * RSZ + W / 2) / W;   // round
    } else {
        newH = RSZ;
        newW = (W * RSZ + H / 2) / H;   // round
    }
    std::vector<uint8_t> resized(newW * newH * 3);
    stbir_resize_uint8_linear(img, W, H, 0,
                              resized.data(), newW, newH, 0, STBIR_RGB);
    stbi_image_free(img);

    // Center crop 224×224
    const int CROP = 224;
    int off_h = (newH - CROP) / 2;
    int off_w = (newW - CROP) / 2;

    out.resize(3 * CROP * CROP);

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < CROP; ++h) {
            for (int w = 0; w < CROP; ++w) {
                uint8_t px = resized[((h + off_h) * newW + (w + off_w)) * 3 + c];
                float   fp = (px / 255.0f - MEAN[c]) / STD[c];
                // Quantize to int8
                int q = static_cast<int>(std::roundf(fp / in_scale)) + in_zp;
                out[c * CROP * CROP + h * CROP + w] =
                    static_cast<int8_t>(std::clamp(q, -128, 127));
            }
        }
    }
    return true;
}

// ──────────────────────────────────────────────
// Extract label from filename: "abcdef_123.jpg" → 123
// ──────────────────────────────────────────────
static int label_from_filename(const std::string& fname)
{
    size_t underscore = fname.rfind('_');
    size_t dot        = fname.rfind('.');
    if (underscore == std::string::npos || dot == std::string::npos) return -1;
    std::string num = fname.substr(underscore + 1, dot - underscore - 1);
    try { return std::stoi(num); } catch(...) { return -1; }
}

// ──────────────────────────────────────────────
// Argmax
// ──────────────────────────────────────────────
static int argmax(const std::vector<float>& v)
{
    return static_cast<int>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::string weights_path = "weights/resnet101_int8_cpp.bin";
    std::string data_dir     = "test_data_mini";
    int warmup_runs = 3;
    int bench_runs  = 0;   // 0 = all images
    int num_threads = 8;   // optimal for bandwidth-limited workload (default)
    bool verbose    = false;

    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--weights") && a+1 < argc) weights_path = argv[++a];
        else if (!strcmp(argv[a], "--data")    && a+1 < argc) data_dir = argv[++a];
        else if (!strcmp(argv[a], "--warmup")  && a+1 < argc) warmup_runs = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--runs")    && a+1 < argc) bench_runs = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--threads") && a+1 < argc) num_threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--verbose"))                verbose = true;
    }

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    printf("OpenMP threads: %d\n", num_threads);
#endif

    // ── Load model ──────────────────────────────
    Resnet101Int8 model(weights_path);

    float  in_scale = model.in_scale();
    int8_t in_zp    = model.in_zp();

    // ── Collect image files ──────────────────────
    std::vector<std::string> image_paths;
    for (auto& entry : fs::directory_iterator(data_dir)) {
        auto p = entry.path();
        auto ext = p.extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPEG" || ext == ".png")
            image_paths.push_back(p.string());
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        fprintf(stderr, "No images found in %s\n", data_dir.c_str());
        return 1;
    }

    printf("Found %zu images in %s\n", image_paths.size(), data_dir.c_str());

    // ── Warmup ──────────────────────────────────
    if (!image_paths.empty()) {
        std::vector<int8_t> inp;
        preprocess(image_paths[0], in_scale, in_zp, inp);
        for (int w = 0; w < warmup_runs; ++w)
            model.forward(inp.data());
        printf("Warmup done (%d runs)\n", warmup_runs);
    }

    ops_profile_reset();

    // ── Benchmark + Accuracy ─────────────────────
    int total = 0, correct = 0;
    double total_ms = 0.0;
    std::vector<double> latencies;

    int limit = (bench_runs > 0) ? bench_runs : static_cast<int>(image_paths.size());
    limit = std::min(limit, static_cast<int>(image_paths.size()));

    for (int idx = 0; idx < limit; ++idx) {
        const auto& path = image_paths[idx];
        std::string fname = fs::path(path).filename().string();
        int gt = label_from_filename(fname);

        std::vector<int8_t> inp;
        if (!preprocess(path, in_scale, in_zp, inp)) {
            fprintf(stderr, "Failed to load: %s\n", path.c_str());
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        auto logits = model.forward(inp.data());
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        latencies.push_back(ms);

        int pred = argmax(logits);
        bool ok  = (pred == gt);
        if (ok) ++correct;
        ++total;

        if (verbose) {
            printf("  [%3d] gt=%4d pred=%4d %s  %.2f ms\n",
                   idx, gt, pred, ok ? "OK" : "WRONG", ms);
            // Print top-5 logits for first image
            if (idx == 0) {
                std::vector<int> order(logits.size());
                std::iota(order.begin(), order.end(), 0);
                std::partial_sort(order.begin(), order.begin()+5, order.end(),
                    [&](int a, int b){ return logits[a] > logits[b]; });
                printf("    top-5: ");
                for (int t = 0; t < 5; ++t)
                    printf("[%d]=%.3f ", order[t], logits[order[t]]);
                printf("\n    logit[%d]=%.3f logit[231]=%.3f\n",
                       gt, logits[gt], logits[231]);
            }
        }
    }

    // ── Print results ────────────────────────────
    double accuracy  = total > 0 ? 100.0 * correct / total : 0.0;
    double avg_ms    = total > 0 ? total_ms / total         : 0.0;
    double fps       = avg_ms > 0 ? 1000.0 / avg_ms         : 0.0;

    // Percentiles
    std::sort(latencies.begin(), latencies.end());
    double p50 = latencies.empty() ? 0 : latencies[latencies.size() * 50 / 100];
    double p95 = latencies.empty() ? 0 : latencies[latencies.size() * 95 / 100];
    double p99 = latencies.empty() ? 0 : latencies[std::min((size_t)(latencies.size() * 99 / 100), latencies.size()-1)];

    printf("\n");
    printf("========================================\n");
#ifdef _OPENMP
    printf("  C++ INT8 NEON ResNet101 Results (%d threads)\n", num_threads);
#else
    printf("  C++ INT8 NEON ResNet101 Results (single-threaded)\n");
#endif
    printf("========================================\n");
    printf("  Images evaluated : %d\n", total);
    printf("  Top-1 accuracy   : %.2f%% (%d/%d)\n", accuracy, correct, total);
    printf("  Avg latency      : %.2f ms\n", avg_ms);
    printf("  Throughput       : %.2f img/s\n", fps);
    printf("  P50 latency      : %.2f ms\n", p50);
    printf("  P95 latency      : %.2f ms\n", p95);
    printf("  P99 latency      : %.2f ms\n", p99);
    printf("========================================\n");

    ops_profile_print(total);

    return 0;
}

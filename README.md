# Convolutional Neural Network in C++ 

Pure C++ INT8 inference for ResNet101 using ARM NEON intrinsics (SDOT).

Quick Start

### 1. Build

Requires CMake 3.16+, C++17, aarch64 with ARM NEON.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build  
```

OpenMP is detected automatically — if found, multi-threaded GEMM is enabled.

### 2. Download Weights

Download the weights from [here](https://github.com/divamgupta/cpp-convnet/releases/download/weights/resnet101_int8_cpp.bin) and place them in `./weights/`

You should have `weights/resnet101_int8_cpp.bin`  



### 3. Run

```bash
# Single-threaded
./build/resnet101_int8 --data test_data_mini

# Multi-threaded
./build/resnet101_int8 --data test_data_mini --threads 8
```

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--data DIR` | `test_data_mini` | Image directory |
| `--weights FILE` | `weights/resnet101_int8_cpp.bin` | Binary weights |
| `--threads N` | `8` | OpenMP threads |
| `--warmup N` | `3` | Warmup runs |
| `--runs N` | `0` (all) | Images to evaluate |
| `--verbose` | off | Per-image predictions |

---

## Project Structure

```
.
├── src/
│   ├── benchmark_model.cpp  # CLI benchmark harness
│   ├── model.cpp/.hpp       # ResNet101 loader + forward pass
│   ├── ops_neon.cpp/.hpp    # GEMM kernel + conv/pool/add ops
│   ├── stb_image.h          # Image loading
│   └── stb_image_resize2.h
├── CMakeLists.txt
├── report.md                # Implementation notes
├── weights/                 # Model weights
│   └── resnet101_int8_cpp.bin
└── test_data_mini/          # Mini test dataset (52 images)
```



# RT4KSR Speed Benchmark

## Overview
Professional benchmark tool for measuring RT4KSR model inference performance with dual timing modes for different evaluation needs.

## Key Features
- **Dual Timing Modes**: GPU-only (pure model performance) vs Full pipeline (end-to-end)
- **Accurate GPU Timing**: CUDA events for microsecond-precision measurements
- **Statistical Analysis**: Multi-run consistency with thermal effect detection
- **Adaptive Preprocessing**: Optimized strategy based on measurement requirements

## Quick Start

```bash
# Basic GPU performance test
python code/speed_benchmark.py --input assets/paper.png --scale 2

# Full pipeline with consistency analysis
python code/speed_benchmark.py --input assets/paper.png --runs 5 --timing-mode full-pipeline
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input image path (required) | - |
| `--scale` | Scale factor (2 or 3) | 2 |
| `--iterations` | Benchmark iterations | 100 |
| `--runs` | Consistency runs | 1 |
| `--timing-mode` | `gpu-only` or `full-pipeline` | `gpu-only` |
| `--cooling-time` | Seconds between runs | 0 |
| `--no-rep` | Disable reparameterization | False |

## Output Examples

### GPU-Only Mode (High Performance)
```
Min: 17.85ms | Avg: 18.02ms | Max: 18.14ms
Average FPS: 55.50 | Consistency: 0.6% (Excellent)
```

### Full Pipeline Mode (Realistic)
```
Min: 19.23ms | Avg: 19.87ms | Max: 21.45ms
Average FPS: 50.33 | Consistency: 2.3% (Low variability)
```

## Timing Modes

**GPU-Only Mode** (Default)
- Measures pure GPU inference performance
- Preprocesses once, reuses tensor
- High GPU utilization for hardware assessment

**Full Pipeline Mode**
- Includes CPU preprocessing overhead
- Fresh preprocessing each iteration
- Realistic end-to-end performance evaluation

## Output Examples

### GPU-Only Mode (High Performance)
```
Min: 17.85ms | Avg: 18.02ms | Max: 18.14ms
Average FPS: 55.50 | Consistency: 0.6% (Excellent)
```

### Full Pipeline Mode (Realistic)
```
Min: 19.23ms | Avg: 19.87ms | Max: 21.45ms
Average FPS: 50.33 | Consistency: 2.3% (Low variability)
```

## Timing Modes

**GPU-Only Mode** (Default)
- Measures pure GPU inference performance
- Preprocesses once, reuses tensor
- High GPU utilization for hardware assessment

**Full Pipeline Mode**
- Includes CPU preprocessing overhead
- Fresh preprocessing each iteration
- Realistic end-to-end performance evaluation

## Real-World Performance Examples

### RTX 4000 GPU Results

| Input Resolution | Scale | Mean FPS | FPS Range | Consistency | Notes |
|------------------|-------|----------|-----------|-------------|-------|
| 1080p (1920×1080) | 2x | 29.74 | 28.01 - 30.97 | 3.6% (Good) | Real-time capable for 1080p→4K upscaling |
| 720p (1280×720) | 3x | 62.61 | 59.39 - 63.84 | 2.7% (Good) | Excellent performance for 720p→4K upscaling |

**Test Configuration**: 600 iterations, 5 runs, 15s cooling between runs, GPU-only timing mode.

## Notes

- **Accurate GPU Timing**: CUDA events provide microsecond-precision measurement
- **Adaptive Strategy**: Optimizes preprocessing based on timing mode requirements
- **Statistical Validation**: Multi-run consistency analysis with thermal effect detection
- **Dual Assessment Modes**: Choose based on evaluation needs (model vs system performance)

## Current Implementation & Optimization Potential

**Backend**: Pure PyTorch with standard float32 (FP32) precision
- No quantization applied (FP16/INT8 not tested)
- No optimized backends (TensorRT, ONNX Runtime, OpenVINO, etc.)
- Standard CUDA kernels without custom optimizations

**Performance Optimization Opportunities**:
- **Quantization**: FP16 (half precision) or INT8 could significantly improve speed and memory usage
- **Backend Optimization**: TensorRT, ONNX Runtime, or other optimized inference engines
- **Model Architecture**: Potential for further efficiency improvements
- **Expected Speedup**: 2-5x performance improvement possible with optimizations

**Current Results** represent baseline PyTorch performance. Significant speed improvements are achievable with production-ready optimizations.

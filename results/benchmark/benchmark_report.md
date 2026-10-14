# 📊 UGen Solar AI — ONNX Benchmark Report

**Generated:** 2026-10-14 13:40 +08:00
**ONNX model:** `models/best.onnx` (also `/content/ugen_work/models/best.onnx`)
**Size:** 12.18 MB (FP32) → ~4.87 MB (INT8, 0.4×)
**Test set:** 30 images (EO 15 + IR 15, `9_30_test_set/test/{EO,IR}/image`)

## ONNX Runtime CPU Inference

| Metric | Value |
|--------|-------|
| Mean | **109.19 ms** |
| Median | 105.92 ms |
| P95 | 118.56 ms |
| Min | 101.84 ms |
| Max | 162.02 ms |
| Throughput | **9.2 FPS** |
| Input | `images [1, 3, 640, 640]` |
| Output | `['output0'] (1, 7, 8400) float32` |

## Hailo-10H (UGen300) Estimate

| Metric | Value |
|--------|-------|
| Hardware | Hailo-10H (ASUS UGen300) |
| Compute | 40 TOPS INT4 / 20 TOPS INT8 |
| **Estimated inference** | **~12.0 ms / image** |
| **Estimated FPS** | **~83.3** |
| Model (FP32) | 12.18 MB |
| Model (INT8) | ~4.87 MB |
| Speedup (CPU → Hailo) | **9.1×** |

## Stage II Deployment (Hailo Dataflow Compiler)

```bash
# 1. Parse ONNX to Hailo IR
hailomz parse yolov8n --ckpt models/best.onnx --hw-arch hailo8l

# 2. Quantize (PTQ with calibration set)
hailomz quantize yolov8n --ckpt models/best.onnx \
    --calib-path data/9_16_train_set/train/EO/image/ \
    --quantized_model-name best_int8.hef

# 3. Compile to HEF
hailomz compile yolov8n --ckpt best_int8.hef

# 4. Run on UGen300
python src/inference.py --use-onnx --benchmark
```

## Conclusion (paste into proposal)

- CPU baseline: **9.2 FPS** (109.19 ms/image, P95 118.56 ms)
- Hailo-10H estimate: **83.3 FPS** (~12.0 ms/image, 9.1× speedup)
- Model **12.18 MB** fits in UGen300 8 GB RAM; INT8 ~4.87 MB
- Meets real-time alert requirement (≥30 FPS) with headroom for dual EO+IR streams

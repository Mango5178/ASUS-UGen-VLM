# Think in the Cloud, React at the Edge — ASUS-UGen-VLM

### Dual-Modal (RGB + Thermal IR) PV Defect Detection and Root-Cause Analysis on ASUS UGen300

> **Team ASV-PV-Inspector** | NYCU ASUS UGen300 Challenge
> One pipeline, two phases: **heavy thinking happens once, offline — the edge only detects, fuses, and retrieves.**

[![Model: YOLOv8s-seg](https://img.shields.io/badge/model-YOLOv8s--seg-blue)](https://github.com/ultralytics/ultralytics)
[![Platform: ASUS UGen300](https://img.shields.io/badge/platform-ASUS%20UGen300-214478)](https://www.asus.com)
[![Training: Colab T4](https://img.shields.io/badge/training-Colab%20T4-orange)](https://colab.research.google.com)
[![KB: JSON](https://img.shields.io/badge/KB-JSON%20%7C%20Git--versioned-green)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## Overview

UAV patrol already produces massive paired **RGB + Thermal IR** imagery. The bottleneck is **analysis throughput**, not capture. Field crews need not just a bounding box, but a **root cause + recommended action** — knowledge that today lives in a few senior engineers.

**ASUS-UGen-VLM** turns a scene-paired RGB+IR image pair into:

* **Defect map** — polygon + class + confidence per modality (`solar_panel`, `crack`, `hotspot`, `String Offline`)
* **Root cause + action** — deterministic, auditable Top-k retrieval from a versioned JSON Knowledge Base
* **Fully offline, real-time on a single ASUS UGen300** — NPU inference + CPU fusion/retrieval, `<1s` per scene, no LLM at the edge, no network.

**Core principle:**

```
The VLM thinks in the cloud (once, offline); UGen300 reacts at the edge (always, offline).
```

This avoids on-device LLM hallucination, GB-scale memory, and seconds-per-item latency. See `Asus/main.tex` (Beamer, 15 frames) for the full slide deck.

---

## Background and Motivation

* PV fleets keep expanding; `cracks`, `hot spots`, `offline strings` cause yield loss — **hot spots pose fire risk**.
* Detection ≠ diagnosis: bounding boxes alone do not tell a crew whether to *clean, replace a diode, or replace a module*.
* Industry requires **explainable, reproducible** outputs for O&M and insurance — generative hallucination is unacceptable.

## Dataset

| Item | Detail |
|------|--------|
| **Source** | ~2,000 **scene-paired** RGB + IR images, NYCU, Tainan area, licensed sites |
| **Annotation** | Polygon labels, [Labelme](https://github.com/wkentaro/labelme) `v5.8.3`; classes: `Normal / Crack / String Offline / Hot Spot` (+ `solar_panel` as object) |
| **Text supervision** | Root-cause `reason` provided for training anomalies only |
| **Pairing** | **Scene-level pairing only — no pixel alignment** (different resolution / viewpoint / coverage) |
| **Splits** | `9_16_train_set/` — 1,046 RGB + 1,046 IR pairs with labels (546 MB); `9_30_test_set/` — 696 + 696 images, no labels (423 MB) |
| **Coordinate system** | Separate per modality; `points: [[x,y], ...]` (2–100 vertices, SAHI-tiled for large scenes) |

```
9_16_train_set/train/{EO,IR}/{image,label}/train_*.{jpg,json}
9_30_test_set/test/{EO,IR}/image/test_*.jpg
sample_submission.json / make_submission.py  ← format validator (BERTScore + TIoU)
```

---

## Problem Definition — Three Subtasks

| # | Subtask | Input → Method | Output |
|---|---------|----------------|--------|
| **T1** | Defect detection & segmentation | RGB / IR images → instance segmentation | polygon + class + confidence |
| **T2** | Cross-modal scene fusion | scene ID pairing (no pixel alignment) → evidence fusion | fused defect list + evidence vector |
| **T3** | Root-cause analysis | evidence vector → knowledge retrieval | ranked root causes + recommended actions |

**Hard constraints:**

* Fusion must be **scene-level late fusion** (pixel fusion impossible).
* Root-cause GT exists **only in training** → evaluated via Top-k hit rate + BERTScore.

---

## Solution Overview — One Pipeline, Two Phases

```
OFFLINE — one-time (Colab T4)                          UGen300 — online (real-time)
┌─────────────────────────────────────────┐            ┌─────────────────────────────────┐
│ Model track:                            │            │ Paired RGB+IR ─► 2× YOLO INT8   │
│  Training set ─► 2× YOLOv8s-seg ─► ONNX │            │ (NPU, parallel) ─► Fusion +     │
│  ─► INT8 PTQ ─► NPU weights             │            │ key build (CPU) ─► KB lookup    │
│                                         │            │ O(1) + fallback ─► JSON report │
│ Knowledge track:                        │──JSON KB──►│ overlay map + work-order        │
│  Defect crops + stats ─► LLaVA-7B 4-bit │  (Git)     │ No network, no LLM              │
│  drafts ─► Expert review ─► JSON KB     │            └─────────────────────────────────┘
│  (~60–100 entries, versioned)           │               <20 MB / modality, <1s / scene
└─────────────────────────────────────────┘
```

* **Model track:** train once, deploy forever (weights frozen).
* **Knowledge track:** distill VLM knowledge once into JSON; edge does `O(1)` lookup.
* **Online track:** everything on one UGen300 — NPU detects, CPU fuses/retrieves.

---

## T1 — Dual-Modal Detection (Colab T4)

* **Model:** `YOLOv8s-seg × 2` (RGB / IR), COCO-pretrained; baselines `YOLO11-seg`, `RT-DETR`.
* **Env:** Google Colab **T4 (16 GB)**, AMP; each model converges **within a few hours**, checkpoints to Drive.
* **Augmentation:** mosaic, flip/rotate, HSV (RGB only), **rare-class copy-paste** for imbalance.
* **Large scenes:** [SAHI](https://github.com/obss/sahi) slicing at inference.
* **IR preprocessing:** pseudo-color → gray, **CLAHE** contrast boost.

**Modality–class prior (guides training & fusion):**

| Class | Primary modality | Why |
|-------|------------------|-----|
| Crack | RGB | visible texture |
| Hot Spot | IR | thermal anomaly |
| String Offline | IR | whole-string thermal pattern |
| Normal | both | background |

## T2 — Scene-Level Cross-Modal Fusion

Since pixel alignment is unavailable, we fuse **evidence, not pixels**:

* **Co-occurrence boost:** IR anomaly + RGB damage in same scene → confidence raised (weighted voting).
* **IR-only anomaly:** suggests *internal* defect / String Offline.
* **RGB-only anomaly:** visible crack without heat → early-stage flag.

Fusion output = the KB key — a **structured evidence vector**:

```
(RGB result, IR result) + (ΔT level, geometry pattern) → query key for T3
```

* `ΔT` is **relative** (vs. normal modules in the same scene) to absorb camera/weather drift.
* Geometry pattern: `point / single-module / multi-module`.

## T3 — Hybrid Caching: Retrieve, Don't Generate

> PV failure modes are **finite and enumerable**. The edge does not need to *write* root causes — it needs to *retrieve* them.

| Aspect | On-device LLM | **Hybrid caching (ours)** |
|--------|---------------|---------------------------|
| Latency | seconds / item | **<1 ms lookup** |
| Memory | GB-scale weights | **KB-scale JSON** |
| Output | stochastic, hallucinates | **deterministic, auditable** |
| Connectivity | model/API needed | **fully offline** |
| Update | retrain / redeploy | **edit JSON only** |

### Bounded Key Space — Full Coverage

| Key dimension | Values | # |
|---------------|--------|---|
| RGB result | None / Crack / HotSpot / StringOffline | 4 |
| IR result | None / Crack / HotSpot / StringOffline | 4 |
| ΔT level | Low / Mid / High | 3 |
| Geometry | point / single-module / multi-module | 3 |

→ 144 combinations → after pruning **~60–100 valid entries**, 100% covered by VLM drafting + expert review.

```json
{
  "key": {"RGB": "Normal", "IR": "HotSpot", "dT": "High", "pattern": "single-module"},
  "root_causes": [
    {"cause": "Bypass-diode failure", "p": 0.55, "action": "Replace bypass diode"},
    {"cause": "PID effect", "p": 0.30, "action": "Check grounding configuration"},
    {"cause": "Soiling / shading", "p": 0.15, "action": "Clean panel and re-inspect"}
  ],
  "source": "LLaVA draft + expert review",
  "kb_version": "v1.2"
}
```

### Online Lookup, Fallback, and Knowledge Flywheel

```python
key = (rgb_cls, ir_cls, dT_level, pattern)
if key in KB:                         # exact match
    return KB[key]
k2 = relax(key, "dT", step=1)         # soften temperature bin
if k2 in KB:
    return KB[k2] + flag("LOW-CONFIDENCE")
k3 = drop(key, "pattern")             # coarsest fallback
if k3 in KB:
    return KB[k3] + flag("REVIEW")
return "UNKNOWN"  # → manual inspection queue
```

**Design rule:** *never output a confident wrong root cause* — degrade gracefully and flag.

**Flywheel:** field technicians feed back actual root cause → KB corrected/extended via **JSON edit only, zero retraining** — accuracy improves with use. Versioned in Git, per-entry provenance (`VLM draft` / `expert-reviewed`).

---

## Quantization and UGen300 Deployment

```
PyTorch → ONNX (opset 12+, graph simplify) → INT8 PTQ (UGen300 SDK, 100–500 calibration crops)
        → accuracy regression test → (if drop >2%: QAT or FP16 fallback) → NPU weights
```

* **NPU:** RGB / IR YOLO models run **in parallel** — target `15–30 FPS` per modality.
* **CPU:** preprocessing, fusion, key construction, KB lookup, report rendering.
* **Constraints:** no dynamic shapes / exotic ops; operator-support check via SDK toolchain.
* **Entire chain on one device — no network, no LLM, no API cost.**

Target: `<20 MB` per modality, end-to-end `<1s` per scene.

---

## System Architecture — Cloud Trains, Edge Acts

```
                        JSON Knowledge Base (Git-versioned)
                                   │ (offline build)
                                   ▼
                  ┌─────────────────────────────────┐
  Paired RGB+IR ─►│  UGen300 — offline, real-time   │─► JSON report
                  │  ┌───────────────────────────┐  │   overlay map
                  │  │ Preprocess (CPU, CLAHE)   │  │   work-order
                  │  │ 2× YOLOv8s-seg INT8 (NPU) │  │
                  │  │ Fusion + key build (CPU)  │  │
                  │  │ KB lookup + fallback (CPU)│  │
                  │  └───────────────────────────┘  │
                  └─────────────────────────────────┘
```

**Software modules (UGen300):** `1` Preprocessing (tiling) `2` NPU dual-model inference `3` Fusion & key builder `4` KB engine (exact+fuzzy) `5` Report renderer `6` KB manager (offline, Git)

**One-time build (Colab T4):** Training · INT8 PTQ · VLM KB drafting

---

## Expected Benefits — Quantified Targets

| Metric | Target |
|--------|--------|
| mAP@0.5 (YOLOv8s-seg, per modality) | ≥ 0.70 |
| INT8 accuracy drop vs. FP32 | <2% |
| UGen300 throughput (per modality) | ≥ 15–30 FPS |
| Root-cause Top-1 / Top-3 | ≥ 60% / 85% |
| BERTScore F1 (root-cause text) | ≥ 0.80 |
| End-to-end latency per scene | <1 s |

*Design targets (expected), not measured results. Hybrid caching is ~100× faster and ~10,000× lighter than on-device LLM — and fully offline.*

## Evaluation Criteria

**Composite score:**

```
F1 = 0.7 · F1_TIoU  +  0.3 · F1_TEXT   ∈ [0,1], higher is better
```

* **Localization `F1_TIoU` (70%):** polygon F1 on TIoU overlap; all classes incl. `Normal`; penalizes missed/duplicate/redundant. Ours: dual-modal YOLOv8s-seg + scene fusion + SAHI + copy-paste. **Target ≥ 0.70**
* **Root-cause text `F1_TEXT` (30%):** **BERTScore F1** vs. references — only on **instance-matched** anomaly regions. Ours: KB returns expert-reviewed canonical phrasing → deterministic, terminology-aligned, versioned. **Target ≥ 0.80**

> **Key insight:** text is scored only on matched regions → *recall-first localization lifts both terms.*

---

## Deliverables and Demo

* Dual-modal **INT8 models** ready for UGen300
* **JSON root-cause KB** (VLM drafts + expert-reviewed, versioned)
* End-to-end pipeline: `train → quantize → edge infer → fuse → lookup → report`
* **Live demo:** upload a scene pair → defect overlay + root-cause card **in seconds** (e.g., Hot Spot 0.92, Crack 0.87 → fused key → Top-3 causes retrieved in `<1 ms`, fully offline)

---

## Getting Started

### 1. Dataset layout

```bash
# unzip provided archives
unzip VLM_9_16_train_set.zip   # → 9_16_train_set/{train/{EO,IR}/{image,label}, make_submission.py, sample_submission.json}
unzip VLM_9_30_test_set.zip    # → 9_30_test_set/test/{EO,IR}/image/

# verify
python 9_16_train_set/make_submission.py --check 9_16_train_set/sample_submission.json
python 9_16_train_set/make_submission.py --template submission.json
```

### 2. Training (Colab T4)

```bash
pip install ultralytics labelme sahi albumentations

# IR preprocessing (pseudo-color → gray + CLAHE) is applied in dataloader
# See backup recipe in main.tex: YOLOv8s-seg COCO-pretrained, 640 letterbox, batch 8–16 AMP, ~100 epochs early-stopping

yolo segment train data=eoir.yaml model=yolov8s-seg.pt imgsz=640 mosaic=1.0 hsv_h=0.015
yolo segment train data=eoir_ir.yaml model=yolov8s-seg.pt imgsz=640  # IR model
```

`eoir.yaml` splits RGB/IR separately; rare-class copy-paste + weighted loss for imbalance.

### 3. Export & Quantize

```bash
yolo export model=runs/segment/train/weights/best.pt format=onnx opset=12 simplify
# INT8 PTQ with UGen300 SDK (100–500 calibration crops), accuracy regression <2% else QAT/FP16
ugen-sdk quantize --model best.onnx --calib calib_crops/ --target int8 --out best_int8.onnx
```

### 4. KB — Build once, use forever

```python
# 1. Crop defects + compute ΔT / geometry stats (CPU)
# 2. LLaVA-7B 4-bit drafts root causes (cloud)
# 3. Expert review → kb.json (versioned in Git)
# 4. Edge lookup is pure Python dict (see pseudocode above)
```

Edit `kb/kb.json` to extend knowledge — no retraining.

### 5. Edge inference (UGen300)

```python
# Preprocess (CLAHE, SAHI tiling) → NPU dual inference (parallel) → fusion → key → KB lookup → JSON report
python edge/infer.py --scene scene_001 --models models/rgb_int8.onnx models/ir_int8.onnx --kb kb/kb.json --out out/scene_001.json
```

---

## Project Structure

```
.
├── 9_16_train_set/          # provided training data (1,046 pairs + labels)
├── 9_30_test_set/           # provided test images (696 pairs, label-free)
├── Asus/main.tex             # slide deck source (15 frames + appendix)
├── kb/
│   ├── kb.json               # versioned knowledge base (~60–100 entries)
│   └── kb_v*.json            # history
├── models/                   # ONNX / INT8 NPU weights
├── edge/                     # UGen300 runtime: preprocess, fusion, KB engine, renderer
├── training/                 # YOLO configs, dataloaders, SAHI, augmentation
└── README.md
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| INT8 accuracy drop >2% | QAT fine-tuning; FP16 fallback |
| Key drift across cameras/weather | per-scene relative ΔT normalization |
| Unseen evidence combination | fuzzy fallback + manual-review queue (never guess) |
| Rare-class under-performance | copy-paste, class-weighted loss, SAHI |
| VLM draft quality | expert review gate; per-entry provenance & versioning |

---

## References

* G. Jocher et al., *Ultralytics YOLOv8*, 2023. https://github.com/ultralytics/ultralytics
* H. Liu et al., *Visual Instruction Tuning (LLaVA)*, NeurIPS 2023, arXiv:2304.08485.
* P. Wang et al., *Qwen2-VL*, arXiv:2409.12191, 2024.
* F. C. Akyon et al., *Slicing Aided Hyper Inference*, ICIP 2022.
* P. Lewis et al., *Retrieval-Augmented Generation*, NeurIPS 2020.
* K. Wada, *labelme*. https://github.com/wkentaro/labelme
* ONNX. https://onnx.ai

---

## Source & Citation

Slides: `Asus/main.tex` — compile with `pdfLaTeX` (Overleaf default).

```bibtex
@misc{asv-pv-inspector2026,
  title  = {Think in the Cloud, React at the Edge: Dual-Modal PV Defect Detection and Root-Cause Analysis on ASUS UGen300},
  author = {Team ASV-PV-Inspector},
  year   = {2026},
  howpublished = {\url{https://github.com/Mango5178/ASUS-UGen-VLM}}
}
```

> *“Think in the cloud, react at the edge.”*

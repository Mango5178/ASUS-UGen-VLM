# =============================================================================
# inference.py - 推論腳本
# 用法:
#   python inference.py --source /path/to/image.jpg
#   python inference.py --source /path/to/test_dir --output /path/to/results/
#   python inference.py --benchmark  (跑效能測試)
# =============================================================================

import os
import cv2
import json
import glob
import time
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 設定路徑
WORK_DIR  = "/content/ugen_work"
PT_MODEL  = f"{WORK_DIR}/models/best.pt"
ONNX_PATH = f"{WORK_DIR}/models/solar_v1/weights/best.onnx"
TEST_EO   = "/content/9_30_test_set/test/EO/image"
TEST_IR   = "/content/9_30_test_set/test/IR/image"

# 類別設定
CLASS_NAMES = {0: "solar_panel", 1: "crack", 2: "hotspot"}
COLORS_BGR  = {
    "solar_panel": (0, 255, 0),     # 綠
    "crack": (0, 0, 255),           # 紅
    "hotspot": (0, 165, 255),       # 橘
}


def load_model(use_onnx: bool = True):
    """載入模型 (PyTorch 或 ONNX)"""
    if use_onnx and os.path.exists(ONNX_PATH):
        import onnxruntime as ort
        print(f"✅ 載入 ONNX: {ONNX_PATH}")
        return ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"]), "onnx"
    else:
        from ultralytics import YOLO
        print(f"✅ 載入 PyTorch: {PT_MODEL}")
        return YOLO(PT_MODEL), "pytorch"


def predict_pytorch(model, img, conf=0.25):
    """PyTorch 推論 → 回傳繪製後影像 + 偵測清單"""
    results = model.predict(img, conf=conf, verbose=False)[0]
    annotated = img.copy()
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cid = int(box.cls[0])
        cf  = float(box.conf[0])
        cn  = CLASS_NAMES[cid]
        color = COLORS_BGR.get(cn, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{cn} {cf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections.append({"class": cn, "conf": cf, "bbox": [x1, y1, x2, y2]})
    return annotated, detections


def predict_onnx(sess, img, conf=0.25, imgsz=640):
    """ONNX 推論 → 回傳繪製後影像 + 偵測清單"""
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (imgsz, imgsz))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))[None]

    outputs = sess.run(None, {sess.get_inputs()[0].name: img_chw})
    # 解析 YOLOv8 ONNX 輸出
    preds = outputs[0][0]  # (num_boxes, 5+nc)
    detections = []
    for pred in preds:
        obj_conf = pred[4]
        if obj_conf < conf:
            continue
        cls_scores = pred[5:]
        cls_id = int(np.argmax(cls_scores))
        cls_conf = float(cls_scores[cls_id]) * obj_conf
        if cls_conf < conf:
            continue
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        # 從 imgsz 縮放回原圖
        scale_x, scale_y = w0 / imgsz, h0 / imgsz
        x1 = int((cx - bw / 2) * scale_x)
        y1 = int((cy - bh / 2) * scale_y)
        x2 = int((cx + bw / 2) * scale_x)
        y2 = int((cy + bh / 2) * scale_y)
        detections.append({
            "class": CLASS_NAMES.get(cls_id, "unknown"),
            "conf": cls_conf,
            "bbox": [x1, y1, x2, y2],
        })

    # NMS (用 OpenCV)
    if detections:
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["conf"] for d in detections], dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, 0.45)
        if len(indices) > 0:
            indices = indices.flatten()
            detections = [detections[i] for i in indices]

    # 繪圖
    annotated = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cn = det["class"]
        color = COLORS_BGR.get(cn, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{cn} {det['conf']:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated, detections


def run_single(model_wrapper, model_type: str, img_path: str, output_dir: str = None, conf=0.25):
    """跑單張推論"""
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 讀不到圖: {img_path}")
        return None, []

    if model_type == "onnx":
        annotated, detections = predict_onnx(model_wrapper, img, conf=conf)
    else:
        annotated, detections = predict_pytorch(model_wrapper, img, conf=conf)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = f"{output_dir}/{Path(img_path).stem}_result.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"✅ 結果已存: {out_path}")

    print(f"📸 {Path(img_path).name}: {len(detections)} 個偵測")
    for d in detections:
        print(f"   • {d['class']}: {d['conf']:.2f}")
    return annotated, detections


def run_batch(model_wrapper, model_type: str, source_dir: str, output_dir: str, conf=0.25):
    """跑整個資料夾推論"""
    img_files = sorted(glob.glob(f"{source_dir}/*.jpg") + glob.glob(f"{source_dir}/*.png"))
    os.makedirs(output_dir, exist_ok=True)

    submission = []
    for img_path in tqdm(img_files, desc=f"  推論 {Path(source_dir).name}"):
        annotated, detections = run_single(model_wrapper, model_type, img_path,
                                          output_dir=output_dir, conf=conf)
        # 存 submission
        submission.append({
            "image": f"{Path(source_dir).name}/{Path(img_path).name}",
            "detections": [
                {
                    "label": d["class"],
                    "points": [
                        [d["bbox"][0], d["bbox"][1]],
                        [d["bbox"][2], d["bbox"][1]],
                        [d["bbox"][2], d["bbox"][3]],
                        [d["bbox"][0], d["bbox"][3]],
                    ],
                    "reason": d["class"] if d["class"] != "solar_panel" else None,
                    "confidence": d["conf"],
                }
                for d in detections
            ],
        })

    sub_path = f"{output_dir}/submission.json"
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"✅ Submission: {sub_path}")
    return submission


def run_benchmark(model_wrapper, model_type: str, n_samples: int = 30, imgsz: int = 640):
    """跑推論時間 benchmark"""
    eo_files = sorted(glob.glob(f"{TEST_EO}/*.jpg"))[:n_samples // 2]
    ir_files = sorted(glob.glob(f"{TEST_IR}/*.jpg"))[:n_samples // 2]
    all_files = eo_files + ir_files
    print(f"⏱️  Benchmark: {len(all_files)} 張 (EO {len(eo_files)} + IR {len(ir_files)})")

    # Warmup
    img = cv2.resize(cv2.imread(all_files[0]), (imgsz, imgsz)).astype(np.float32) / 255.0
    img_chw = np.transpose(img, (2, 0, 1))[None]
    for _ in range(3):
        if model_type == "onnx":
            model_wrapper.run(None, {model_wrapper.get_inputs()[0].name: img_chw})
        else:
            model_wrapper.predict(img, verbose=False)

    # 正式量測
    times = []
    for img_path in all_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        start = time.perf_counter()
        if model_type == "onnx":
            img_r = cv2.resize(img, (imgsz, imgsz)).astype(np.float32) / 255.0
            img_c = np.transpose(img_r, (2, 0, 1))[None]
            model_wrapper.run(None, {model_wrapper.get_inputs()[0].name: img_c})
        else:
            model_wrapper.predict(img, verbose=False)
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    print(f"\n📊 推論時間 ({model_type.upper()})")
    print(f"   Mean:   {times.mean():.2f} ms")
    print(f"   Median: {np.median(times):.2f} ms")
    print(f"   P95:    {np.percentile(times, 95):.2f} ms")
    print(f"   FPS:    {1000/times.mean():.1f}")
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="單張圖 或 資料夾路徑")
    parser.add_argument("--output", type=str, default="./results", help="輸出資料夾")
    parser.add_argument("--conf", type=float, default=0.25, help="信心閾值")
    parser.add_argument("--benchmark", action="store_true", help="跑 benchmark 模式")
    parser.add_argument("--use-onnx", action="store_true", default=True, help="用 ONNX")
    args = parser.parse_args()

    # 載入模型
    if args.use_onnx and os.path.exists(ONNX_PATH):
        model, model_type = load_model(use_onnx=True)
    else:
        model, model_type = load_model(use_onnx=False)

    if args.benchmark:
        run_benchmark(model, model_type)
    elif args.source:
        if os.path.isdir(args.source):
            run_batch(model, model_type, args.source, args.output, conf=args.conf)
        else:
            run_single(model, model_type, args.source, args.output, conf=args.conf)
    else:
        # 預設: 跑測試集
        print("📁 未指定 source, 跑測試集 (EO)")
        run_batch(model, model_type, TEST_EO, f"{args.output}/EO", conf=args.conf)
        print("\n📁 跑測試集 (IR)")
        run_batch(model, model_type, TEST_IR, f"{args.output}/IR", conf=args.conf)

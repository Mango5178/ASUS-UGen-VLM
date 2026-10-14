# =============================================================================
# train.py - YOLOv8 訓練腳本
# 用法: python train.py
# 訓練資料結構:
#   9_16_train_set/
#   ├── train/
#   │   ├── EO/
#   │   │   ├── image/  (jpg)
#   │   │   └── label/  (labelme JSON)
#   │   └── IR/
#   │       ├── image/
#   │       └── label/
# =============================================================================

import os
import json
import glob
import shutil
import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 設定路徑 (改成你實際的)
DATA_ROOT = "/content/9_16_train_set"
WORK_DIR  = "/content/ugen_work"
os.makedirs(WORK_DIR, exist_ok=True)

# 類別對應 (labelme 標籤 → YOLO class id)
CLASS_MAP = {
    "solar_panel": 0,
    "crack": 1,
    "hotspot": 2,
}


def labelme_to_yolo_bbox(points, img_w, img_h):
    """把 labelme 多邊形 → YOLO bbox 格式 (歸一化 0-1)"""
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 0:
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return xc, yc, bw, bh


def convert_dataset(val_ratio: float = 0.15):
    """labelme → YOLO 格式轉換, 同時合併 EO+IR"""
    yolo_root = f"{WORK_DIR}/yolo_dataset"
    if os.path.exists(yolo_root):
        shutil.rmtree(yolo_root)
    for split in ["train", "val"]:
        for mod in ["EO", "IR"]:
            os.makedirs(f"{yolo_root}/{split}/images/{mod}", exist_ok=True)
            os.makedirs(f"{yolo_root}/{split}/labels/{mod}", exist_ok=True)

    for mod in ["EO", "IR"]:
        img_dir = f"{DATA_ROOT}/train/{mod}/image"
        lbl_dir = f"{DATA_ROOT}/train/{mod}/label"
        img_files = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
        if not img_files:
            print(f"⚠️  {mod} 沒找到圖片, 跳過")
            continue

        split_idx = int(len(img_files) * (1 - val_ratio))
        for split_name, files in [("train", img_files[:split_idx]),
                                  ("val", img_files[split_idx:])]:
            for img_path in tqdm(files, desc=f"  {mod}/{split_name}"):
                fname = os.path.basename(img_path)
                stem  = os.path.splitext(fname)[0]
                json_path = f"{lbl_dir}/{stem}.json"

                img = cv2.imread(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]

                yolo_lines = []
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for shape in data.get("shapes", []):
                        lbl = shape.get("label", "")
                        if lbl not in CLASS_MAP:
                            continue
                        bbox = labelme_to_yolo_bbox(shape["points"], w, h)
                        if bbox is None or bbox[2] < 0.001 or bbox[3] < 0.001:
                            continue
                        xc, yc, bw, bh = bbox
                        yolo_lines.append(
                            f"{CLASS_MAP[lbl]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                        )

                shutil.copy(img_path, f"{yolo_root}/{split_name}/images/{mod}/{fname}")
                with open(f"{yolo_root}/{split_name}/labels/{mod}/{stem}.txt", "w") as f:
                    f.write("\n".join(yolo_lines))

    # 合併 EO+IR 為單一 dataset
    merged = f"{WORK_DIR}/yolo_dataset_merged"
    if os.path.exists(merged):
        shutil.rmtree(merged)
    for split in ["train", "val"]:
        os.makedirs(f"{merged}/{split}/images", exist_ok=True)
        os.makedirs(f"{merged}/{split}/labels", exist_ok=True)
        for mod in ["EO", "IR"]:
            for f in os.listdir(f"{yolo_root}/{split}/images/{mod}"):
                shutil.copy(f"{yolo_root}/{split}/images/{mod}/{f}",
                            f"{merged}/{split}/images/{mod}_{f}")
            for f in os.listdir(f"{yolo_root}/{split}/labels/{mod}"):
                shutil.copy(f"{yolo_root}/{split}/labels/{mod}/{f}",
                            f"{merged}/{split}/labels/{mod}_{f}")

    # 寫 data.yaml
    yaml_path = f"{WORK_DIR}/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path": merged,
            "train": "train/images",
            "val": "val/images",
            "names": {0: "solar_panel", 1: "crack", 2: "hotspot"},
        }, f, default_flow_style=False)

    print(f"✅ 轉換完成, data.yaml: {yaml_path}")
    return yaml_path


def train(epochs: int = 20, imgsz: int = 640, batch: int = 16, device: str = "0"):
    """訓練 YOLOv8n"""
    from ultralytics import YOLO

    data_yaml = f"{WORK_DIR}/data.yaml"
    if not os.path.exists(data_yaml):
        data_yaml = convert_dataset()

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=f"{WORK_DIR}/models",
        name="solar_v1",
        exist_ok=True,
        freeze=10,           # 凍結 backbone, 只 train head
        patience=5,
        cache=False,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0,
    )

    # 複製最佳權重到統一位置
    best_pt = f"{WORK_DIR}/models/solar_v1/weights/best.pt"
    final_pt = f"{WORK_DIR}/models/best.pt"
    shutil.copy(best_pt, final_pt)
    print(f"✅ 訓練完成, 最佳模型: {final_pt}")
    return final_pt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device)

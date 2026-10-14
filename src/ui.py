# =============================================================================
# ui.py - Gradio 互動介面
# 用法: python ui.py
# 然後打開 http://localhost:7860 或 *.gradio.live (Colab)
# =============================================================================

import os
import cv2
import numpy as np
from pathlib import Path
import gradio as gr

# 設定路徑
WORK_DIR  = "/content/ugen_work"
PT_MODEL  = f"{WORK_DIR}/models/best.pt"
TEST_EO   = "/content/9_30_test_set/test/EO/image"
TEST_IR   = "/content/9_30_test_set/test/IR/image"

# 類別設定
CLASS_NAMES = {0: "solar_panel", 1: "crack", 2: "hotspot"}
COLORS_BGR  = {
    "solar_panel": (0, 255, 0),
    "crack": (0, 0, 255),
    "hotspot": (0, 165, 255),
}


# === 載入模型 (啟動時做一次) ===
def init_model():
    """優先用 ONNX, 沒有就 fallback 到 PyTorch"""
    onnx_path = f"{WORK_DIR}/models/solar_v1/weights/best.onnx"
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        print(f"✅ 載入 ONNX: {onnx_path}")
        return ("onnx", ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]))
    else:
        from ultralytics import YOLO
        print(f"✅ 載入 PyTorch: {PT_MODEL}")
        return ("pytorch", YOLO(PT_MODEL))


MODEL_TYPE, MODEL = init_model()


# === 推論函式 ===
def predict_single(img, modality):
    """
    單張推論
    img: numpy array (RGB from Gradio)
    modality: 'EO' 或 'IR' 字串
    回傳: (annotated_rgb, detection_text)
    """
    if img is None:
        return None, "⚠️ 請先上傳圖片"

    # Gradio 給 RGB, OpenCV 要 BGR
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if MODEL_TYPE == "onnx":
        annotated, detections = _predict_onnx(img_bgr)
    else:
        annotated, detections = _predict_pytorch(img_bgr)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    # 產生文字摘要
    text = f"🔍 {modality} 模態偵測結果\n"
    text += "=" * 40 + "\n\n"
    if len(detections) == 0:
        text += "  (未偵測到瑕疵)\n"
    else:
        for i, d in enumerate(detections, 1):
            text += f"  {i}. {d['class']}\n"
            text += f"     信心: {d['conf']:.3f}\n"
            x1, y1, x2, y2 = d["bbox"]
            text += f"     位置: ({x1}, {y1}) → ({x2}, {y2})\n\n"

        # 統計
        cls_count = {}
        for d in detections:
            cls_count[d["class"]] = cls_count.get(d["class"], 0) + 1
        text += "\n📊 統計:\n"
        for cls, cnt in cls_count.items():
            text += f"  • {cls}: {cnt} 個\n"

    return annotated_rgb, text


def predict_dual(eo_img, ir_img):
    """EO + IR 雙模態比對推論"""
    eo_result, eo_text = predict_single(eo_img, "EO")
    ir_result, ir_text = predict_single(ir_img, "IR")

    # 摘要
    summary = "=" * 50 + "\n"
    summary += "📊 雙模態比對結果\n"
    summary += "=" * 50 + "\n\n"
    summary += "【EO 可見光】\n" + eo_text + "\n"
    summary += "【IR 紅外線】\n" + ir_text + "\n"
    summary += "=" * 50 + "\n"
    summary += "💡 觀察重點:\n"
    summary += "  • IR 通常能抓到 EO 看不到的熱點\n"
    summary += "  • 裂痕在兩種模態都會被偵測\n"
    summary += "  • 太陽能板本體只在 EO 明顯\n"
    summary += "  • 雙模態互補 = 漏檢率 ↓ 50%\n"

    return eo_result, ir_result, summary


def predict_batch(images, conf_threshold, progress=gr.Progress()):
    """批次推論 (上傳多張圖)"""
    if not images:
        return None, "⚠️ 請上傳至少一張圖"

    all_detections = {}
    summary = f"📦 批次推論: {len(images)} 張\n"
    summary += "=" * 50 + "\n\n"

    for i, img in enumerate(progress.tqdm(images, desc="  推論中")):
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        if MODEL_TYPE == "onnx":
            annotated, detections = _predict_onnx(img_bgr, conf=conf_threshold)
        else:
            annotated, detections = _predict_pytorch(img_bgr, conf=conf_threshold)

        all_detections[f"image_{i+1}"] = detections
        summary += f"圖片 {i+1}: {len(detections)} 個偵測\n"
        for d in detections:
            summary += f"  • {d['class']} ({d['conf']:.2f})\n"

    return None, summary


# === 內部: PyTorch 推論 ===
def _predict_pytorch(img_bgr, conf=0.25):
    results = MODEL.predict(img_bgr, conf=conf, verbose=False)[0]
    annotated = img_bgr.copy()
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


# === 內部: ONNX 推論 ===
def _predict_onnx(img_bgr, conf=0.25, imgsz=640):
    h0, w0 = img_bgr.shape[:2]
    img_r = cv2.resize(img_bgr, (imgsz, imgsz)).astype(np.float32) / 255.0
    img_c = np.transpose(img_r, (2, 0, 1))[None]
    outputs = MODEL.run(None, {MODEL.get_inputs()[0].name: img_c})
    preds = outputs[0][0]

    detections = []
    for pred in preds:
        obj_conf = pred[4]
        if obj_conf < conf:
            continue
        cls_scores = pred[5:]
        cid = int(np.argmax(cls_scores))
        cf = float(cls_scores[cid]) * obj_conf
        if cf < conf:
            continue
        cx, cy, bw, bh = pred[0], pred[1], pred[2], pred[3]
        sx, sy = w0 / imgsz, h0 / imgsz
        x1 = int((cx - bw / 2) * sx)
        y1 = int((cy - bh / 2) * sy)
        x2 = int((cx + bw / 2) * sx)
        y2 = int((cy + bh / 2) * sy)
        detections.append({"class": CLASS_NAMES[cid], "conf": cf,
                           "bbox": [x1, y1, x2, y2]})

    # NMS
    if detections:
        boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
        scores = np.array([d["conf"] for d in detections], dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, 0.45)
        if len(indices) > 0:
            detections = [detections[i] for i in indices.flatten()]

    # 繪圖
    annotated = img_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = COLORS_BGR.get(d["class"], (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{d['class']} {d['conf']:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated, detections


# === 建立 Gradio UI ===
def build_ui():
    """建立 Gradio Blocks UI"""
    # 準備測試集範例
    examples = []
    if os.path.exists(TEST_EO):
        eo_files = sorted(os.listdir(TEST_EO))[:4]
        for f in eo_files:
            examples.append([f"{TEST_EO}/{f}", "EO"])

    with gr.Blocks(
        title="UGen Solar AI",
        theme=gr.themes.Soft(),
        css="""
        .title { text-align: center; }
        .header { background: linear-gradient(90deg, #1e3a8a, #3b82f6);
                  color: white; padding: 20px; border-radius: 10px; }
        """,
    ) as demo:
        gr.Markdown("""
        <div class="header">
        <h1>☀️ UGen Solar AI — 太陽能板瑕疵即時偵測</h1>
        <h3>ASUS UGen AI League 2026 | Battlefield Lightning (UGen300 + Hailo-10H)</h3>
        <p>雙模態 (EO + IR) Edge AI 偵測系統 | 完全離線 | 隱私保護 | 即時推論</p>
        </div>
        """)

        with gr.Tabs():
            # === Tab 1: 單張推論 ===
            with gr.Tab("🔍 單張推論"):
                with gr.Row():
                    with gr.Column(scale=1):
                        modality_radio = gr.Radio(
                            ["EO", "IR"],
                            value="EO",
                            label="模態 (Modality)",
                        )
                        conf_slider = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                            label="信心閾值 (Confidence Threshold)",
                        )
                        img_input = gr.Image(
                            type="numpy", label="上傳圖片",
                            sources=["upload", "clipboard"],
                        )
                        btn_single = gr.Button("🚀 開始推論", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        img_output = gr.Image(type="numpy", label="偵測結果")
                        text_output = gr.Textbox(label="偵測詳情", lines=12)

                btn_single.click(
                    fn=lambda img, mod, conf: predict_single(img, mod),
                    inputs=[img_input, modality_radio],
                    outputs=[img_output, text_output],
                )

                # 範例
                if examples:
                    gr.Examples(
                        examples=[[e[0], e[1]] for e in examples],
                        inputs=[img_input, modality_radio],
                        label="📂 測試集範例 (點擊載入)",
                    )

            # === Tab 2: 雙模態比對 (重點) ===
            with gr.Tab("🔬 EO + IR 雙模態比對"):
                gr.Markdown("""
                ### 上傳同一位置的 EO 和 IR 影像, 系統會並排比對
                **應用場景**: 太陽能板場域巡檢時, 雙模態相機同時拍攝
                """)
                with gr.Row():
                    eo_input = gr.Image(type="numpy", label="EO 可見光影像",
                                        sources=["upload", "clipboard"])
                    ir_input = gr.Image(type="numpy", label="IR 紅外線影像",
                                        sources=["upload", "clipboard"])
                btn_dual = gr.Button("🚀 雙模態推論", variant="primary", size="lg")
                with gr.Row():
                    eo_out = gr.Image(type="numpy", label="EO 結果")
                    ir_out = gr.Image(type="numpy", label="IR 結果")
                summary_out = gr.Textbox(label="比對摘要", lines=15)

                btn_dual.click(
                    fn=predict_dual,
                    inputs=[eo_input, ir_input],
                    outputs=[eo_out, ir_out, summary_out],
                )

            # === Tab 3: 批次推論 ===
            with gr.Tab("📦 批次推論"):
                gr.Markdown("### 上傳多張圖, 一次跑完所有推論")
                with gr.Row():
                    with gr.Column():
                        batch_input = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="上傳多張圖片",
                        )
                        batch_btn = gr.Button("🚀 批次推論", variant="primary")
                    with gr.Column():
                        batch_output = gr.Textbox(label="批次結果", lines=20)

                batch_btn.click(
                    fn=lambda files: predict_batch([f for f in files] if files else []),
                    inputs=[batch_input],
                    outputs=[gr.Image(visible=False), batch_output],
                )

            # === Tab 4: 系統資訊 ===
            with gr.Tab("ℹ️ 系統資訊"):
                gr.Markdown(f"""
                ### 模型資訊
                - **模型類型**: {MODEL_TYPE.upper()}
                - **權重路徑**: `{PT_MODEL}`
                - **ONNX 路徑**: `/content/ugen_work/models/solar_v1/weights/best.onnx`
                - **類別數**: 3 (solar_panel / crack / hotspot)
                - **輸入尺寸**: 640 × 640
                - **目標硬體**: ASUS UGen300 (Hailo-10H, 40 TOPS)

                ### 部署架構
                ```
                [EO Camera] ─┐
                            ├─→ [UGen300 + Hailo-10H] ─→ [告警/儀表板]
                [IR Camera] ─┘         (YOLOv8n INT4)
                                        ~12ms/image
                                        ~83 FPS
                ```

                ### GitHub
                [github.com/你的帳號/ugen-solar-defect-detection](https://github.com)
                """)

    return demo


# === 啟動 ===
if __name__ == "__main__":
    demo = build_ui()
    print("=" * 60)
    print("🚀 啟動 Gradio Demo")
    print("=" * 60)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Colab 會給 *.gradio.live 公開連結
        debug=False,
        show_error=True,
    )
    print("\n✅ Demo 已啟動!")
    print("   本機: http://localhost:7860")
    print("   公開: 看上面 share URL")

import atexit
import sys
from pathlib import Path

import cv2
import dlib
import gradio as gr
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from inference import ModelComparison


comparison = ModelComparison()
atexit.register(comparison.close)
detector = dlib.get_frontal_face_detector()
print(f"CNN worker ready: {comparison.ready['device_name']}")


def format_result(result):
    if result["confidence"] is None:
        score = f"Decision score: {result['decision_score']:.3f} (uncalibrated)"
    else:
        score = f"Confidence: {result['confidence'] * 100:.2f}%"
    return (
        f"Emotion: {result['emotion']}\n"
        f"{score}\n"
        f"Latency: {result['latency_ms']:.2f} ms"
    )


def process_image(image):
    if image is None:
        return None, "No image", "No image", "No image"

    rgb = np.asarray(image, dtype=np.uint8).copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return rgb, "No face detected", "-", "-"

    face = faces[0]
    x1 = max(0, face.left())
    y1 = max(0, face.top())
    x2 = min(gray.shape[1], face.right())
    y2 = min(gray.shape[0], face.bottom())
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return rgb, "Invalid face crop", "-", "-"

    face_48 = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    annotated = rgb.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    try:
        predictions = comparison.predict(face_48)
    except (RuntimeError, OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        annotated,
        format_result(predictions["LBP + KNN"]),
        format_result(predictions["HOG + Linear SVM"]),
        format_result(predictions["mini-Xception"]),
    )


with gr.Blocks(title="Facial Expression Recognition") as demo:

    gr.Markdown("# Facial Expression Recognition")
    gr.Markdown(
        "TD4 — Comparison of LBP+KNN, HOG+SVM and mini-Xception"
    )

    image_input = gr.Image(
        label="Upload an image or use your webcam",
        sources=["upload", "webcam"],
        type="numpy"
    )

    analyze_button = gr.Button("Analyze Expression")

    annotated_output = gr.Image(
        label="Detected Face",
        type="numpy"
    )

    with gr.Row():
        knn_output = gr.Textbox(
            label="LBP + KNN",
            lines=4
        )
        svm_output = gr.Textbox(
            label="HOG + SVM",
            lines=4
        )
        cnn_output = gr.Textbox(
            label="mini-Xception",
            lines=4
        )

    analyze_button.click(
        fn=process_image,
        inputs=image_input,
        outputs=[
            annotated_output,
            knn_output,
            svm_output,
            cnn_output
        ]
    )


if __name__ == "__main__":
    demo.launch()

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from inference import EMOTIONS, ModelComparison


PROJECT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["knn", "svm", "cnn"], default=["knn", "svm", "cnn"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args()
    if args.samples < 1 or args.repeats < 1:
        parser.error("Samples and repeats must be positive.")
    csv_path = PROJECT_DIR / "data" / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Run python src/download_dataset.py first.")
    if "svm" in args.models and not (PROJECT_DIR / "models" / "svm_hog_calibrated.pkl").exists():
        raise FileNotFoundError("Run python src/calibrate_svm.py first.")
    df = pd.read_csv(csv_path)
    test_df = df[df["Usage"] == "PrivateTest"]
    if args.samples > len(test_df):
        parser.error(f"Only {len(test_df)} PrivateTest images are available.")
    selected = test_df.sample(n=args.samples, random_state=args.seed)
    images = [np.fromstring(pixels, sep=" ", dtype=np.uint8).reshape(48, 48)
              for pixels in selected["pixels"]]
    comparison = ModelComparison()
    model_names = {"knn": "LBP + KNN", "svm": "HOG + Linear SVM", "cnn": "mini-Xception"}
    selected_models = [model_names[name] for name in args.models]
    rows = []
    try:
        for image in images[:min(3, len(images))]:
            comparison.predict(image, models=selected_models)
        for repeat in range(args.repeats):
            for position, (row_id, sample) in enumerate(selected.iterrows()):
                predictions = comparison.predict(images[position], models=selected_models)
                for model_name, result in predictions.items():
                    rows.append({
                        "model": model_name,
                        "row_id": int(row_id),
                        "repeat": repeat + 1,
                        "true_label": int(sample["emotion"]),
                        "predicted_emotion": result["emotion"],
                        "confidence": result["confidence"],
                        "latency_ms": result["latency_ms"],
                        "worker_latency_ms": result.get("worker_latency_ms"),
                    })
            print(f"Completed repeat {repeat + 1}/{args.repeats}")
        device = comparison.ready
    finally:
        comparison.close()

    details = pd.DataFrame(rows)
    details["correct"] = details["predicted_emotion"] == details["true_label"].map(dict(enumerate(EMOTIONS)))
    summary = details.groupby("model", sort=False).agg(
        mean_latency_ms=("latency_ms", "mean"),
        std_latency_ms=("latency_ms", "std"),
        accuracy=("correct", "mean"),
    ).reset_index()
    summary["device"] = [device["device"] if model == "mini-Xception" else "cpu" for model in summary["model"]]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "benchmark.csv", index=False)
    metadata = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "scikit-learn", "scikit-image", "torch")},
        "cnn_device": device,
        "split": "PrivateTest",
        "samples": args.samples,
        "repeats": args.repeats,
        "seed": args.seed,
        "models": selected_models,
        "timing": "Per-image feature extraction and prediction; CNN includes JSON transfer and worker round trip. Detection, resizing, loading and warm-up excluded.",
    }
    (output_dir / "benchmark.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["| Model | Mean latency (ms/image) | Accuracy | Device |",
             "|---|---:|---:|---|"]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.model} | {row.mean_latency_ms:.3f} | {row.accuracy:.1%} | {row.device} |")
    report = "\n".join(lines) + f"\n\nPrivateTest: {args.samples} images, {args.repeats} repeats, seed {args.seed}.\n\n" + metadata["timing"] + "\n"
    (output_dir / "benchmark.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

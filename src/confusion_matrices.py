import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent.parent
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PROJECT_DIR / "results" / "predictions.csv")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.predictions)
    required = {"model", "row_id", "repeat", "true_label", "predicted_emotion"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Expected a nonempty benchmark predictions CSV.")
    if not data["true_label"].isin(range(7)).all() or not data["predicted_emotion"].isin(EMOTIONS).all():
        raise ValueError("Unknown emotion label in predictions.")
    data = data[data["repeat"] == 1].copy()
    if data.empty or data.duplicated(["model", "row_id"]).any():
        raise ValueError("Expected exactly one prediction per image and model in repeat 1.")
    image_sets = data.groupby("model")["row_id"].apply(set).tolist()
    if any(images != image_sets[0] for images in image_sets):
        raise ValueError("Models must be evaluated on the same images.")
    if (data.groupby("row_id")["true_label"].nunique() != 1).any():
        raise ValueError("Inconsistent ground truth across models.")

    output_dir = args.output_dir or args.predictions.parent / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = list(data.groupby("model", sort=False))
    matrices = []
    label_ids = list(range(7))
    for name, group in groups:
        predicted = group["predicted_emotion"].map(dict(zip(EMOTIONS, label_ids)))
        counts = confusion_matrix(group["true_label"], predicted, labels=label_ids)
        support = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, support, out=np.zeros_like(counts, dtype=float), where=support != 0)
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        pd.DataFrame(counts, index=EMOTIONS, columns=EMOTIONS).rename_axis("True emotion").to_csv(output_dir / f"{slug}_counts.csv")
        pd.DataFrame(normalized, index=EMOTIONS, columns=EMOTIONS).rename_axis("True emotion").to_csv(output_dir / f"{slug}_normalized.csv")
        report = classification_report(group["true_label"], predicted, labels=label_ids, target_names=EMOTIONS, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output_dir / f"{slug}_classification_report.csv")
        matrices.append((name, counts, normalized))
        if counts.sum() != len(group):
            raise ValueError("Confusion matrix does not match the sample count.")

    sample_count = len(image_sets[0])
    for normalized in (False, True):
        fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 6), squeeze=False, layout="constrained")
        for axis, (name, counts, fractions) in zip(axes[0], matrices):
            values = fractions * 100 if normalized else counts
            heatmap = axis.imshow(values, cmap="Blues", vmin=0, vmax=100 if normalized else max(int(matrix[1].max()) for matrix in matrices))
            axis.set_xticks(label_ids, EMOTIONS, rotation=45, ha="right")
            axis.set_yticks(label_ids, EMOTIONS)
            axis.set_xlabel("Predicted emotion")
            axis.set_ylabel("True emotion")
            accuracy = np.trace(counts) / counts.sum()
            axis.set_title(f"{name}\nAccuracy: {accuracy:.2%} | n = {sample_count}")
            threshold = heatmap.norm.vmax / 2
            for row in label_ids:
                for col in label_ids:
                    text = f"{values[row, col]:.0f}%" if normalized else str(counts[row, col])
                    if normalized and counts[row].sum() == 0:
                        text = "N/A"
                    axis.text(col, row, text, ha="center", va="center", color="white" if values[row, col] > threshold else "black", fontsize=9)
            fig.colorbar(heatmap, ax=axis, shrink=0.75, label="Percent of true class" if normalized else "Images")
        fig.suptitle("FER-2013 PrivateTest — " + ("Row-normalized confusion matrices" if normalized else "Confusion matrices"), fontsize=16)
        filename = "normalized.png" if normalized else "counts.png"
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    sample_note = (
        "A 50-image sample has limited support per class; use the full CNN test for a more stable per-class analysis."
        if sample_count <= 50 else
        "This full PrivateTest evaluation gives more stable per-class estimates than the 50-image comparison."
    )
    note = (
        f"# Confusion matrices\n\n{sample_count} unique PrivateTest images per model. "
        "Only repeat 1 is used; timing repetitions do not add independent test images.\n\n"
        "Rows are true labels; columns are predicted labels. The diagonal contains correct predictions. "
        "Normalized values show the percentage within each true class. N/A indicates a class with no test examples.\n\n"
        "![Counts](counts.png)\n\n![Row-normalized percentages](normalized.png)\n\n"
        "CSV files contain exact counts, normalized fractions, and classification reports. "
        f"{sample_note}\n"
    )
    (output_dir / "README.md").write_text(note, encoding="utf-8")
    print(f"Confusion matrices saved to: {output_dir}")


if __name__ == "__main__":
    main()

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from train_svm import prepare_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
BASE_PATH = PROJECT_DIR / "models" / "svm_hog.pkl"
OUTPUT_PATH = PROJECT_DIR / "models" / "svm_hog_calibrated.pkl"


def main():
    csv_path = PROJECT_DIR / "data" / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Run python src/download_dataset.py first.")
    df = pd.read_csv(csv_path)
    calibration_df = df[df["Usage"] == "PublicTest"]
    if set(calibration_df["emotion"].unique()) != set(range(7)):
        raise ValueError("PublicTest must include all seven emotion classes.")

    base_model = joblib.load(BASE_PATH)
    print(f"Extracting HOG features for {len(calibration_df)} calibration images...")
    features, labels = prepare_data(calibration_df)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(features, labels)
    probabilities = calibrated.predict_proba(features[:10])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid calibrated probabilities.")

    joblib.dump(calibrated, OUTPUT_PATH)
    metadata = {
        "calibration_split": "PublicTest",
        "calibration_images": len(calibration_df),
        "method": "sigmoid",
        "base_model_sha256": hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(),
        "evaluation_split": "PrivateTest",
    }
    OUTPUT_PATH.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Calibrated model saved to: {OUTPUT_PATH}")
    print("Restart the app to load calibrated SVM confidence.")


if __name__ == "__main__":
    main()

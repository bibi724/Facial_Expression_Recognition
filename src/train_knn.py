import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from time import perf_counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from lbp_features import extract_lbp


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "fer2013.csv"
MODEL_PATH = PROJECT_DIR / "models" / "knn_lbp.pkl.gz"

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]


def prepare_data(df):

    X = np.zeros((len(df), 4096), dtype=np.float32)
    y = df["emotion"].to_numpy(dtype=np.int64)

    for i, pixels in enumerate(df["pixels"]):

        image = np.fromstring(
            pixels, sep=" ", dtype=np.uint8
        ).reshape(48, 48)

        X[i] = extract_lbp(image)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(df)} images")

    return X, y


def main():

    print("Loading FER-2013...")
    df = pd.read_csv(DATA_PATH)

    train_df = df[df["Usage"] == "Training"]
    test_df = df[df["Usage"] == "PublicTest"]

    print("Training images:", len(train_df))
    print("Validation images:", len(test_df))

    print("\nExtracting training features...")
    X_train, y_train = prepare_data(train_df)

    print("\nExtracting validation features...")
    X_test, y_test = prepare_data(test_df)

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )

    print("\nFitting KNN...")
    start = perf_counter()
    knn.fit(X_train, y_train)
    print(f"Fit time: {perf_counter() - start:.2f} s")

    print("\nEvaluating KNN...")
    start = perf_counter()
    predictions = knn.predict(X_test)
    elapsed = perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Total prediction time: {elapsed:.2f} s")
    print(f"Mean prediction time: {1000 * elapsed / len(X_test):.2f} ms/image")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(7)),
        target_names=EMOTIONS,
        zero_division=0
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(knn, MODEL_PATH, compress=("gzip", 3))

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

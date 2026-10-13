import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from time import perf_counter
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from hog_features import extract_hog


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "fer2013.csv"
MODEL_PATH = PROJECT_DIR / "models" / "svm_hog.pkl"

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]


def prepare_data(df):

    X = np.zeros((len(df), 900), dtype=np.float32)
    y = df["emotion"].to_numpy(dtype=np.int64)

    for i, pixels in enumerate(df["pixels"]):

        image = np.fromstring(
            pixels,
            sep=" ",
            dtype=np.uint8
        ).reshape(48, 48)

        X[i] = extract_hog(image)

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

    print("\nExtracting training HOG features...")
    X_train, y_train = prepare_data(train_df)

    print("\nExtracting validation HOG features...")
    X_test, y_test = prepare_data(test_df)

    svm = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LinearSVC(
                C=0.1,
                max_iter=10000,
                random_state=42
            ),
            n_jobs=-1
        )
    )

    print("\nTraining HOG + SVM...")
    start = perf_counter()
    svm.fit(X_train, y_train)
    print(f"Training time: {perf_counter() - start:.2f} s")

    print("\nEvaluating HOG + SVM...")
    start = perf_counter()
    predictions = svm.predict(X_test)
    elapsed = perf_counter() - start

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nValidation accuracy: {accuracy:.4f}")
    print(f"Total prediction time: {elapsed:.2f} s")
    print(f"Mean prediction time: {elapsed * 1000 / len(X_test):.3f} ms/image")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(7)),
        target_names=EMOTIONS,
        zero_division=0
    ))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

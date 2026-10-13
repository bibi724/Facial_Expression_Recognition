# Facial Expression Recognition

A Gradio application that compares LBP + KNN, HOG + Linear SVM (One-vs-Rest), and a pretrained mini-Xception CNN on the same detected face.

Upload an image or capture a webcam snapshot, then click **Analyze Expression**. The interface displays the detected face and the three predictions side by side.

Repository: [bibi724/Facial_Expression_Recognition](https://github.com/bibi724/Facial_Expression_Recognition).

**Authors**: Letícia Maria Resende, Beatriz Araujo Cavalcante -- ENSTA Paris, 2026

## What the application does

| Feature | Description |
|---|---|
| Image upload and webcam snapshot | Accepts a still image from either source. |
| Face processing | Dlib detects the first face; the application crops it, converts it to grayscale, and resizes it to 48 x 48. |
| Three predictions | LBP + KNN, HOG + Linear SVM, and mini-Xception process the same face. |
| Scores and latency | Each model displays a label, a confidence estimate, and inference-related latency. |
| Evaluation | Reproducible benchmark and confusion matrices are included in `results/`. |

The webcam input captures a snapshot. The application does not perform continuous video inference. Only the first detected face is processed. Landmark alignment is not implemented.

Personal sample images in `images/` are excluded from Git. The optional `src/face_detection.py` example expects a local `images/demo.jpg`; the Gradio interface works without it.

## Pipeline

```text
Image / webcam snapshot
    -> Dlib face detection
    -> Crop first detected face
    -> Grayscale and resize to 48 x 48
    -> LBP + KNN / HOG + SVM / mini-Xception
    -> Label, confidence or decision score, and latency
```

| Model | Features and classifier | Output score |
|---|---|---|
| LBP + KNN | 4 x 4 blocks, 256-bin LBP histograms, 4,096 features; distance-weighted KNN with 5 neighbors | Neighbor-vote probability estimate |
| HOG + Linear SVM | 9 orientations, 8 x 8 cells, 2 x 2 blocks, 900 features; StandardScaler + OneVsRestClassifier(LinearSVC) | Sigmoid-calibrated probability estimate |
| mini-Xception | Pretrained PyTorch CNN; grayscale pixels normalized to [0, 1] | Softmax score |

Scores are confidence estimates, not guarantees of correctness. The SVM uses sigmoid calibration on PublicTest, separate from its Training split. The app prefers models/svm_hog_calibrated.pkl. Without that file, it keeps the original model available and explicitly displays an uncalibrated decision score.

## Installation on Windows

Run the commands below in PowerShell from the project root.

The tested setup uses Python **3.13.14** and a `.venv` environment.

### 1. Create and activate the environment

For a fresh clone:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the environment already exists, run only the activation command. If PowerShell blocks activation, allow scripts for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

In VS Code, use **Python: Select Interpreter** and select `.venv\Scripts\python.exe`.

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` records the direct package versions installed on the current computer. It is not a full dependency lockfile. Dlib may be compiled during installation; this completed successfully on the current computer.

The current computer has Intel Iris Xe graphics and runs the CNN on **CPU**. The worker also supports CUDA when a compatible NVIDIA GPU, driver, and PyTorch build are available. For GPU installation, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) instead of the CPU command above.

### 3. Prepare the trained models

The repository includes the trained KNN model as `models/knn_lbp.pkl.gz` (about 33 MB), along with the base and calibrated SVM models. Joblib reads the compressed KNN file directly. The local uncompressed model is excluded from Git.

Download the mini-Xception source and checkpoint into the expected directory:

```powershell
git clone https://github.com/yoshidan/pytorch-facial-expression-recognition.git external/pytorch-facial-expression-recognition-master
```

Run the clone command only if that directory does not already exist. The application expects:

```text
external/pytorch-facial-expression-recognition-master/src/model.py
external/pytorch-facial-expression-recognition-master/trained/private_model_233_66.t7
```

Use the same scikit-learn version that created the saved classical models when possible. If loading reports an incompatible version, regenerate them using the training commands below rather than assuming cross-version compatibility.

### 4. Start the interface

```powershell
python app.py
```

Open the local URL printed in the terminal, normally `http://127.0.0.1:7860`.

1. Upload a clear, front-facing image or capture a webcam snapshot.
2. Click **Analyze Expression**.
3. Check the face rectangle and the three model outputs.
4. Try another expression using the same person and lighting.

Stop the application with **Ctrl+C** in the terminal.

The CNN runs in a separate persistent process to isolate PyTorch from the classical model stack after an OpenMP conflict on the original development machine. Images and results are exchanged as JSON lines. The worker is started automatically and warmed up once; do not start it manually.

## Dataset and training

The interface uses the saved models and does not require the training dataset. Training and evaluation scripts require the FER-2013 CSV at:

```text
data/fer2013.csv
```

The CSV must contain `emotion`, `pixels`, and `Usage`, with split names `Training`, `PublicTest`, and `PrivateTest`. Labels are ordered as:

```text
0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral
```

The dataset source documented during earlier development is [chitradrishti/fer2013](https://huggingface.co/datasets/chitradrishti/fer2013). Download it with the command below; the CSV and ZIP are ignored by Git.

```powershell
python src/download_dataset.py
python src/check_dataset.py
python src/train_knn.py
python src/train_svm.py
python src/test_cnn.py
```

The training scripts train on `Training`, evaluate on `PublicTest`, and overwrite their corresponding files in `models/`. `test_cnn.py` checks one `PrivateTest` image; it does not compute test-set accuracy.

The [mini-Xception source repository](https://github.com/yoshidan/pytorch-facial-expression-recognition) reports 66% accuracy on the FER-2013 private split. This is an upstream result, not a measurement reproduced by this project. Local evaluation on all 3,589 PrivateTest images achieved **64.56% accuracy**, within the approximate 60-67% lab target. Details are in [the full CNN report](results/cnn_full/benchmark.md).

## Latency and evaluation

The current interface measures:

- **KNN:** LBP feature extraction and classifier probability prediction.
- **SVM:** HOG feature extraction and calibrated probability prediction (or raw decision scores when calibration is unavailable).
- **CNN:** The full request to the worker, including JSON transfer, tensor preparation, inference, and response parsing. The worker-only timing is also saved separately in results/predictions.csv.

Face detection, cropping, resizing, model loading, and UI rendering are excluded from all model timings. The interface shows individual timings; the benchmark computes their means over repeated predictions using the same inference code.

Calibrate the SVM once, then run the benchmark:

```powershell
python src/download_dataset.py
python src/calibrate_svm.py
python src/benchmark.py --samples 50 --repeats 3
```

Restart the interface with `python app.py` after calibration. If you retrain the base SVM, rerun calibration before using the calibrated artifact.

Calibration uses all 3,589 PublicTest images. Evaluation uses a reproducible random sample from PrivateTest, with seed 42 by default. The same 50 images are evaluated in three repetitions; this is 50 unique images, not 150. Warm-up uses up to three selected images and is excluded from the measurements. FER-2013 already provides 48 x 48 face crops, so the benchmark skips face detection.

Generated files:

- `results/benchmark.md`: mean latency and small-sample accuracy table.
- `results/benchmark.csv`: mean latency, standard deviation, accuracy, and device per model.
- `results/predictions.csv`: individual predictions, image row IDs, confidence, and timings.
- `results/benchmark.json`: machine, package versions, device, sampling configuration, and timing definition.

CNN latency includes interprocess communication; worker inference time is recorded separately. The benchmark measures the implementation used by the interface, rather than pure classifier compute alone. Accuracy on 50 images is a small-sample observation and cannot establish full FER-2013 accuracy. To evaluate the CNN on the entire private split without overwriting the three-model table, run `python src/benchmark.py --samples 3589 --repeats 1 --models cnn --output-dir results/cnn_full`. The same command without `--models cnn` evaluates all three models and takes longer.

| Model | Mean latency (ms/image) | Accuracy | Device |
|---|---:|---:|---|
| LBP + KNN | 216.968 | 30.0% | cpu |
| HOG + Linear SVM | 6.080 | 32.0% | cpu |
| mini-Xception | 12.840 | 50.0% | cpu |

PrivateTest: 50 images, 3 repeats, seed 42.

Per-image feature extraction and prediction; CNN includes JSON transfer and worker round trip. Detection, resizing, loading and warm-up excluded.

See [the measured results](results/benchmark.md). Measurements apply to the hardware and package versions recorded in `results/benchmark.json`.

## Confusion matrices

Generate count and row-normalized plots from the saved predictions:

```powershell
python src/confusion_matrices.py
python src/confusion_matrices.py --predictions results/cnn_full/predictions.csv
```

The plots use only repeat 1, so each test image is counted once. Rows are true emotions and columns are predictions. The diagonal shows correct predictions; normalized rows show class recall and which other labels receive the errors.

- [Three-model comparison on the same 50 images](results/confusion_matrices/README.md).
- [CNN evaluation on all 3,589 PrivateTest images](results/cnn_full/confusion_matrices/README.md).

The full CNN evaluation recognizes 89.3% of Happy images correctly. Disgust and Fear have lower recall, at 27.3% and 29.9%. Of the 55 Disgust images, 25 are predicted as Angry; of the 528 Fear images, 143 are predicted as Sad. These class differences are not apparent from the overall 64.56% accuracy alone. The three-model plots use a much smaller sample and should be interpreted with their class supports in mind.

Exact matrices and per-class precision, recall, F1, and support are exported as CSV files alongside the plots.

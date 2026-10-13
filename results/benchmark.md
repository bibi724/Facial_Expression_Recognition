| Model | Mean latency (ms/image) | Accuracy | Device |
|---|---:|---:|---|
| LBP + KNN | 216.968 | 30.0% | cpu |
| HOG + Linear SVM | 6.080 | 32.0% | cpu |
| mini-Xception | 12.840 | 50.0% | cpu |

PrivateTest: 50 images, 3 repeats, seed 42.

Per-image feature extraction and prediction; CNN includes JSON transfer and worker round trip. Detection, resizing, loading and warm-up excluded.

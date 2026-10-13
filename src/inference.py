import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from hog_features import extract_hog
from lbp_features import extract_lbp


PROJECT_DIR = Path(__file__).resolve().parent.parent
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class ModelComparison:
    def __init__(self):
        self.knn = joblib.load(PROJECT_DIR / "models" / "knn_lbp.pkl.gz")
        calibrated_path = PROJECT_DIR / "models" / "svm_hog_calibrated.pkl"
        base_path = PROJECT_DIR / "models" / "svm_hog.pkl"
        self.svm = joblib.load(calibrated_path if calibrated_path.exists() else base_path)
        self.calibrated = hasattr(self.svm, "predict_proba")
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(PROJECT_DIR / "src" / "cnn_worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_DIR),
        )
        threading.Thread(target=self._read_responses, daemon=True).start()
        try:
            self.ready = self._receive(timeout=120)
            if self.ready.get("status") != "ready":
                raise RuntimeError(f"CNN initialization failed: {self.ready}")
        except Exception:
            self.close()
            raise

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def _receive(self, timeout=30):
        try:
            line = self.responses.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("CNN worker timed out. Check terminal errors.") from exc
        if line is None:
            raise RuntimeError("CNN worker stopped. Check terminal errors.")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def predict(self, image, models=None):
        image = np.asarray(image, dtype=np.uint8)
        if image.shape != (48, 48):
            raise ValueError("Expected a 48 x 48 grayscale face.")

        results = {}
        selected = set(models) if models is not None else {"LBP + KNN", "HOG + Linear SVM", "mini-Xception"}
        for name, model, extractor in (
            ("LBP + KNN", self.knn, extract_lbp),
            ("HOG + Linear SVM", self.svm, extract_hog),
        ):
            if name not in selected:
                continue
            start = perf_counter()
            features = extractor(image).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(features)[0]
                index = int(np.argmax(scores))
                label = int(model.classes_[index])
                confidence = float(scores[index])
                decision_score = None
            else:
                scores = model.decision_function(features)[0]
                index = int(np.argmax(scores))
                label = int(model.classes_[index])
                confidence = None
                decision_score = float(scores[index])
            results[name] = {
                "emotion": EMOTIONS[label],
                "confidence": confidence,
                "decision_score": decision_score,
                "latency_ms": (perf_counter() - start) * 1000,
            }

        if "mini-Xception" not in selected:
            return results
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("CNN worker has stopped.")
            start = perf_counter()
            self.process.stdin.write(json.dumps({"image": image.tolist()}) + "\n")
            self.process.stdin.flush()
            response = self._receive()
            response["worker_latency_ms"] = response["latency_ms"]
            response["latency_ms"] = (perf_counter() - start) * 1000
        results["mini-Xception"] = response
        return results

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

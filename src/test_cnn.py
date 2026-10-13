import sys
from pathlib import Path
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent

REPO_DIR = (
    PROJECT_DIR
    / "external"
    / "pytorch-facial-expression-recognition-master"
)

sys.path.insert(0, str(REPO_DIR / "src"))

from model import Model

WEIGHTS_PATH = (
    REPO_DIR
    / "trained"
    / "private_model_233_66.t7"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = Model(num_classes=7).to(device)

checkpoint = torch.load(
    WEIGHTS_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["net"])
model.eval()

print("Mini-Xception loaded successfully!")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

csv_path = PROJECT_DIR / "data" / "fer2013.csv"
df = pd.read_csv(csv_path)

test_df = df[df["Usage"] == "PrivateTest"]
sample = test_df.iloc[0]

image = np.fromstring(
    sample["pixels"],
    sep=" ",
    dtype=np.uint8
).reshape(48, 48)

true_label = int(sample["emotion"])

input_tensor = torch.from_numpy(
    image.astype(np.float32) / 255.0
).unsqueeze(0).unsqueeze(0).to(device)

with torch.inference_mode():
    model(input_tensor)

if device.type == "cuda":
    torch.cuda.synchronize()

start = perf_counter()

with torch.inference_mode():
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)

if device.type == "cuda":
    torch.cuda.synchronize()

latency_ms = (perf_counter() - start) * 1000

predicted_label = probabilities.argmax(dim=1).item()
confidence = probabilities[0, predicted_label].item()

print("\n===== MINI-XCEPTION RESULT =====")
print("True emotion:", EMOTIONS[true_label])
print("Predicted emotion:", EMOTIONS[predicted_label])
print(f"Confidence: {confidence * 100:.2f}%")
print(f"Inference latency: {latency_ms:.3f} ms")

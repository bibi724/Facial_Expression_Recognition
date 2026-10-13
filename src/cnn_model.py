import sys
from pathlib import Path
import numpy as np
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

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]


class EmotionCNN:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = Model(num_classes=7).to(self.device)

        checkpoint = torch.load(
            WEIGHTS_PATH,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(checkpoint["net"])
        self.model.eval()

        print(f"Mini-Xception loaded on {self.device}")

    def predict(self, image):
        image = np.asarray(image, dtype=np.uint8)

        tensor = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)

        predicted_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()

        return EMOTIONS[predicted_class], confidence

if __name__ == "__main__":
    cnn = EmotionCNN()

    image = np.random.randint(
        0, 256, (48, 48), dtype=np.uint8
    )

    emotion, confidence = cnn.predict(image)

    print("Predicted emotion:", emotion)
    print(f"Confidence: {confidence * 100:.2f}%")

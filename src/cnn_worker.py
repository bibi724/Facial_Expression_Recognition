import sys
import json
from time import perf_counter
import numpy as np
import torch
from cnn_model import EmotionCNN


def main():
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        cnn = EmotionCNN()

    use_cuda = cnn.device.type == "cuda"

    dummy = np.zeros((48, 48), dtype=np.uint8)
    cnn.predict(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    print(json.dumps({
        "status": "ready",
        "device": str(cnn.device),
        "device_name": torch.cuda.get_device_name(0) if use_cuda else "CPU"
    }), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            image = np.asarray(
                request["image"],
                dtype=np.uint8
            ).reshape(48, 48)

            if use_cuda:
                torch.cuda.synchronize()
            start = perf_counter()

            emotion, confidence = cnn.predict(image)

            if use_cuda:
                torch.cuda.synchronize()
            latency_ms = (perf_counter() - start) * 1000

            print(json.dumps({
                "emotion": emotion,
                "confidence": confidence,
                "latency_ms": latency_ms
            }), flush=True)

        except Exception as exc:
            print(json.dumps({
                "error": str(exc)
            }), flush=True)


if __name__ == "__main__":
    main()

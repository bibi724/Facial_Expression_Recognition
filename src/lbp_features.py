import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp(image):

    P = 8
    R = 1

    lbp = local_binary_pattern(
        image,
        P=P,
        R=R,
        method="default"
    )

    histograms = []

    for row in range(4):
        for col in range(4):

            block = lbp[
                row * 12:(row + 1) * 12,
                col * 12:(col + 1) * 12
            ]

            hist, _ = np.histogram(
                block,
                bins=256,
                range=(0, 256)
            )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-10

            histograms.append(hist)

    features = np.concatenate(histograms)

    return features


if __name__ == "__main__":

    image = np.random.randint(
        0, 256,
        (48, 48),
        dtype=np.uint8
    )

    features = extract_lbp(image)

    print("Image shape:", image.shape)
    print("Feature vector shape:", features.shape)

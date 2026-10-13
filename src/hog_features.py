import numpy as np
from skimage.feature import hog

def extract_hog(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features.astype(np.float32)


if __name__ == "__main__":

    image = np.random.randint(
        0, 256,
        (48, 48),
        dtype=np.uint8
    )

    features = extract_hog(image)

    print("Image shape:", image.shape)
    print("HOG feature vector:", features.shape)

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
csv_path = project_dir / "data" / "fer2013.csv"

df = pd.read_csv(csv_path)

print("Dataset dimensions:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df[["emotion", "Usage"]].head())

print("\nNumber of images per emotion:")
print(df["emotion"].value_counts().sort_index())

print("\nData split:")
print(df["Usage"].value_counts())

pixels = df.iloc[0]["pixels"]
image_array = np.fromstring(pixels, sep=" ", dtype=np.uint8)
image = image_array.reshape(48, 48)

print("\nImage dimensions:", image.shape)

plt.imshow(image, cmap="gray", vmin=0, vmax=255)
plt.title("FER-2013 - First image")
plt.axis("off")
plt.show()

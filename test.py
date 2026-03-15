import rasterio
import numpy as np
import matplotlib.pyplot as plt

image_path = "data/raw_sar/scene2.tif"   # change to one of your scenes

with rasterio.open(image_path) as src:
    img = src.read(1).astype("float32")

#same preprocessing as dataset
img = np.abs(img)
img = np.log1p(img)
img = img / 10.0

plt.figure(figsize=(6,4))
plt.hist(img.flatten(), bins=100)
plt.title("Pixel Intensity Distribution in SAR Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

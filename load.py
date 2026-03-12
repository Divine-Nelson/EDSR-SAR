import rasterio
import numpy as np

PATCH_SIZE = 512

with rasterio.open("data/raw_sar/scene1.tif") as src:
    image = src.read(1)
    print(image.shape)

# normalize
image = image.astype("float32")
image = (image - image.min()) / (image.max() - image.min())

patches = [image]

for y in range(0, image.shape[0] - PATCH_SIZE, PATCH_SIZE):
    for x in range(0, image.shape[1] - PATCH_SIZE, PATCH_SIZE):
        patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        patches.append(patch)

print("Number of patches:", len(patches))
import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.edsr import EDSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load model
# ----------------------------

model = EDSR().to(device)
model.load_state_dict(torch.load("models/edsr_sar.pth", map_location=device))
model.eval()

# ----------------------------
# Load SAR image
# ----------------------------

with rasterio.open("data/test_sar/s1_7/scene15.tif") as src:
    image = src.read(1)

print("Image stats:", image.min(), image.max(), image.mean())

# ----------------------------
# Normalize (works for any dataset)
# ----------------------------

image = image.astype("float32")
image = (image - image.min()) / (image.max() - image.min() + 1e-8)

patch = image
h, w = patch.shape

# ----------------------------
# Create low resolution version
# ----------------------------

scale = 4

lr = cv2.resize(
    patch,
    (w//scale, h//scale),
    interpolation=cv2.INTER_AREA
)

bicubic = cv2.resize(
    lr,
    (w, h),
    interpolation=cv2.INTER_CUBIC
)

# ----------------------------
# Run EDSR
# ----------------------------

input_tensor = torch.tensor(bicubic).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    sr = model(input_tensor)

sr = sr.squeeze().cpu().numpy()
sr = np.clip(sr, 0, 1)

# ----------------------------
# Compute metrics
# ----------------------------

psnr_bicubic = peak_signal_noise_ratio(patch, bicubic, data_range=1.0)
psnr_edsr = peak_signal_noise_ratio(patch, sr, data_range=1.0)

ssim_bicubic = structural_similarity(patch, bicubic, data_range=1.0)
ssim_edsr = structural_similarity(patch, sr, data_range=1.0)

print("\nResults")
print("------------------------")

print("Bicubic PSNR:", psnr_bicubic)
print("EDSR PSNR:", psnr_edsr)

print("Bicubic SSIM:", ssim_bicubic)
print("EDSR SSIM:", ssim_edsr)

# ----------------------------
# Visualization
# ----------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Ground Truth")
plt.imshow(patch, cmap="gray")
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Low Resolution")
plt.imshow(cv2.resize(lr,(w,h)), cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Bicubic")
plt.imshow(bicubic, cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("EDSR")
plt.imshow(sr, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models.edsr import EDSR
import matplotlib.patches as patches
from torchinfo import summary



# --------------------------------------------------
# Setup
# --------------------------------------------------

np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = EDSR().to(device)
model.load_state_dict(
    torch.load("models/best_edsr_sar.pth", map_location=device, weights_only=True)
)
model.eval()
#print(sum(p.numel() for p in model.parameters()))

#print("\nModel Summary\n")
#summary(model, input_size=(1,1,256,256))
# --------------------------------------------------
# Load SAR scene
# --------------------------------------------------

with rasterio.open("data/raw_sar/scene20.tif") as src:
    image = src.read(1).astype(np.float32)

h, w = image.shape


# --------------------------------------------------
# Evaluation settings
# --------------------------------------------------

num_tests = 500
patch_size = 256
lr_size = 64
crop = 4

psnr_bicubic = []
psnr_edsr = []

ssim_bicubic = []
ssim_edsr = []


# zoom settings
zoom_x = 120
zoom_y = 140
zoom_size = 40
zoom_scale = 6


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------

saved_example = False

for _ in range(num_tests):

    # -------------------------
    # Random patch selection
    # -------------------------

    if h > patch_size and w > patch_size:

        for _ in range(50):  # safe retry loop
            y = np.random.randint(0, h - patch_size)
            x = np.random.randint(0, w - patch_size)

            patch = image[y:y+patch_size, x:x+patch_size]

            if patch.std() > 50:
                break
    else:
        patch = image.copy()


    # -------------------------
    # SAR preprocessing
    # -------------------------

    patch = np.log1p(np.abs(patch))
    patch = patch / 10.0
    patch = np.clip(patch, 0, 1)


    # -------------------------
    # Generate LR
    # -------------------------

    lr = cv2.resize(patch, (lr_size, lr_size), interpolation=cv2.INTER_AREA)

    bicubic = cv2.resize(lr, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)


    # -------------------------
    # Model inference
    # -------------------------

    input_tensor = torch.from_numpy(bicubic).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        residual = model(input_tensor)
        sr = input_tensor + residual

    sr = sr.squeeze().cpu().numpy()
    sr = np.clip(sr, 0, 1)

    sr_vis = sr.copy()


    # -------------------------
    # Crop borders for metrics
    # -------------------------

    gt = patch[crop:-crop, crop:-crop]
    bc = bicubic[crop:-crop, crop:-crop]
    sr_eval = sr[crop:-crop, crop:-crop]


    # -------------------------
    # Metrics
    # -------------------------

    psnr_bicubic.append(
        peak_signal_noise_ratio(gt, bc, data_range=1.0)
    )

    psnr_edsr.append(
        peak_signal_noise_ratio(gt, sr_eval, data_range=1.0)
    )

    ssim_bicubic.append(
        structural_similarity(gt, bc, data_range=1.0, win_size=11)
    )

    ssim_edsr.append(
        structural_similarity(gt, sr_eval, data_range=1.0, win_size=11)
    )


    # -------------------------
    # Save one example for visualization
    # -------------------------

    if not saved_example:

        ex_patch = patch
        ex_lr = lr
        ex_bicubic = bicubic
        ex_sr = sr_vis

        # ensure zoom region valid
        zx = min(zoom_x, patch_size - zoom_size)
        zy = min(zoom_y, patch_size - zoom_size)

        gt_zoom = ex_patch[zy:zy+zoom_size, zx:zx+zoom_size]
        bicubic_zoom = ex_bicubic[zy:zy+zoom_size, zx:zx+zoom_size]
        edsr_zoom = ex_sr[zy:zy+zoom_size, zx:zx+zoom_size]

        gt_zoom = cv2.resize(gt_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
        bicubic_zoom = cv2.resize(bicubic_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
        edsr_zoom = cv2.resize(edsr_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)

        saved_example = True



# --------------------------------------------------
# Convert metrics
# --------------------------------------------------

psnr_bicubic = np.array(psnr_bicubic)
psnr_edsr = np.array(psnr_edsr)

# Compute error maps
error_bicubic = np.abs(ex_patch - ex_bicubic)
error_edsr = np.abs(ex_patch - ex_sr)

# remove infinite values
valid_mask = np.isfinite(psnr_bicubic) & np.isfinite(psnr_edsr)

psnr_bicubic = psnr_bicubic[valid_mask]
psnr_edsr = psnr_edsr[valid_mask]

psnr_gain = psnr_edsr - psnr_bicubic

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nAverage Results")
print("----------------")

print("Bicubic PSNR:", np.mean(psnr_bicubic))
print("EDSR PSNR:", np.mean(psnr_edsr))

print("Bicubic SSIM:", np.mean(ssim_bicubic))
print("EDSR SSIM:", np.mean(ssim_edsr))

print("\nEDSR PSNR std:", np.std(psnr_edsr))
print("EDSR PSNR min:", np.min(psnr_edsr))
print("EDSR PSNR max:", np.max(psnr_edsr))


# --------------------------------------------------
# Visual comparison
# --------------------------------------------------

plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Ground Truth")
plt.imshow(ex_patch, cmap="gray")
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Low Resolution")
plt.imshow(ex_lr, cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("Bicubic")
plt.imshow(ex_bicubic, cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("EDSR")
plt.imshow(ex_sr, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
plt.show()


# --------------------------------------------------
# PSNR histogram
# --------------------------------------------------

plt.figure()
plt.hist(psnr_edsr, bins=30)
plt.xlabel("PSNR")
plt.ylabel("Number of Patches")
plt.title("PSNR Distribution Across Patches (EDSR)")
plt.show()


# --------------------------------------------------
# PSNR improvement histogram
# --------------------------------------------------

plt.figure()
plt.hist(psnr_gain, bins=30)
plt.xlabel("PSNR Improvement (EDSR - Bicubic)")
plt.title("PSNR Gain Distribution")
plt.show()


# --------------------------------------------------
# Zoom comparison
# --------------------------------------------------

plt.figure(figsize=(10,6))

plt.subplot(1,3,1)
plt.title("GT (zoom)")
plt.imshow(gt_zoom, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Bicubic (zoom)")
plt.imshow(bicubic_zoom, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("EDSR (zoom)")
plt.imshow(edsr_zoom, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots()

ax.imshow(ex_patch, cmap="gray")

rect = patches.Rectangle(
    (zoom_x, zoom_y),
    zoom_size,
    zoom_size,
    linewidth=2,
    edgecolor="red",
    facecolor="none"
)

ax.add_patch(rect)

plt.title("Ground Truth with Zoom Region")
plt.axis("off")
plt.show()

#Error display:
plt.figure(figsize=(18,4))

plt.subplot(1,5,1)
plt.title("Ground Truth")
plt.imshow(ex_patch, cmap="gray")
plt.axis("off")

plt.subplot(1,5,2)
plt.title("Bicubic Error")
plt.imshow(error_bicubic, cmap="hot")
plt.colorbar()
plt.axis("off")

plt.subplot(1,5,3)
plt.title("EDSR Error")
plt.imshow(error_edsr, cmap="hot")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()
import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#from models.srcnn import SRCNN
from models.res_srcnn import ResidualSRCNN
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

model = ResidualSRCNN().to(device)
model.load_state_dict(
    torch.load("models/best_res_srcnn_sar.pth", map_location=device, weights_only=True)
    #torch.load("models/res_srcnn_sar_final.pth", map_location=device, weights_only=True)
)
model.eval()
#print(sum(p.numel() for p in model.parameters()))

#print("\nModel Summary\n")
#summary(model, input_size=(1,1,256,256))
# --------------------------------------------------
# Load SAR scene
# --------------------------------------------------

#with rasterio.open("data/raw_sar/scene10.tif") as src:
with rasterio.open("data/test_sar/s1_7/scene10.tif") as src:

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
psnr_SRCNN = []

ssim_bicubic = []
ssim_SRCNN = []


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

    psnr_SRCNN.append(
        peak_signal_noise_ratio(gt, sr_eval, data_range=1.0)
    )

    ssim_bicubic.append(
        structural_similarity(gt, bc, data_range=1.0, win_size=11)
    )

    ssim_SRCNN.append(
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
        SRCNN_zoom = ex_sr[zy:zy+zoom_size, zx:zx+zoom_size]

        gt_zoom = cv2.resize(gt_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
        bicubic_zoom = cv2.resize(bicubic_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)
        SRCNN_zoom = cv2.resize(SRCNN_zoom, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_NEAREST)

        saved_example = True



# --------------------------------------------------
# Convert metrics
# --------------------------------------------------

psnr_bicubic = np.array(psnr_bicubic)
psnr_SRCNN = np.array(psnr_SRCNN)

# Compute error maps
error_bicubic = np.abs(ex_patch - ex_bicubic)
error_SRCNN = np.abs(ex_patch - ex_sr)

# remove infinite values
valid_mask = np.isfinite(psnr_bicubic) & np.isfinite(psnr_SRCNN)

psnr_bicubic = psnr_bicubic[valid_mask]
psnr_SRCNN = psnr_SRCNN[valid_mask]

psnr_gain = psnr_SRCNN - psnr_bicubic

# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nAverage Results")
print("----------------")

print("Bicubic PSNR:", np.mean(psnr_bicubic))
print("SRCNN PSNR:", np.mean(psnr_SRCNN))

print("Bicubic SSIM:", np.mean(ssim_bicubic))
print("SRCNN SSIM:", np.mean(ssim_SRCNN))

print("\nSRCNN PSNR std:", np.std(psnr_SRCNN))
print("SRCNN PSNR min:", np.min(psnr_SRCNN))
print("SRCNN PSNR max:", np.max(psnr_SRCNN))
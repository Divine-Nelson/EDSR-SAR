import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
from rasterio.windows import Window

from datasets import SARSuperResolutionDataset
from models.edsr import EDSR


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    np.random.seed(42)
    torch.manual_seed(42)

    dataset = SARSuperResolutionDataset(
        "data/raw_sar",
        patch_size=256
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model = EDSR().to(device)
    
    
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 60

    # ---------------------------------
    # Create validation patches
    # ---------------------------------

    num_val_patches = 30
    val_patches = []

    with rasterio.open("data/raw_sar/scene3.tif") as src:

        h, w = src.height, src.width

        for _ in range(num_val_patches):

            y = np.random.randint(0, h - 256)
            x = np.random.randint(0, w - 256)

            window = Window(x, y, 256, 256)

            #patch = src.read(1, window=window).astype("float32")
            patch = np.abs(src.read(1, window=window)).astype("float32")
            
            patch = np.log1p(patch)
            patch = patch / 10.0

            if patch.std() < 1e-3:
                continue

            val_patches.append(patch)

    # ---------------------------------
    # Metrics storage
    # ---------------------------------

    losses = []
    psnr_history = []

    best_psnr = 0

    # ---------------------------------
    # Training loop
    # ---------------------------------

    print("Entering training loop...")
    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for i, (bicubic, hr) in enumerate(loader):

            bicubic = bicubic.to(device)
            hr = hr.to(device)

            residual = hr - bicubic

            pred_residual = model(bicubic)

            loss = criterion(pred_residual, residual)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 200 == 0:
                print(f"Epoch {epoch+1} Batch {i}/{len(loader)} Loss {loss.item():.5f}")

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1} Loss {avg_loss:.5f}")

        # ---------------------------------
        # Validation
        # ---------------------------------

        model.eval()

        psnr_list = []

        with torch.no_grad():

            for patch in val_patches:

                lr = cv2.resize(patch, (64,64), interpolation=cv2.INTER_AREA)

                bicubic = cv2.resize(
                    lr,
                    (256,256),
                    interpolation=cv2.INTER_CUBIC
                )

                inp = torch.from_numpy(bicubic).unsqueeze(0).unsqueeze(0).float().to(device)

                residual = model(inp)
                sr = inp + residual

                sr = sr.squeeze().cpu().numpy()
                sr = np.clip(sr, 0, 1)

                # Debug (optional – remove later)
                print("SR range:", sr.min(), sr.max())
                print("HR range:", patch.min(), patch.max())
                

                psnr = peak_signal_noise_ratio(
                    patch,
                    sr,
                    #data_range=patch.max() - patch.min()
                    #data_range = max(patch.max() - patch.min(), 1e-6)
                    data_range=1.0
                )

                psnr_list.append(psnr)

        avg_psnr = np.mean(psnr_list)

        psnr_history.append(avg_psnr)

        print("Validation PSNR:", avg_psnr)

        if avg_psnr > best_psnr:

            best_psnr = avg_psnr

            torch.save(
                model.state_dict(),
                "models/best_edsr_sar.pth"
            )

    torch.save(model.state_dict(), "models/edsr_sar_final.pth")

    # ---------------------------------
    # Plot curves
    # ---------------------------------

    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(psnr_history, marker='o')
    plt.title("Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig("training_curves.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
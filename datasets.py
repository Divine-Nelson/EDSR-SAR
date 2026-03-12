import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import cv2
import random
from rasterio.windows import Window


class SARSuperResolutionDataset(Dataset):

    def __init__(self, data_dir, patch_size=256, scale=4, samples_per_epoch=10000):

        random.seed(42)
        np.random.seed(42)

        self.patch_size = patch_size
        self.scale = scale
        self.samples_per_epoch = samples_per_epoch

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".tif")
        ])

        print("Scenes loaded:", len(self.files))

        # read image sizes
        self.scene_sizes = []
        for f in self.files:
            with rasterio.open(f) as src:
                self.scene_sizes.append((src.height, src.width))

        self.datasets = {}

        # prepare patch coordinates
        self.patch_index = []

        patches_per_scene = samples_per_epoch // len(self.files)

        for scene_id, (h, w) in enumerate(self.scene_sizes):

            for _ in range(patches_per_scene):

                y = random.randint(0, h - patch_size - 1)
                x = random.randint(0, w - patch_size - 1)

                self.patch_index.append((scene_id, x, y))

        print("Total patches prepared:", len(self.patch_index))


    def __len__(self):
        return len(self.patch_index)


    def _get_dataset(self, scene_id):

        if scene_id not in self.datasets:
            self.datasets[scene_id] = rasterio.open(self.files[scene_id])

        return self.datasets[scene_id]


    def __getitem__(self, idx):

        scene_id, x, y = self.patch_index[idx]

        src = self._get_dataset(scene_id)

        window = Window(x, y, self.patch_size, self.patch_size)

        #hr = src.read(1, window=window).astype("float32")
        hr = np.abs(src.read(1, window=window)).astype("float32")

        # SAR log transform
        hr = np.log1p(np.abs(hr))

        # global scaling (no per-patch normalization)
        hr = hr / 10.0

        # generate LR
        lr = cv2.resize(
            hr,
            (self.patch_size // self.scale, self.patch_size // self.scale),
            interpolation=cv2.INTER_AREA
        )

        # bicubic upsample (network input)
        bicubic = cv2.resize(
            lr,
            (self.patch_size, self.patch_size),
            interpolation=cv2.INTER_CUBIC
        )

        hr = torch.from_numpy(hr).unsqueeze(0)
        bicubic = torch.from_numpy(bicubic).unsqueeze(0)

        return bicubic.float(), hr.float()
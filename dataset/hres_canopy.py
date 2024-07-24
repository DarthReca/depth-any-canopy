from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kornia.augmentation import Normalize
from PIL import Image
from torch import nn
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.transforms import AugmentationSequential
from torchvision.transforms import functional as TF


class HResCanopyDatamodule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        mean: list[float, float, float] = (0, 0, 0),
        std: list[float, float, float] = (1, 1, 1),
        **kwargs,
    ):
        mean = list(mean)
        std = list(std)
        super().__init__(
            NeonDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.test_aug = AugmentationSequential(
            Normalize(mean, std), data_keys=["image"]
        )


class NeonDataset(NonGeoDataset):
    def __init__(
        self,
        new_norm=True,
        src_img="neon",
        trained_rgb=False,
        no_norm=True,
        root_dir="data/hres_canopy",
        max_depth: None | float = None,
        as_uint8=False,
        **kwargs,
    ):
        self.no_norm = no_norm
        self.model_norm = nn.Identity()
        self.new_norm = new_norm
        self.trained_rgb = trained_rgb
        self.size = 256
        self.src_img = src_img

        self.path = root_dir + "/images/"
        self.root_dir = Path(self.path)
        self.df_path = root_dir + "/neon_test_data.csv"
        self.df = pd.read_csv(self.df_path, index_col=0)

        # number of times crops can be used horizontally
        self.size_multiplier = 6
        self.max_depth = max_depth
        self.as_uint8 = as_uint8

    def __len__(self):
        if self.src_img == "neon":
            return 30 * len(self.df)
        return len(self.df)

    def __getitem__(self, i):
        n = self.size_multiplier
        ix, jx, jy = i // (n**2), (i % (n**2)) // n, (i % (n**2)) % n
        if self.src_img == "neon":
            l = self.df.iloc[ix]
        x = list(range(l.bord_x, l.imsize - l.bord_x - self.size, self.size))[jx]
        y = list(range(l.bord_y, l.imsize - l.bord_y - self.size, self.size))[jy]
        img = TF.to_tensor(
            Image.open(self.root_dir / l[self.src_img]).crop(
                (x, y, x + self.size, y + self.size)
            )
        )
        chm = TF.to_tensor(
            Image.open(self.root_dir / l.chm).crop((x, y, x + self.size, y + self.size))
        )
        chm[chm < 0] = 0

        if not self.trained_rgb:
            if self.src_img == "neon":
                if self.no_norm:
                    normIn = img
                else:
                    if self.new_norm:
                        # image image normalization using learned quantiles of pairs of Maxar/Neon images
                        x = torch.unsqueeze(img, dim=0)
                        norm_img = self.model_norm(x).detach()
                        p5I = [
                            norm_img[0][0].item(),
                            norm_img[0][1].item(),
                            norm_img[0][2].item(),
                        ]
                        p95I = [
                            norm_img[0][3].item(),
                            norm_img[0][4].item(),
                            norm_img[0][5].item(),
                        ]
                    else:
                        # apply image normalization to aerial images, matching color intensity of maxar images
                        I = TF.to_tensor(
                            Image.open(self.root_dir / l["maxar"]).crop(
                                (x, y, x + s, y + s)
                            )
                        )
                        p5I = [np.percentile(I[i, :, :].flatten(), 5) for i in range(3)]
                        p95I = [
                            np.percentile(I[i, :, :].flatten(), 95) for i in range(3)
                        ]
                    p5In = [np.percentile(img[i, :, :].flatten(), 5) for i in range(3)]

                    p95In = [
                        np.percentile(img[i, :, :].flatten(), 95) for i in range(3)
                    ]
                    normIn = img.clone()
                    for i in range(3):
                        normIn[i, :, :] = (img[i, :, :] - p5In[i]) * (
                            (p95I[i] - p5I[i]) / (p95In[i] - p5In[i])
                        ) + p5I[i]

        if self.max_depth:
            chm /= self.max_depth
        if self.as_uint8:
            img *= 255

        return {"image": img, "mask": chm}

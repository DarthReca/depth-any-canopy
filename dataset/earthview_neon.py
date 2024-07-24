import pathlib
import re
from typing import Callable

import kornia.augmentation as K
import matplotlib.pyplot as plt
import polars as pl
import torch
from datasets import Dataset, config
from kornia.geometry import resize
from torch import Tensor
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.transforms import AugmentationSequential


class EarthViewNEON(NonGeoDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ):
        number = re.compile(r"train-0{2}(\d{3})-of")

        self.samples = pl.read_parquet(root + "/earthview_splits.parquet")
        self.datasets = {
            number.search(f.stem).group(1): Dataset.from_file(str(f))
            .with_format("torch")
            .select_columns(["rgb", "chm"])
            for f in pathlib.Path(root).rglob("*.arrow")
        }

        # Keep good samples only
        self.samples = self.samples.filter(pl.col("score") > 2.5)
        # Keep only the split we want
        self.samples = self.samples.filter(pl.col("split") == split)
        # Filter the samples with some canopy height model data
        chm = pl.read_parquet(root + "/chm_canopy_sum.parquet").filter(
            pl.col("canopy_sum") > 0
        )
        self.samples = self.samples.join(chm, on="key", how="semi")
        self.transforms = transforms

    def __getitem__(self, index):
        key, index, revisit = tuple(self.samples.item(index, "key").split("_"))
        index, revisit = int(index), int(revisit)
        sample = self.datasets[key][index]

        image = sample["rgb"][revisit].float()

        mask = sample["chm"][revisit].float()
        mask = resize(mask, image.shape[-2:], interpolation="nearest")

        sample = {"image": image, "mask": mask / 255.0 + 1e-6}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return self.samples.height

    def _download(self):
        raise NotImplementedError("Download not supported for this dataset")

    def plot(self, image, mask, prediction=None, show_titles=True):
        if prediction is not None:
            prediction = prediction.clip(0, 1).float()
        # Convert image to [0, 1] range
        image = image.float()
        image = image - image.min()
        image = image / image.max()

        mask = mask.float()

        showing_prediction = prediction is not None
        ncols = 2 + int(showing_prediction)
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[0].axis("off")
        axs[1].imshow(
            mask.squeeze(), interpolation="none", cmap="Spectral_r", vmin=0, vmax=1
        )
        axs[1].axis("off")
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Mask")

        if showing_prediction:
            axs[2].imshow(
                prediction.squeeze(),
                interpolation="none",
                cmap="Spectral_r",
                vmin=0,
                vmax=1,
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")
        return fig


class EarthViewNEONDatamodule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        size: int = 518,
        mean: tuple[float, float, float] = (0.420, 0.411, 0.296),
        std: tuple[float, float, float] = (0.213, 0.156, 0.143),
        **kwargs,
    ):
        mean = list(mean)
        std = list(std)
        super().__init__(
            EarthViewNEON, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.train_aug = AugmentationSequential(
            K.Normalize(mean, std), data_keys=["image"]
        )

        self.val_aug = AugmentationSequential(
            K.Normalize(mean, std), data_keys=["image"]
        )

        self.test_aug = AugmentationSequential(
            K.Normalize(mean, std), data_keys=["image"]
        )

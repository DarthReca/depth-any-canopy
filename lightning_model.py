import argparse
import logging
import os
import pprint
import random
import warnings
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import transformers
from kornia.geometry.transform import resize
from lightning import LightningModule
from model import DepthAnythingV2
from torch import nn
from torchmetrics import MetricCollection, classification, regression


class DepthAnythingV2Module(LightningModule):
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    size_map = {
        "vits": "depth-anything/Depth-Anything-V2-Small-hf",
        "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
        "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
        "vitg": None,
    }

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        min_depth: float = 1e-4,
        max_depth: float = 20,
        lr: float = 0.000005,
        use_huggingface: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if not use_huggingface:
            pretrained_from = f"base-checkpoints/{encoder}.pth"
            self.model = DepthAnythingV2(**{**self.model_configs[encoder]})
            self.model.load_state_dict(
                {
                    k: v
                    for k, v in torch.load(pretrained_from, map_location="cpu").items()
                    if "pretrained" in k
                },
                strict=False,
            )
        else:
            self.model = transformers.AutoModelForDepthEstimation.from_pretrained(
                self.size_map[encoder], cache_dir="cache"
            ).train()

        self.loss = nn.MSELoss()
        self.metric = MetricCollection(
            [regression.MeanSquaredError(), regression.MeanAbsoluteError()]
        )
        self.classification_metrics = MetricCollection(
            [classification.JaccardIndex(task="binary")]
        )
        self.corr = MetricCollection([regression.PearsonCorrCoef()])
        self.predictions = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr,
            pct_start=0.05,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        loss = self.loss(pred, depth)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        loss = self.loss(pred, depth)
        self.log("val_loss", loss)
        self.metric(pred, depth)
        self.log_dict(self.metric)

        if batch_idx < 10 and self.logger is not None:
            fig = self.trainer.datamodule.val_dataset.plot(
                img[0].cpu().detach(), depth[0].cpu().detach(), pred[0].cpu().detach()
            )
            self.logger.experiment.log_figure(
                figure=fig, figure_name=f"val_{batch_idx}"
            )
            # fig.savefig(f"logs/val_{batch_idx}.png")
            plt.close(fig)

        return loss

    def test_step(self, batch, batch_idx):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        self.metric(pred, depth)
        self.log_dict(self.metric)

        self.classification_metrics(pred > 1e-4, depth > 1e-4)
        self.log_dict(self.classification_metrics)

        # self.corr(pred[depth > 1e-4].flatten(), depth[depth > 1e-4].flatten())
        # self.log_dict(self.corr)

        self.predictions.append(
            {
                "prediction": pred[depth > 1e-4].flatten().detach().cpu(),
                "depth": depth[depth > 1e-4].flatten().detach().cpu(),
            }
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img).predicted_depth

        pred = resize(pred, depth.shape[-2:], interpolation="bilinear").clamp(0, 1)

        return pred

    def _preprocess_batch(self, batch):
        img, depth = batch["image"], batch["mask"]

        img = resize(img, (518, 518), interpolation="bilinear")

        depth = torch.clamp(
            depth, min=self.hparams.min_depth, max=self.hparams.max_depth
        )

        return img, depth

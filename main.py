import time

import comet_ml
import hydra
import lightning as pl
import torch
from dataset import EarthViewNEONDatamodule
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CometLogger
from lightning_model import DepthAnythingV2Module
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(args: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    data_module = EarthViewNEONDatamodule(**args.dataset)
    model = DepthAnythingV2Module(**args.model)

    experiment_id = time.strftime("%Y%m%d-%H%M%S")
    logger = False
    if args.logger:
        logger = CometLogger(
            project_name="depth-any-canopy",
            workspace="",
            experiment_name="",
            save_dir="comet-logs",
            offline=False,
        )
        experiment_id = logger.experiment.id

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{experiment_id}",
        filename="depth-any-canopy-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [checkpoint_callback, early_stopping]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callback,
        log_every_n_steps=50,
        precision="32-true" if args.model.encoder == "vitl" else "32-true",
        limit_val_batches=50,
        val_check_interval=500,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local LightningCLI entry for the unofficial PixelGen SeaCache-style port."""

from lightning.pytorch.cli import LightningCLI

from seacache_style.pixelgen_lightning import SeaCacheLightningModel
from src.lightning_data import DataModule


if __name__ == "__main__":
    LightningCLI(
        SeaCacheLightningModel,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )

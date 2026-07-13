#!/usr/bin/env python3
"""Local LightningCLI entry for the unofficial PixelGen DiCache-style port."""

from lightning.pytorch.cli import LightningCLI

from dicache_style.pixelgen_lightning import DiCachePixelGenLightning
from src.lightning_data import DataModule


if __name__ == "__main__":
    LightningCLI(
        DiCachePixelGenLightning,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )


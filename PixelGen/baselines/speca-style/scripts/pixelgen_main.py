#!/usr/bin/env python3
"""Local LightningCLI entry for the unofficial PixelGen SpeCa-style port."""

from lightning.pytorch.cli import LightningCLI

from speca_style.pixelgen_lightning import SpeCaPixelGenLightning
from src.lightning_data import DataModule


if __name__ == "__main__":
    LightningCLI(
        SpeCaPixelGenLightning,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )

#!/usr/bin/env python3
"""Local LightningCLI entry for the unofficial PixelGen TaylorSeer-style port."""

from lightning.pytorch.cli import LightningCLI

from taylorseer_style.pixelgen_lightning import TaylorSeerPixelGenLightning
from src.lightning_data import DataModule


if __name__ == "__main__":
    LightningCLI(
        TaylorSeerPixelGenLightning,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )

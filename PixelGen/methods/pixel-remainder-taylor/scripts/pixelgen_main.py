"""LightningCLI entry for Pixel-Remainder Taylor."""

from lightning.pytorch.cli import LightningCLI
from src.lightning_data import DataModule

from pixel_remainder_taylor.pixelgen_lightning import PixelRemainderTaylorLightning


if __name__ == "__main__":
    LightningCLI(
        PixelRemainderTaylorLightning,
        DataModule,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )

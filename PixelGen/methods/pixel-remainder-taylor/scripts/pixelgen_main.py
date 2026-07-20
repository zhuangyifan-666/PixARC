"""LightningCLI entry for Pixel-Remainder Taylor."""

from lightning.pytorch.cli import LightningCLI

from pixel_remainder_taylor.pixelgen_lightning import PixelRemainderTaylorLightning


if __name__ == "__main__":
    LightningCLI(
        PixelRemainderTaylorLightning,
        auto_configure_optimizers=False,
        save_config_callback=None,
    )

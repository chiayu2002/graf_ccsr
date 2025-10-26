"""
GRAF Models Module

Available models:
- Generator: NeRF generator
- Discriminator: GAN discriminator
- CCSR: Simple consistency-controlled super-resolution
- CCSR_ESRGAN: Hybrid CCSR and ESRGAN model
- ESRGANWrapper: Standalone ESRGAN wrapper
- RRDB: Residual-in-Residual Dense Block

Usage:
    from graf.models.ccsr import CCSR_ESRGAN
    from graf.models.esrgan_model import RRDB
    from graf.models.generator import Generator
"""

# Empty __init__.py to avoid circular imports
# Import directly from submodules: from graf.models.xxx import YYY

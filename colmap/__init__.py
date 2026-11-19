"""Incremental Structure-from-Motion.

Implements the reconstruction pipeline from Schönberger and Frahm,
"Structure-from-Motion Revisited", CVPR 2016.
"""

from colmap.device import configure_device, get_device
from colmap.mapper import Mapper
from colmap.options import MapperOptions, VerificationOptions

__all__ = ["Mapper", "MapperOptions", "VerificationOptions", "configure_device", "get_device"]
__version__ = "0.1.0"

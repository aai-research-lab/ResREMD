"""Reservoir replica exchange molecular dynamics for OpenMM.

Every setting is declared once in :mod:`resremd.options`.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # an uninstalled checkout
    __version__ = "0.0.0"

from .errors import ResRemdError
from .options import GENERATE, IMPORT, RUN
from .reservoir import Reservoir
from .sampler import run

__all__ = ["__version__", "run", "Reservoir", "ResRemdError",
           "RUN", "GENERATE", "IMPORT"]

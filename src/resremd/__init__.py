"""Reservoir replica exchange molecular dynamics for OpenMM.

    import resremd

    resremd.generate_reservoir("setup/", output="reservoir",
                               temperature_K=500, duration_ns=200)
    resremd.run("setup/", output="remd", reservoir="reservoir",
                temperature_min_K=300, n_replicas=8, duration_ns=100)
    print(resremd.format_summary(resremd.summarize("remd")))

Every setting is declared once in :mod:`resremd.options`.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # an uninstalled checkout
    __version__ = "0.0.0"

from .analysis import format_summary, summarize
from .build import generate as generate_reservoir
from .build import import_trajectories as import_reservoir
from .errors import ResRemdError
from .options import GENERATE, IMPORT, RUN
from .reservoir import Reservoir
from .sampler import run

__all__ = [
    "__version__", "run", "generate_reservoir", "import_reservoir",
    "Reservoir", "summarize", "format_summary", "ResRemdError",
    "RUN", "GENERATE", "IMPORT",
]

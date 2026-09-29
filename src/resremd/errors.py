"""Errors, each with a stable code.

The message is for a person. The code is for a program: a caller such as
FastMDXplora can map ``resremd.reservoir.mismatch`` onto its own error types
without matching on wording that is free to improve.
"""

from __future__ import annotations


class ResRemdError(Exception):
    """Base class. ``code`` names the failure; the message explains it."""

    code = "resremd.error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class InputError(ResRemdError):
    """A setting, or a combination of settings, that cannot be run."""

    code = "resremd.input.invalid"


class UnsupportedSystem(ResRemdError):
    """A System this method cannot simulate correctly."""

    code = "resremd.system.unsupported"


class ReservoirError(ResRemdError):
    """A reservoir that is malformed or does not belong to this system."""

    code = "resremd.reservoir.invalid"


class ResumeError(ResRemdError):
    """A run directory that cannot be continued as asked."""

    code = "resremd.resume.invalid"


class BackendUnavailable(ResRemdError):
    """An optional package needed for this step is not installed."""

    code = "resremd.environment.backend.missing"


def require(module: str, purpose: str, extra: str):
    """Import an optional module or say how to install it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise BackendUnavailable(
            f"{purpose} needs {module}. Install it with "
            f"`conda install -c conda-forge {module}` or "
            f"`pip install resremd[{extra}]`."
        ) from exc

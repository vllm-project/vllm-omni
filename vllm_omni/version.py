"""
Version information for vLLM-Omni.

The version is automatically generated from git tags via setuptools_scm
and written to _version.py during package build.
"""

import warnings

try:
    # Import auto-generated version from _version.py (created by setuptools_scm)
    from ._version import __version__, __version_tuple__
except ImportError as e:
    warnings.warn(
        f"Failed to import version from _version.py: {e}\n"
        "This typically happens in development mode before building.\n"
        "Using fallback version 'dev'.",
        RuntimeWarning,
        stacklevel=2,
    )

    __version__ = "dev"
    __version_tuple__ = (0, 0, "dev")


def warn_if_misaligned_vllm_version():
    """Warn if vLLM and vllm-omni versions don't match (major.minor)."""
    # Import vllm lazily since import order may be sensitive with current monkeypatching,
    # but we want to check this before potentially breaking imports run.
    from vllm import __version__ as vllm_version
    from vllm import __version_tuple__ as vllm_version_tuple

    omni_ver: tuple[str | int, ...] = __version_tuple__[:2]
    vllm_ver: tuple[str | int, ...] = vllm_version_tuple[:2]
    # Skip if either version is dev (0, 0)
    if omni_ver == (0, 0) or vllm_ver == (0, 0):
        return

    # Compare major.minor
    if omni_ver != vllm_ver:
        warnings.warn(
            "vLLM and vLLM-Omni appear to have mismatched major/minor versions:\n"
            f" --> vLLM-Omni version {__version__}\n"
            f" --> vLLM version {vllm_version}\n"
            "This will likely cause compatibility issues.",
            RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["__version__", "__version_tuple__"]

# Run version check automatically when this module is imported
try:
    warn_if_misaligned_vllm_version()
except ModuleNotFoundError:
    # vLLM not installed (e.g., documentation builds)
    pass

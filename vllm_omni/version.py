"""
Version information for vLLM-Omni.

The version is automatically generated from git tags via setuptools_scm
and written to _version.py during package build.
"""

try:
    # Import auto-generated version from _version.py (created by setuptools_scm)
    from ._version import __version__, __version_tuple__
except ImportError as e:
    import warnings

    warnings.warn(
        f"Failed to import version from _version.py: {e}\n"
        "This typically happens in development mode before building.\n"
        "Using fallback version 'dev'.",
        RuntimeWarning,
        stacklevel=2,
    )

    __version__ = "dev"
    __version_tuple__ = (0, 0, "dev")

__all__ = ["__version__", "__version_tuple__"]

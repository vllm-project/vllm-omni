"""
Setup script for vLLM-Omni with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`pip install vllm-omni` and automatically receive the correct platform-specific
dependencies (CUDA/ROCm/CPU/XPU/NPU) without requiring extras like `[cuda]`.
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools_scm import get_version


def uninstall_onnxruntime() -> None:
    """
    Uninstall onnxruntime package if it exists.

    This is necessary for ROCm environments where onnxruntime may conflict
    with ROCm-specific dependencies.
    """
    try:
        import pkg_resources

        try:
            pkg_resources.get_distribution("onnxruntime")
            print("Found onnxruntime installed, uninstalling for ROCm compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Successfully uninstalled onnxruntime")
        except pkg_resources.DistributionNotFound:
            print("onnxruntime not installed, skipping uninstall")
    except Exception as e:
        print(f"Warning: Failed to uninstall onnxruntime: {e}")


def detect_target_device() -> str:
    """
    Detect the target device for installation following RFC priority rules.

    Priority order:
    1. VLLM_OMNI_TARGET_DEVICE environment variable (highest priority)
    2. Torch backend detection (cuda, rocm, npu, xpu)
    3. CPU fallback (default)

    Returns:
        str: Device name ('cuda', 'rocm', 'npu', 'xpu', or 'cpu')
    """
    # Priority 1: Explicit override via environment variable
    target_device = os.environ.get("VLLM_OMNI_TARGET_DEVICE")
    if target_device:
        valid_devices = ["cuda", "rocm", "npu", "xpu", "cpu"]
        if target_device.lower() in valid_devices:
            print(f"Using target device from VLLM_OMNI_TARGET_DEVICE: {target_device.lower()}")
            return target_device.lower()
        else:
            print(f"Warning: Invalid VLLM_OMNI_TARGET_DEVICE '{target_device}', falling back to auto-detection")

    # Priority 2: Torch backend detection
    # This is a code path for when user is using
    # --no-build-isolation flag
    try:
        import torch

        # Check for CUDA
        if torch.version.cuda is not None:
            print("Detected CUDA backend from torch")
            return "cuda"

        # Check for ROCm (AMD)
        if torch.version.hip is not None:
            print("Detected ROCm backend from torch")
            uninstall_onnxruntime()
            return "rocm"

        # Check for NPU (Ascend)
        if hasattr(torch, "npu"):
            try:
                if torch.npu.is_available():
                    print("Detected NPU backend from torch")
                    return "npu"
            except Exception:
                pass

        # Check for XPU (Intel)
        if hasattr(torch, "xpu"):
            try:
                if torch.xpu.is_available():
                    print("Detected XPU backend from torch")
                    return "xpu"
            except Exception:
                pass

        print("No GPU backend detected in torch, defaulting to CPU")
        return "cpu"

    except ImportError:
        print("PyTorch not found, defaulting to CUDA installation")
        return "cuda"


def get_vllm_omni_version() -> str:
    """
    Get the vLLM-Omni version with device-specific suffix.

    Version format: {base_version}+{device}
    Examples:
        - 0.14.0+cuda (release version with CUDA)
        - 0.14.1.dev23+g1a2b3c4.rocm (dev version with ROCm)
        - 0.15.0+npu (release version with NPU)

    Environment variables:
        VLLM_OMNI_VERSION_OVERRIDE: Override version completely
        VLLM_OMNI_TARGET_DEVICE: Override device detection

    Returns:
        Version string with device suffix
    """
    # Allow complete version override via environment variable
    if env_version := os.getenv("VLLM_OMNI_VERSION_OVERRIDE"):
        print(f"Overriding vLLM-Omni version with {env_version} from VLLM_OMNI_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        # Get version without device suffix for override case
        version = get_version(write_to="vllm_omni/_version.py")
    else:
        # Generate version from git tags via setuptools_scm (without writing yet)
        try:
            version = get_version()
        except Exception as e:
            print(f"Warning: Failed to get version from git, using fallback: {e}")
            version = "dev"

        # Determine separator: '+' for normal versions, '.' for dev versions with '+'
        sep = "+" if "+" not in version else "."

        # Append device-specific suffix
        device = detect_target_device()

        if device == "cuda":
            # if it is cuda, following vLLM
            # we don't need to add any suffix
            pass
        elif device == "rocm":
            version += f"{sep}rocm"
        elif device == "npu":
            version += f"{sep}npu"
        elif device == "xpu":
            version += f"{sep}xpu"
        elif device == "cpu":
            version += f"{sep}cpu"
        else:
            raise RuntimeError(f"Unknown target device: {device}")

    # Tell setuptools_scm to use this version
    # This will be picked up by write_to and written to _version.py
    os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
    get_version(write_to="vllm_omni/_version.py")  # Write the correct version
    print(f"Generated version: {version}")
    return version


def load_requirements(file_path: Path) -> list[str]:
    """
    Load requirements from a file, supporting -r directive for recursive loading.

    Args:
        file_path: Path to the requirements file

    Returns:
        List of requirement strings
    """
    requirements = []

    if not file_path.exists():
        print(f"Warning: Requirements file not found: {file_path}")
        return requirements

    with open(file_path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Handle -r directive for recursive loading
            if line.startswith("-r "):
                nested_file = line[3:].strip()
                nested_path = file_path.parent / nested_file
                requirements.extend(load_requirements(nested_path))
            else:
                requirements.append(line)

    return requirements


def get_install_requires() -> list[str]:
    """
    Get the list of dependencies based on detected platform.

    Returns:
        List of requirement strings for the detected platform
    """
    device = detect_target_device()
    requirements_dir = Path(__file__).parent / "requirements"
    requirements_file = requirements_dir / f"{device}.txt"

    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)

    if not requirements:
        print(f"Warning: No requirements loaded for device '{device}'")
    else:
        print(f"Loaded {len(requirements)} requirements for {device}")

    return requirements


if __name__ == "__main__":
    # Get platform-specific dependencies
    install_requires = get_install_requires()

    # Setup configuration
    setup(
        version=get_vllm_omni_version(),
        install_requires=install_requires,
    )

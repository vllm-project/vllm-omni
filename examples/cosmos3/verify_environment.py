"""Verify the shared Cosmos3 environment and record actual import locations."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "plugins/cosmos3_client")]
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    os.environ["VLLM_TARGET_DEVICE"] = args.device
    # Keep the same LLVM load order as the simulation client.
    import llvmlite.binding

    if args.device == "npu":
        import torch_npu  # noqa: F401

    import mujoco
    import numpy
    import scipy
    import torch
    import torchvision
    import vllm
    import vllm_omni_cosmos3_client
    from vllm_omni_cosmos3_client.closed_loop_eval import _import_libero

    import vllm_omni

    assert Path(vllm_omni.__file__).resolve().is_relative_to(root / "vllm_omni")
    assert Path(vllm_omni_cosmos3_client.__file__).resolve().is_relative_to(root / "plugins/cosmos3_client")
    _import_libero()
    import numba
    from robosuite.utils.transform_utils import mat2quat

    numpy.testing.assert_array_equal(mat2quat(numpy.eye(3)), numpy.array([0.0, 0.0, 0.0, 1.0], dtype=numpy.float32))
    context = mujoco.GLContext(32, 32)
    context.make_current()
    from OpenGL import GL

    rendering = {
        "renderer": GL.glGetString(GL.GL_RENDERER).decode(),
        "version": GL.glGetString(GL.GL_VERSION).decode(),
        "runtime_root": os.environ.get("COSMOS3_MESA_ROOT"),
    }
    context.free()
    del context
    device = f"{args.device}:0"
    assert torch.ones(4, device=device).sum().item() == 4
    locations = {
        "vllm_omni": vllm_omni.__file__,
        "client": vllm_omni_cosmos3_client.__file__,
        "vllm": vllm.__file__,
    }
    if args.device == "npu":
        import vllm_ascend

        locations["vllm_ascend"] = vllm_ascend.__file__
        assert Path(vllm_ascend.__file__).resolve().is_relative_to(root)
    report = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "device": device,
        "cann_root": os.environ.get("ASCEND_HOME_PATH"),
        "host_native": not Path("/.dockerenv").exists(),
        "shared_server_client_environment": True,
        "control_jit": {
            "disabled": numba.config.DISABLE_JIT,
            "optimization_level": int(numba.config.OPT),
            "compiled_signatures": str(getattr(mat2quat, "signatures", [])),
            "llvm_version": llvmlite.binding.llvm_version_info,
        },
        "rendering": rendering,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "vllm": vllm.__version__,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            **({"torch-npu": importlib.metadata.version("torch-npu")} if args.device == "npu" else {}),
        },
        "locations": locations,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

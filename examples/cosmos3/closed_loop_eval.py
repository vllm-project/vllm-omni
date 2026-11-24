"""Compatibility entry point for the optional Cosmos3 client plugin."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins/cosmos3_client"))
from vllm_omni_cosmos3_client import closed_loop_eval as _implementation  # noqa: E402

if __name__ == "__main__" and hasattr(_implementation, "main"):
    _implementation.main()
else:
    sys.modules[__name__] = _implementation

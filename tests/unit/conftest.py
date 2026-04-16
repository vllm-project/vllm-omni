"""conftest.py for unit tests — stubs out heavy vllm_omni init."""

import sys
import types

# Provide a lightweight stub for vllm_omni so that submodule imports
# (e.g. vllm_omni.model_executor.stage_input_processors) don't trigger the
# full package __init__.py which requires a complete vLLM installation.
_stub = types.ModuleType("vllm_omni")
_stub.__path__ = []
_stub.__spec__ = None
sys.modules.setdefault("vllm_omni", _stub)

# Stub out vllm_omni.inputs.data.OmniTokensPrompt
_inputs = types.ModuleType("vllm_omni.inputs")
_inputs_data = types.ModuleType("vllm_omni.inputs.data")
_inputs_data.OmniTokensPrompt = dict  # type: ignore[attr-defined]
sys.modules.setdefault("vllm_omni.inputs", _inputs)
sys.modules.setdefault("vllm_omni.inputs.data", _inputs_data)

"""
Reproduction test for GitHub Issue #1569:
  llm2code2wav_async_chunk() got an unexpected keyword argument 'is_finished'

The call site in chunk_transfer_adapter.py:211 passes is_finished=is_finished,
but the function signature in mimo_audio.py:53 does NOT accept this parameter.
This causes a TypeError crash when using MiMo-Audio in async chunk mode.

We import the module directly via importlib to bypass the vllm_omni __init__.py
which has heavy dependencies (aenum, vllm, etc.) not available in test env.
"""

import importlib.util
import inspect
import os
import sys
import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import torch

# Worktree root
WORKTREE = os.path.dirname(os.path.abspath(__file__))

# Will be set in setUpModule / restored in tearDownModule
_saved_sys_modules = None
llm2code2wav_async_chunk = None


def _load_module_directly(module_name: str, file_path: str):
    """Load a Python module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)

    # Provide stubs for dependencies that would fail to import
    if "vllm.inputs" not in sys.modules:
        stub = types.ModuleType("vllm.inputs")
        stub.TextPrompt = type("TextPrompt", (), {})
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.inputs"] = stub
    if "vllm.logger" not in sys.modules:
        vllm_logger = types.ModuleType("vllm.logger")
        import logging
        vllm_logger.init_logger = lambda name: logging.getLogger(name)
        sys.modules["vllm.logger"] = vllm_logger
    if "vllm_omni.inputs.data" not in sys.modules:
        stub_data = types.ModuleType("vllm_omni.inputs.data")
        stub_data.OmniTokensPrompt = type("OmniTokensPrompt", (), {})
        sys.modules.setdefault("vllm_omni", types.ModuleType("vllm_omni"))
        sys.modules.setdefault("vllm_omni.inputs", types.ModuleType("vllm_omni.inputs"))
        sys.modules["vllm_omni.inputs.data"] = stub_data
    if "vllm_omni.model_executor.models.mimo_audio.config_mimo_audio" not in sys.modules:
        stub_config = types.ModuleType("vllm_omni.model_executor.models.mimo_audio.config_mimo_audio")
        stub_config.TALKER_CODEC_PAD_TOKEN_ID = 0  # Default pad token ID
        sys.modules.setdefault("vllm_omni.model_executor", types.ModuleType("vllm_omni.model_executor"))
        sys.modules.setdefault("vllm_omni.model_executor.models", types.ModuleType("vllm_omni.model_executor.models"))
        sys.modules.setdefault(
            "vllm_omni.model_executor.models.mimo_audio",
            types.ModuleType("vllm_omni.model_executor.models.mimo_audio"),
        )
        sys.modules["vllm_omni.model_executor.models.mimo_audio.config_mimo_audio"] = stub_config

    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def setUpModule():
    """Snapshot sys.modules, then load the target module with stubs."""
    global _saved_sys_modules, llm2code2wav_async_chunk
    _saved_sys_modules = sys.modules.copy()

    mimo_audio_path = os.path.join(
        WORKTREE, "vllm_omni", "model_executor", "stage_input_processors", "mimo_audio.py"
    )
    mimo_audio = _load_module_directly("mimo_audio_test_target", mimo_audio_path)
    llm2code2wav_async_chunk = mimo_audio.llm2code2wav_async_chunk


def tearDownModule():
    """Restore sys.modules to its original state so other tests are unaffected."""
    sys.modules.clear()
    sys.modules.update(_saved_sys_modules)


class TestMiMoAudioIsFinishedParam(unittest.TestCase):
    """Test that llm2code2wav_async_chunk accepts is_finished keyword argument."""

    def _make_transfer_manager(self):
        """Create a mock transfer_manager with the attributes used by the function."""
        tm = MagicMock()
        tm.code_prompt_token_ids = defaultdict(list)
        return tm

    def _make_request(self, is_finished=False, external_req_id="test-req-001"):
        """Create a mock request object."""
        req = MagicMock()
        req.is_finished.return_value = is_finished
        req.external_req_id = external_req_id
        return req

    def _make_pooling_output_with_codes(self):
        """Create a pooling_output dict with valid code_predictor_codes (shape: [1,1,8,4])."""
        codes = torch.ones(1, 1, 8, 4, dtype=torch.long)
        return {"code_predictor_codes": codes}

    def test_calling_with_is_finished_kwarg_raises_typeerror(self):
        """
        REPRODUCTION: This test demonstrates the bug from Issue #1569.
        The call site passes is_finished=True, but the function
        does not accept this parameter, causing TypeError.
        """
        tm = self._make_transfer_manager()
        request = self._make_request(is_finished=True)
        pooling_output = self._make_pooling_output_with_codes()

        # This is exactly how _send_single_request calls the function
        # (chunk_transfer_adapter.py:207-211)
        try:
            result = llm2code2wav_async_chunk(
                transfer_manager=tm,
                pooling_output=pooling_output,
                request=request,
                is_finished=True,
            )
            # If we get here, the fix is applied
            print("PASS: llm2code2wav_async_chunk accepts is_finished parameter (fix applied)")
            self.assertIsNotNone(result)
        except TypeError as e:
            if "unexpected keyword argument 'is_finished'" in str(e):
                self.fail(
                    f"BUG REPRODUCED (Issue #1569): {e}\n"
                    "The function does not accept 'is_finished' keyword argument, "
                    "but chunk_transfer_adapter.py:211 passes it."
                )
            else:
                raise

    def test_is_finished_true_sets_finished_flag_in_payload(self):
        """When is_finished=True, the returned payload should have finished=True."""
        tm = self._make_transfer_manager()
        request = self._make_request(is_finished=False, external_req_id="test-req-002")
        pooling_output = self._make_pooling_output_with_codes()

        result = llm2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
            is_finished=True,
        )

        # With is_finished=True, the function should emit a payload even if chunk is not full
        self.assertIsNotNone(result, "Should return payload when is_finished=True")
        self.assertIn("finished", result)
        self.assertTrue(
            result["finished"].item(),
            "finished flag should be True when is_finished=True"
        )

    def test_is_finished_false_buffers_until_chunk_full(self):
        """When is_finished=False and chunk is not full, should return None (buffering)."""
        tm = self._make_transfer_manager()
        request = self._make_request(is_finished=False, external_req_id="test-req-003")
        pooling_output = self._make_pooling_output_with_codes()

        result = llm2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
            is_finished=False,
        )

        self.assertIsNone(result, "Should return None when chunk is not full and not finished")

    def test_empty_codes_with_is_finished_returns_sentinel(self):
        """When codes are empty but is_finished=True, should return a finished sentinel."""
        tm = self._make_transfer_manager()
        request = self._make_request(is_finished=False)
        pooling_output = {"code_predictor_codes": None}

        result = llm2code2wav_async_chunk(
            transfer_manager=tm,
            pooling_output=pooling_output,
            request=request,
            is_finished=True,
        )

        self.assertIsNotNone(result, "Should return sentinel when is_finished=True and codes are None")
        self.assertTrue(result["finished"].item())

    def test_signature_matches_thinker2talker_pattern(self):
        """
        Verify function signature matches the pattern used by thinker2talker_async_chunk,
        which is the working reference implementation in qwen3_omni.py.
        """
        sig = inspect.signature(llm2code2wav_async_chunk)
        params = list(sig.parameters.keys())

        self.assertIn("is_finished", params,
                       "is_finished must be in function signature (matches thinker2talker_async_chunk pattern)")

        is_finished_param = sig.parameters["is_finished"]
        self.assertEqual(is_finished_param.default, False,
                         "is_finished should default to False")


if __name__ == "__main__":
    unittest.main(verbosity=2)

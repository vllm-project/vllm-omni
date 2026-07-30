# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration coverage for the shared task examples' ``_apply_ar_stage_inputs``.

The unit tests in ``test_model_extras.py`` cover ``get_ar_input_builder`` and
``build_ar_stage_inputs`` in isolation, but nothing previously exercised the
*glue* in ``text_to_image.py`` / ``image_edit.py`` that wires them together:
resolving the model's declared ``ar_input_builder`` via the registry, loading
the AR tokenizer, and writing the result onto ``prompt_dict`` /
``sampling_params_list``. A regression in that glue -- e.g. the registry key
mismatch fixed in this PR (``HunyuanImage3ForCausalMM`` vs
``HunyuanImage3Pipeline``), or the ``trust_remote_code`` opt-in being
silently bypassed -- would not have been caught by the isolated unit tests
above, since they call ``build_ar_stage_inputs`` directly rather than going
through the shared script's own code path.

These tests dynamically load the example scripts by file path (``examples/``
is not an installed package) and call ``_apply_ar_stage_inputs`` the same way
``main()`` does, with a fake tokenizer standing in for a real HF download.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relpath: str):
    path = _REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeHunyuanTokenizer:
    """Mirrors test_model_extras.py's fixture: distinct ids, no network I/O."""

    SPECIAL = {"<|startoftext|>": 1, "<img>": 2, "<think>": 3, "<recaption>": 4}

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.SPECIAL.get(tok, 0)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [500 + len(text)]


@dataclass
class _FakeARSamplingParams:
    """Stands in for an LLM (non-diffusion) stage's sampling params."""

    stop_token_ids: list[int] | None = None


def _patch_tokenizer_loader(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[dict[str, Any]]:
    """Patch the ``AutoTokenizer`` the module imports locally, recording calls."""
    calls: list[dict[str, Any]] = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str, trust_remote_code: bool = False) -> _FakeHunyuanTokenizer:
            calls.append({"model": model, "trust_remote_code": trust_remote_code})
            return _FakeHunyuanTokenizer()

    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return calls


@pytest.mark.parametrize(
    "module_name,relpath",
    [
        ("_shared_text_to_image", "examples/offline_inference/text_to_image/text_to_image.py"),
        ("_shared_image_edit", "examples/offline_inference/image_to_image/image_edit.py"),
    ],
)
def test_apply_ar_stage_inputs_routes_through_real_registry(
    monkeypatch: pytest.MonkeyPatch, module_name: str, relpath: str
) -> None:
    """The shared script's glue must resolve HunyuanImage3's *real* registry
    entry (keyed on the runtime ``model_class_name``) and write its output
    onto the request -- not a stub, not a hardcoded id."""
    from vllm_omni.model_extras import get_ar_input_builder

    module = _load_module(module_name, relpath)
    tokenizer_calls = _patch_tokenizer_loader(monkeypatch, module)

    ar_input_builder = get_ar_input_builder("HunyuanImage3ForCausalMM")
    assert ar_input_builder is not None, (
        "get_ar_input_builder('HunyuanImage3ForCausalMM') returned None -- "
        "if this regresses, the shared script silently skips AR-stage "
        "input building entirely (this is exactly how the registry-key bug "
        "this PR fixes went undetected)."
    )

    prompt_dict: dict[str, Any] = {}
    ar_params = _FakeARSamplingParams()
    diffusion_params = OmniDiffusionSamplingParams()
    sampling_params_list: list[Any] = [ar_params, diffusion_params]

    module._apply_ar_stage_inputs(
        ar_input_builder,
        model="tencent/HunyuanImage-3.0-Instruct",
        prompt_text="a red panda",
        extra_body={"bot_task": "think", "use_system_prompt": "en_recaption"},
        num_images=0,
        height=1024,
        width=1024,
        prompt_dict=prompt_dict,
        sampling_params_list=sampling_params_list,
        trust_remote_code=False,
    )

    # The tokenizer was loaded exactly once, for the model the user asked for.
    assert len(tokenizer_calls) == 1
    assert tokenizer_calls[0]["model"] == "tencent/HunyuanImage-3.0-Instruct"

    # Byte-for-byte tokenizer path: prompt_token_ids populated, not the string form.
    assert prompt_dict.get("prompt_token_ids"), "AR prefill token-ids were not written onto the request"
    assert "prompt" not in prompt_dict
    assert prompt_dict.get("use_system_prompt") == "en_recaption"
    assert prompt_dict.get("modalities") == ["image"]

    # stop_token_ids reached the AR-stage params, and only those.
    assert ar_params.stop_token_ids, "AR stage did not receive stop_token_ids"
    assert diffusion_params.stop_token_ids is None


@pytest.mark.parametrize(
    "module_name,relpath,cli_flag",
    [
        ("_shared_text_to_image_trc", "examples/offline_inference/text_to_image/text_to_image.py", False),
        ("_shared_text_to_image_trc", "examples/offline_inference/text_to_image/text_to_image.py", True),
    ],
)
def test_apply_ar_stage_inputs_respects_trust_remote_code_optin(
    monkeypatch: pytest.MonkeyPatch, module_name: str, relpath: str, cli_flag: bool
) -> None:
    """The tokenizer load must use the caller's resolved --trust-remote-code
    value, not silently default to True regardless of user intent."""
    from vllm_omni.model_extras import get_ar_input_builder

    module = _load_module(module_name, relpath)
    tokenizer_calls = _patch_tokenizer_loader(monkeypatch, module)
    ar_input_builder = get_ar_input_builder("HunyuanImage3ForCausalMM")

    module._apply_ar_stage_inputs(
        ar_input_builder,
        model="tencent/HunyuanImage-3.0-Instruct",
        prompt_text="a red panda",
        extra_body={},
        num_images=0,
        height=1024,
        width=1024,
        prompt_dict={},
        sampling_params_list=[_FakeARSamplingParams(), OmniDiffusionSamplingParams()],
        trust_remote_code=cli_flag,
    )

    assert tokenizer_calls[0]["trust_remote_code"] is cli_flag


@pytest.mark.parametrize(
    "module_name,relpath",
    [
        ("_shared_text_to_image_validate", "examples/offline_inference/text_to_image/text_to_image.py"),
        ("_shared_image_edit_validate", "examples/offline_inference/image_to_image/image_edit.py"),
    ],
)
def test_apply_ar_stage_inputs_runs_declared_tokenizer_validator(
    monkeypatch: pytest.MonkeyPatch, module_name: str, relpath: str
) -> None:
    """get_ar_tokenizer_validator's hook must run on a real tokenizer load,
    and a validation failure must propagate -- not get swallowed by the
    tokenizer-load try/except (which would silently fall back to the string
    prompt instead of surfacing the drift)."""
    from vllm_omni.model_extras import get_ar_input_builder, get_ar_tokenizer_validator

    module = _load_module(module_name, relpath)
    _patch_tokenizer_loader(monkeypatch, module)
    ar_input_builder = get_ar_input_builder("HunyuanImage3ForCausalMM")
    validator = get_ar_tokenizer_validator("HunyuanImage3ForCausalMM")
    assert validator is not None

    # The fake tokenizer's ids intentionally don't match
    # HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS, so the real validator must reject it.
    with pytest.raises(ValueError, match="no longer match"):
        module._apply_ar_stage_inputs(
            ar_input_builder,
            model="tencent/HunyuanImage-3.0-Instruct",
            prompt_text="a red panda",
            extra_body={},
            num_images=0,
            height=1024,
            width=1024,
            prompt_dict={},
            sampling_params_list=[_FakeARSamplingParams(), OmniDiffusionSamplingParams()],
            trust_remote_code=False,
            validate_tokenizer=validator,
        )

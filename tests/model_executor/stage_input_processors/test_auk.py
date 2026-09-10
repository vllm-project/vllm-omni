# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the AuK encoder -> diffusion stage input processor."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.auk import DUMP_DIR_ENV, encoder2dit

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 2048


def _request_output(
    mm_output: Any,
    completion_mm_output: Any = None,
    request_id: str | None = None,
) -> SimpleNamespace:
    """Build a minimal encoder-stage output mock."""
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[0], multimodal_output=completion_mm_output)],
        multimodal_output=mm_output,
        request_id=request_id,
    )


def _payload(text_cond: Any) -> dict[str, Any]:
    """The slot the encoder actually emits: hidden_states.output."""
    return {"hidden_states": {"output": text_cond}}


def _prompt(
    *,
    audio: bool = False,
    gen_seconds: float | None = None,
    has_audio: bool | None = None,
) -> dict[str, Any]:
    knobs: dict[str, Any] = {
        "gen_seconds": gen_seconds,
        "sway": -1.0,
        "t_grid": None,
        "vae_sample": False,
    }
    if has_audio is not None:
        knobs["has_audio"] = has_audio
    prompt: dict[str, Any] = {
        "prompt": "<|im_start|>system\n",
        "additional_information": {"auk": knobs},
    }
    if audio:
        prompt["multi_modal_data"] = {"audio": (np.zeros(24000, dtype=np.float32), 24000)}
    return prompt


class TestTextCondExtraction:
    def test_tensor_output(self):
        cond = torch.zeros(7, HIDDEN, dtype=torch.bfloat16)
        out = encoder2dit([_request_output(_payload(cond))], _prompt())
        assert out is not None
        assert out["prompt"] == ""
        assert out["prompt_embeds"].shape == (7, HIDDEN)
        assert out["prompt_embeds"].dtype == torch.bfloat16
        assert not out["prompt_embeds"].is_cuda

    def test_list_of_tensors_is_concatenated(self):
        parts = [torch.zeros(3, HIDDEN), torch.ones(4, HIDDEN)]
        out = encoder2dit([_request_output(_payload(parts))], _prompt())
        assert out["prompt_embeds"].shape == (7, HIDDEN)
        assert torch.equal(out["prompt_embeds"][3:], torch.ones(4, HIDDEN))

    def test_single_element_list(self):
        out = encoder2dit([_request_output(_payload([torch.zeros(5, HIDDEN)]))], _prompt())
        assert out["prompt_embeds"].shape == (5, HIDDEN)

    def test_numpy_output(self):
        cond = np.zeros((4, HIDDEN), dtype=np.float32)
        out = encoder2dit([_request_output(_payload(cond))], _prompt())
        assert isinstance(out["prompt_embeds"], torch.Tensor)
        assert out["prompt_embeds"].shape == (4, HIDDEN)

    def test_completion_output_fallback(self):
        cond = torch.zeros(6, HIDDEN)
        out = encoder2dit(
            [_request_output(None, completion_mm_output=_payload(cond))],
            _prompt(),
        )
        assert out["prompt_embeds"].shape == (6, HIDDEN)

    def test_hidden_states_as_a_struct(self):
        """A deserialized payload arrives as a struct, not a mapping."""
        cond = torch.zeros(9, HIDDEN)
        mm_output = {"hidden_states": SimpleNamespace(output=cond, layers=None)}
        out = encoder2dit([_request_output(mm_output)], _prompt())
        assert out["prompt_embeds"].shape == (9, HIDDEN)

    def test_flattened_payload_key(self):
        cond = torch.zeros(8, HIDDEN)
        out = encoder2dit([_request_output({"hidden_states.output": cond})], _prompt())
        assert out["prompt_embeds"].shape == (8, HIDDEN)

    def test_direct_text_cond_key(self):
        cond = torch.zeros(2, HIDDEN)
        out = encoder2dit([_request_output({"text_cond": cond})], _prompt())
        assert out["prompt_embeds"].shape == (2, HIDDEN)

    def test_no_source_outputs_returns_none(self):
        assert encoder2dit([], _prompt()) is None

    def test_missing_payload_raises(self):
        with pytest.raises(ValueError, match="hidden_states.output"):
            encoder2dit([_request_output({})], _prompt())

    def test_empty_hidden_states_slot_raises(self):
        with pytest.raises(ValueError, match="hidden_states.output"):
            encoder2dit([_request_output({"hidden_states": {}})], _prompt())

    def test_empty_multimodal_output_raises(self):
        with pytest.raises(ValueError, match="hidden_states.output"):
            encoder2dit([_request_output(None)], _prompt())

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty text condition"):
            encoder2dit([_request_output(_payload([]))], _prompt())

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match=r"\[tokens, hidden\]"):
            encoder2dit([_request_output(_payload(torch.zeros(2, 7, HIDDEN)))], _prompt())


class TestKnobPropagation:
    def test_knobs_are_forwarded(self):
        prompt = _prompt(audio=True, gen_seconds=6.0)
        prompt["additional_information"]["auk"]["t_grid"] = [0.0, 0.5, 1.0]
        prompt["additional_information"]["auk"]["vae_sample"] = True
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], prompt)
        knobs = out["additional_information"]["auk"]
        assert knobs["gen_seconds"] == 6.0
        assert knobs["sway"] == -1.0
        assert knobs["t_grid"] == [0.0, 0.5, 1.0]
        assert knobs["vae_sample"] is True

    def test_sampling_knobs_are_not_copied(self):
        """nfe/cfg/seed ride the stage-1 sampling params, not the prompt."""
        prompt = _prompt()
        prompt["additional_information"]["auk"].update({"nfe": 8, "cfg": 1.0, "seed": 3})
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], prompt)
        knobs = out["additional_information"]["auk"]
        assert set(knobs) == {"gen_seconds", "sway", "t_grid", "vae_sample", "has_audio"}

    def test_multi_modal_data_is_forwarded(self):
        prompt = _prompt(audio=True)
        out = encoder2dit(
            [_request_output(_payload(torch.zeros(3, HIDDEN)))],
            prompt,
            requires_multimodal_data=True,
        )
        assert out["multi_modal_data"] is prompt["multi_modal_data"]

    def test_text_only_request_has_no_multi_modal_data(self):
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], _prompt())
        assert out["multi_modal_data"] is None
        assert out["additional_information"]["auk"]["has_audio"] is False

    def test_has_audio_declared_by_the_prompt_wins(self):
        prompt = _prompt(audio=True, has_audio=True)
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], prompt)
        assert out["additional_information"]["auk"]["has_audio"] is True

    def test_has_audio_inferred_from_multi_modal_data(self):
        prompt = _prompt(audio=True)
        prompt["additional_information"]["auk"].pop("has_audio", None)
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], prompt)
        assert out["additional_information"]["auk"]["has_audio"] is True

    def test_prompt_as_list(self):
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], [_prompt(gen_seconds=2.0)])
        assert out["additional_information"]["auk"]["gen_seconds"] == 2.0

    def test_missing_prompt_yields_empty_knobs(self):
        out = encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], None)
        knobs = out["additional_information"]["auk"]
        assert knobs["gen_seconds"] is None
        assert knobs["has_audio"] is False


class TestDebugDump:
    def test_off_by_default(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(DUMP_DIR_ENV, raising=False)
        encoder2dit([_request_output(_payload(torch.zeros(3, HIDDEN)))], _prompt())
        assert list(tmp_path.iterdir()) == []

    def test_dump_written_when_set(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path / "dumps"))
        cond = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        prompt = _prompt(gen_seconds=4.0)
        encoder2dit([_request_output(_payload(cond), request_id="req-7")], prompt)

        saved = torch.load(tmp_path / "dumps" / "req-7.pt", weights_only=False)
        assert torch.equal(saved["text_cond"], cond)
        assert saved["request_id"] == "req-7"
        assert saved["prompt"] == prompt["prompt"]

    def test_missing_request_id_still_dumps(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(DUMP_DIR_ENV, str(tmp_path))
        encoder2dit([_request_output(_payload(torch.zeros(2, HIDDEN)))], _prompt())
        assert (tmp_path / "unknown.pt").is_file()

    def test_unwritable_dir_does_not_fail_the_request(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory")
        monkeypatch.setenv(DUMP_DIR_ENV, str(blocker))
        out = encoder2dit([_request_output(_payload(torch.zeros(2, HIDDEN)))], _prompt())
        assert out["prompt_embeds"].shape == (2, HIDDEN)

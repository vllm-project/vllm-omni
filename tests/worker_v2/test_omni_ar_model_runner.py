"""Unit tests for OmniARModelRunner v2."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.data_entry_keys import unflatten_payload
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.omni_ar_model_runner import (
    OmniARModelRunner,
    _async_copy_mm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------
# _build_pooler_output_from_cpu (was _build_pooler_output)
# ---------------------------------------------------------------


def test_reconstruct_raw_model_output_preserves_omni_multimodal_outputs():
    hidden = torch.randn(3, 4)
    latent = torch.randn(3, 4)
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs={"latent": latent},
        aux=None,
    )

    assert isinstance(raw, OmniOutput)
    assert raw.text_hidden_states is hidden
    assert raw.multimodal_outputs["latent"] is latent


def test_reconstruct_raw_model_output_keeps_aux_tuple_without_multimodal_outputs():
    hidden = torch.randn(3, 4)
    aux = {"layers": torch.randn(3, 2)}
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs=None,
        aux=aux,
    )

    assert raw == (hidden, aux)


def test_reconstruct_raw_model_output_ignores_empty_multimodal_outputs():
    hidden = torch.randn(3, 4)
    raw = OmniARModelRunner._reconstruct_raw_model_output(
        hidden_states=hidden,
        multimodal_outputs={},
        aux=None,
    )

    assert raw is hidden


def test_build_pooler_output_basic():
    """Verify _build_pooler_output_from_cpu slices per-request hidden + mm."""
    hidden = torch.randn(6, 8)
    mm = {"audio": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )

    assert len(pooler) == 2
    assert pooler[0]["hidden"].shape == (3, 8)
    assert pooler[1]["hidden"].shape == (3, 8)
    assert pooler[0]["audio"].shape == (3, 2)


def test_build_pooler_output_empty_mm():
    hidden = torch.randn(4, 8)

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        {},
        query_start_loc_np=np.array([0]),
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_reqs=1,
    )
    assert len(pooler) == 1
    assert "hidden" in pooler[0]
    assert len(pooler[0]) == 1


# ---------------------------------------------------------------
# _async_copy_mm (was _copy_mm_to_cpu)
# ---------------------------------------------------------------


def test_copy_mm_to_cpu_tensor():
    total = 10
    t = torch.randn(10, 4)
    result = _async_copy_mm({"feat": t}, total)
    assert "feat" in result
    assert result["feat"].shape == (10, 4)
    assert result["feat"].device == torch.device("cpu")


def test_copy_mm_to_cpu_dict():
    total = 10
    d = {"inner": torch.randn(10, 2)}
    result = _async_copy_mm({"nested": d}, total)
    assert "nested" in result
    assert "inner" in result["nested"]


def test_copy_mm_to_cpu_list():
    result = _async_copy_mm({"items": [torch.randn(3), "text"]}, 10)
    assert "items" in result
    assert isinstance(result["items"][0], torch.Tensor)
    assert result["items"][1] == "text"


def test_copy_mm_to_cpu_empty():
    assert _async_copy_mm({}, 10) == {}


# ---------------------------------------------------------------
# Slicing via _build_pooler_output_from_cpu (was _slice_mm_payload)
# ---------------------------------------------------------------


def test_slice_mm_payload_tensor():
    hidden = torch.randn(6, 4)
    mm_cpu = {"feat": torch.randn(6, 2)}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[0]["feat"].shape == (3, 2)
    assert pooler[1]["feat"].shape == (3, 2)


def test_slice_mm_payload_list():
    hidden = torch.randn(6, 4)
    mm_cpu = {"items": [torch.randn(2), torch.randn(3)]}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert isinstance(pooler[0]["items"], torch.Tensor)
    assert isinstance(pooler[1]["items"], torch.Tensor)


def test_slice_mm_payload_dict():
    hidden = torch.randn(6, 4)
    mm_cpu = {"nested": {"a": torch.randn(6, 2)}}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3, 3], dtype=np.int32),
        num_reqs=2,
    )
    assert pooler[1]["nested"]["a"].shape == (3, 2)


def test_build_pooler_output_flattens_nested_payload_for_msgspec():
    hidden = torch.randn(4, 4)
    mm_cpu = {"codes": {"audio": torch.randn(4, 16)}}

    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 2]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_reqs=2,
    )

    assert "codes" not in pooler[0]
    assert pooler[0]["codes.audio"].shape == (2, 16)
    assert pooler[1]["codes.audio"].shape == (2, 16)


def test_build_pooler_output_preserves_qwen3_nested_payload():
    hidden = torch.randn(4, 4)
    mm = {
        "hidden_states": {
            "layers": {
                0: torch.randn(4, 4),
                24: torch.randn(4, 4),
            },
        },
        "embed": {
            "tts_bos": [torch.randn(1, 1, 4)],
            "tts_eos": [torch.randn(1, 1, 4)],
            "tts_pad": [torch.randn(1, 1, 4)],
        },
    }

    mm_cpu = _async_copy_mm(mm, total_tokens=4)
    pooler = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden,
        mm_cpu,
        query_start_loc_np=np.array([0, 2]),
        num_scheduled_tokens=np.array([2, 2], dtype=np.int32),
        num_reqs=2,
    )

    payload = unflatten_payload(pooler[0])
    assert payload["hidden_states"]["layers"][0].shape == (2, 4)
    assert payload["hidden_states"]["layers"][24].shape == (2, 4)
    assert payload["embed"]["tts_bos"].shape == (1, 1, 4)


def test_clamp_sampling_prompt_token_ids_to_logits_vocab():
    prompt_token_ids = torch.tensor([[3, 99, 152064]])
    input_batch = SimpleNamespace(
        vocab_size=152064,
        sampling_metadata=SimpleNamespace(
            no_penalties=False,
            prompt_token_ids=prompt_token_ids,
        ),
    )

    OmniARModelRunner._clamp_sampling_prompt_token_ids(
        input_batch,
        logits_vocab_size=10,
    )

    assert prompt_token_ids.tolist() == [[3, 9, 9]]


def test_sample_clamps_prompt_token_ids_from_actual_logits_vocab():
    runner = object.__new__(OmniARModelRunner)

    class Model:
        def compute_logits(self, *_args, **_kwargs):
            return torch.empty(1, 7)

    runner.model = Model()
    prompt_token_ids = torch.tensor([[3, 99, 152064]])
    input_batch = SimpleNamespace(
        vocab_size=152064,
        sampling_metadata=SimpleNamespace(
            no_penalties=False,
            prompt_token_ids=prompt_token_ids,
        ),
    )
    sample_seen_prompt_ids = []

    def sample(text_hidden, input_batch_arg, grammar_output):
        assert grammar_output is None
        runner.model.compute_logits(text_hidden)
        sample_seen_prompt_ids.append(input_batch_arg.sampling_metadata.prompt_token_ids.clone())
        return "sampler", "num_sampled", "num_rejected"

    runner.sample = sample

    result = runner._sample_with_prompt_token_compat(
        torch.empty(1, 4),
        input_batch,
        None,
    )

    assert result == ("sampler", "num_sampled", "num_rejected")
    assert prompt_token_ids.tolist() == [[3, 6, 6]]
    assert sample_seen_prompt_ids[0].tolist() == [[3, 6, 6]]
    assert runner.model.compute_logits.__func__ is Model.compute_logits
    assert "compute_logits" not in runner.model.__dict__

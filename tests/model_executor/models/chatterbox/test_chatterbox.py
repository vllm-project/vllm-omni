# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections.abc import Callable

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces_base import is_text_generation_model

from tests.model_executor.models.chatterbox.helpers import DEPLOY, build_stage, stage_vllm_config
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.model_executor.models.chatterbox.chatterbox import STAGES, ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import ChatterboxS3Gen
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import ChatterboxT3ForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_ORIGINAL_PIPELINE, CHATTERBOX_TURBO_PIPELINE
from vllm_omni.model_executor.models.registry import OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BuiltStage = Callable[[str, int], ChatterboxForConditionalGeneration]


@pytest.mark.parametrize(
    ("variant", "stage_id", "stage", "flags", "hooks", "undefined"),
    [
        (
            "turbo",
            0,
            ChatterboxT3ForConditionalGeneration,
            # Without the first the runner never calls preprocess; without the
            # second it drops make_omni_output's payload; with the third true it
            # ships every hidden state to stage 1, which reads none.
            {"has_preprocess": True, "have_multimodal_outputs": True, "omni_pooler_payload_include_hidden": False},
            ["preprocess", "preprocess_decode_batch", "preprocess_decode_batch_mrv2", "make_omni_output"],
            # Either would change which of the hooks above Model Runner V2 calls.
            ["mrv2_decode_preprocess_is_identity", "make_omni_output_mrv2", "on_requests_finished"],
        ),
        (
            "original",
            0,
            ChatterboxOriginalT3,
            # The last two have the runner sample through the stage's guidance
            # sampler and hand it each request's guidance pair.
            {
                "has_preprocess": True,
                "have_multimodal_outputs": True,
                "omni_pooler_payload_include_hidden": False,
                "prefer_model_sampler": True,
                "model_sampler_wants_extra_args": True,
            },
            ["preprocess", "preprocess_decode_batch", "preprocess_decode_batch_mrv2", "sample", "on_requests_finished"],
            ["mrv2_decode_preprocess_is_identity", "make_omni_output_mrv2"],
        ),
        (
            "turbo",
            1,
            ChatterboxS3Gen,
            # Without requires_request_ids the runner passes no ids, and
            # decode_step returns empty audio for every request with no error.
            {
                "have_multimodal_outputs": True,
                "requires_request_ids": True,
                "enable_update_additional_information": True,
            },
            ["on_requests_finished"],
            ["has_preprocess", "make_omni_output"],
        ),
    ],
    ids=["chatterbox_t3", "chatterbox_original_t3", "chatterbox_s3gen"],
)
def test_the_runners_find_each_stages_hooks_on_the_registered_class(
    built_stage: BuiltStage,
    variant: str,
    stage_id: int,
    stage: type[nn.Module],
    flags: dict[str, bool],
    hooks: list[str],
    undefined: list[str],
) -> None:
    """Both runners look hooks up on the class the registry names, with ``getattr`` and ``hasattr``.

    The wrapper is built from each deploy file's stage, as the engine builds it.
    """
    unified = built_stage(variant, stage_id)

    assert type(unified.model) is stage
    assert {name: getattr(unified, name) for name in flags} == flags
    for name in hooks:
        assert getattr(unified, name) == getattr(unified.model, name), name
    for name in undefined:
        assert not hasattr(unified, name), name


@pytest.mark.usefixtures("process_groups")
def test_the_stage_is_built_under_the_wrappers_prefix(model_dir: str) -> None:
    """vLLM names every attention layer by its module path; the stage sits under ``model``."""
    config = stage_vllm_config(model_dir, DEPLOY["turbo"], 0)

    unified = build_stage(config, "cpu")

    layers = config.compilation_config.static_forward_context
    assert set(layers) == {f"model.tfmr.h.{layer}.attn.attn" for layer in range(unified.model.config.num_hidden_layers)}
    assert all(layers[name] is unified.get_submodule(name) for name in layers)
    assert all(name.startswith("model.") for name, _ in unified.named_parameters())


def test_forward_passes_runner_keywords_through(built_stage: BuiltStage) -> None:
    """Stage 1's audio comes back through the wrapper as the stage returns it.

    Its keywords are the runner's (``seq_token_counts``, the payloads, the
    request ids); the vocoder draws its noise, so both calls are seeded alike.
    """
    unified = built_stage("turbo", 1)
    torch.manual_seed(4)
    voice = {
        "speech_token": torch.randint(0, 6561, (1, 20)),
        "speech_feat": torch.randn(1, 40, 80),
        "embedding": torch.randn(1, 192),
    }
    payload = {
        "meta": {"stream_finished": torch.tensor(True), "left_context_size": 0, "num_processed_tokens": 0},
        "embed": voice,
    }
    tokens = torch.randint(0, 6561, (23,))

    def decode(module: nn.Module, request_id: str) -> dict:
        torch.manual_seed(5)
        output = module(
            tokens,
            torch.arange(23),
            seq_token_counts=[23, 0],
            model_intermediate_buffer=[payload, {}],
            request_ids=[request_id, "done"],
        )
        return output.multimodal_outputs

    through_wrapper, direct = decode(unified, "a"), decode(unified.model, "b")

    assert [audio.shape for audio in through_wrapper["audio"]] == [(2 * (23 + 3) * 480,), (0,)]
    assert all(torch.equal(a, b) for a, b in zip(through_wrapper["audio"], direct["audio"], strict=True))
    assert [int(rate) for rate in through_wrapper["sr"]] == [24000, 24000]
    assert unified.model.streams == {}


def test_unknown_stage_is_refused_by_name(model_dir: str) -> None:
    config = OmniEngineArgs(
        model=model_dir,
        model_arch="ChatterboxForConditionalGeneration",
        model_stage="chatterbox_vocoder",
        worker_type="generation",
        skip_tokenizer_init=True,
    ).create_engine_config()

    with pytest.raises(KeyError, match="chatterbox_vocoder"):
        build_stage(config, "cpu")


def test_one_registered_architecture_wraps_every_stage() -> None:
    """The pipelines name the wrapper for every stage; the stage classes are not architectures."""
    pipelines = (CHATTERBOX_TURBO_PIPELINE, CHATTERBOX_ORIGINAL_PIPELINE)
    assert {stage.model_stage for pipeline in pipelines for stage in pipeline.stages} == set(STAGES)
    assert {pipeline.model_arch for pipeline in pipelines} == {"ChatterboxForConditionalGeneration"}
    archs = [name for name in OmniModelRegistry.get_supported_archs() if name.startswith("Chatterbox")]
    assert archs == ["ChatterboxForConditionalGeneration"]
    assert OmniModelRegistry.models[archs[0]].load_model_cls() is ChatterboxForConditionalGeneration


def test_vllm_gives_the_registered_class_the_generate_runner() -> None:
    """vLLM inspects the class, not an instance, so ``__getattr__`` cannot answer for it.

    ``embed_input_ids``, ``forward`` and ``compute_logits`` on the class make
    it a text-generation model.
    """
    assert is_text_generation_model(ChatterboxForConditionalGeneration)

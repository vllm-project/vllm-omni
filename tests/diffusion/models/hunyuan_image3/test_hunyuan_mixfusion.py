# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hunyuan's existing layout/mask semantics at the MixFusion boundary."""

from dataclasses import dataclass, replace

import pytest
import torch
from transformers import GenerationConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from tests.diffusion.models.hunyuan_image3.test_diffusion_kv_request import _components, _prepare, _request
from tests.diffusion.models.hunyuan_image3.test_hunyuan_image3_step_execution import _pipeline, _state
from vllm_omni.diffusion.data import AttentionScheduleConfig, DiffusionKVCacheMode
from vllm_omni.diffusion.mixfusion.plan import MixFusionSequenceRequest
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionCapability, MixFusionSequenceInfo, fits_mixfusion_budget
from vllm_omni.diffusion.mixfusion.sequence import build_sequence_plan
from vllm_omni.diffusion.mixfusion.setup import prepare_mixfusion_admission
from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as pipeline_module
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_PROMPT_KV,
    HunyuanImage3Pipeline,
    _PreparedModelForward,
    get_hunyuan_image_3_pre_process_func,
)
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _Config:
    model: str = "model"
    model_class_name: str = "HunyuanImage3ForCausalMM"
    revision: str | None = None
    diffusion_kv_mode: DiffusionKVCacheMode = DiffusionKVCacheMode.DENSE_LEGACY
    diffusion_attention_schedule: AttentionScheduleConfig | None = None
    enable_mixfusion: bool = True
    max_num_seqs: int = 2


@dataclass
class _AdmissionEngine:
    od_config: _Config
    mixfusion_capability: MixFusionCapability = MixFusionCapability(MixFusionSequenceRequest())


@dataclass
class _LegacyKV:
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]


def _preprocessor(monkeypatch, *, enabled=True):
    tokenizer, image_processor = _components([12, 14])
    config = _Config(enable_mixfusion=enabled)
    hf_config = PretrainedConfig(
        vae_downsample_factor=(8, 8), patch_size=2, image_base_size=1024, vae={"latent_channels": 4}
    )
    monkeypatch.setattr(pipeline_module, "get_config", lambda *_a, **_kw: hf_config)
    monkeypatch.setattr(pipeline_module, "HunyuanImage3ImageProcessor", lambda _config: image_processor)
    monkeypatch.setattr(pipeline_module, "TokenizerWrapper", lambda *_a, **_kw: tokenizer)
    monkeypatch.setattr(
        pipeline_module.GenerationConfig,
        "from_pretrained",
        lambda *_a, **_kw: GenerationConfig(sequence_template="instruct", drop_think=False),
    )
    return config, get_hunyuan_image_3_pre_process_func(config), tokenizer


@pytest.mark.parametrize("server,flag,expected", [(True, None, True), (False, True, True), (True, False, False)])
def test_host_sequence_summary_and_request_opt_in(monkeypatch, server, flag, expected):
    config, preprocess, tokenizer = _preprocessor(monkeypatch, enabled=server)
    request = _request(guidance_scale=5.0)
    if flag is not None:
        request.sampling_params.extra_args = {"enable_mixfusion": flag}
    preprocess(request)
    engine = _AdmissionEngine(config)
    prepare_mixfusion_admission(engine, request)
    assert (request.mixfusion_info is not None) == expected
    assert not request.diffusion_kv_requests
    if expected:
        assert request.mixfusion_info == MixFusionSequenceInfo((1, 4, 96, 128), 16, (12, 14), (32, 34), 1)
        assert len(tokenizer.calls) == 1
    else:
        assert not tokenizer.calls


def test_sequence_scheduler_ignores_chunk_limits_and_refreshes_members(monkeypatch):
    config, preprocess, _ = _preprocessor(monkeypatch)
    engine = _AdmissionEngine(config)
    scheduler = StepScheduler()
    scheduler.initialize(config)
    requests = []
    for i, (h, w) in enumerate([(768, 1024), (1024, 768), (512, 512)]):
        request = _request(guidance_scale=5.0, request_id=str(i))
        request.sampling_params.height, request.sampling_params.width = h, w
        request.sampling_params.extra_args = {"mixfusion_min_chunk_tokens": 99999, "mixfusion_max_chunks": 1}
        preprocess(request)
        prepare_mixfusion_admission(engine, request)
        assert request.mixfusion_info is not None
        requests.append(request)
    scheduler.add_request(requests[0])
    assert scheduler.schedule().scheduled_request_ids == ["0"]
    scheduler.add_request(requests[1])
    wave = scheduler.schedule()
    assert set(wave.scheduled_request_ids) == {"0", "1"} and wave.use_mixfusion
    scheduler.add_request(requests[2])
    assert set(scheduler.schedule().scheduled_request_ids) == {"0", "1"}
    scheduler.finish_requests("0", DiffusionRequestStatus.FINISHED_ABORTED)
    assert set(scheduler.schedule().scheduled_request_ids) == {"1", "2"}


@pytest.mark.parametrize("kind", ["step", "sigma"])
@pytest.mark.parametrize("disabled", [False, True])
def test_sequence_admission_preserves_effective_attention_schedules(monkeypatch, kind, disabled):
    config, preprocess, _ = _preprocessor(monkeypatch)
    config.diffusion_attention_schedule = AttentionScheduleConfig(
        profiles={"p": {"default": "TORCH_SDPA"}},
        default=[{"start": 0, "end": 2, "profile": "p"}] if kind == "step" else [],
        sigma=[{"low": 0.0, "high": 1.0, "profile": "p"}] if kind == "sigma" else [],
    )
    request = _request(guidance_scale=5.0)
    if disabled:
        if kind == "step":
            request.sampling_params.attention_schedule = ()
        else:
            request.sampling_params.attention_sigma_schedule = ()
    preprocess(request)
    assert request.prepared_layout.mixfusion_sequence is not None
    prepare_mixfusion_admission(_AdmissionEngine(config), request)
    assert (request.mixfusion_info is not None) == disabled


def test_legacy_ar_preparation_validates_token_prefix():
    tokenizer, processor = _components([12, 14])
    request = _request(guidance_scale=5.0)
    request.sampling_params.past_key_values = _LegacyKV(key_cache=[torch.zeros(3)], value_cache=[torch.zeros(3)])
    _prepare(request, tokenizer, processor)
    assert tokenizer.calls[0]["validate_ar_kv_prefix"] is True


@pytest.mark.parametrize("reused", [0, 7])
def test_first_and_later_sequence_masks_match_existing_hunyuan_masks(monkeypatch, reused):
    pipeline = _pipeline()
    monkeypatch.setattr(HunyuanImage3Pipeline, "device", property(lambda self: torch.device("cpu")))
    tokenizer, processor = _components([12, 14])
    layout = _prepare(_request(guidance_scale=5.0), tokenizer, processor)
    output = layout.tokenizer_output
    width = output.tokens.shape[1]
    positions = torch.arange(width)[None].expand(2, -1)
    model_kwargs = {
        "mode": "gen_image",
        "tokenizer_output": output,
        "custom_pos_emb": (torch.ones(2, width, 4), torch.zeros(2, width, 4)),
        "position_ids": positions,
        "image_mask": output.gen_image_mask,
        "gen_timestep_scatter_index": output.gen_timestep_scatter_index,
        "num_image_tokens": 17,
        "num_special_tokens": 1,
        "ar_kv_reuse_len": reused,
    }
    mask = pipeline._prepare_attention_mask_for_generation(output.tokens, None, model_kwargs)
    model_kwargs["attention_mask"] = mask
    state = _state("a", 0)
    state.mixfusion_info = MixFusionSequenceInfo(tuple(state.latents.shape), 16, (12, 14), (32, 34), 1)
    state.extra[_STEP_AR_KV] = [[(torch.randn(reused, 2, 4), torch.randn(reused, 2, 4)) for _ in range(2)]]
    first_values = dict(model_kwargs, position_ids=positions[:, reused:])
    first_values["custom_pos_emb"] = tuple(t[:, reused:] for t in model_kwargs["custom_pos_emb"])
    prepared = _PreparedModelForward(
        first_values,
        torch.zeros(2, width - reused, 8),
        output.gen_image_mask[:, reused:],
        torch.ones(2),
        4,
        4,
        True,
        1,
        "gen_image",
        False,
        True,
    )
    docs = pipeline._mixfusion_documents(state, prepared)

    def plan_mask(doc):
        plan = build_sequence_plan((doc.geometry,), torch.device("cpu"))
        mask = torch.zeros(
            doc.geometry.query_length, doc.geometry.prefix_length + doc.geometry.query_length, dtype=torch.bool
        )
        for segment in plan.segments:
            qo = ko = 0
            for qlen, klen in zip(segment.query_lengths, segment.key_lengths, strict=True):
                qi = segment.query_indices[qo : qo + qlen]
                ki = segment.kv_indices[ko : ko + klen]
                block = torch.ones(qlen, klen, dtype=torch.bool)
                if segment.causal:
                    block = block.tril(klen - qlen)
                mask[qi[:, None], ki] = block
                qo += qlen
                ko += klen
        return mask

    for branch, doc in enumerate(docs):
        end = state.mixfusion_info.query_lengths[branch]
        assert torch.equal(plan_mask(doc), mask[branch, 0, reused:end, :end])
        assert doc.geometry.capture_length == (12, 14)[branch]
    # Existing Hunyuan update selects timestep+image positions and removes
    # boi/eoi/tail tokens. Compare against that owner, not a copied formula.
    later = pipeline._update_model_kwargs_for_generation(BaseModelOutputWithPast(), model_kwargs)
    state.step_index = 1
    state.extra[_STEP_PROMPT_KV] = [{"key": torch.randn(2, 14, 2, 4), "value": torch.randn(2, 14, 2, 4)}]
    later.update(ar_kv_reuse_len=0)
    prepared = replace(prepared, model_inputs=later, inputs_embeds=torch.zeros(2, 17, 8), first_step=False)
    docs = pipeline._mixfusion_documents(state, prepared)
    for branch, doc in enumerate(docs):
        prefix = doc.geometry.prefix_length
        reference = torch.cat(
            (later["attention_mask"][branch, 0, :, :prefix], later["attention_mask"][branch, 0, :, 14:]), dim=-1
        )
        assert torch.equal(plan_mask(doc), reference)
        assert doc.geometry.capture_length is None
        assert doc.prefix_kv[0][0].shape[0] == prefix


@pytest.mark.parametrize(
    "changes", [{"image_tokens": 0}, {"prefix_lengths": ()}, {"query_lengths": (1,)}, {"latent_shape": (1, 4, 0, 8)}]
)
def test_invalid_host_sequence_never_admitted(changes):
    info = MixFusionSequenceInfo((1, 4, 8, 8), 16, (12,), (32,), 1)
    assert not fits_mixfusion_budget([replace(info, **changes)])

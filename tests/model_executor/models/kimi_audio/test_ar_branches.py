# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio dual-stream routing and Model Runner V2 sampling."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.kimi_audio import kimi_audio_ar_stage
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams, sample_kimi_audio_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class InplaceLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states.clone()
        else:
            residual.add_(hidden_states)
        hidden_states.copy_(self.projection(residual))
        return hidden_states, residual


class InplaceNorm(nn.Module):
    def forward(self, hidden_states, residual):
        residual.add_(hidden_states)
        hidden_states.copy_(residual)
        return hidden_states, residual


@pytest.mark.parametrize("branch", ["shared", "text", "audio"])
@pytest.mark.parametrize("num_tokens", [1, 3])
@torch.inference_mode()
def test_forward_branch_isolation(monkeypatch, branch, num_tokens):
    monkeypatch.setattr(
        kimi_audio_ar_stage, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        stage = kimi_audio_ar_stage.KimiAudioARStage.__new__(kimi_audio_ar_stage.KimiAudioARStage)
        nn.Module.__init__(stage)
        stage.config = SimpleNamespace(hidden_size=4)
        stage.start_layer, stage.end_layer, stage.branch_layer = 0, 3, 0
        stage.mimo_start_layer, stage.mimo_end_layer = 0, 1
        stage.layers = nn.ModuleList(InplaceLayer() for _ in range(3))
        stage.mimo_layers = nn.ModuleList([InplaceLayer()])
        stage.norm, stage.mimo_norm = InplaceNorm(), InplaceNorm()
        inputs = torch.randn(num_tokens, 4)

    positions = torch.arange(num_tokens)
    before = stage(None, positions, inputs_embeds=inputs.clone())
    layer = {"shared": stage.layers[0], "text": stage.layers[1], "audio": stage.mimo_layers[0]}[branch]
    layer.projection.bias.add_(1)
    after = stage(None, positions, inputs_embeds=inputs.clone())

    assert before.shape == after.shape == (num_tokens, 8)
    for index, stream in enumerate(("text", "audio")):
        expected_change = branch in ("shared", stream)
        start = index * 4
        assert torch.equal(before[:, start : start + 4], after[:, start : start + 4]) != expected_change


SPECIAL = dict(
    msg_end=0,
    media_begin=1,
    media_end=2,
    kimia_text_blank=3,
    kimia_text_eos=4,
    kimia_user_msg_start=5,
    kimia_assistant_msg_start=6,
    kimia_speech_ct_id=7,
    kimia_speech_ctd_id=8,
)
VOCAB, AUDIO_OFFSET, TEXT_TOKEN, AUDIO_TOKEN = 32, 16, 10, 20


@pytest.fixture
def stage():
    stage = kimi_audio_ar_stage.KimiAudioARStage.__new__(kimi_audio_ar_stage.KimiAudioARStage)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(
        vocab_size=VOCAB, kimia_mimo_audiodelaytokens=0, kimia_token_offset=AUDIO_OFFSET, eos_token_ids=[9]
    )
    stage.vllm_config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=100))
    stage._sampling_context = stage._audio_logits = None
    return stage


def _info(req_id: str) -> dict:
    generation = {
        "special_tokens": SPECIAL,
        "seed": 0,
        "max_tokens": 8,
        "prompt_len": 4,
        "output_type": "both",
        "text_history": [],
        "audio_history": [],
        "scheduler_history": [],
        "text_finished": False,
        "finished": False,
    }
    return {
        "req_id": req_id,
        "kimi_audio_request_validated": True,
        "sampling_params": SamplingParams(temperature=0.0, top_k=0),
        "kimi_audio_generation": generation,
    }


def _one_hot(rows: int, token: int) -> torch.Tensor:
    logits = torch.zeros(rows, VOCAB)
    logits[:, token] = 1.0
    return logits


def test_sample_reports_text_and_keeps_audio_in_the_payload(stage):
    infos = [_info("decode"), _info("prefill"), {"req_id": "_warmup_0"}]
    output = stage.make_omni_output(
        torch.zeros(3, 4), model_intermediate_buffer=infos, request_sample_eligible=[True, False, False]
    )
    stage._audio_logits = _one_hot(3, AUDIO_TOKEN)

    tokens = stage.sample(_one_hot(3, TEXT_TOKEN))

    # Visible text for the decode row, a blank for the partial prefill row,
    # and any token for the runner's warmup row.
    assert tokens.tolist() == [[TEXT_TOKEN], [SPECIAL["kimia_text_blank"]], [TEXT_TOKEN]]
    decode = infos[0]["kimi_audio_generation"]
    assert decode["text_history"] == [TEXT_TOKEN] and decode["audio_history"] == [AUDIO_TOKEN]
    assert decode["rng_state"] is not None
    assert infos[1]["kimi_audio_generation"]["scheduler_history"] == []
    payload = output.multimodal_outputs
    assert payload["codes"]["audio"][0].tolist() == [AUDIO_TOKEN]
    assert payload["codes"]["audio"][1].numel() == 0
    assert [flag.item() for flag in payload["meta"]["finished"]] == [False, False, False]
    assert stage._sampling_context is None and stage._audio_logits is None


class _Native:
    """The native MRv2 sampler: called only when Kimi has no request handoff."""

    def __init__(self):
        self.req_states = SimpleNamespace(prefill_len=SimpleNamespace(gpu=None))
        self.result = object()
        self.calls = 0

    def __call__(self, logits, input_batch):
        self.calls += 1
        return self.result


def test_sampler_counts_rows_and_falls_back_without_a_handoff(stage, monkeypatch):
    def counts(num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len):
        # The native kernel marks chunked-prefill rows as producing no token.
        return torch.tensor([1, 0]), torch.zeros(2, dtype=torch.long)

    monkeypatch.setattr(kimi_audio_ar_stage, "get_num_sampled_and_rejected", counts)
    native = _Native()
    sampler, rejection_sampler = stage.mrv2_custom_sampler(native)
    assert rejection_sampler is None
    batch = SimpleNamespace(num_reqs=2, seq_lens=torch.tensor([5, 2]), cu_num_logits=None, idx_mapping=None)

    # Sampler warmup and profiling have no request handoff.
    assert sampler(_one_hot(2, TEXT_TOKEN), batch) is native.result

    stage.make_omni_output(
        torch.zeros(2, 4),
        model_intermediate_buffer=[_info("decode"), _info("prefill")],
        request_sample_eligible=[True, False],
    )
    stage._audio_logits = _one_hot(2, AUDIO_TOKEN)
    result = sampler(_one_hot(2, TEXT_TOKEN), batch)

    assert result.sampled_token_ids.dtype == torch.long
    assert result.sampled_token_ids.tolist() == [[TEXT_TOKEN], [SPECIAL["kimia_text_blank"]]]
    assert result.num_sampled.tolist() == [1, 0] and result.logprobs_tensors is None
    assert native.calls == 1


def test_seeded_request_continues_its_random_stream(stage):
    # A seeded request must continue one generator across steps, not reseed it.
    info = _info("seeded")
    info["sampling_params"] = SamplingParams(temperature=1.0, top_k=0)
    text_logits, audio_logits = torch.zeros(1, VOCAB), _one_hot(1, AUDIO_TOKEN)

    generator = torch.Generator().manual_seed(info["kimi_audio_generation"]["seed"])
    params = KimiAudioSamplingParams.from_sampling_params(info["sampling_params"])
    expected, text_history, audio_history = [], [], []
    for _ in range(2):
        result = sample_kimi_audio_step(
            text_logits[0],
            audio_logits[0],
            text_history=text_history,
            audio_history=audio_history,
            text_finished=False,
            output_type="both",
            special_tokens=KimiAudioSpecialTokens(**SPECIAL),
            audio_delay=0,
            params=params,
            generator=generator,
        )
        assert not result.text_finished
        expected.append(result.text_token)
        text_history, audio_history = [*text_history, result.text_token], [*audio_history, result.audio_token]
    # Reseeding every step would repeat the first draw.
    assert expected[0] != expected[1]

    for _ in range(2):
        stage.make_omni_output(torch.zeros(1, 4), model_intermediate_buffer=[info], request_sample_eligible=[True])
        stage._audio_logits = audio_logits.clone()
        stage.sample(text_logits.clone())

    assert info["kimi_audio_generation"]["text_history"] == expected


def test_unvalidated_request_is_rejected(stage):
    info = _info("decode")
    info["kimi_audio_request_validated"] = False
    with pytest.raises(ValueError, match="prepare_kimi_audio_request"):
        stage.make_omni_output(torch.zeros(1, 4), model_intermediate_buffer=[info], request_sample_eligible=[True])

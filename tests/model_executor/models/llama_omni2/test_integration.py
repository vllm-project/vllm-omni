# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.llama_omni2.llama_omni2_code2wav import (
    LlamaOmni2Code2Wav,
    LlamaOmni2Code2WavCore,
)
from vllm_omni.model_executor.models.llama_omni2.llama_omni2_talker import (
    LlamaOmni2TalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.llama_omni2.llama_omni2_thinker import (
    LlamaOmni2ThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.stage_input_processors.llama_omni2 import (
    talker2code2wav_async_chunk,
    thinker2talker_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Request:
    def __init__(self, request_id: str, output_token_ids: list[int]) -> None:
        self.request_id = request_id
        self.external_req_id = request_id
        self.output_token_ids = output_token_ids


class _TransferManager:
    pass


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1).repeat(1, self.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions, intermediate_tensors
        if inputs_embeds is not None:
            return inputs_embeds
        assert input_ids is not None
        return self.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = hidden_states.new_zeros((hidden_states.shape[0], self.vocab_size))
        logits[:, : self.hidden_size] = hidden_states
        return logits


class _FakeFlow(torch.nn.Module):
    token_mel_ratio = 2
    pre_lookahead_len = 3

    def inference(self, *, token, **kwargs):
        del kwargs
        mel = token.to(torch.float32).repeat_interleave(self.token_mel_ratio, dim=1)
        return mel.unsqueeze(1), None


class _FakeHift(torch.nn.Module):
    def inference(self, *, speech_feat, cache_source):
        speech = speech_feat.repeat_interleave(480, dim=-1)
        return speech, torch.cat([cache_source, speech], dim=-1)


def _bare_thinker() -> LlamaOmni2ThinkerForConditionalGeneration:
    model = object.__new__(LlamaOmni2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.have_multimodal_outputs = True
    model.language_model = _FakeLanguageModel(hidden_size=4, vocab_size=32)
    return model


def _bare_talker() -> LlamaOmni2TalkerForConditionalGeneration:
    model = object.__new__(LlamaOmni2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.have_multimodal_outputs = True
    model.language_model = _FakeLanguageModel(hidden_size=4, vocab_size=64)
    model.language_model.config.vocab_size = 158227
    model.input_proj = torch.nn.Sequential(
        torch.nn.Identity(),
        torch.nn.Identity(),
        torch.nn.Identity(),
    )
    model.gate = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias=False),
        torch.nn.Sigmoid(),
    )
    torch.nn.init.zeros_(model.gate[0].weight)
    return model


def _bare_code2wav() -> LlamaOmni2Code2Wav:
    model = object.__new__(LlamaOmni2Code2Wav)
    torch.nn.Module.__init__(model)
    model.have_multimodal_outputs = True
    model.enable_update_additional_information = True
    model.requires_raw_input_tokens = True
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(get_hidden_size=lambda: 4))
    model.core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=2,
        source_cache_len=4,
    )
    return model


def test_text_only_three_stage_runtime_contract_reaches_streaming_audio():
    thinker = _bare_thinker()
    thinker_hidden = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )
    thinker_output = thinker.make_omni_output(thinker_hidden)
    thinker._last_sampled_token_ids = torch.tensor([[13]])
    thinker_update = thinker.postprocess(thinker_hidden)

    assert isinstance(thinker_output, OmniOutput)
    assert thinker.compute_logits(thinker_output).shape == (3, 32)
    assert thinker_update["embed"]["decode"].shape == (1, 4)
    assert thinker_update["hidden_states"]["output"].shape == (1, 4)

    manager = _TransferManager()
    thinker_request = _Request("request-a", [11, 12, 13])
    talker_payload = thinker2talker_async_chunk(
        manager,
        {
            "embed": {
                "decode": torch.cat(
                    [
                        torch.full((1, 4), 11.0),
                        torch.full((1, 4), 12.0),
                        thinker_update["embed"]["decode"],
                    ]
                )
            },
            "hidden_states": {
                "output": torch.cat(
                    [
                        torch.full((1, 4), 1.0),
                        torch.full((1, 4), 2.0),
                        thinker_update["hidden_states"]["output"],
                    ]
                )
            },
        },
        thinker_request,
    )

    talker = _bare_talker()
    talker_hidden = talker(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        inputs_embeds=talker.preprocess(
            input_ids=torch.zeros(3, dtype=torch.long),
            input_embeds=None,
            ids={"output": talker_payload.ids.output},
            embed={"decode": talker_payload.embed.decode},
            hidden_states={"output": talker_payload.hidden_states.output},
            meta={"finished": talker_payload.meta.finished},
        )[1],
    )
    talker_output = talker.make_omni_output(talker_hidden)
    talker._last_sampled_token_ids = torch.tensor([[151769]])
    talker_update = talker.postprocess(talker_hidden)

    assert isinstance(talker_output, OmniOutput)
    assert talker.compute_logits(talker_output).shape == (3, 64)
    assert talker_update["codes"]["audio"].tolist() == [151769]

    talker_request = _Request("request-a", [151769])
    code2wav_payload = talker2code2wav_async_chunk(
        manager,
        talker_update,
        talker_request,
        is_finished=True,
    )
    code2wav = _bare_code2wav()
    audio_output = code2wav(
        runtime_additional_information=[
            {
                "codes": {"audio": code2wav_payload.codes.audio},
                "meta": {
                    "request_id": code2wav_payload.meta.request_id,
                    "finished": code2wav_payload.meta.finished,
                },
            }
        ]
    )

    assert audio_output.multimodal_outputs["sr"][0].item() == 24000
    assert audio_output.multimodal_outputs["finished"][0].item() is True
    assert audio_output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "request-a" not in code2wav.core


def test_talker_terminal_separator_does_not_require_a_hidden_row():
    talker = _bare_talker()
    input_ids, inputs_embeds, _ = talker.preprocess(
        input_ids=torch.zeros(4, dtype=torch.long),
        input_embeds=None,
        ids={"output": [11, 12, 13, 151665]},
        embed={"decode": torch.ones(3, 4)},
        hidden_states={"output": torch.full((3, 4), 2.0)},
        meta={"finished": torch.tensor(True)},
    )
    output = talker(
        input_ids=input_ids,
        positions=torch.arange(4),
        inputs_embeds=inputs_embeds,
    )

    assert output.shape == (4, 4)


def test_talker_cached_decode_does_not_reuse_prefill_handoff():
    talker = _bare_talker()
    decode_ids = torch.tensor([103], dtype=torch.long)

    input_ids, inputs_embeds, update = talker.preprocess(
        input_ids=decode_ids,
        input_embeds=None,
        ids={"prompt": [7]},
        embed={"persistent": torch.ones(1, 4)},
        hidden_states={"last": torch.ones(1, 4)},
        meta={"finished": False},
        _omni_is_prefill=False,
    )

    assert torch.equal(input_ids, decode_ids)
    assert torch.equal(inputs_embeds, talker.language_model.embed_input_ids(decode_ids))
    assert update == {}


def test_speech_placeholder_path_publishes_decode_rows_after_splice():
    thinker = _bare_thinker()
    input_ids = torch.tensor([7, 151665, 8])
    inputs_embeds = thinker.embed_input_ids(
        input_ids,
        multimodal_embeddings=[torch.full((1, 4), 9.0)],
        is_multimodal=input_ids == 151665,
    )
    hidden = inputs_embeds + 1

    output = thinker.make_omni_output(hidden)
    thinker._last_sampled_token_ids = torch.tensor([[23]])
    update = thinker.postprocess(hidden)

    assert isinstance(output, OmniOutput)
    assert update["embed"]["decode"].shape == (1, 4)
    assert update["hidden_states"]["output"].shape == (1, 4)

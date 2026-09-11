# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU output boundaries with a substituted acoustic runtime, not real audio inference."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import msgspec
import pytest
import torch
from vllm.model_executor import model_loader

from tests.model_executor.models.kimi_audio.runtime import registered_model_runtime as registered_model_runtime
from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.detokenizer import PrefixStreamingFlowMatchingDetokenizer
from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_decoder import KimiAudioDecoder
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder, KimiAudioSpecialTokens
from vllm_omni.model_executor.stage_input_processors.kimi_audio import kimi_audio_to_decoder
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))
SPECIAL = KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"])
OFFSET = REFERENCE["input_config"]["audio_token_offset"]
VOCAB = REFERENCE["input_config"]["audio_vocab_size"]


class RecordingAcoustics:
    """Expose chunk boundaries and retained state; no neural computation."""

    def __init__(self):
        self.semantic_fm = SimpleNamespace(speech_model=torch.nn.Linear(1, 1))
        self.vocoder = SimpleNamespace(vocoder=torch.nn.Linear(1, 1), h={"sampling_rate": 24000})
        self.calls = []
        self.history = []
        self.fail = False

    def clear_states(self):
        self.history.clear()

    def detokenize_streaming(self, codes, *, upsample_factor, is_final):
        assert not torch.is_grad_enabled()
        self.calls.append((codes.clone(), upsample_factor, is_final, len(self.history)))
        self.history.extend(codes[0].tolist())
        if self.fail:
            raise RuntimeError("injected acoustic failure")
        return codes.float()


@pytest.fixture
def runtime(monkeypatch, tmp_path, registered_model_runtime):
    acoustic = RecordingAcoustics()
    loads, downloads = [], []

    def load(**kwargs):
        loads.append(kwargs)
        return acoustic

    def download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(PrefixStreamingFlowMatchingDetokenizer, "from_pretrained", load)
    weights = ModuleType("vllm_omni.model_executor.model_loader.weight_utils")
    weights.download_weights_from_hf_specific = download
    monkeypatch.setitem(sys.modules, weights.__name__, weights)
    config = registered_model_runtime(
        model_config=SimpleNamespace(
            model_stage="kimi_audio_decoder",
            model=str(tmp_path),
            revision="fixture-revision",
            async_chunk=False,
            hf_config=SimpleNamespace(
                vocab_size=OFFSET + VOCAB,
                kimia_token_offset=OFFSET,
                architectures=["MoonshotKimiaForCausalLM"],
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        load_config=SimpleNamespace(download_dir=None),
        quant_config=None,
    )
    return SimpleNamespace(acoustic=acoustic, loads=loads, downloads=downloads, config=config)


@pytest.mark.parametrize("remote", [False, True])
def test_acoustic_load_is_lazy_scoped_and_registered(runtime, remote):
    if remote:
        runtime.config.model_config.model = "moonshotai/Kimi-Audio-7B-Instruct"
    model = KimiAudioForConditionalGeneration(vllm_config=runtime.config)
    stage = model.model
    assert isinstance(stage, KimiAudioDecoder)
    assert stage.vllm_config is not runtime.config
    assert stage.vllm_config.model_config.hf_config.architectures == ["KimiAudioDecoder"]
    assert runtime.config.model_config.hf_config.architectures == ["MoonshotKimiaForCausalLM"]
    assert stage.detokenizer is None and not runtime.loads and not runtime.downloads
    assert model.requires_raw_input_tokens and not model.has_preprocess
    assert not hasattr(model, "sample") and not hasattr(model, "make_omni_output")

    def root_llm_weights():
        pytest.fail("Audio decoding must not read the root LLM weights")
        yield

    loaded = model.load_weights(root_llm_weights())
    model_loader.DefaultModelLoader.track_weights_loading(None, model, loaded)
    assert loaded == set(dict(model.named_parameters()))
    assert len(runtime.loads) == 1
    assert stage.speech_model is runtime.acoustic.semantic_fm.speech_model
    assert stage.vocoder is runtime.acoustic.vocoder.vocoder
    assert runtime.loads[0]["look_ahead_tokens"] == 12
    assert runtime.loads[0]["device"] == torch.device("cpu")
    assert bool(runtime.downloads) == remote
    if remote:
        assert runtime.downloads[0]["revision"] == "fixture-revision"
        assert set(runtime.downloads[0]["allow_patterns"]) == {
            "audio_detokenizer/config.yaml",
            "audio_detokenizer/model.pt",
            "vocoder/config.json",
            "vocoder/model.pt",
        }
    model.load_weights(root_llm_weights())
    assert len(runtime.loads) == 1


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_ar_conversion_roundtrip_and_complete_request_decode(runtime, finish_reason):
    builder = KimiAudioPromptBuilder(REFERENCE["text_tokens"].__getitem__, SPECIAL, **REFERENCE["input_config"])
    prompt = prepare_kimi_audio_inputs(
        [{"role": "user", "message_type": "text", "content": "你好"}], builder, output_type="both"
    )
    # Include a real codebook zero, control markers, a 30-token boundary,
    # an empty request, and a second request that must start with clean state.
    requests = [list(range(31)), [], [9]]
    sources = [
        SimpleNamespace(
            finished=True,
            outputs=[
                SimpleNamespace(
                    finish_reason=finish_reason,
                    token_ids=[SPECIAL.kimia_text_blank],
                    multimodal_output={
                        "codes": {
                            "audio": torch.tensor(
                                [SPECIAL.kimia_text_blank] + [OFFSET + code for code in codes] + [SPECIAL.media_end]
                            )
                        }
                    },
                )
            ],
        )
        for codes in requests
    ]
    inputs = kimi_audio_to_decoder(sources, prompt)
    # These inputs cross the generic request IPC as builtins, including the
    # distinct empty sequence and its one scheduling placeholder.
    inputs = msgspec.msgpack.decode(msgspec.msgpack.encode(inputs))
    assert inputs[1]["prompt_token_ids"] == [0]
    assert inputs[1]["model_intermediate_buffer"]["codes"]["audio"] == []
    assert inputs[0]["prompt_token_ids"] == requests[0]
    stage = KimiAudioForConditionalGeneration(vllm_config=runtime.config)
    stage.load_weights(iter(()))
    counts = [len(item["prompt_token_ids"]) for item in inputs]
    ids = torch.tensor([code for item in inputs for code in item["prompt_token_ids"]])
    runner = object.__new__(GPUGenerationModelRunner)
    runner.model = stage
    runner.input_batch = SimpleNamespace(sampling_metadata=None)
    runner.sampler = None
    # The eager model call replaces only GPU execution/graph wrapping. The
    # generation dispatch method forwards the actual framework kwargs.
    runner._model_forward = stage
    output = runner._run_generation_model(
        input_ids=ids,
        positions=None,
        inputs_embeds=None,
        intermediate_tensors=None,
        logits_indices=torch.empty(0, dtype=torch.long),
        model_kwargs={
            "seq_token_counts": counts,
            "model_intermediate_buffer": [item["model_intermediate_buffer"] for item in inputs],
        },
    )
    for actual, expected in zip(output.multimodal_outputs["model_outputs"], requests, strict=True):
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))
    assert [sr.item() for sr in output.multimodal_outputs["sr"]] == [24000] * 3
    assert [(code.shape[1], factor, final, prefix) for code, factor, final, prefix in runtime.acoustic.calls] == [
        (30, 4, False, 0),
        (1, 4, True, 30),
        (1, 4, True, 0),
    ]
    assert runtime.acoustic.history == []


def test_acoustic_failure_clears_state_before_next_request(runtime):
    stage = KimiAudioDecoder(vllm_config=runtime.config)
    stage.load_weights(iter(()))
    runtime.acoustic.fail = True
    with pytest.raises(RuntimeError, match="injected acoustic failure"):
        stage(torch.tensor([4, 5]), seq_token_counts=[2])
    assert runtime.acoustic.history == []
    runtime.acoustic.fail = False
    stage(torch.tensor([8]), seq_token_counts=[1])
    assert runtime.acoustic.calls[-1][-1] == 0


def test_incomplete_audio_and_offset_ids_are_rejected(runtime):
    runtime.config.model_config.async_chunk = True
    with pytest.raises(ValueError, match="async_chunk"):
        KimiAudioDecoder(vllm_config=runtime.config)
    runtime.config.model_config.async_chunk = False
    stage = KimiAudioDecoder(vllm_config=runtime.config)
    stage.load_weights(iter(()))
    with pytest.raises(ValueError, match="raw codebook IDs"):
        stage(torch.tensor([OFFSET]), seq_token_counts=[1])
    with pytest.raises(ValueError, match="complete semantic sequence"):
        stage(torch.tensor([4]), seq_token_counts=[1], model_intermediate_buffer=[{"meta": {"finished": False}}])
    assert runtime.acoustic.calls == []

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.deepseek_janus import pipeline_janus
from vllm_omni.diffusion.models.deepseek_janus.pipeline_janus import JanusPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        assert prompt.endswith("<img>")
        return [1, 2, 3]


class _FakeProcessor:
    sft_format = "deepseek"
    image_start_tag = "<img>"
    pad_id = 0

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_sft_template_for_multi_turn_prompts(self, conversations, sft_format, system_prompt):
        del conversations, sft_format, system_prompt
        return "stub"


class _FakeLanguageModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=1,
            num_hidden_layers=1,
        )

    def get_input_embeddings(self):
        def _embed(tokens: torch.Tensor) -> torch.Tensor:
            batch, seq = tokens.shape
            return torch.ones((batch, seq, 4), dtype=torch.float32)

        return _embed


class _FakeTransformer:
    def __call__(self, *, inputs_embeds, use_cache, past_key_values, return_dict, cache_position=None):
        del use_cache, past_key_values, return_dict, cache_position
        batch = inputs_embeds.shape[0]
        return SimpleNamespace(last_hidden_state=torch.zeros((batch, 1, 4), dtype=inputs_embeds.dtype))


class _FakeGenVisionModel:
    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.generated_shapes: list[tuple[int, ...]] = []

    def decode_code(self, generated: torch.Tensor, shape: list[int]) -> torch.Tensor:
        self.generated_shapes.append(tuple(generated.shape))
        self.calls.append(shape)
        h = shape[2] * 16
        w = shape[3] * 16
        return torch.zeros((shape[0], 3, h, w), dtype=torch.float32)


class _FakeMMModel:
    def __init__(self, gen_vision_model: _FakeGenVisionModel) -> None:
        self.language_model = _FakeLanguageModel()
        self.gen_vision_model = gen_vision_model

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def gen_head(self, hidden: torch.Tensor) -> torch.Tensor:
        batch = hidden.shape[0]
        logits = torch.zeros((batch, 6), dtype=hidden.dtype)
        logits[:, 0] = 1.0
        return logits

    def prepare_gen_img_embeds(self, stacked: torch.Tensor) -> torch.Tensor:
        return torch.ones((stacked.shape[0], 4), dtype=torch.float32)


def _build_pipeline() -> tuple[JanusPipeline, _FakeGenVisionModel]:
    pipe = JanusPipeline.__new__(JanusPipeline)
    nn.Module.__init__(pipe)
    pipe.processor = _FakeProcessor()
    pipe.mm_model = _FakeMMModel(_FakeGenVisionModel())
    pipe.transformer = _FakeTransformer()
    pipe.od_config = SimpleNamespace(enforce_eager=True)
    pipe._prefill_chunk_size = 2048
    pipe._cudagraph_wrapper = None
    pipe._stage_durations = {}
    return pipe, pipe.mm_model.gen_vision_model


def _request_batch(request: OmniDiffusionRequest) -> DiffusionRequestBatch:
    return DiffusionRequestBatch(requests=[request])


def _payload_images(output) -> list:
    assert isinstance(output.output, dict)
    assert isinstance(output.output["payload"], dict)
    return output.output["payload"]["image"]


def test_janus_prefill_chunk_size_defaults_to_2048() -> None:
    assert pipeline_janus._resolve_prefill_chunk_size(SimpleNamespace()) == 2048


def test_janus_prefill_chunk_size_uses_extras() -> None:
    od_config = SimpleNamespace(extras={"max_prefill_chunk_size": 512})

    assert pipeline_janus._resolve_prefill_chunk_size(od_config) == 512


@pytest.mark.parametrize(
    "extras",
    [
        {"max_prefill_chunk_size": 0},
        {"max_prefill_chunk_size": "invalid"},
        {"max_prefill_chunk_size": True},
        {"max_prefill_chunk_size": 1.5},
    ],
)
def test_janus_prefill_chunk_size_rejects_invalid_value(extras) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        pipeline_janus._resolve_prefill_chunk_size(SimpleNamespace(extras=extras))


def test_janus_prefill_chunk_size_rejects_non_mapping_extras() -> None:
    with pytest.raises(TypeError, match="extras must be a mapping"):
        pipeline_janus._resolve_prefill_chunk_size(SimpleNamespace(extras=[("max_prefill_chunk_size", 512)]))


def test_janus_chunked_prefill_returns_last_chunk_hidden() -> None:
    pipe, _ = _build_pipeline()
    pipe._prefill_chunk_size = 2
    calls: list[tuple[int, list[int]]] = []

    class _RecordingTransformer:
        def __call__(self, *, inputs_embeds, use_cache, past_key_values, cache_position, return_dict):
            del use_cache, past_key_values, return_dict
            calls.append((inputs_embeds.shape[1], cache_position.tolist()))
            hidden = torch.full(
                (inputs_embeds.shape[0], inputs_embeds.shape[1], inputs_embeds.shape[-1]),
                float(len(calls)),
                dtype=inputs_embeds.dtype,
            )
            return SimpleNamespace(last_hidden_state=hidden)

    pipe.transformer = _RecordingTransformer()

    hidden = pipe._chunked_prefill(
        inputs_embeds=torch.zeros((2, 5, 4), dtype=torch.float32),
        past_kv=object(),
        input_len=5,
    )

    assert calls == [(2, [0, 1]), (2, [2, 3]), (1, [4])]
    assert torch.equal(hidden, torch.full((2, 4), 3.0))


def test_janus_pipeline_rejects_prompt_extra_image_geometry_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompt={"prompt": "p", "extra": {"img_size": 512, "patch_size": 32}},
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
        request_id="req-1",
    )

    with pytest.raises(ValueError, match="fixed 576 image tokens"):
        pipe.forward(_request_batch(req))

    assert gen_vision_model.calls == []


def test_janus_pipeline_rejects_prompt_extra_token_count_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompt={"prompt": "p", "extra": {"image_token_num": 256}},
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
        request_id="req-token-count",
    )

    with pytest.raises(ValueError, match="fixed 576 image tokens"):
        pipe.forward(_request_batch(req))

    assert gen_vision_model.calls == []


def test_janus_pipeline_rejects_standard_size_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompt="p",
        sampling_params=OmniDiffusionSamplingParams(
            height=512,
            width=512,
            num_outputs_per_prompt=1,
        ),
        request_id="req-size",
    )

    with pytest.raises(ValueError, match="fixed 576 image tokens"):
        pipe.forward(_request_batch(req))

    assert gen_vision_model.calls == []


def test_janus_pipeline_rejects_prompt_mm_processor_size_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    req = OmniDiffusionRequest(
        prompt={"prompt": "p", "mm_processor_kwargs": {"target_h": 384, "target_w": 512}},
        sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1),
        request_id="req-mm-size",
    )

    with pytest.raises(ValueError, match="fixed 576 image tokens"):
        pipe.forward(_request_batch(req))

    assert gen_vision_model.calls == []


def test_janus_pipeline_prefers_sampling_extra_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    monkeypatch.setattr(JanusPipeline, "_decode_manual", lambda self, **kwargs: kwargs["generated"])
    req = OmniDiffusionRequest(
        prompt={"prompt": "p", "extra": {"img_size": 128, "patch_size": 8}},
        sampling_params=OmniDiffusionSamplingParams(
            num_outputs_per_prompt=1,
            extra_step_kwargs={"img_size": 384, "patch_size": 16},
        ),
        request_id="req-2",
    )

    output = pipe.forward(_request_batch(req))

    assert output.error is None
    assert gen_vision_model.calls == [[1, 8, 24, 24]]
    assert gen_vision_model.generated_shapes == [(1, 576)]
    assert len(_payload_images(output)) == 1


def test_janus_pipeline_returns_all_requested_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    pipe, gen_vision_model = _build_pipeline()
    monkeypatch.setattr(JanusPipeline, "_decode_manual", lambda self, **kwargs: kwargs["generated"])
    req = OmniDiffusionRequest(
        prompt="p",
        sampling_params=OmniDiffusionSamplingParams(
            num_outputs_per_prompt=2,
        ),
        request_id="req-3",
    )

    output = pipe.forward(_request_batch(req))

    assert output.error is None
    assert gen_vision_model.calls == [[2, 8, 24, 24]]
    assert gen_vision_model.generated_shapes == [(2, 576)]
    assert len(_payload_images(output)) == 2


def test_janus_pipeline_uses_seeded_generator_for_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_janus, "StaticCache", lambda **kwargs: object())
    captured_generators = []

    def _fake_multinomial(probs, num_samples, *, generator=None):
        captured_generators.append(generator)
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

    monkeypatch.setattr(pipeline_janus.torch, "multinomial", _fake_multinomial)
    pipe, _ = _build_pipeline()
    monkeypatch.setattr(JanusPipeline, "_decode_manual", lambda self, **kwargs: kwargs["generated"])
    req = OmniDiffusionRequest(
        prompt="p",
        sampling_params=OmniDiffusionSamplingParams(
            seed=123,
        ),
        request_id="req-seed",
    )

    output = pipe.forward(_request_batch(req))

    assert output.error is None
    assert captured_generators
    assert all(generator is not None for generator in captured_generators)
    assert captured_generators[0].initial_seed() == 123


def test_janus_init_defers_weight_loading_to_vllm_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(language_config=SimpleNamespace(_attn_implementation="sdpa"))
    fake_model = nn.Module()
    fake_model.param = nn.Parameter(torch.zeros(1))
    fake_model.language_model = SimpleNamespace(model=nn.Identity())

    monkeypatch.setattr(pipeline_janus, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_janus.AutoConfig, "from_pretrained", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(pipeline_janus.AutoModelForCausalLM, "from_config", lambda config, **kwargs: fake_model)
    monkeypatch.setattr(
        pipeline_janus.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("JanusPipeline should not load checkpoint weights in __init__"),
    )
    monkeypatch.setattr(pipeline_janus, "_build_janus_vl_chat_processor", lambda *args, **kwargs: _FakeProcessor())

    od_config = SimpleNamespace(
        model="/tmp/janus",
        revision=None,
        trust_remote_code=True,
        dtype=torch.float32,
        quantization_config=None,
        enable_layerwise_offload=False,
        enable_diffusion_pipeline_profiler=False,
        enforce_eager=True,
    )

    pipe = JanusPipeline(od_config)

    assert pipe.mm_model is fake_model
    assert pipe.weights_sources[0].prefix == ""
    assert pipe.weights_sources[0].model_or_path == "/tmp/janus"


def test_janus_init_compiles_decode_without_compiling_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(language_config=SimpleNamespace(_attn_implementation="sdpa"))
    raw_transformer = nn.Identity()
    compiled_transformer = nn.Identity()
    fake_model = nn.Module()
    fake_model.param = nn.Parameter(torch.zeros(1))
    fake_model.language_model = SimpleNamespace(model=raw_transformer)
    captured: dict[str, object] = {}

    class _FakeCUDAGraphWrapper:
        def __init__(self, runnable, vllm_config, runtime_mode):
            captured["runnable"] = runnable
            captured["vllm_config"] = vllm_config
            captured["runtime_mode"] = runtime_mode
            self.vllm_config = vllm_config

    def _fake_compile(module, **kwargs):
        assert module is raw_transformer
        assert kwargs["mode"] == "reduce-overhead"
        return compiled_transformer

    monkeypatch.setattr(pipeline_janus, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_janus.AutoConfig, "from_pretrained", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(pipeline_janus.AutoModelForCausalLM, "from_config", lambda config, **kwargs: fake_model)
    monkeypatch.setattr(pipeline_janus, "_build_janus_vl_chat_processor", lambda *args, **kwargs: _FakeProcessor())
    monkeypatch.setattr(
        pipeline_janus,
        "current_omni_platform",
        SimpleNamespace(supports_torch_inductor=lambda: True),
    )
    monkeypatch.setattr(pipeline_janus.torch, "compile", _fake_compile)
    monkeypatch.setattr(pipeline_janus, "CUDAGraphWrapper", _FakeCUDAGraphWrapper)
    monkeypatch.setattr(JanusPipeline, "_build_minimal_vllm_config", lambda self: object())

    od_config = SimpleNamespace(
        model="/tmp/janus",
        revision=None,
        trust_remote_code=True,
        dtype=torch.float32,
        quantization_config=None,
        enable_layerwise_offload=False,
        enable_diffusion_pipeline_profiler=False,
        enforce_eager=False,
    )

    pipe = JanusPipeline(od_config)

    assert pipe.transformer is raw_transformer
    assert pipe._decode_transformer is compiled_transformer
    assert pipe._decode_wrapper is not None
    assert pipe._decode_wrapper.transformer is compiled_transformer
    assert captured["runnable"] is pipe._decode_wrapper


def test_janus_minimal_vllm_config_builds() -> None:
    pipe, _ = _build_pipeline()

    vllm_config = pipe._build_minimal_vllm_config()

    assert vllm_config.cache_config.block_size == 16
    assert vllm_config.scheduler_config.max_num_seqs == 8
    assert vllm_config.scheduler_config.async_scheduling is False


def test_janus_cudagraph_wrapper_uses_public_vllm_config(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeGraphWrapper:
        vllm_config = object()
        clear_calls = 0

        def clear_graphs(self):
            self.clear_calls += 1

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace(last_hidden_state=torch.zeros((2, 1, 4), dtype=torch.float32))

    def _fake_forward_context(*args, **kwargs):
        assert args[1] is _FakeGraphWrapper.vllm_config

        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Context()

    monkeypatch.setattr(pipeline_janus, "set_forward_context", _fake_forward_context)
    pipe, _ = _build_pipeline()
    pipe._cudagraph_wrapper = _FakeGraphWrapper()
    generated = torch.zeros((1, 2), dtype=torch.long)

    output = pipe._decode_with_cudagraph(
        inputs_embeds=torch.ones((2, 1, 4), dtype=torch.float32),
        past_kv=object(),
        generated=generated,
        input_len=3,
        image_token_num=2,
        cfg_weight=5.0,
        temperature=1.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
    )

    assert output.shape == (1, 2)


def test_janus_cudagraph_decode_recaptures_for_same_shape_second_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class _CacheSensitiveMMModel(_FakeMMModel):
        def gen_head(self, hidden: torch.Tensor) -> torch.Tensor:
            batch = hidden.shape[0]
            logits = torch.zeros((batch, 6), dtype=hidden.dtype)
            token_ids = (hidden[:, 0] > 2).to(torch.long)
            logits[torch.arange(batch), token_ids] = 10.0
            return logits

    class _FakeGraphWrapper:
        vllm_config = object()

        def __init__(self) -> None:
            self.clear_calls = 0
            self._captured_past_kv = None

        def clear_graphs(self):
            self.clear_calls += 1
            self._captured_past_kv = None

        def __call__(self, inputs_embeds, past_kv, cache_position):
            del cache_position
            if self._captured_past_kv is None:
                self._captured_past_kv = past_kv
            value = float(self._captured_past_kv.value)
            hidden = torch.full(
                (inputs_embeds.shape[0], 1, inputs_embeds.shape[-1]),
                value,
                dtype=inputs_embeds.dtype,
            )
            return SimpleNamespace(last_hidden_state=hidden)

    def _argmax_multinomial(probs, num_samples, *, generator=None):
        del generator
        return torch.argmax(probs, dim=1, keepdim=True)[:, :num_samples]

    def _fake_forward_context(*args, **kwargs):
        class _Context:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Context()

    monkeypatch.setattr(pipeline_janus, "set_forward_context", _fake_forward_context)
    monkeypatch.setattr(pipeline_janus.torch, "multinomial", _argmax_multinomial)

    pipe, gen_vision_model = _build_pipeline()
    pipe.mm_model = _CacheSensitiveMMModel(gen_vision_model)
    graph_wrapper = _FakeGraphWrapper()
    pipe._cudagraph_wrapper = graph_wrapper

    first = pipe._decode_with_cudagraph(
        inputs_embeds=torch.ones((2, 1, 4), dtype=torch.float32),
        past_kv=SimpleNamespace(value=1),
        generated=torch.zeros((1, 2), dtype=torch.long),
        input_len=3,
        image_token_num=2,
        cfg_weight=5.0,
        temperature=1.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
    )
    second = pipe._decode_with_cudagraph(
        inputs_embeds=torch.ones((2, 1, 4), dtype=torch.float32),
        past_kv=SimpleNamespace(value=5),
        generated=torch.zeros((1, 2), dtype=torch.long),
        input_len=3,
        image_token_num=2,
        cfg_weight=5.0,
        temperature=1.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=None,
    )

    assert graph_wrapper.clear_calls == 2
    assert first[0, 1].item() == 0
    assert second[0, 1].item() == 1

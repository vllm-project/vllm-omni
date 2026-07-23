# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""L1 unit tests for the native Boogu-Image pipeline.

Two groups:

1. Constructor tests: all ``from_pretrained`` calls are mocked (Ovis pattern)
   so the pipeline ``__init__`` wiring is exercised without downloading
   weights or building the real transformer.
2. Prompt-encoding tests: a pipeline shell (``object.__new__``) is wired with
   small deterministic fakes so the ported chat templating, processor kwargs,
   CFG handling, and reshape logic can be verified numerically on CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_MODULE = "vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image"

_EMBED_DIM = 8
_SEQ_LEN = 6


# ---------------------------------------------------------------------------
# Constructor tests (mocked from_pretrained)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dependencies(mocker, monkeypatch):
    """Mock external components so ``__init__`` runs without real weights."""
    # The full VLM wrapper has an ``lm_head``; the pipeline must strip it and
    # keep the inner ``.model`` as the encoder.
    inner_encoder = mocker.MagicMock(name="inner_qwen3vl_model")
    inner_encoder.dtype = torch.float32
    mllm_wrapper = mocker.MagicMock(name="qwen3vl_wrapper")
    mllm_wrapper.model.to.return_value = inner_encoder

    mock_processor = mocker.MagicMock(name="processor")

    mock_vae = mocker.MagicMock(name="vae")
    mock_vae.config.block_out_channels = [128, 256, 512, 512]  # scale factor 8
    mock_vae.to.return_value = mock_vae

    mock_scheduler = mocker.MagicMock(name="scheduler")

    monkeypatch.setattr(
        f"{_MODULE}.FlowMatchEulerDiscreteScheduler.from_pretrained",
        lambda *a, **k: mock_scheduler,
    )
    monkeypatch.setattr(
        f"{_MODULE}.Qwen3VLForConditionalGeneration.from_pretrained",
        lambda *a, **k: mllm_wrapper,
    )
    monkeypatch.setattr(
        f"{_MODULE}.Qwen3VLProcessor.from_pretrained",
        lambda *a, **k: mock_processor,
    )
    monkeypatch.setattr(
        f"{_MODULE}.AutoencoderKL.from_pretrained",
        lambda *a, **k: mock_vae,
    )

    mock_transformer_cls = mocker.MagicMock(name="transformer_cls")
    mock_transformer_instance = mocker.MagicMock(name="transformer")
    mock_transformer_cls.return_value = mock_transformer_instance
    monkeypatch.setattr(f"{_MODULE}.BooguImageTransformer2DModel", mock_transformer_cls)

    # Treat the dummy model id as a local path: skips hub prefetch.
    mocker.patch("os.path.exists", return_value=True)

    return {
        "inner_encoder": inner_encoder,
        "mllm_wrapper": mllm_wrapper,
        "processor": mock_processor,
        "vae": mock_vae,
        "scheduler": mock_scheduler,
        "transformer": mock_transformer_instance,
    }


@pytest.fixture
def boogu_pipeline(mock_dependencies):
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        BooguImagePipeline,
    )

    od_config = OmniDiffusionConfig(
        model="dummy-boogu",
        tf_model_config=TransformerConfig(params={}),
        dtype=torch.float32,
        num_gpus=1,
    )
    return BooguImagePipeline(od_config=od_config)


def test_boogu_image_pipeline_import():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    assert BooguImagePipeline is not None


def test_component_discovery_declarations():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline

    # CPU offload / HSDP discovery must find ``mllm`` (there is no
    # ``text_encoder`` attribute on this pipeline).
    assert BooguImagePipeline._dit_modules == ["transformer"]
    assert BooguImagePipeline._encoder_modules == ["mllm"]
    assert BooguImagePipeline._vae_modules == ["vae"]


def test_constructor_wires_components(boogu_pipeline, mock_dependencies):
    assert boogu_pipeline.scheduler is mock_dependencies["scheduler"]
    assert boogu_pipeline.processor is mock_dependencies["processor"]
    assert boogu_pipeline.vae is mock_dependencies["vae"]
    assert boogu_pipeline.transformer is mock_dependencies["transformer"]
    assert boogu_pipeline.vae_scale_factor == 8
    assert boogu_pipeline.default_sample_size == 128
    assert hasattr(boogu_pipeline, "load_weights")


def test_constructor_strips_mllm_lm_head(boogu_pipeline, mock_dependencies):
    # Upstream encodes with the inner Qwen3VLModel, not the generation wrapper.
    assert boogu_pipeline.mllm is mock_dependencies["inner_encoder"]


def test_constructor_weights_sources(boogu_pipeline):
    (source,) = boogu_pipeline.weights_sources
    assert source.model_or_path == "dummy-boogu"
    assert source.subfolder == "transformer"
    assert source.prefix == "transformer."
    assert source.fall_back_to_pt is True


@pytest.mark.parametrize(
    ("parallel_config", "cache_backend", "message"),
    [
        (DiffusionParallelConfig(tensor_parallel_size=2), "none", "Tensor parallelism"),
        (DiffusionParallelConfig(ulysses_degree=2), "none", "Sequence parallelism"),
        (DiffusionParallelConfig(ring_degree=2), "none", "Sequence parallelism"),
        (DiffusionParallelConfig(cfg_parallel_size=2), "none", "CFG parallelism"),
        (
            DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2),
            "none",
            "HSDP",
        ),
        (DiffusionParallelConfig(), "cache_dit", "Cache backend 'cache_dit'"),
        (DiffusionParallelConfig(), "tea_cache", "Cache backend 'tea_cache'"),
    ],
)
def test_constructor_rejects_unsupported_execution_modes(
    mock_dependencies,
    parallel_config,
    cache_backend,
    message,
):
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        BooguImagePipeline,
    )

    od_config = OmniDiffusionConfig(
        model="dummy-boogu",
        tf_model_config=TransformerConfig(params={}),
        dtype=torch.float32,
        parallel_config=parallel_config,
        cache_backend=cache_backend,
    )

    with pytest.raises(NotImplementedError, match=message):
        BooguImagePipeline(od_config=od_config)

    # Validation happens before any checkpoint component is constructed.
    mock_dependencies["mllm_wrapper"].model.to.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt-encoding tests (deterministic fakes, no constructor)
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    """Fake Qwen3VLProcessor: deterministic token ids derived from the text."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, prompts, **kwargs):
        self.calls.append({"prompts": prompts, "kwargs": kwargs})
        batch = len(prompts)
        input_ids = torch.zeros(batch, _SEQ_LEN, dtype=torch.long)
        for i, messages in enumerate(prompts):
            system_text = messages[0]["content"][0]["text"]
            user_text = messages[1]["content"][0]["text"]
            input_ids[i, 0] = len(system_text) % 997
            input_ids[i, 1] = len(user_text) % 997
            input_ids[i, 2:] = torch.arange(2, _SEQ_LEN) + i
        attention_mask = torch.ones(batch, _SEQ_LEN, dtype=torch.long)
        attention_mask[:, -1] = 0  # fake right-padding
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeMLLM:
    """Fake Qwen3VLModel: hidden states are a deterministic function of ids."""

    dtype = torch.bfloat16

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        self.calls.append({"output_hidden_states": output_hidden_states, **kwargs})
        base = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, _EMBED_DIM)
        hidden = base + torch.arange(_EMBED_DIM, dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden - 1, hidden))


def _make_encode_pipeline(num_instruction_feature_layers: int = 1):
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        SYSTEM_PROMPT_4_T2I_UNIFIED,
        SYSTEM_PROMPT_4_TI2I_UNIFIED,
        BooguImagePipeline,
    )

    pipeline = object.__new__(BooguImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline._execution_device = torch.device("cpu")
    pipeline.processor = _RecordingProcessor()
    pipeline.mllm = _FakeMLLM()
    pipeline.transformer = SimpleNamespace(
        instruction_feature_configs={
            "instruction_feat_dim": _EMBED_DIM,
            "num_instruction_feature_layers": num_instruction_feature_layers,
            "reduce_type": "mean",
        },
        dtype=torch.float32,
    )
    pipeline.SYSTEM_PROMPT_4_T2I = SYSTEM_PROMPT_4_T2I_UNIFIED
    pipeline.SYSTEM_PROMPT_DROP = SYSTEM_PROMPT_4_TI2I_UNIFIED
    return pipeline


def test_apply_chat_template_system_prompt_selection():
    pipeline = _make_encode_pipeline()

    messages = pipeline._apply_chat_template("a cat on a mat")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_T2I
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["text"] == "a cat on a mat"

    # Empty and whitespace-only instructions select the DROP prompt (this is
    # the path the default negative prompt "" takes).
    for empty in ("", "   ", None):
        messages = pipeline._apply_chat_template(empty)
        assert messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_DROP


def test_processor_called_with_upstream_kwargs():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog", do_classifier_free_guidance=False)

    (call,) = pipeline.processor.calls
    kwargs = call["kwargs"]
    assert kwargs["padding"] == "longest"
    assert kwargs["padding_side"] == "right"
    assert kwargs["truncation"] is False
    assert kwargs["max_length"] == 1280  # upstream __call__ default
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is True
    assert kwargs["return_tensors"] == "pt"


def test_encode_prompt_shapes_and_dtype():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt(["a dog", "a cat"])

    assert embeds.shape == (2, _SEQ_LEN, _EMBED_DIM)
    assert mask.shape == (2, _SEQ_LEN)
    assert neg_embeds.shape == (2, _SEQ_LEN, _EMBED_DIM)
    assert neg_mask.shape == (2, _SEQ_LEN)
    # Cast to the MLLM dtype, mask passed through from the processor.
    assert embeds.dtype == torch.bfloat16
    assert neg_embeds.dtype == torch.bfloat16
    assert torch.equal(mask[:, -1], torch.zeros(2, dtype=torch.long))


def test_single_layer_encoding_requests_hidden_states_once():
    pipeline = _make_encode_pipeline()

    pipeline._get_instruction_feature_embeds("a dog")

    assert pipeline.mllm.calls == [{"output_hidden_states": True, "return_dict": True}]


def test_single_layer_encoding_propagates_mllm_failure():
    pipeline = _make_encode_pipeline()
    failure = RuntimeError("mllm failed")

    class _FailingMLLM:
        dtype = torch.bfloat16

        def __init__(self):
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            raise failure

    pipeline.mllm = _FailingMLLM()

    with pytest.raises(RuntimeError, match="mllm failed") as exc_info:
        pipeline._get_instruction_feature_embeds("a dog")

    assert exc_info.value is failure
    assert pipeline.mllm.calls == 1


def test_encode_prompt_cfg_negative_default_is_empty_string():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog")

    assert len(pipeline.processor.calls) == 2
    negative_messages = pipeline.processor.calls[1]["prompts"]
    assert len(negative_messages) == 1
    # The default "" negative prompt goes through the DROP system prompt.
    assert negative_messages[0][0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_DROP
    assert negative_messages[0][1]["content"][0]["text"] == ""


def test_encode_prompt_without_cfg_skips_negative():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt("a dog", do_classifier_free_guidance=False)

    assert len(pipeline.processor.calls) == 1
    assert embeds.shape == (1, _SEQ_LEN, _EMBED_DIM)
    assert neg_embeds is None
    assert neg_mask is None


def test_encode_prompt_explicit_negative_prompt():
    pipeline = _make_encode_pipeline()
    pipeline.encode_prompt("a dog", negative_prompt="blurry, low quality")

    negative_messages = pipeline.processor.calls[1]["prompts"]
    assert negative_messages[0][0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_T2I
    assert negative_messages[0][1]["content"][0]["text"] == "blurry, low quality"


def test_encode_prompt_negative_batch_mismatch_raises():
    pipeline = _make_encode_pipeline()
    with pytest.raises(ValueError, match="batch size"):
        pipeline.encode_prompt(["a dog", "a cat"], negative_prompt=["only one"])


def test_encode_prompt_num_images_per_prompt_repeats_embeds():
    pipeline = _make_encode_pipeline()
    embeds, mask, neg_embeds, neg_mask = pipeline.encode_prompt("a dog", num_images_per_prompt=3)

    assert embeds.shape == (3, _SEQ_LEN, _EMBED_DIM)
    assert mask.shape == (3, _SEQ_LEN)
    assert torch.equal(embeds[0], embeds[1])
    assert torch.equal(embeds[0], embeds[2])
    assert torch.equal(mask[0], mask[1])
    assert neg_embeds.shape == (3, _SEQ_LEN, _EMBED_DIM)


def test_encode_prompt_precomputed_embeds_bypass_encoder():
    pipeline = _make_encode_pipeline()
    precomputed = torch.randn(1, _SEQ_LEN, _EMBED_DIM)
    precomputed_mask = torch.ones(1, _SEQ_LEN, dtype=torch.long)
    neg_precomputed = torch.randn(1, _SEQ_LEN, _EMBED_DIM)

    embeds, mask, neg_embeds, _ = pipeline.encode_prompt(
        "ignored",
        prompt_embeds=precomputed,
        prompt_attention_mask=precomputed_mask,
        negative_prompt_embeds=neg_precomputed,
    )

    assert len(pipeline.processor.calls) == 0
    assert torch.equal(embeds, precomputed)
    assert torch.equal(neg_embeds, neg_precomputed)


def test_reshape_embeds_and_mask_list_branch():
    # Multi-layer configs return a list of per-layer tensors; the reshape
    # helper must handle both forms.
    pipeline = _make_encode_pipeline()
    layer = torch.arange(2 * _SEQ_LEN * _EMBED_DIM, dtype=torch.float32).view(2, _SEQ_LEN, _EMBED_DIM)
    mask = torch.ones(2, _SEQ_LEN, dtype=torch.long)

    batch_size, seq_len, reshaped, reshaped_mask = pipeline._reshape_embeds_and_mask([layer, layer + 1], mask, 2)

    assert batch_size == 2
    assert seq_len == _SEQ_LEN
    assert isinstance(reshaped, list) and len(reshaped) == 2
    assert reshaped[0].shape == (4, _SEQ_LEN, _EMBED_DIM)
    # repeat(1, n, 1).view(b*n, ...) interleaves per-sample: rows [s0, s0, s1, s1]
    assert torch.equal(reshaped[0][0], reshaped[0][1])
    assert torch.equal(reshaped[0][2], reshaped[0][3])


class _FakeTransformer:
    """Fake denoiser: velocity 0 (identity flow) with the config the loop reads."""

    in_channels = 4
    axes_dim_rope = (8, 4, 4)
    axes_lens = (32, 16, 16)
    dtype = torch.float32
    instruction_feature_configs = {
        "instruction_feat_dim": _EMBED_DIM,
        "num_instruction_feature_layers": 1,
        "reduce_type": "mean",
    }

    def __call__(self, latents, timestep, instruction_embeds, freqs_cis, instruction_attention_mask, **kwargs):
        return torch.zeros_like(latents)


class _FakeScheduler:
    def set_timesteps(self, num_inference_steps, device=None, num_tokens=None):
        self.timesteps = torch.linspace(0, 1, num_inference_steps + 1)[:-1]

    def step(self, model_output, t, latents, return_dict=False):
        return (latents,)


class _FakeDecodeVAE:
    dtype = torch.float32

    def __init__(self):
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0, block_out_channels=[128, 256, 512, 512])

    def decode(self, latents, return_dict=False):
        batch = latents.shape[0]
        return (torch.zeros(batch, 3, 16, 16),)


def _make_forward_pipeline():
    pipeline = _make_encode_pipeline()
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.vae = _FakeDecodeVAE()
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    return pipeline


def _make_request_batch(prompt, **sampling_overrides):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    return SimpleNamespace(prompts=[prompt], sampling_params=sampling)


def test_forward_returns_diffusion_output():
    from vllm_omni.diffusion.data import DiffusionOutput

    pipeline = _make_forward_pipeline()
    req = _make_request_batch("a cat", height=64, width=64, num_inference_steps=2)

    out = pipeline.forward(req)

    assert isinstance(out, DiffusionOutput)
    assert isinstance(out.output, torch.Tensor)
    assert out.output.shape[0] == 1
    assert torch.isfinite(out.output).all()
    # CFG default is on (text guidance 4.0), so the encoder ran twice (pos + neg).
    assert len(pipeline.processor.calls) == 2


def test_forward_cfg_off_when_guidance_one():
    pipeline = _make_forward_pipeline()
    req = _make_request_batch("a cat", height=64, width=64, num_inference_steps=2, guidance_scale=1.0)
    # guidance_scale=1.0 is falsy-adjacent but explicitly disables CFG; the
    # request layer would set guidance_scale_provided, so emulate that here.
    req.sampling_params.guidance_scale_provided = True

    pipeline.forward(req)

    # Only the positive prompt is encoded when CFG is off.
    assert len(pipeline.processor.calls) == 1


# ---------------------------------------------------------------------------
# Editing / TI2I: pre-process function
# ---------------------------------------------------------------------------


def _make_edit_od_config(tmp_path, block_out_channels=(128, 256, 512, 512)):
    import json

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"block_out_channels": list(block_out_channels)}))
    return OmniDiffusionConfig(
        model=str(tmp_path),
        tf_model_config=TransformerConfig(params={}),
        dtype=torch.float32,
        num_gpus=1,
    )


@pytest.mark.parametrize(
    "factory_name",
    ["get_boogu_image_pre_process_func", "get_boogu_image_post_process_func"],
)
@pytest.mark.parametrize("config_contents", [None, "{not valid json"])
def test_process_factory_reports_invalid_vae_config(tmp_path, factory_name, config_contents):
    import importlib

    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    if config_contents is not None:
        (vae_dir / "config.json").write_text(config_contents)

    od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        tf_model_config=TransformerConfig(params={}),
        dtype=torch.float32,
        num_gpus=1,
    )
    factory = getattr(importlib.import_module(_MODULE), factory_name)

    with pytest.raises(RuntimeError, match=r"Failed to load Boogu VAE config from .*vae/config\.json"):
        factory(od_config)


@pytest.mark.parametrize(("height", "width"), [(0, 16), (16, 0), (0, 0)])
def test_image_processor_rejects_non_positive_dimensions(height, width):
    from vllm_omni.diffusion.models.boogu_image.image_processor import BooguImageProcessor

    image_processor = BooguImageProcessor()
    image = torch.zeros(1, 3, 16, 16)

    with pytest.raises(ValueError, match=rf"height={height}, width={width}"):
        image_processor.get_new_height_width(image, height=height, width=width)


def _make_diffusion_request(prompt, **sampling_overrides):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(**sampling_overrides)
    return OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="req-0")


def test_pre_process_no_image_is_noop(tmp_path):
    import PIL.Image  # noqa: F401  (import guard: PIL must be available)

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        get_boogu_image_pre_process_func,
    )

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))

    # Text-to-image request: no multimodal image -> returned unchanged.
    req = _make_diffusion_request({"prompt": "a cat"}, height=123, width=456)
    out = pre(req)
    assert "additional_information" not in out.prompt
    assert out.sampling_params.height == 123
    assert out.sampling_params.width == 456

    # A plain-string prompt cannot carry an image either.
    str_req = _make_diffusion_request("a cat")
    assert pre(str_req).prompt == "a cat"


def test_pre_process_populates_reference_and_align_res(tmp_path):
    import PIL.Image

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        get_boogu_image_pre_process_func,
    )

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))

    image = PIL.Image.new("RGB", (1000, 500))  # (width, height)
    req = _make_diffusion_request({"prompt": "make it winter", "multi_modal_data": {"image": image}})
    out = pre(req)

    ai = out.prompt["additional_information"]
    assert "prompt_image" in ai and "preprocessed_image" in ai

    # VLM copy is a PIL image, downscaled, never upscaled, aligned to 16.
    prompt_image = ai["prompt_image"]
    assert isinstance(prompt_image, PIL.Image.Image)
    assert prompt_image.width % 16 == 0 and prompt_image.height % 16 == 0
    assert prompt_image.width <= 1000 and prompt_image.height <= 500
    assert max(prompt_image.width, prompt_image.height) <= 768

    # VAE copy is a normalized [1, C, H, W] tensor aligned to 16.
    vae = ai["preprocessed_image"]
    assert isinstance(vae, torch.Tensor) and vae.ndim == 4 and vae.shape[0] == 1
    assert vae.shape[-1] % 16 == 0 and vae.shape[-2] % 16 == 0
    assert -1.0 <= float(vae.min()) and float(vae.max()) <= 1.0

    # align_res: the request resolution follows the VAE-encoded reference dims.
    assert out.sampling_params.height == vae.shape[-2]
    assert out.sampling_params.width == vae.shape[-1]


def test_pre_process_rejects_multiple_images(tmp_path):
    import PIL.Image

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        get_boogu_image_pre_process_func,
    )

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))
    imgs = [PIL.Image.new("RGB", (64, 64)), PIL.Image.new("RGB", (64, 64))]
    req = _make_diffusion_request({"prompt": "combine", "multi_modal_data": {"image": imgs}})

    with pytest.raises(ValueError, match="single reference image"):
        pre(req)


def test_pre_process_single_image_in_list_is_accepted(tmp_path):
    import PIL.Image

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        get_boogu_image_pre_process_func,
    )

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))
    req = _make_diffusion_request({"prompt": "edit", "multi_modal_data": {"image": [PIL.Image.new("RGB", (128, 128))]}})
    out = pre(req)
    assert "preprocessed_image" in out.prompt["additional_information"]


# ---------------------------------------------------------------------------
# Editing / TI2I: chat template + image-aware encoding
# ---------------------------------------------------------------------------


def _make_edit_encode_pipeline():
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import (
        SYSTEM_PROMPT_4_TI2I_UNIFIED,
    )

    pipeline = _make_encode_pipeline()
    pipeline.SYSTEM_PROMPT_4_TI2I = SYSTEM_PROMPT_4_TI2I_UNIFIED
    pipeline.SYSTEM_PROMPT_4_I2I = SYSTEM_PROMPT_4_TI2I_UNIFIED
    return pipeline


def test_apply_chat_template_ti2i_places_image_before_text():
    import PIL.Image

    pipeline = _make_edit_encode_pipeline()
    image = PIL.Image.new("RGB", (16, 16))

    messages = pipeline._apply_chat_template("turn day into night", [image])
    assert messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_TI2I
    user_content = messages[1]["content"]
    # Image content comes first, then the instruction text.
    assert user_content[0]["type"] == "image"
    assert user_content[0]["image"] is image
    assert user_content[-1] == {"type": "text", "text": "turn day into night"}

    # Empty instruction with an image selects the I2I system prompt.
    empty_messages = pipeline._apply_chat_template("", [image])
    assert empty_messages[0]["content"][0]["text"] == pipeline.SYSTEM_PROMPT_4_I2I


class _ImageAwareRecordingProcessor:
    """Records whether reference images reached the processor."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, prompts, **kwargs):
        has_image = []
        for messages in prompts:
            user_content = messages[1]["content"]
            has_image.append(any(c.get("type") == "image" for c in user_content))
        self.calls.append({"prompts": prompts, "kwargs": kwargs, "has_image": has_image})
        batch = len(prompts)
        input_ids = torch.arange(batch * _SEQ_LEN, dtype=torch.long).view(batch, _SEQ_LEN)
        attention_mask = torch.ones(batch, _SEQ_LEN, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_encode_prompt_attaches_images_to_positive_only():
    import PIL.Image

    pipeline = _make_edit_encode_pipeline()
    pipeline.processor = _ImageAwareRecordingProcessor()
    image = PIL.Image.new("RGB", (16, 16))

    pipeline.encode_prompt(
        "add a rainbow",
        do_classifier_free_guidance=True,
        input_images=[[image]],
    )

    # Positive call carries the image; the negative (CFG) call is image-free.
    assert pipeline.processor.calls[0]["has_image"] == [True]
    assert pipeline.processor.calls[1]["has_image"] == [False]


# ---------------------------------------------------------------------------
# Editing / TI2I: reference-latent VAE encode
# ---------------------------------------------------------------------------


class _FakeRefVAE:
    """Fake VAE whose ``encode`` yields a known latent for shape/scaling checks."""

    dtype = torch.float32

    def __init__(self, scaling_factor=2.0, shift_factor=0.5):
        self.config = SimpleNamespace(scaling_factor=scaling_factor, shift_factor=shift_factor)
        self._latent = torch.ones(1, 4, 3, 5)

    def encode(self, img):
        dist = SimpleNamespace(sample=lambda generator=None: self._latent.clone())
        return SimpleNamespace(latent_dist=dist)


def test_build_ref_latents_shape_and_scaling():
    pipeline = _make_encode_pipeline()
    pipeline.vae = _FakeRefVAE(scaling_factor=2.0, shift_factor=0.5)

    preprocessed = torch.zeros(1, 3, 48, 80)  # normalized image tensor
    ref_latents = pipeline._build_ref_latents([preprocessed], num_images_per_prompt=1, device=torch.device("cpu"))

    assert len(ref_latents) == 1
    (sample_latents,) = ref_latents
    assert isinstance(sample_latents, list) and len(sample_latents) == 1
    latent = sample_latents[0]
    # squeeze(0) -> [C, H, W]; (1 - shift) * scaling = (1 - 0.5) * 2 = 1.0
    assert latent.shape == (4, 3, 5)
    assert torch.allclose(latent, torch.ones(4, 3, 5))


def test_build_ref_latents_expands_per_output_and_handles_none():
    pipeline = _make_encode_pipeline()
    pipeline.vae = _FakeRefVAE()

    preprocessed = torch.zeros(1, 3, 48, 80)
    ref_latents = pipeline._build_ref_latents([preprocessed, None], num_images_per_prompt=2, device=torch.device("cpu"))

    # Two samples x 2 outputs each = 4 entries; the None sample stays None.
    assert len(ref_latents) == 4
    assert ref_latents[0] is ref_latents[1]  # same sample repeated
    assert ref_latents[2] is None and ref_latents[3] is None


# ---------------------------------------------------------------------------
# Editing / TI2I: forward CFG branch selection
# ---------------------------------------------------------------------------


class _RecordingRefTransformer(_FakeTransformer):
    """Counts predictions per step and records the reference-latent argument."""

    def __init__(self):
        self.calls = []

    def __call__(self, latents, timestep, instruction_embeds, freqs_cis, instruction_attention_mask, **kwargs):
        self.calls.append(kwargs.get("ref_image_hidden_states"))
        return torch.zeros_like(latents)


class _EditForwardVAE(_FakeDecodeVAE):
    """Adds a fake ``encode`` so the editing forward path can build ref latents."""

    def encode(self, img):
        dist = SimpleNamespace(sample=lambda generator=None: torch.zeros(1, 4, 8, 8))
        return SimpleNamespace(latent_dist=dist)


def _make_edit_forward_pipeline():
    pipeline = _make_edit_encode_pipeline()
    # Image-aware processor: reference images appear as ``{"type": "image"}``
    # content entries, which the default text-first fake cannot parse.
    pipeline.processor = _ImageAwareRecordingProcessor()
    pipeline.transformer = _RecordingRefTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.vae = _EditForwardVAE()
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    return pipeline


def _make_edit_request(**sampling_overrides):
    import PIL.Image

    image = PIL.Image.new("RGB", (64, 64))
    prompt = {
        "prompt": "make it winter",
        "additional_information": {
            "prompt_image": image,
            "preprocessed_image": torch.zeros(1, 3, 64, 64),
        },
    }
    sampling = _sampling(**sampling_overrides)
    return SimpleNamespace(prompts=[prompt], sampling_params=sampling)


def _sampling(**overrides):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionSamplingParams(**overrides)


def _count_per_step(calls, num_steps):
    assert len(calls) % num_steps == 0
    return len(calls) // num_steps


def test_forward_ti2i_text_only_two_predictions_with_ref():
    pipeline = _make_edit_forward_pipeline()
    num_steps = 2
    req = _make_edit_request(height=64, width=64, num_inference_steps=num_steps, guidance_scale=5.0)
    req.sampling_params.guidance_scale_provided = True

    pipeline.forward(req)

    # Text-only ti2i: cond+ref and neg+ref -> 2 predictions/step, both carry ref.
    assert _count_per_step(pipeline.transformer.calls, num_steps) == 2
    assert all(ref is not None for ref in pipeline.transformer.calls)


def test_forward_ti2i_double_guidance_three_predictions():
    pipeline = _make_edit_forward_pipeline()
    num_steps = 2
    req = _make_edit_request(
        height=64, width=64, num_inference_steps=num_steps, guidance_scale=5.0, guidance_scale_2=2.0
    )
    req.sampling_params.guidance_scale_provided = True
    req.sampling_params.guidance_scale_2_provided = True

    pipeline.forward(req)

    # Double guidance: cond+ref, neg+ref, neg+no-ref -> 3 predictions/step.
    assert _count_per_step(pipeline.transformer.calls, num_steps) == 3
    # Exactly one of the three per step drops the reference (neg+no-ref).
    per_step = [pipeline.transformer.calls[i : i + 3] for i in range(0, len(pipeline.transformer.calls), 3)]
    for step_calls in per_step:
        assert sum(ref is None for ref in step_calls) == 1


def test_forward_ti2i_image_only_two_predictions_drop_ref():
    pipeline = _make_edit_forward_pipeline()
    num_steps = 2
    # text guidance 1.0 (off, provided) + image guidance 2.0 (provided).
    req = _make_edit_request(
        height=64, width=64, num_inference_steps=num_steps, guidance_scale=1.0, guidance_scale_2=2.0
    )
    req.sampling_params.guidance_scale_provided = True
    req.sampling_params.guidance_scale_2_provided = True

    pipeline.forward(req)

    # Image-only ti2i: cond+ref and cond+no-ref -> 2 predictions/step.
    assert _count_per_step(pipeline.transformer.calls, num_steps) == 2
    per_step = [pipeline.transformer.calls[i : i + 2] for i in range(0, len(pipeline.transformer.calls), 2)]
    for step_calls in per_step:
        assert sum(ref is None for ref in step_calls) == 1


def test_forward_ti2i_no_guidance_single_prediction_with_ref():
    pipeline = _make_edit_forward_pipeline()
    num_steps = 2
    # Both guidances off/unprovided -> image guidance forced to 1.0, no CFG.
    req = _make_edit_request(height=64, width=64, num_inference_steps=num_steps, guidance_scale=1.0)
    req.sampling_params.guidance_scale_provided = True

    pipeline.forward(req)

    assert _count_per_step(pipeline.transformer.calls, num_steps) == 1
    assert all(ref is not None for ref in pipeline.transformer.calls)


def test_forward_image_guidance_ignored_without_reference():
    # guidance_scale_2 is set but there is no reference image (t2i request);
    # image guidance must be forced off so this stays plain t2i CFG.
    pipeline = _make_edit_forward_pipeline()
    num_steps = 2
    sampling = _sampling(height=64, width=64, num_inference_steps=num_steps, guidance_scale=5.0, guidance_scale_2=2.0)
    sampling.guidance_scale_provided = True
    sampling.guidance_scale_2_provided = True
    req = SimpleNamespace(prompts=[{"prompt": "a cat"}], sampling_params=sampling)

    pipeline.forward(req)

    # t2i text CFG: cond + uncond -> 2 predictions/step, no ref anywhere.
    assert _count_per_step(pipeline.transformer.calls, num_steps) == 2
    assert all(ref is None for ref in pipeline.transformer.calls)

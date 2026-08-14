from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import ErnieImageTransformer2DModel
from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import ErnieImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TokenInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.last_chat_input = None

    def apply_chat_template(self, messages, *_args, **_kwargs):
        # Store the last input for checking prompt enhance
        self.last_chat_input = messages[0]["content"]
        return "chat prompt"

    def __call__(self, *_args, **_kwargs):
        return _TokenInputs(input_ids=torch.tensor([[1, 2]]))

    def decode(self, *_args, **_kwargs):
        return "enhanced-prompt"


class _FakePEModel:
    def __init__(self):
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return torch.tensor([[1, 2, 3, 4]])


def test_enhance_prompt_uses_rank0_result_in_distributed(monkeypatch):
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe.pe_model = None
    pipe.pe_tokenizer = None

    fake_model = _FakePEModel()
    fake_tokenizer = _FakeTokenizer()
    pipe._load_pe = lambda: (fake_model, fake_tokenizer)

    broadcasts = []

    def fake_broadcast_object_list(values, src=0):
        broadcasts.append((list(values), src))
        values[0] = "rank0-enhanced"

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", fake_broadcast_object_list)

    enhanced = pipe._enhance_prompt("original", torch.device("cpu"))

    assert enhanced == "rank0-enhanced"
    assert fake_model.calls == 0
    assert broadcasts == [([None], 0)]


def _make_batch(*requests):
    return SimpleNamespace(
        num_reqs=len(requests),
        requests=list(requests),
        request_id=requests[0].request_id if requests else "test",
    )


def _make_request(extra_args=None, *, request_id="real_req"):
    return SimpleNamespace(
        request_id=request_id,
        sampling_params=SimpleNamespace(extra_args=extra_args or {}),
    )


def test_should_apply_pe_true_when_requested():
    """Ensure that use_prompt_upscaling forwards through."""
    batch = _make_batch(_make_request({"use_prompt_upscaling": True}))
    assert ErnieImagePipeline._should_apply_pe(batch) is True


def test_should_apply_pe_false_when_not_requested():
    """Ensure that use_prompt_upscaling is False by default."""
    batch = _make_batch(_make_request({}))
    assert ErnieImagePipeline._should_apply_pe(batch) is False


def test_should_apply_pe_disables_dummy_warmup_request():
    """Ensure that use_prompt_upscaling is False during warmup."""
    batch = _make_batch(
        _make_request(
            {"use_prompt_upscaling": True},
            request_id="dummy_req_id",
        )
    )
    assert ErnieImagePipeline._should_apply_pe(batch) is False


def test_enhance_prompt_triggers_lazy_load(monkeypatch):
    """_enhance_prompt lazy-loads the PE model on first call."""
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe.pe_model = None
    pipe.pe_tokenizer = None

    fake_model = _FakePEModel()
    fake_tokenizer = _FakeTokenizer()
    pipe._load_pe = lambda: (fake_model, fake_tokenizer)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)

    result = pipe._enhance_prompt("a cat sitting on a mat", torch.device("cpu"))

    assert pipe.pe_model is fake_model
    assert fake_model.calls == 1
    # Should get the enhanced result back after getting our input
    assert result == "enhanced-prompt"
    assert "a cat sitting on a mat" in fake_tokenizer.last_chat_input


def test_ensure_pe_loaded_skips_when_disabled():
    """Ensure has pe loaded is False when external upscaler loading is not enabled."""
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe.pe_model = None
    pipe.pe_tokenizer = None
    pipe.has_external_prompt_upscaler = True
    pipe._load_pe = None

    assert pipe._ensure_pe_loaded() is False
    assert pipe.pe_model is None


def test_hybrid_ring_slices_full_attention_mask(monkeypatch):
    import vllm_omni.diffusion.models.ernie_image.ernie_image_transformer as ernie_transformer

    mask = torch.arange(8).view(1, 8)

    monkeypatch.setattr(ernie_transformer, "_get_ring_parallel_info", lambda: (2, 1))

    sliced = ErnieImageTransformer2DModel._slice_attention_mask_for_ring(mask)

    assert sliced.tolist() == [[4, 5, 6, 7]]
    assert sliced.is_contiguous()


def test_resolve_model_path_downloads_repo_id_for_pe_discovery(monkeypatch):
    calls = []

    def fake_exists(path):
        return path == "/cache/baidu/ERNIE-Image" or path == "/cache/baidu/ERNIE-Image/pe"

    def fake_download(model, revision, allow_patterns):
        calls.append((model, revision, allow_patterns))
        return "/cache/baidu/ERNIE-Image"

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image.download_weights_from_hf_specific",
        fake_download,
    )

    from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import _resolve_model_path_for_optional_pe

    assert _resolve_model_path_for_optional_pe("baidu/ERNIE-Image", None) == "/cache/baidu/ERNIE-Image"
    assert calls == [("baidu/ERNIE-Image", None, ["pe/*", "pe_tokenizer/*"])]


def test_resize_dimensions_rounds_down_to_vae_scale_factor():
    pipe = ErnieImagePipeline.__new__(ErnieImagePipeline)
    pipe.vae_scale_factor = 16

    assert pipe._resize_dimensions(1025, 1024) == (1024, 1024)
    assert pipe._resize_dimensions(1024, 1025) == (1024, 1024)
    assert pipe._resize_dimensions(1025, 1031) == (1024, 1024)


def test_load_weights_accepts_transformer_prefix():
    model = ErnieImageTransformer2DModel.__new__(ErnieImageTransformer2DModel)
    param = torch.nn.Parameter(torch.zeros(1))

    model.named_parameters = lambda: [("proj.weight", param)]
    model.named_buffers = lambda: []

    loaded = model.load_weights([("transformer.proj.weight", torch.ones(1))])

    assert loaded == {"proj.weight"}
    assert torch.equal(param.data, torch.ones(1))

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional FastWAM kernel backends (``VLLM_OMNI_FASTWAM_KERNELS``) on the CPU.

Covers the flag parsing, the registry, the fallback to the stock path (the ``hub`` backend cannot run
on a CPU, without ``kernels`` or on a non-gfx950 GPU) and the pipeline routing through a backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest
import torch
from torch import nn

from tests.diffusion.models.fastwam import tiny
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.fastwam import config as fastwam_config
from vllm_omni.diffusion.models.fastwam import hub_kernels, kernel_backend, pipeline_fastwam
from vllm_omni.diffusion.models.fastwam.kernel_backend import (
    FASTWAM_KERNELS_ENV,
    FastWAMKernelContext,
    fastwam_kernel_backends,
    load_fastwam_kernel_backend,
    parse_fastwam_kernels,
    register_fastwam_kernel_backend,
    requested_fastwam_kernels,
    unregister_fastwam_kernel_backend,
)
from vllm_omni.diffusion.models.fastwam.pipeline_fastwam import FastWAMPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ----------------------------------------------------------------------------- flag parsing
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "off"),
        ("", "off"),
        ("  ", "off"),
        ("off", "off"),
        ("OFF", "off"),
        ("0", "off"),
        ("false", "off"),
        ("no", "off"),
        ("none", "off"),
        ("stock", "off"),
        ("hub", "hub"),
        (" HUB\n", "hub"),
        ("Other", "other"),
    ],
)
def test_parse_fastwam_kernels(value, expected):
    assert parse_fastwam_kernels(value) == expected


def test_requested_backend_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv(FASTWAM_KERNELS_ENV, raising=False)
    assert requested_fastwam_kernels() == "off"
    monkeypatch.setenv(FASTWAM_KERNELS_ENV, "Hub")
    assert requested_fastwam_kernels() == "hub"
    assert requested_fastwam_kernels({FASTWAM_KERNELS_ENV: "off"}) == "off"


def test_hub_backend_is_built_in():
    assert "hub" in fastwam_kernel_backends()


def test_off_is_reserved():
    with pytest.raises(ValueError, match="reserved"):
        register_fastwam_kernel_backend(_FakeBackend(name="off"))


# ----------------------------------------------------------------------------- registry and fallback
def _ctx(cfg=None, device="cpu", dtype=torch.bfloat16) -> FastWAMKernelContext:
    return FastWAMKernelContext(
        cfg=cfg or fastwam_config.PRESETS["cae_fastwam_libero"],
        model_dir="/nonexistent/model",
        base_model_dir="/nonexistent/base",
        device=torch.device(device),
        dtype=dtype,
    )


@dataclass
class _FakeLoaded:
    name: str
    components: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)
    description: str = "fake"

    def infer_action(self, image, prompt, proprio, *, num_steps, generator, suite_id):
        self.calls.append(
            {"prompt": prompt, "steps": num_steps, "seed": generator.initial_seed(), "suite_id": suite_id}
        )
        return torch.full((8, 7), 0.25)


@dataclass
class _FakeBackend:
    name: str = "fake"
    reason: str | None = None
    error: Exception | None = None
    loaded: _FakeLoaded | None = None

    def unsupported_reason(self, ctx):
        return self.reason

    def load(self, ctx):
        if self.error is not None:
            raise self.error
        return self.loaded or _FakeLoaded(self.name)


@pytest.fixture
def warnings_seen(mocker):
    return mocker.patch.object(kernel_backend.logger, "warning")


@pytest.fixture
def fake_backend():
    backend = _FakeBackend()
    register_fastwam_kernel_backend(backend)
    yield backend
    unregister_fastwam_kernel_backend(backend.name)


def _warning_text(mock) -> str:
    return "\n".join(call.args[0] % call.args[1:] for call in mock.call_args_list)


def test_off_loads_nothing_and_logs_nothing(monkeypatch, warnings_seen):
    monkeypatch.delenv(FASTWAM_KERNELS_ENV, raising=False)
    assert load_fastwam_kernel_backend(_ctx()) is None
    assert load_fastwam_kernel_backend(_ctx(), "off") is None
    warnings_seen.assert_not_called()


def test_unknown_backend_falls_back(warnings_seen):
    assert load_fastwam_kernel_backend(_ctx(), "nope") is None
    text = _warning_text(warnings_seen)
    assert "no such backend" in text and "stock PyTorch path" in text


def test_hub_backend_falls_back_on_cpu(warnings_seen):
    assert load_fastwam_kernel_backend(_ctx(), "hub") is None
    text = _warning_text(warnings_seen)
    assert "'hub'" in text and "needs a ROCm GPU" in text and "stock PyTorch path" in text


def test_registered_backend_is_loaded(fake_backend, warnings_seen):
    loaded = load_fastwam_kernel_backend(_ctx(), "FAKE")
    assert isinstance(loaded, _FakeLoaded)
    warnings_seen.assert_not_called()


def test_unsupported_backend_falls_back(fake_backend, warnings_seen):
    fake_backend.reason = "not on this GPU"
    assert load_fastwam_kernel_backend(_ctx(), "fake") is None
    assert "not on this GPU" in _warning_text(warnings_seen)


def test_failing_load_falls_back_and_cleans_up(fake_backend, warnings_seen):
    fake_backend.error = OSError("401 Client Error: repository not found")
    cleaned = []
    assert load_fastwam_kernel_backend(_ctx(), "fake", on_fallback=lambda: cleaned.append(True)) is None
    assert cleaned == [True]
    text = _warning_text(warnings_seen)
    assert "load failed" in text and "401" in text


def test_failing_support_check_falls_back(fake_backend, warnings_seen, mocker):
    mocker.patch.object(fake_backend, "unsupported_reason", side_effect=RuntimeError("boom"))
    assert load_fastwam_kernel_backend(_ctx(), "fake") is None
    assert "support check failed" in _warning_text(warnings_seen)


# ----------------------------------------------------------------------------- hub backend support checks
@pytest.fixture
def rocm_gfx950(mocker):
    """Pretend to run on a ROCm gfx950 GPU with every dependency importable."""
    mocker.patch.object(hub_kernels.torch.version, "hip", "7.0")
    mocker.patch.object(hub_kernels, "gpu_arch", return_value="gfx950")
    mocker.patch.object(hub_kernels, "_module_available", return_value=True)
    return mocker


def test_hub_backend_supported_on_rocm_gfx950(rocm_gfx950):
    assert hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda")) is None


def test_hub_backend_pins_the_kernel_version():
    assert (hub_kernels.HUB_KERNELS_BACKEND.repo_id, hub_kernels.HUB_KERNELS_BACKEND.version) == (
        "amd/cae-fastwam-kernels",
        1,
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a", None])
def test_hub_backend_needs_gfx950(rocm_gfx950, arch):
    rocm_gfx950.patch.object(hub_kernels, "gpu_arch", return_value=arch)
    reason = hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda"))
    assert reason is not None and "gfx950" in reason and str(arch or "unknown") in reason


def test_hub_backend_needs_rocm(rocm_gfx950):
    rocm_gfx950.patch.object(hub_kernels.torch.version, "hip", None)
    assert "ROCm" in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda"))


@pytest.mark.parametrize(("missing", "match"), [("kernels", "`kernels` package"), ("fastwam", "faster-action-models")])
def test_hub_backend_needs_its_packages(rocm_gfx950, missing, match):
    rocm_gfx950.patch.object(hub_kernels, "_module_available", side_effect=lambda name: name != missing)
    assert match in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda"))


def test_hub_backend_needs_cae_fastwam_in_bfloat16(rocm_gfx950):
    original = fastwam_config.PRESETS["fastwam_libero_uncond"]
    assert "cae_fastwam_libero only" in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(cfg=original))
    assert "bfloat16" in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda", dtype=torch.float32))


def test_hub_backend_must_install_before_the_reference_import(rocm_gfx950, monkeypatch):
    monkeypatch.setitem(hub_kernels.sys.modules, hub_kernels.REFERENCE_MODEL_MODULE, object())
    assert "imported before" in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(device="cuda"))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("encoder.conv_in.weight", "model.encoder.conv1.weight"),
        ("quant_conv.bias", "model.conv1.bias"),
        ("encoder.down_blocks.1.resnets.0.conv1.weight", "model.encoder.downsamples.1.downsamples.0.residual.2.weight"),
        (
            "encoder.down_blocks.2.downsampler.resample.1.weight",
            "model.encoder.downsamples.2.downsamples.2.resample.1.weight",
        ),
        ("encoder.mid_block.resnets.1.norm2.gamma", "model.encoder.middle.2.residual.3.gamma"),
        ("encoder.mid_block.attentions.0.to_qkv.weight", "model.encoder.middle.1.to_qkv.weight"),
        ("decoder.conv_in.weight", None),
    ],
)
def test_vae_keys_map_to_the_reference_layout(name, expected):
    assert hub_kernels.vae_key_to_reference(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("shared.weight", "token_embedding.weight"),
        ("encoder.final_layer_norm.weight", "norm.weight"),
        ("encoder.block.3.layer.0.SelfAttention.q.weight", "blocks.3.attn.q.weight"),
        ("encoder.block.3.layer.1.DenseReluDense.wi_0.weight", "blocks.3.ffn.gate.0.weight"),
        ("encoder.block.3.layer.1.unknown.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_t5_keys_map_to_the_reference_layout(name, expected):
    assert hub_kernels.t5_key_to_reference(name) == expected


class _RecordingReference(nn.Module):
    """Stands in for the reference model: records the autograd / inference-mode state of every call."""

    def __init__(self):
        super().__init__()
        self.mot, self.text_encoder, self.vae = nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1)
        self.proprio_encoder, self.action_vae = nn.Linear(1, 1), nn.Linear(1, 1)
        self.device, self.torch_dtype = torch.device("cpu"), torch.float32
        self.calls = []

    def _state(self, *tensors):
        return {
            "inference_mode": torch.is_inference_mode_enabled(),
            "grad": torch.is_grad_enabled(),
            "inference_tensors": [t.is_inference() for t in tensors],
        }

    def encode_prompt(self, prompt):
        self.calls.append(("encode_prompt", self._state()))
        return torch.zeros(1, 4, 2), torch.ones(1, 4, dtype=torch.bool)

    def infer_action(self, *, input_image, proprio, context, context_mask, seed, num_inference_steps, **kwargs):
        self.calls.append(("infer_action", self._state(input_image, proprio, context), seed, num_inference_steps))
        return {"action": torch.zeros(8, 7)}


def test_hub_backend_leaves_inference_mode():
    # The pipeline calls the backend under torch.inference_mode(); the package's CUDA-graph regions need tensors with
    # version counters, so the reference model must run with inference mode off and autograd off.
    reference = _RecordingReference()
    backend = hub_kernels.HubKernelsFastWAM(reference, kernel=object(), action_horizon=8)
    with torch.inference_mode():
        image = torch.zeros(1, 3, 16, 32, dtype=torch.uint8)
        proprio = torch.zeros(1, 8)
        out = backend.infer_action(
            image, "pick the bowl", proprio, num_steps=3, generator=torch.Generator().manual_seed(7), suite_id=1
        )
    assert out.shape == (8, 7) and out.dtype == torch.float32
    (_, encode_state), (_, infer_state, seed, steps) = reference.calls
    for state in (encode_state, infer_state):
        assert state["inference_mode"] is False and state["grad"] is False
    assert infer_state["inference_tensors"] == [False, False, False]
    assert (seed, steps, reference.action_vae_suite_id) == (7, 3, 1)
    # The prompt context is cached: a second call does not re-encode.
    backend.infer_action(image, "pick the bowl", proprio, num_steps=3, generator=None, suite_id=1)
    assert [c[0] for c in reference.calls].count("encode_prompt") == 1


# ----------------------------------------------------------------------------- pipeline routing
@pytest.fixture
def tiny_assets(tmp_path, monkeypatch):
    cfg = tiny.tiny_cae_config()
    monkeypatch.setitem(fastwam_config.PRESETS, "tiny_cae", cfg)
    monkeypatch.setattr(pipeline_fastwam, "get_local_device", lambda: torch.device("cpu"))
    model = tiny.build_tiny_model(cfg, seed=0)
    model_dir, base_dir = tiny.write_tiny_assets(tmp_path, cfg, model)
    return cfg, model_dir, base_dir


def _pipeline(tiny_assets) -> FastWAMPipeline:
    _, model_dir, base_dir = tiny_assets
    od_config = OmniDiffusionConfig(
        model=model_dir,
        model_class_name="FastWAMPipeline",
        dtype=torch.float32,
        model_config={
            "fastwam_variant": "tiny_cae",
            "base_model": base_dir,
            "task_suite": "suite_a",
            "num_inference_steps": 3,
        },
    )
    return FastWAMPipeline(od_config=od_config)


def _request(seed=42):
    rng = np.random.RandomState(0)
    robot_obs = {
        "observation/image": rng.randint(0, 256, (40, 40, 3), dtype=np.uint8),
        "observation/wrist_image": rng.randint(0, 256, (40, 40, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-1, 1, 8).astype(np.float32),
        "prompt": "pick the bowl",
        "task_suite": "suite_b",
    }
    params = OmniDiffusionSamplingParams(seed=seed, extra_args={"robot_obs": robot_obs})
    return DiffusionRequestBatch(requests=[OmniDiffusionRequest(prompt="", sampling_params=params, request_id="r")])


def test_pipeline_serves_the_stock_path_when_the_backend_falls_back(tiny_assets, monkeypatch, warnings_seen):
    monkeypatch.delenv(FASTWAM_KERNELS_ENV, raising=False)
    stock = _pipeline(tiny_assets).forward(_request()).output["actions"]
    monkeypatch.setenv(FASTWAM_KERNELS_ENV, "hub")
    pipeline = _pipeline(tiny_assets)
    assert pipeline.kernel_backend is None and pipeline.model is not None
    assert not hasattr(pipeline, "setup_compile")
    text = _warning_text(warnings_seen)
    assert "'hub'" in text and "stock PyTorch path" in text
    np.testing.assert_array_equal(pipeline.forward(_request()).output["actions"], stock)


def test_pipeline_routes_inference_to_the_backend(tiny_assets, monkeypatch, fake_backend):
    cfg, _, _ = tiny_assets
    components = {name: nn.Linear(2, 2) for name in ("mot", "text_encoder", "vae_encoder", "proprio_encoder")}
    components["action_codec"] = None
    fake_backend.loaded = _FakeLoaded("fake", components=components)
    monkeypatch.setenv(FASTWAM_KERNELS_ENV, "fake")
    pipeline = _pipeline(tiny_assets)

    assert pipeline.kernel_backend is fake_backend.loaded
    assert pipeline.model is None  # the stock model is not built
    assert pipeline.mot is components["mot"] and pipeline.vae_encoder is components["vae_encoder"]
    pipeline.setup_compile()  # the runner's compile is replaced by a log line

    output = pipeline.forward(_request(seed=7))
    assert output.error is None
    call = fake_backend.loaded.calls[-1]
    assert call == {"prompt": "pick the bowl", "steps": 3, "seed": 7, "suite_id": cfg.suite_ids["suite_b"]}
    # Observation processing and post-processing stay in vLLM-Omni: the LIBERO gripper column is binarised.
    actions = output.output["actions"]
    assert actions.shape == (cfg.action_horizon, cfg.action_dim) and actions.dtype == np.float32
    assert set(np.unique(actions[:, -1])) <= {-1.0, 0.0, 1.0}
    with pytest.raises(RuntimeError, match="kernel backend"):
        pipeline.policy.encode_prompt("pick the bowl")


def test_tiny_config_is_not_supported_by_the_hub_backend():
    cfg = replace(tiny.tiny_cae_config())
    assert "cae_fastwam_libero only" in hub_kernels.HUB_KERNELS_BACKEND.unsupported_reason(_ctx(cfg=cfg))

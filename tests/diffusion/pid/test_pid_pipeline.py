# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plan-B integration tests: pipelines stay PiD-free, the Runner orchestrates.

Covers:
1. QwenImage pipeline reversion regression (no mixin / no PiD symbols).
2. flux2 family latent-branch semantics: unpack + BN denorm + unpatchify run
   BEFORE the output_type branch, so ``output_type="latent"`` returns the
   original VAE-ready grid regardless of PiD; the LatentForm converts it
   back to PiD's BN-normalized patchified form at the Runner layer only.
3. Full batch passthrough flow: gate -> force latent -> forward -> restore
   -> PiD decode, on a mocked pipeline.
"""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.pid import PidDecodeConfig
from vllm_omni.diffusion.pid.runner_integration import maybe_pid_passthrough

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# -- QwenImage reversion regression ------------------------------------------


def test_qwen_image_pipeline_has_no_pid_intrusion():
    """Plan B keeps pipelines PiD-free: no mixin, no backbone, no hooks."""
    assert not any("Pid" in cls.__name__ or "pid" in cls.__name__ for cls in QwenImagePipeline.__mro__)
    assert not hasattr(QwenImagePipeline, "PID_BACKBONE")
    assert not hasattr(QwenImagePipeline, "maybe_pid_decode")
    assert not hasattr(QwenImagePipeline, "_init_pid_decoder")
    source_has_pid = False
    src = inspect.getsource(QwenImagePipeline)
    for token in ("maybe_pid_decode", "_pid_override", "_pid_caption", "PidDecodeMixin"):
        if token in src:
            source_has_pid = True
    assert not source_has_pid


def test_flux2_pipelines_are_pid_free():
    """PiD must not alter pipeline behavior: zero PiD symbols in flux2 sources."""
    for module_name, cls_name in (
        ("vllm_omni.diffusion.models.flux2.pipeline_flux2", "Flux2Pipeline"),
        ("vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein", "Flux2KleinPipeline"),
    ):
        cls = getattr(importlib.import_module(module_name), cls_name)
        src = inspect.getsource(cls)
        for token in ("maybe_pid_decode", "pid_decode", "PidDecode", "_pid_", "PidPassthrough"):
            assert token not in src, f"{cls_name} source contains PiD token {token!r}"


# -- flux2 latent-branch semantics ---------------------------------------------


def _flux2_klein_like(mocker, vae_dtype=torch.bfloat16):
    """Build a Flux2Klein-like object exposing the unpack/denorm/unpatch path."""
    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline

    pipe = object.__new__(Flux2KleinPipeline)
    torch.nn.Module.__init__(pipe)
    pipe.vae = mocker.Mock()
    pipe.vae.dtype = vae_dtype
    pipe.vae.decode = mocker.Mock(return_value=(torch.zeros(1, 3, 512, 512),))
    pipe.vae.bn = SimpleNamespace(
        running_mean=torch.randn(128),
        running_var=torch.rand(128) + 0.5,
    )
    pipe.vae.config = SimpleNamespace(batch_norm_eps=1e-5)
    return pipe


def _loop_tokens(batch=1, height=512, width=512, channels=128):
    """Flux2 loop output: BN-normalized 2x2-patchified tokens [B, T, C].

    Geometry: VAE 8x compression -> VAE grid h=H/8; patchify 2x2 ->
    tokens (h/2)x(w/2) of 4*latent_ch channels (Flux2: 32ch VAE -> 128).
    """
    h = height // 8
    w = width // 8
    return torch.randn(batch, (h // 2) * (w // 2), channels)


def _canonical_ids(latents):
    """Row-major canonical grid ids for a [B, T, C] token tensor."""
    b, t, _ = latents.shape
    side = int(t**0.5)
    ids = []
    for _ in range(b):
        h_ids = torch.arange(t) // side
        w_ids = torch.arange(t) % side
        ids.append(torch.stack([torch.zeros_like(h_ids), h_ids, w_ids], dim=1))
    return torch.stack(ids)


def _pipeline_latent_branch_path(pipe, tokens):
    """Replicate the restored pipeline: unpack -> BN denorm -> unpatchify.

    These transforms run BEFORE the output_type branch, so this is exactly
    what ``output_type="latent"`` returns (VAE-ready grid).
    """
    latents = Flux2KleinPipeline._unpack_latents_with_ids(tokens, _canonical_ids(tokens))
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
    latents = latents * std + mean
    return Flux2KleinPipeline._unpatchify_latents(latents)


def test_flux2_latent_branch_returns_vae_ready_grid(mocker):
    """Restored semantics: latent branch output is the VAE-ready 32ch grid."""
    pipe = _flux2_klein_like(mocker)
    tokens = _loop_tokens()  # [1, 1024, 128] for 512x512

    grid = _pipeline_latent_branch_path(pipe, tokens)

    assert grid.shape == (1, 32, 64, 64)  # 128ch unpatchified -> 32ch, 2x grid


def test_flux2_latent_form_roundtrips_pipeline_latent(mocker):
    """LatentForm._patchify_and_normalize inverts the pipeline's denorm+unpatchify.

    The form receives the pipeline's latent-branch output (VAE-ready grid) and
    must recover the BN-normalized patchified loop latents the PiD checkpoint
    expects — without the pipeline itself doing anything PiD-specific.
    """
    from vllm_omni.diffusion.pid.latent_forms import _patchify_and_normalize

    pipe = _flux2_klein_like(mocker)
    tokens = _loop_tokens()  # [1, 1024, 128] BN-normalized loop latents

    vae_ready = _pipeline_latent_branch_path(pipe, tokens)
    x0, pid_h, pid_w = _patchify_and_normalize(vae_ready, 512, 512, 8, pipeline=pipe)

    unpacked = Flux2KleinPipeline._unpack_latents_with_ids(tokens, _canonical_ids(tokens))
    assert x0.shape == (1, 128, 32, 32)
    assert torch.allclose(x0, unpacked, atol=1e-4)
    assert (pid_h, pid_w) == (512, 512)


def test_flux2_klein_vae_branch_still_decodes(mocker):
    """output_type="pil" consumes the same VAE-ready grid via VAE decode."""
    pipe = _flux2_klein_like(mocker)
    tokens = _loop_tokens()

    latents = _pipeline_latent_branch_path(pipe, tokens)
    assert latents.shape == (1, 32, 64, 64)

    latents = latents.to(pipe.vae.dtype)
    image = pipe.vae.decode(latents, return_dict=False)[0]
    assert image.shape == (1, 3, 512, 512)


# -- full batch passthrough flow (Runner orchestration) ------------------------


class FluxPipeline:
    """Stands in for a pipeline family registered in LATENT_FORMS."""

    vae_scale_factor = 8
    supports_request_batch = True

    def __init__(self, decoder, config):
        self._pid_decoder = decoder
        self._pid_config = config
        self._resident_modules = []
        self.seen_output_types = []

    def forward(self, reqs):
        """Mimic a pipeline's latent branch: return raw latents per request."""
        self.seen_output_types = [r.sampling_params.output_type for r in reqs]
        outputs = []
        for _ in reqs:
            # Flux-family packed tokens: [B, T, 4C] = [1, 1024, 64]
            outputs.append(DiffusionOutput(output=torch.zeros(1, 1024, 64)))
        return outputs


def test_batch_passthrough_end_to_end(mocker):
    """gate -> force -> forward sees "latent" -> restore -> PiD replaces output."""
    decoder = mocker.Mock()
    config = PidDecodeConfig(enabled=True, checkpoint_path="/tmp/pid.pth", gemma_model="/tmp/g")
    pipe = FluxPipeline(decoder, config)

    reqs = [
        SimpleNamespace(
            request_id="r0",
            prompt="a cat",
            sampling_params=SimpleNamespace(
                output_type="pil",
                pid_decode=None,
                height=512,
                width=512,
                latents=None,
                image_latent=None,
                strength=None,
            ),
        )
    ]
    od_config = SimpleNamespace(pid_decode=config, enforce_eager=True)

    pt = maybe_pid_passthrough(pipe, reqs, od_config)
    assert pt is not None

    pt.force_latent_output(reqs)
    outputs = pipe.forward(reqs)
    assert pipe.seen_output_types == ["latent"]  # pipeline took its latent branch
    pt.restore_output_type(reqs)
    assert reqs[0].sampling_params.output_type == "pil"

    mocker.patch(
        "vllm_omni.diffusion.pid.runner_integration.decode_with_pid",
        return_value=torch.zeros(1, 3, 2048, 2048),
    )
    outputs = pt.decode_outputs(outputs, reqs)
    assert outputs[0].output.shape == (1, 3, 2048, 2048)

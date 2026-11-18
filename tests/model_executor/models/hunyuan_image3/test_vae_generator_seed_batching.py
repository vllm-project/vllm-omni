# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the HunyuanImage3 VAE generator seeding path (issue #9362).

When independent single-image requests are batched, ``embed_multimodal`` must
give each image a fresh ``torch.Generator`` seeded with that image's own
request-local ``vae_generator_seed`` entry. Sharing one generator across the
per-image loop leaks the first request's seed to the rest of the batch and
advances the RNG stream across images, so latents depend on batch composition
and order.

The tests extract the production methods verbatim from the checkout sources
(``_parse_and_validate_image_input``, ``_vae_encode``, ``embed_multimodal``,
and the real ``DiagonalGaussianDistribution`` posterior sampling) so CI runs
against the actual code under fix. Everything runs on CPU with synthetic
network stubs: no GPU, no model weights, no vLLM installation.

Invariants enforced here:

- mixed-seed batches, reversed ordering, and same-seed independent requests
  match their standalone latents and embeddings;
- a cache-miss subset (one request omitted) matches the standalone latents;
- a single-image request samples with its own seed through a fresh generator
  (reference-equal latent), i.e. single-image behavior is unchanged;
- unseeded requests still encode without a generator.
"""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODEL_FILE = _REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "hunyuan_image3" / "hunyuan_image3.py"
_VAE_FILE = _REPO_ROOT / "vllm_omni" / "model_executor" / "models" / "hunyuan_image3" / "autoencoder_kl_3d.py"


def _compile_members(source: str, filename: str, names: set[str], namespace: dict, class_name: str | None = None):
    """Exec the named members of ``source`` (optionally of one class) into ``namespace``.

    ``from __future__ import annotations`` is prepended so unbound annotation
    names never evaluate. A missing member means the source layout drifted and
    this regression test must be updated, so it raises instead of skipping.
    """
    tree = ast.parse(source, filename=filename)
    body = tree.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    nodes = [node for node in body if getattr(node, "name", None) in names]
    found = {node.name for node in nodes}
    if found != names:
        raise RuntimeError(
            f"Source layout changed in {filename}: missing {sorted(names - found)}; "
            "update tests/model_executor/models/hunyuan_image3/test_vae_generator_seed_batching.py"
        )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, filename, "exec"), namespace)


def _load_production_namespace() -> dict:
    """Namespace with the production methods and the real posterior sampler."""
    namespace: dict = {
        "torch": torch,
        "np": np,
        "randn_tensor": randn_tensor,
        "logger": logging.getLogger("hunyuan_image3.vae_seed_regression"),
        # The TensorSchema wrapper is pass-through for the dict it carries; a
        # plain dict keeps the extracted methods runnable without vLLM.
        "HunyuanImage3PixelInputs": dict,
    }
    _compile_members(
        _MODEL_FILE.read_text(encoding="utf-8"),
        str(_MODEL_FILE),
        {"_parse_and_validate_image_input", "_vae_encode", "embed_multimodal"},
        namespace,
        class_name="HunyuanImage3ForConditionalGeneration",
    )
    _compile_members(
        _VAE_FILE.read_text(encoding="utf-8"),
        str(_VAE_FILE),
        {"DiagonalGaussianDistribution"},
        namespace,
    )
    return namespace


_NS = _load_production_namespace()
DiagonalGaussianDistribution = _NS["DiagonalGaussianDistribution"]


class _TinyVAE:
    """Records the generator actually used for each posterior sample."""

    dtype = torch.float32
    config = SimpleNamespace()

    def __init__(self, records: list[dict]):
        self.records = records

    def encode(self, images: torch.Tensor):
        # Real posterior shape math, synthetic parameters: mean=pixels, std=1.
        mean = images[:, :1]
        posterior = DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))

        def sample(generator=None):
            latent = posterior.sample(generator)
            self.records.append(
                {
                    "used_seed": None if generator is None else generator.initial_seed(),
                    "latent": latent.detach().clone(),
                }
            )
            return latent

        return SimpleNamespace(latent_dist=SimpleNamespace(sample=sample))


class _TinyHunyuanModel:
    """Binds the extracted production methods to deterministic stubs."""

    _parse_and_validate_image_input = _NS["_parse_and_validate_image_input"]
    _vae_encode = _NS["_vae_encode"]
    embed_multimodal = _NS["embed_multimodal"]
    _mrope_joint_img_sep_token_id = 0

    def __init__(self):
        self.records: list[dict] = []
        self.config = SimpleNamespace(vae_downsample_factor=(1, 1), patch_size=1)
        self.vae = _TinyVAE(self.records)
        self.model = SimpleNamespace(embed_input_ids=lambda ids: torch.zeros((len(ids), 2)))

    def _vit_encode(self, pixel_values, attention_mask, spatial_shapes):
        return torch.zeros((len(pixel_values), 1, 2))

    def time_embed(self, timestep):
        return torch.zeros((1, 2))

    def patch_embed(self, latent, time_embedding):
        return latent.reshape(1, 2, 2), None, None


# Each entry is one independent single-image request: distinct synthetic pixels
# and its own vae_generator_seed, exactly what the processor's
# MultiModalFieldConfig.batched("image") grouping hands to embed_multimodal.
_REQUESTS = {
    "A": {"marker": 1.0, "seed": 42},
    "B": {"marker": 2.0, "seed": 123},
    "C": {"marker": 3.0, "seed": 42},
    "U": {"marker": 5.0, "seed": None},
}


def _request_kwargs(label: str) -> dict:
    """Kwargs payload for one grouped single-image request item."""
    request = _REQUESTS[label]
    kwargs = {
        "vit_pixel_values": torch.full((1, 1, 2), request["marker"]),
        "vit_pixel_attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "vit_spatial_shapes": torch.tensor([[1, 1]]),
        "vae_pixel_values": torch.full((12,), request["marker"]),
        "vae_pixel_size": torch.tensor([12]),
        "vae_token_grid_hw": torch.tensor([[2, 2]]),
    }
    if request["seed"] is not None:
        kwargs["vae_generator_seed"] = torch.tensor([request["seed"]], dtype=torch.long)
    return kwargs


def _batch_requests(labels: list[str]) -> dict:
    """Merge single-request payloads into one batched embed_multimodal call.

    Mirrors what ``group_and_batch_mm_kwargs`` produces for same-shaped items:
    per-image fields are concatenated along the image dimension, and seeded
    requests are never grouped with unseeded ones.
    """
    single = [_request_kwargs(label) for label in labels]
    batch = {
        "vit_pixel_values": torch.cat([kw["vit_pixel_values"] for kw in single], dim=0),
        "vit_pixel_attention_mask": torch.cat([kw["vit_pixel_attention_mask"] for kw in single], dim=0),
        "vit_spatial_shapes": torch.cat([kw["vit_spatial_shapes"] for kw in single], dim=0),
        "vae_pixel_values": torch.cat([kw["vae_pixel_values"] for kw in single], dim=0),
        "vae_pixel_size": torch.cat([kw["vae_pixel_size"] for kw in single], dim=0),
        "vae_token_grid_hw": torch.cat([kw["vae_token_grid_hw"] for kw in single], dim=0),
    }
    if all("vae_generator_seed" in kw for kw in single):
        batch["vae_generator_seed"] = torch.cat([kw["vae_generator_seed"] for kw in single], dim=0)
    return batch


def _run(labels: list[str]) -> list[dict]:
    """Run one embed_multimodal call and return per-request latent records."""
    model = _TinyHunyuanModel()
    outputs = model.embed_multimodal(**_batch_requests(labels))
    assert len(outputs) == len(labels), f"expected {len(labels)} embeddings, got {len(outputs)}"
    assert len(model.records) == len(labels), f"expected {len(labels)} VAE samples, got {len(model.records)}"
    results = []
    for label, record, output in zip(labels, model.records, outputs, strict=True):
        assert record["latent"].shape == (1, 1, 2, 2)
        assert output.shape == (4, 2)
        results.append(
            {
                "label": label,
                "seed": _REQUESTS[label]["seed"],
                "used_seed": record["used_seed"],
                "latent": record["latent"],
                "embedding": output.detach().clone(),
            }
        )
    return results


def _assert_matches_standalone(observed: dict, standalone: dict) -> None:
    assert observed["used_seed"] == standalone["used_seed"], (
        f"{observed['label']}: generator seed {observed['used_seed']} does not match "
        f"standalone seed {standalone['used_seed']}"
    )
    max_diff = (observed["latent"] - standalone["latent"]).abs().max().item()
    assert torch.equal(observed["latent"], standalone["latent"]), (
        f"{observed['label']}: latent depends on batch composition (max_abs_diff={max_diff})"
    )
    assert torch.equal(observed["embedding"], standalone["embedding"]), (
        f"{observed['label']}: embedding depends on batch composition"
    )


def test_single_image_uses_fresh_generator_with_request_seed():
    """Single-image behavior: the latent is the reference posterior sample of a
    fresh generator seeded with the request's own seed."""
    for label in ("A", "B"):
        seed = _REQUESTS[label]["seed"]
        [observed] = _run([label])

        vae_image = torch.full((1, 3, 2, 2), _REQUESTS[label]["marker"])
        mean = vae_image[:, :1]
        posterior = DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))
        reference = posterior.sample(torch.Generator(device="cpu").manual_seed(seed))

        assert observed["used_seed"] == seed, f"{label}: used seed {observed['used_seed']} != request seed {seed}"
        assert torch.equal(observed["latent"], reference), f"{label}: latent is not the seed-{seed} reference sample"


def test_mixed_seed_batch_matches_standalone():
    standalone = {r["label"]: r for r in (_run(["A"])[0], _run(["B"])[0])}
    for observed, expected in zip(_run(["A", "B"]), standalone.values(), strict=True):
        _assert_matches_standalone(observed, expected)


def test_reversed_order_batch_matches_standalone():
    standalone = {r["label"]: r for r in (_run(["B"])[0], _run(["A"])[0])}
    for observed, expected in zip(_run(["B", "A"]), standalone.values(), strict=True):
        _assert_matches_standalone(observed, expected)


def test_same_seed_batch_matches_standalone():
    """Two different images with the same seed must not share one RNG stream."""
    standalone = {r["label"]: r for r in (_run(["A"])[0], _run(["C"])[0])}
    for observed, expected in zip(_run(["A", "C"]), standalone.values(), strict=True):
        _assert_matches_standalone(observed, expected)


def test_three_image_batch_matches_standalone():
    labels = ["A", "B", "C"]
    standalone = {r["label"]: r for r in (_run([label])[0] for label in labels)}
    for observed, expected in zip(_run(labels), standalone.values(), strict=True):
        _assert_matches_standalone(observed, expected)


def test_batch_subset_matches_standalone():
    """A cache-miss subset (one request omitted) must not change the rest."""
    standalone = {r["label"]: r for r in (_run(["B"])[0], _run(["C"])[0])}
    for observed, expected in zip(_run(["B", "C"]), standalone.values(), strict=True):
        _assert_matches_standalone(observed, expected)


@pytest.mark.parametrize("seeds", [torch.tensor([42]), torch.tensor([42, 123, 456])])
def test_seed_count_must_match_image_count(seeds: torch.Tensor):
    kwargs = _batch_requests(["A", "B"])
    kwargs["vae_generator_seed"] = seeds

    model = _TinyHunyuanModel()
    with pytest.raises(
        ValueError,
        match=rf"Number of VAE generator seeds \({seeds.numel()}\) does not match number of VAE images \(2\)",
    ):
        model.embed_multimodal(**kwargs)


def test_unseeded_requests_run_without_generator():
    [single] = _run(["U"])
    assert single["used_seed"] is None

    pair = _run(["U", "U"])
    assert len(pair) == 2
    assert all(record["used_seed"] is None for record in pair)

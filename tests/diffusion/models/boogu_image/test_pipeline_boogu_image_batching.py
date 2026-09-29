# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Boogu-Image request-batching tests: compatibility keys, cross-request
isolation, CFG-parallel dispatch, and variable-length padding.

Split out from ``test_pipeline_boogu_image.py`` to keep that file's
constructor/prompt-encoding focus separate from these request-batch-specific
cases. Shared pipeline/fake builders are imported from there rather than
duplicated.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .test_pipeline_boogu_image import (
    _content_id,
    _EditForwardVAE,
    _FakeScheduler,
    _FakeTransformer,
    _make_edit_forward_pipeline,
    _make_edit_od_config,
    _make_forward_pipeline,
    _sampling,
    _wrap_request_batch,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _RefContentVAE(_EditForwardVAE):
    """Fake VAE whose ``encode`` reflects the input image's mean into the latent."""

    def encode(self, img):
        val = float(img.float().mean().item())
        dist = SimpleNamespace(sample=lambda generator=None: torch.full((1, 4, 8, 8), val))
        return SimpleNamespace(latent_dist=dist)


def _edit_prompt(text, neg, ref_fill):
    import PIL.Image

    return {
        "prompt": text,
        "negative_prompt": neg,
        "additional_information": {
            "prompt_image": PIL.Image.new("RGB", (64, 64)),
            "preprocessed_image": torch.full((1, 3, 64, 64), ref_fill),
        },
    }


# ---------------------------------------------------------------------------
# Request-batch: compatibility key
# ---------------------------------------------------------------------------


def test_boogu_batch_compatibility_key_t2i_and_ti2i_stable_and_isolated():
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import _boogu_batch_compatibility_key

    # Each task shares a stable key; guidance compatibility is checked upstream.
    assert _boogu_batch_compatibility_key(False) == ("boogu_image", "t2i")

    assert _boogu_batch_compatibility_key(True) == ("boogu_image", "ti2i")

    # T2I and TI2I use different denoise paths.
    assert _boogu_batch_compatibility_key(False) != _boogu_batch_compatibility_key(True)


def test_pre_process_key_wiring_t2i_and_ti2i_batch(tmp_path):
    # Verify preprocessing keys reach the scheduler and keep T2I/TI2I apart.
    import PIL.Image

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import get_boogu_image_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))

    def condition_key(prompt, rid):
        req = OmniDiffusionRequest(
            prompt=prompt, sampling_params=OmniDiffusionSamplingParams(height=512, width=512), request_id=rid
        )
        pre(req)
        return build_request_batch_sampling_params_key(req).condition_key

    t2i_a = condition_key({"prompt": "a cat"}, "t-a")
    t2i_b = condition_key({"prompt": "a dog"}, "t-b")
    assert t2i_a == t2i_b and t2i_a[1] == "t2i"

    img = PIL.Image.new("RGB", (64, 64))
    ti2i_a = condition_key({"prompt": "edit", "multi_modal_data": {"image": img}}, "e-a")
    ti2i_b = condition_key({"prompt": "edit", "multi_modal_data": {"image": img}}, "e-b")
    assert ti2i_a == ti2i_b and ti2i_a[1] == "ti2i"
    assert t2i_a != ti2i_a


def test_ti2i_batch_key_distinguishes_explicit_image_guidance(tmp_path):
    import PIL.Image

    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import get_boogu_image_pre_process_func
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched.request_scheduler import build_request_batch_sampling_params_key
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pre = get_boogu_image_pre_process_func(_make_edit_od_config(tmp_path))
    image = PIL.Image.new("RGB", (64, 64))

    def make_request(guidance_scale_2: float | None) -> OmniDiffusionRequest:
        request = OmniDiffusionRequest(
            prompt={"prompt": "edit", "multi_modal_data": {"image": image}},
            sampling_params=OmniDiffusionSamplingParams(
                height=512,
                width=512,
                seed=123,
                guidance_scale=2.0,
                guidance_scale_2=guidance_scale_2,
            ),
            request_id="ti2i",
        )
        pre(request)
        return request

    omitted = build_request_batch_sampling_params_key(make_request(None))
    explicit = build_request_batch_sampling_params_key(make_request(2.0))

    # Omission auto-fills the same value but must not enable Boogu's image-guidance branch.
    assert omitted.condition_key == explicit.condition_key == ("boogu_image", "ti2i")
    assert omitted.guidance_scale_2 == explicit.guidance_scale_2 == 2.0
    assert omitted.guidance_scale_2_provided is False
    assert explicit.guidance_scale_2_provided is True
    assert omitted != explicit


# ---------------------------------------------------------------------------
# Request-batch: cross-request isolation
# ---------------------------------------------------------------------------


def test_forward_batch_isolation_partner_content_and_seed(monkeypatch):
    """Keep A unchanged under B's prompt, negative-prompt, or seed changes.

    Exercise the real CFG/encoding path at B=2 with signal-sensitive fakes.
    """

    class _ContentAwareTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            content = instruction_embeds.mean(dim=(1, 2)).view(-1, 1, 1, 1)
            return latents + content

    class _ApplyingScheduler(_FakeScheduler):
        def step(self, model_output, t, latents, return_dict=False):
            return (model_output,)

    def run(prompt_a, seed_a, neg_a, prompt_b, seed_b, neg_b):
        pipeline = _make_forward_pipeline()
        pipeline.transformer = _ContentAwareTransformer()
        pipeline.scheduler = _ApplyingScheduler()
        kw = dict(height=64, width=64, num_inference_steps=2, guidance_scale=4.0, output_type="latent")
        req = _wrap_request_batch(
            [
                (
                    {"prompt": prompt_a, "negative_prompt": neg_a},
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_a)),
                ),
                (
                    {"prompt": prompt_b, "negative_prompt": neg_b},
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_b)),
                ),
            ]
        )
        return pipeline.forward(req)[0].output

    baseline = run("a cat on a mat", 1, "ugly", "a dog in a park", 2, "blurry")
    assert torch.equal(baseline, run("a cat on a mat", 1, "ugly", "a totally different scene", 2, "blurry"))
    assert torch.equal(baseline, run("a cat on a mat", 1, "ugly", "a dog in a park", 999, "blurry"))
    assert torch.equal(baseline, run("a cat on a mat", 1, "ugly", "a dog in a park", 2, "watermark"))
    # Same-length partner rewrites exercise content rather than length.
    assert torch.equal(baseline, run("a cat on a mat", 1, "ugly", "a fox in a cave", 2, "blurry"))
    assert torch.equal(baseline, run("a cat on a mat", 1, "ugly", "a dog in a park", 2, "grainy"))

    # Sensitivity control: A's own prompt and negative prompt must affect A.
    assert not torch.equal(baseline, run("a totally different scene", 1, "ugly", "a dog in a park", 2, "blurry"))
    assert not torch.equal(baseline, run("a cat on a mat", 1, "watermark", "a dog in a park", 2, "blurry"))

    # Test seed separately: BF16 content magnitudes can hide O(1) noise.
    class _LatentsOnlyTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            return latents

    def run_seed(seed_a, seed_b):
        pipeline = _make_forward_pipeline()
        pipeline.transformer = _LatentsOnlyTransformer()
        pipeline.scheduler = _ApplyingScheduler()
        kw = dict(height=64, width=64, num_inference_steps=2, guidance_scale=4.0, output_type="latent")
        req = _wrap_request_batch(
            [
                ("a cat on a mat", _sampling(**kw, generator=torch.Generator().manual_seed(seed_a))),
                ("a dog in a park", _sampling(**kw, generator=torch.Generator().manual_seed(seed_b))),
            ]
        )
        return pipeline.forward(req)[0].output

    seed_baseline = run_seed(1, 2)
    assert torch.equal(seed_baseline, run_seed(1, 999))  # B's seed must not affect A
    assert not torch.equal(seed_baseline, run_seed(999, 2))  # A's own seed must affect A

    # RED: compare partner seeds under the same reversed-generator mutation.
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    real_collate = DiffusionRequestBatch.collate_request_generators

    def reversed_collate(self, num_outputs_per_prompt, default_generator):
        result = real_collate(self, num_outputs_per_prompt, default_generator)
        return list(reversed(result)) if isinstance(result, list) and len(result) > 1 else result

    monkeypatch.setattr(DiffusionRequestBatch, "collate_request_generators", reversed_collate)
    buggy_seed_baseline = run_seed(1, 2)
    assert not torch.equal(buggy_seed_baseline, run_seed(1, 999))


def test_forward_batched_ti2i_partner_reference_isolation(monkeypatch):
    """Keep A independent of B's content, seed, and reference at B=2.

    Double guidance exercises all three prediction branches; separate fakes
    keep seed/reference sensitivity independent of content magnitude.
    """

    class _ContentRefTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            out = latents + instruction_embeds.mean(dim=(1, 2)).view(-1, 1, 1, 1)
            # Reference latents are request-major: one optional list per row.
            ref = kwargs.get("ref_image_hidden_states")
            if ref is not None:
                ref_means = torch.tensor(
                    [float(r[0].float().mean()) if r is not None else 0.0 for r in ref],
                    dtype=out.dtype,
                    device=out.device,
                ).view(-1, 1, 1, 1)
                out = out + ref_means
            return out

    class _ApplyingScheduler(_FakeScheduler):
        def step(self, model_output, t, latents, return_dict=False):
            return (model_output,)

    def run(prompt_a, neg_a, seed_a, ref_fill_a, prompt_b, neg_b, seed_b, ref_fill_b):
        pipeline = _make_edit_forward_pipeline()
        pipeline.transformer = _ContentRefTransformer()
        pipeline.scheduler = _ApplyingScheduler()
        pipeline.vae = _RefContentVAE()
        kw = dict(
            height=64,
            width=64,
            num_inference_steps=2,
            guidance_scale=5.0,
            guidance_scale_2=2.0,
            output_type="latent",
        )
        req = _wrap_request_batch(
            [
                (
                    _edit_prompt(prompt_a, neg_a, ref_fill_a),
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_a)),
                ),
                (
                    _edit_prompt(prompt_b, neg_b, ref_fill_b),
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_b)),
                ),
            ]
        )
        return pipeline.forward(req)[0].output

    A = ("make it winter", "ugly", 1, 0.25)
    baseline = run(*A, "a dog in a park", "blurry", 2, 0.75)
    assert torch.equal(baseline, run(*A, "a totally different scene", "blurry", 2, 0.75))
    assert torch.equal(baseline, run(*A, "a dog in a park", "watermark", 2, 0.75))
    assert torch.equal(baseline, run(*A, "a dog in a park", "blurry", 999, 0.75))
    assert torch.equal(baseline, run(*A, "a dog in a park", "blurry", 2, 0.5))
    # Same-length partner rewrites exercise content rather than length.
    assert torch.equal(baseline, run(*A, "a fox in a cave", "blurry", 2, 0.75))
    assert torch.equal(baseline, run(*A, "a dog in a park", "grainy", 2, 0.75))

    # Sensitivity control: A's own prompt and negative prompt must affect A.
    B = ("a dog in a park", "blurry", 2, 0.75)
    assert not torch.equal(baseline, run("a totally different scene", "ugly", 1, 0.25, *B))
    assert not torch.equal(baseline, run("make it winter", "watermark", 1, 0.25, *B))

    # Keep reference and seed signals separate from the larger BF16 content term.
    class _RefOnlyTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            ref = kwargs.get("ref_image_hidden_states")
            if ref is None:
                # Branches [r, r, 0] give 2r at image guidance 2.
                # Returning latents would cancel r after two scheduler steps.
                return torch.zeros_like(latents)
            ref_means = torch.tensor(
                [float(r[0].float().mean()) if r is not None else 0.0 for r in ref],
                dtype=latents.dtype,
                device=latents.device,
            ).view(-1, 1, 1, 1)
            return torch.zeros_like(latents) + ref_means

    class _LatentsOnlyTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            return latents

    def run_isolated(transformer_cls, ref_fill_a, seed_a, ref_fill_b, seed_b):
        pipeline = _make_edit_forward_pipeline()
        pipeline.transformer = transformer_cls()
        pipeline.scheduler = _ApplyingScheduler()
        pipeline.vae = _RefContentVAE()
        kw = dict(
            height=64, width=64, num_inference_steps=2, guidance_scale=5.0, guidance_scale_2=2.0, output_type="latent"
        )
        req = _wrap_request_batch(
            [
                (
                    _edit_prompt("make it winter", "ugly", ref_fill_a),
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_a)),
                ),
                (
                    _edit_prompt("a dog in a park", "blurry", ref_fill_b),
                    _sampling(**kw, generator=torch.Generator().manual_seed(seed_b)),
                ),
            ]
        )
        return pipeline.forward(req)[0].output

    ref_baseline = run_isolated(_RefOnlyTransformer, 0.25, 1, 0.75, 2)
    assert torch.equal(ref_baseline, run_isolated(_RefOnlyTransformer, 0.25, 1, 0.5, 2))  # B's ref must not affect A
    assert not torch.equal(
        ref_baseline, run_isolated(_RefOnlyTransformer, 0.75, 1, 0.75, 2)
    )  # A's own ref must affect A

    seed_baseline = run_isolated(_LatentsOnlyTransformer, 0.25, 1, 0.75, 2)
    # Only A's seed changes A.
    assert torch.equal(seed_baseline, run_isolated(_LatentsOnlyTransformer, 0.25, 1, 0.75, 999))
    assert not torch.equal(seed_baseline, run_isolated(_LatentsOnlyTransformer, 0.25, 999, 0.75, 2))

    # RED: compare baseline and partner changes under each active mutation.
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import BooguImagePipeline
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    real_build_ref_latents = BooguImagePipeline._build_ref_latents

    def reversed_build_ref_latents(self, preprocessed_images, num_images_per_prompt, device, generators=None):
        swapped = list(reversed(preprocessed_images))
        return real_build_ref_latents(self, swapped, num_images_per_prompt, device, generators)

    monkeypatch.setattr(BooguImagePipeline, "_build_ref_latents", reversed_build_ref_latents)
    buggy_ref_baseline = run_isolated(_RefOnlyTransformer, 0.25, 1, 0.75, 2)
    assert not torch.equal(buggy_ref_baseline, run_isolated(_RefOnlyTransformer, 0.25, 1, 0.5, 2))
    monkeypatch.undo()

    real_collate = DiffusionRequestBatch.collate_request_generators

    def reversed_collate(self, num_outputs_per_prompt, default_generator):
        result = real_collate(self, num_outputs_per_prompt, default_generator)
        return list(reversed(result)) if isinstance(result, list) and len(result) > 1 else result

    monkeypatch.setattr(DiffusionRequestBatch, "collate_request_generators", reversed_collate)
    buggy_seed_baseline = run_isolated(_LatentsOnlyTransformer, 0.25, 1, 0.75, 2)
    assert not torch.equal(buggy_seed_baseline, run_isolated(_LatentsOnlyTransformer, 0.25, 1, 0.75, 999))


# ---------------------------------------------------------------------------
# Request-batch: CFG-parallel dispatch
# ---------------------------------------------------------------------------


def _patch_cfg_parallel_state(monkeypatch, *, world_size, rank, fake_group):
    """Make ``CFGParallelMixin`` see an active M-rank CFG group.

    ``is_cfg_group_initialized()`` reads ``parallel_state._CFG`` through that
    function's own closure, so patching ``_CFG`` flips it regardless of which
    module imported the function. ``get_classifier_free_guidance_world_size``,
    ``get_classifier_free_guidance_rank``, and ``get_cfg_group`` are each
    rebound by name inside ``cfg_parallel.py`` (``from ...parallel_state
    import ...``), so those three must be patched on both modules or
    ``cfg_parallel``'s copy keeps resolving to the real (single-rank)
    implementation.
    """
    from vllm_omni.diffusion.distributed import cfg_parallel as cfg_parallel_mod
    from vllm_omni.diffusion.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "_CFG", fake_group)
    for module in (parallel_state, cfg_parallel_mod):
        monkeypatch.setattr(module, "get_classifier_free_guidance_world_size", lambda: world_size)
        monkeypatch.setattr(module, "get_classifier_free_guidance_rank", lambda: rank)
        monkeypatch.setattr(module, "get_cfg_group", lambda: fake_group)


class _SlotTableCfgGroup:
    """Fake CFG process group backed by a precomputed per-rank slot table.

    Mirrors the real ``all_gather`` contract: called once per (slot, tensor
    element) in slot-major order. Validates that the calling rank's local
    prediction (``input_``) matches what that rank was supposed to compute
    for this slot, and uses the *actual* ``input_`` (not the precomputed
    value) for its own rank in the returned list -- a dispatch bug (wrong
    branch id, wrong rank assignment) fails the assertion below instead of
    being silently overwritten by the precomputed "correct" table.
    """

    def __init__(self, world_size, rank_in_group, slots_per_rank):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self._slots_per_rank = slots_per_rank
        self.gather_calls = 0

    def all_gather(self, input_, dim=0, separate_tensors=False):
        del dim
        assert separate_tensors
        slot_idx = self.gather_calls
        self.gather_calls += 1
        expected_local = self._slots_per_rank[self.rank_in_group][slot_idx]
        torch.testing.assert_close(
            input_,
            expected_local,
            rtol=0,
            atol=0,
            msg=(
                f"rank {self.rank_in_group} computed the wrong local branch prediction "
                f"for slot {slot_idx} (dispatch bug)"
            ),
        )
        return [
            input_ if rank == self.rank_in_group else self._slots_per_rank[rank][slot_idx]
            for rank in range(self.world_size)
        ]


def _make_counting_predict_noise(real_predict_noise, call_log):
    def counting_predict_noise(**kwargs):
        call_log.append(kwargs)
        return real_predict_noise(**kwargs)

    return counting_predict_noise


def test_forward_batched_ti2i_double_guidance_cfg_parallel_matches_sequential(monkeypatch):
    """CFG-parallel dispatch/gather must reproduce the sequential result.

    Builds a real TI2I B=2 double-guidance request batch (distinct prompts,
    negatives, seeds, and references per row, via the same signal-visible
    fakes as ``test_forward_batched_ti2i_partner_reference_isolation``), then
    replays the exact same branch kwargs through the real
    ``_dispatch_branches``/``all_gather`` path for 2- and 3-rank CFG-parallel
    layouts. A wrongly dispatched branch or a row swap during gather would change
    the result, since request A and B carry distinct content/reference
    signals into every branch.
    """
    from vllm_omni.diffusion.distributed.cfg_parallel import _dispatch_branches
    from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import BooguImagePipeline

    class _ContentRefTransformer(_FakeTransformer):
        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            # FP32 throughout: the encode path is BF16, and the content term
            # (derived from crc32-scaled token ids) reaches several hundred in
            # magnitude, which would round away the 0.25/0.75 reference delta
            # in BF16 and make the with-ref/without-ref branches indistinguishable.
            out = latents.float() + instruction_embeds.float().mean(dim=(1, 2)).view(-1, 1, 1, 1)
            ref = kwargs.get("ref_image_hidden_states")
            if ref is not None:
                ref_means = torch.tensor(
                    [float(r[0].float().mean()) if r is not None else 0.0 for r in ref],
                    dtype=out.dtype,
                    device=out.device,
                ).view(-1, 1, 1, 1)
                out = out + ref_means
            return out

    pipeline = _make_edit_forward_pipeline()
    pipeline.transformer = _ContentRefTransformer()
    pipeline.vae = _RefContentVAE()
    kw = dict(
        height=64,
        width=64,
        num_inference_steps=1,
        guidance_scale=5.0,
        guidance_scale_2=2.0,
        output_type="latent",
    )
    req = _wrap_request_batch(
        [
            (
                _edit_prompt("make it winter", "ugly", 0.25),
                _sampling(**kw, generator=torch.Generator().manual_seed(1)),
            ),
            (
                _edit_prompt("a dog in a park", "blurry", 0.75),
                _sampling(**kw, generator=torch.Generator().manual_seed(2)),
            ),
        ]
    )

    # Step 1: run the real sequential (single-rank) forward once, capturing
    # the exact double-guidance branch kwargs and the resulting model_pred
    # for this request batch's one denoising step.
    captured: dict[str, Any] = {}
    real_multi_branch = BooguImagePipeline.predict_noise_with_multi_branch_cfg

    def capturing_multi_branch(
        self, do_true_cfg, true_cfg_scale, branches_kwargs, cfg_normalize=False, output_slice=None
    ):
        result = real_multi_branch(
            self, do_true_cfg, true_cfg_scale, branches_kwargs, cfg_normalize=cfg_normalize, output_slice=output_slice
        )
        captured.update(
            branches_kwargs=branches_kwargs,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=cfg_normalize,
            output_slice=output_slice,
            sequential_pred=result,
        )
        return result

    monkeypatch.setattr(BooguImagePipeline, "predict_noise_with_multi_branch_cfg", capturing_multi_branch)
    pipeline.forward(req)
    monkeypatch.undo()

    branches_kwargs = captured["branches_kwargs"]
    assert len(branches_kwargs) == 3  # cond+ref, neg+ref, neg+no-ref
    sequential_pred = captured["sequential_pred"]

    # Step 2: replay the identical branches through the real dispatch/gather
    # path for 2- and 3-rank layouts (rank 0: [0, 2] / [0]; rank 1: [1] / [1];
    # rank 2 only exists for the 3-rank layout: [2]).
    n_branches = len(branches_kwargs)
    # Branch predictions are a deterministic function of (pipeline, branches_kwargs)
    # alone; compute once and reuse across both rank-layout replays below.
    with torch.no_grad():
        branch_preds = [pipeline.predict_noise(**branch_kw) for branch_kw in branches_kwargs]

    # If any two branches ever collapsed to the same tensor, a wrongly
    # dispatched or wrongly gathered branch below could go unnoticed even with exact-equality
    # checks -- confirm the fakes are actually signal-visible.
    for i in range(n_branches):
        for j in range(i + 1, n_branches):
            assert not torch.equal(branch_preds[i], branch_preds[j]), (
                f"branches {i} and {j} are numerically identical; the fakes cannot "
                "distinguish a branch mix-up from a correct dispatch"
            )

    for cfg_world_size in (2, 3):
        assignments = _dispatch_branches(n_branches, cfg_world_size)
        max_per_rank = max(len(a) for a in assignments)

        slots_per_rank = []
        for branch_ids in assignments:
            slots = [branch_preds[bid] for bid in branch_ids]
            while len(slots) < max_per_rank:
                slots.append(torch.zeros_like(branch_preds[0]))
            slots_per_rank.append(slots)

        rank_outputs = []
        for cfg_rank in range(cfg_world_size):
            fake_group = _SlotTableCfgGroup(cfg_world_size, cfg_rank, slots_per_rank)
            _patch_cfg_parallel_state(monkeypatch, world_size=cfg_world_size, rank=cfg_rank, fake_group=fake_group)

            call_log: list[dict] = []
            pipeline.predict_noise = _make_counting_predict_noise(pipeline.predict_noise, call_log)
            try:
                with torch.no_grad():
                    rank_pred = pipeline.predict_noise_with_multi_branch_cfg(
                        do_true_cfg=True,
                        true_cfg_scale=captured["true_cfg_scale"],
                        branches_kwargs=branches_kwargs,
                        cfg_normalize=captured["cfg_normalize"],
                        output_slice=captured["output_slice"],
                    )
            finally:
                del pipeline.predict_noise  # drop the instance override, restore the class method
                monkeypatch.undo()

            # A silent fallback to the sequential path (e.g. the CFG-group
            # patch not taking effect) would compute every branch locally and
            # never call all_gather; catch that instead of letting it pass
            # only because sequential also happens to match the baseline.
            expected_calls = len(assignments[cfg_rank])
            assert len(call_log) == expected_calls, (
                f"cfg_world_size={cfg_world_size} rank={cfg_rank} called predict_noise "
                f"{len(call_log)}x, expected {expected_calls}x (silent sequential fallback?)"
            )
            assert fake_group.gather_calls == max_per_rank, (
                f"cfg_world_size={cfg_world_size} rank={cfg_rank} issued {fake_group.gather_calls} "
                f"all_gather call(s), expected {max_per_rank} (silent sequential fallback?)"
            )
            rank_outputs.append(rank_pred)

        for cfg_rank, rank_pred in enumerate(rank_outputs):
            torch.testing.assert_close(
                rank_pred,
                sequential_pred,
                rtol=0,
                atol=0,
                msg=(
                    f"cfg_world_size={cfg_world_size} rank={cfg_rank} diverged from the "
                    "sequential (single-rank) baseline"
                ),
            )


# ---------------------------------------------------------------------------
# Request-batch: variable-length padding
# ---------------------------------------------------------------------------


def test_forward_batched_ti2i_variable_length_partner_padding(monkeypatch):
    """A must be invariant to B's real padded length, not just same-length content.

    ``test_forward_batched_ti2i_partner_reference_isolation`` pins every fake
    encoder call to a fixed ``_SEQ_LEN`` with an all-ones mask, so it never
    exercises production's ``padding="longest"`` batching or a real 0/1
    ``instruction_attention_mask``. Here the fake processor tokenizes by
    content length and right-pads to the batch max, and the fake transformer
    does a masked reduction over ``instruction_attention_mask`` instead of a
    blind ``.mean()`` -- so A's row can only stay correct if the real
    encode/reshape/CFG-duplication plumbing threads the real per-row mask
    through unchanged when B's length (or B's negative-prompt length) grows.
    """

    class _VariableLengthImageAwareProcessor:
        """Tokenizes by word count and right-pads to the batch max length."""

        def __init__(self, force_all_ones_mask=False):
            self.calls = []
            self.force_all_ones_mask = force_all_ones_mask

        def apply_chat_template(self, prompts, **kwargs):
            has_image = []
            rows = []
            for messages in prompts:
                user_content = messages[1]["content"]
                has_image.append(any(c.get("type") == "image" for c in user_content))
                system_text = messages[0]["content"][0]["text"]
                user_text = next(c["text"] for c in messages[1]["content"] if c.get("type") == "text")
                length = 2 + len(user_text.split())
                ids = torch.zeros(length, dtype=torch.long)
                ids[0] = _content_id(system_text)
                ids[1] = _content_id(user_text)
                if length > 2:
                    ids[2:] = torch.arange(2, length)
                rows.append(ids)

            batch = len(prompts)
            max_len = max(len(r) for r in rows)
            input_ids = torch.zeros(batch, max_len, dtype=torch.long)
            attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
            for i, ids in enumerate(rows):
                input_ids[i, : len(ids)] = ids
                attention_mask[i, : len(ids)] = 1
            if self.force_all_ones_mask:
                attention_mask = torch.ones_like(attention_mask)

            self.calls.append(
                {"prompts": prompts, "kwargs": kwargs, "has_image": has_image, "lengths": [len(r) for r in rows]}
            )
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _MaskedContentRefTransformer(_FakeTransformer):
        """Masked-mean over real padding, unlike the fixed-``_SEQ_LEN`` fakes."""

        def __call__(self, latents, timestep, instruction_embeds, freqs_real, instruction_attention_mask, **kwargs):
            mask = instruction_attention_mask.float().unsqueeze(-1)
            valid_counts = mask.sum(dim=1).clamp(min=1.0)
            content = ((instruction_embeds.float() * mask).sum(dim=1) / valid_counts).mean(dim=-1).view(-1, 1, 1, 1)
            out = latents.float() + content
            ref = kwargs.get("ref_image_hidden_states")
            if ref is not None:
                ref_means = torch.tensor(
                    [float(r[0].float().mean()) if r is not None else 0.0 for r in ref],
                    dtype=out.dtype,
                    device=out.device,
                ).view(-1, 1, 1, 1)
                out = out + ref_means
            return out

    class _ApplyingScheduler(_FakeScheduler):
        def step(self, model_output, t, latents, return_dict=False):
            return (model_output,)

    A = _edit_prompt("make it winter", "ugly", 0.25)

    def run(partner=None, force_all_ones_mask=False):
        pipeline = _make_edit_forward_pipeline()
        pipeline.processor = _VariableLengthImageAwareProcessor(force_all_ones_mask=force_all_ones_mask)
        pipeline.transformer = _MaskedContentRefTransformer()
        pipeline.scheduler = _ApplyingScheduler()
        pipeline.vae = _RefContentVAE()
        kw = dict(
            height=64,
            width=64,
            num_inference_steps=1,
            guidance_scale=5.0,
            guidance_scale_2=2.0,
            output_type="latent",
        )
        items = [(A, _sampling(**kw, generator=torch.Generator().manual_seed(1)))]
        if partner is not None:
            items.append((partner, _sampling(**kw, generator=torch.Generator().manual_seed(2))))
        req = _wrap_request_batch(items)
        outs = pipeline.forward(req)
        return outs[0].output, pipeline.processor.calls

    a_alone, _ = run(partner=None)

    short_b = _edit_prompt("a dog", "blurry", 0.75)
    long_b = _edit_prompt("a dog running through a large green park full of tall autumn trees", "blurry", 0.75)
    long_neg_b = _edit_prompt(
        "a dog in a park", "extremely blurry ugly low quality watermark grainy artifact noisy", 0.75
    )

    a_with_short_b, calls_short = run(partner=short_b)
    a_with_long_b, calls_long = run(partner=long_b)
    a_with_long_neg_b, calls_long_neg = run(partner=long_neg_b)

    # calls[0] is the positive-instruction batch call, calls[1] the negative.
    width_short = max(calls_short[0]["lengths"])
    width_long = max(calls_long[0]["lengths"])
    assert width_long > width_short, "B's longer instruction must actually widen the padded batch"

    a_len_in_short = calls_short[0]["lengths"][0]
    a_len_in_long = calls_long[0]["lengths"][0]
    assert a_len_in_short == a_len_in_long  # A's own content length never changes...
    assert width_long > a_len_in_long  # ...but A now sits inside a wider, padded row.

    neg_width_short = max(calls_short[1]["lengths"])
    neg_width_long_neg = max(calls_long_neg[1]["lengths"])
    assert neg_width_long_neg > neg_width_short, "B's longer negative prompt must widen the negative-branch batch too"

    assert torch.equal(a_alone, a_with_short_b)
    assert torch.equal(a_alone, a_with_long_b)
    assert torch.equal(a_alone, a_with_long_neg_b)

    # Negative control: force an all-ones mask (ignore real padding). A's row
    # then sees B's padding-region positions as "real" content, so this must
    # break invariance -- otherwise the checks above could pass vacuously
    # even with no masking at all, exactly like the fixed-_SEQ_LEN fakes.
    a_with_long_b_broken_mask, _ = run(partner=long_b, force_all_ones_mask=True)
    assert not torch.equal(a_alone, a_with_long_b_broken_mask)

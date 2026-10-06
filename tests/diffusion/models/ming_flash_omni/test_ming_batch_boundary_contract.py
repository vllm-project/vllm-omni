# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Guide acceptance contracts for admission and per-row reference/CFG routing.

The scheduler deliberately isolates reference requests. Multi-reference hook
tests therefore verify the model boundary, not production reference admission.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.diffusion.models.ming_flash_omni.test_ming_imagegen_model_contract import (
    distributed as distributed,
)
from tests.diffusion.models.ming_flash_omni.test_ming_imagegen_model_contract import real_pipeline
from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import (
    assert_euler_tick,
    prepared,
    request,
)
from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import (
    pipeline as pipeline,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    get_ming_image_post_process_func,
    get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.diffusion.worker.input_batch import InputBatch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def reference_states(pipe):
    """Actual bridge -> preparation -> VAE reference latents, no fake states."""
    return [
        prepared(
            pipe,
            request(
                rid,
                reference=Image.new("RGB", (16, 16), color),
                seed=seed,
                negative=True,
                extra_args={"cfg_truncation": 0.3},
            ),
        )
        for rid, color, seed in [("A", "red", 11), ("B", "blue", 22), ("C", "green", 33), ("D", "yellow", 44)]
    ]


@pytest.mark.cpu
@pytest.mark.parametrize("broadcast_reference", [False, True])
def test_interleaved_reference_cfg_groups_obey_independent_euler(pipeline, monkeypatch, broadcast_reference):  # noqa: F811 - imported pytest fixture
    """Regression: reference sub-batches must follow row indices, not the first reference.

    Input source: genuine thinker bridge, VAE, prepare_encode and InputBatch.
    Why valid: model hook accepts homogeneous reference geometry at independent ticks.
    Expected behavior/source: declared analytic DiT + Euler equation + inclusive CFG
    threshold; each request owns its reference and scheduler. Both groups have two
    distinguishable rows. The deliberate broadcast variant must fail that same oracle.
    """
    states = reference_states(pipeline)
    with set_forward_context(omni_diffusion_config=pipeline.od_config):
        assert_euler_tick(pipeline, [states[0], states[2]], threshold=0.3)
        ordered = [states[2], states[1], states[0], states[3]]
        assert [s.step_index for s in ordered] == [1, 0, 1, 0]
        assert [float((1000 - s.current_timestep) / 1000) <= 0.3 for s in ordered] == [False, True, False, True]
        refs = [s.extra["ming_reference_latent"] for s in ordered]
        assert all(not torch.equal(a, b) for i, a in enumerate(refs) for b in refs[i + 1 :])
        if broadcast_reference:
            original = pipeline.transformer.forward

            def wrong_forward(*args, **kwargs):
                context = get_forward_context()
                saved = context.ref_latent
                context.ref_latent = saved[:1].expand_as(saved)
                try:
                    return original(*args, **kwargs)
                finally:
                    context.ref_latent = saved

            monkeypatch.setattr(pipeline.transformer, "forward", wrong_forward)
            with pytest.raises(AssertionError):
                assert_euler_tick(pipeline, ordered, threshold=0.3)
        else:
            assert_euler_tick(pipeline, ordered, threshold=0.3)
            assert [s.step_index for s in ordered] == [2, 1, 2, 1]
        assert get_forward_context().ref_latent is None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_actual_dit_interleaved_reference_cfg_groups_preserve_rows(pipeline, distributed, dtype):  # noqa: F811 - imported pytest fixture
    """Regression: different-tick CFG grouping must preserve actual DiT reference rows.

    Input source: genuine bridge/VAE states; real scheduler advances A/C once.
    Expected source: independent request/reference ownership and permutation
    equivariance. Singleton evaluation retains the same physical capacity four.
    CPU companion separately pins the equation and kills first-reference broadcast.
    """
    pipeline.od_config.max_num_seqs = 4
    pipe = real_pipeline(pipeline, dtype)
    states = reference_states(pipe)
    with set_forward_context(omni_diffusion_config=pipe.od_config), torch.inference_mode():
        for state in states[::2]:
            noise = pipe.denoise_step(InputBatch.make_batch([state]))
            pipe.step_scheduler(state, noise)
        ordered = [states[2], states[1], states[0], states[3]]
        assert [s.step_index for s in ordered] == [1, 0, 1, 0]
        assert [float((1000 - s.current_timestep) / 1000) <= 0.3 for s in ordered] == [False, True, False, True]
        singleton = torch.cat([pipe.denoise_step(InputBatch.make_batch([s])) for s in ordered])
        batched = pipe.denoise_step(InputBatch.make_batch(ordered))
        reversed_batch = pipe.denoise_step(InputBatch.make_batch(ordered[::-1]))
        tolerance = 1e-4 if dtype == torch.float32 else 3e-2
        assert torch.isfinite(batched).all() and batched.abs().max() > 0
        torch.testing.assert_close(batched, singleton, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(reversed_batch.flip(0), singleton, rtol=tolerance, atol=tolerance)
        assert get_forward_context().ref_latent is None


@pytest.mark.cpu
@pytest.mark.parametrize(
    "different", [{"height": 32}, {"width": 32}, {"num_inference_steps": 5}, {"guidance_scale": 3.0}]
)
def test_real_admission_separates_geometry_schedule_and_cfg(pipeline, different):  # noqa: F811 - imported pytest fixture
    """Regression: admission must not merge requests with incompatible sampling controls.

    Input source: genuine thinker bridge request -> registered preprocessing ->
    actual RequestScheduler. Changed values are legal public sampling parameters.
    Expected source: guide minimum matrix and scheduler homogeneous-wave contract;
    the first wave must contain A alone, leaving the incompatible B waiting.
    """
    pre = get_ming_image_pre_process_func(pipeline.od_config)
    scheduler = RequestScheduler()
    pipeline.od_config.request_batch_max_wait_ms = 0
    scheduler.initialize(pipeline.od_config)
    scheduler.add_request(pre(request("A")))
    scheduler.add_request(pre(request("B", **different)))
    assert scheduler.schedule().scheduled_request_ids == ["A"]


@pytest.mark.cpu
@pytest.mark.parametrize("relative_path", ["config.json", "custom_vae/config.json"])
def test_config_permission_failure_keeps_path_and_cause(pipeline, monkeypatch, relative_path):  # noqa: F811 - imported pytest fixture
    """Regression: unreadable configuration must not silently become default scale.

    Input source: existing real checkpoint-config fixture used by registered
    postprocessing. Inject only filesystem PermissionError, since the server
    runs as root and chmod cannot reliably reproduce access denial there.
    Expected source: guide B2-06/config minimum matrix: fail clearly with the
    actual path and preserve the OS error, without returning a default processor.
    """
    pipeline.od_config.tf_model_config = None
    denied = Path(pipeline.od_config.model) / relative_path
    original_open = Path.open

    def restricted_open(path, *args, **kwargs):
        if path == denied:
            raise PermissionError(f"access denied: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", restricted_open)
    with pytest.raises(ValueError) as failure:
        get_ming_image_post_process_func(pipeline.od_config)
    assert str(denied) in str(failure.value)
    assert isinstance(failure.value.__cause__, PermissionError)

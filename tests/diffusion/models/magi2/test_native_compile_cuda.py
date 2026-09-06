# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.magi2.test_native_packing import (
    _longer_text_tensors,
    _sampler_tensors,
    _tiny_model,
    _tiny_sampler,
)
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.magi2.preview_data_proxy import Magi2PackedLayout
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _cuda_steps(sampler, layout, tensors=_sampler_tensors):
    timesteps = (torch.tensor([900.0]), torch.tensor([600.0]), torch.tensor([300.0]))
    return [
        sampler.prepare_model_input(
            **{name: value.cuda() for name, value in tensors(seed).items()},
            t=t,
            cfg_config=CFGConfig(),
            layout=layout,
        )
        for seed, t in enumerate(timesteps)
    ]


def test_inductor_regions_match_eager_in_deterministic_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAGI2_DETERMINISTIC", "1")
    model = _tiny_model(params_dtype=torch.bfloat16).cuda()
    sampler = _tiny_sampler(model)
    steps = _cuda_steps(sampler, Magi2PackedLayout())
    expected = [sampler.forward(step) for step in steps]
    new_layout_step = _cuda_steps(sampler, Magi2PackedLayout())[0]
    longer_text_step = _cuda_steps(sampler, Magi2PackedLayout(), _longer_text_tensors)[0]
    expected_new_layout = sampler.forward(new_layout_step)
    expected_longer_text = sampler.forward(longer_text_step)

    torch._dynamo.reset()
    graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    # The pipeline compiles with inductor emulating eager's intermediate
    # low-precision casts, so fused bf16 chains keep eager's rounding.
    model.compile_regions(fullgraph=True, dynamic=True, options={"emulate_precision_casts": True})
    steps = _cuda_steps(sampler, Magi2PackedLayout())
    first = sampler.forward(steps[0])
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled = [first, sampler.forward(steps[1]), sampler.forward(steps[2])]
        repeated = [sampler.forward(step) for step in steps]
        # A second request brings a new layout object; a third changes the
        # text length and therefore every token count.
        new_layout_step.layout = Magi2PackedLayout()
        longer_text_step.layout = Magi2PackedLayout()
        compiled_new_layout = sampler.forward(new_layout_step)
        compiled_longer_text = sampler.forward(longer_text_step)

    for (video, audio), (video_ref, audio_ref) in zip(compiled, expected, strict=True):
        assert torch.equal(video, video_ref)
        torch.testing.assert_close(audio, audio_ref, atol=1e-6, rtol=0)
    for (video, audio), (video_again, audio_again) in zip(compiled, repeated, strict=True):
        assert torch.equal(video, video_again)
        assert torch.equal(audio, audio_again)
    for (video, audio), (video_ref, audio_ref) in (
        (compiled_new_layout, expected_new_layout),
        (compiled_longer_text, expected_longer_text),
    ):
        assert torch.equal(video, video_ref)
        torch.testing.assert_close(audio, audio_ref, atol=1e-6, rtol=0)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] - graphs_before == 7

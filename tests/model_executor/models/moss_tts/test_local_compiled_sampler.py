# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.tts,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_explicit_generator_bypasses_compiled_audio_sampler():
    from transformers import GPT2Config

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
        MossTTSLocalDepthTransformer,
    )

    model = MossTTSLocalDepthTransformer(GPT2Config(n_embd=80, n_head=1, n_inner=160)).cuda().bfloat16().eval()
    heads = torch.nn.ModuleList([torch.nn.Linear(80, 32) for _ in range(2)]).cuda().bfloat16()
    embeddings = torch.nn.ModuleList([torch.nn.Embedding(32, 80) for _ in range(2)]).cuda().bfloat16()
    binary = torch.nn.Linear(80, 2).cuda().bfloat16()
    hidden = torch.randn(1, 80, device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(17)
    expected = model.generate_frame(hidden, heads, embeddings, binary, n_vq=2, generator=generator)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled path selected")

    model._compiled_audio_sampler = forbidden
    generator.manual_seed(17)
    actual = model.generate_frame(hidden, heads, embeddings, binary, n_vq=2, generator=generator)
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    with pytest.raises(AssertionError, match="compiled path selected"):
        model.generate_frame(hidden, heads, embeddings, binary, n_vq=2)

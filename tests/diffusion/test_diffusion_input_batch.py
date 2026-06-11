# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def _make_state(
    request_id: str,
    *,
    rows: int,
    prompt_lens: list[int],
    img_shapes: list,
) -> DiffusionRequestState:
    return DiffusionRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(),
        prompts=[f"prompt-{request_id}-{i}" for i in range(rows)],
        latents=torch.zeros((rows, 4, 8), dtype=torch.float32),
        timesteps=torch.tensor([1.0, 0.5], dtype=torch.float32),
        step_index=0,
        prompt_embeds=torch.ones((rows, 3, 4), dtype=torch.float32),
        prompt_embeds_mask=torch.tensor(
            [[idx < prompt_lens[min(row, len(prompt_lens) - 1)] for idx in range(3)] for row in range(rows)],
            dtype=torch.bool,
        ),
        img_shapes=img_shapes,
        txt_seq_lens=prompt_lens,
    )


def test_input_batch_flattens_multi_prompt_request_metadata() -> None:
    first = _make_state(
        "multi",
        rows=2,
        prompt_lens=[2, 3],
        img_shapes=[["shape-a"], ["shape-b"]],
    )
    second = _make_state(
        "single",
        rows=1,
        prompt_lens=[1],
        img_shapes=[["shape-c"]],
    )

    batch = InputBatch.make_batch([first, second])

    assert batch.latents.shape[0] == 3
    assert batch.prompt_embeds.shape[0] == 3
    assert batch.timesteps.tolist() == [1.0, 1.0, 1.0]
    assert batch.txt_seq_lens == [2, 3, 1]
    assert batch.img_shapes == [["shape-a"], ["shape-b"], ["shape-c"]]


def test_input_batch_repeats_request_metadata_for_multiple_outputs_per_prompt() -> None:
    state = _make_state(
        "multi-output",
        rows=4,
        prompt_lens=[2, 3],
        img_shapes=[["shape-a"], ["shape-b"]],
    )
    state.prompt_embeds_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ],
        dtype=torch.bool,
    )
    state.prompt_embeds = torch.ones((4, 3, 4), dtype=torch.float32)

    batch = InputBatch.make_batch([state])

    assert batch.txt_seq_lens == [2, 2, 3, 3]
    assert batch.img_shapes == [["shape-a"], ["shape-a"], ["shape-b"], ["shape-b"]]

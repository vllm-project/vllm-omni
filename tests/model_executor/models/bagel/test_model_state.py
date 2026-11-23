# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The MRv2 entry point must use BAGEL's prompt positions and preserve IDs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration
from vllm_omni.model_executor.models.bagel.model_state import BagelModelState
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_runner_selects_bagel_state_with_one_position_channel(mocker):
    model = object.__new__(OmniBagelForConditionalGeneration)
    torch.nn.Module.__init__(model)
    cfg = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=3),
        model_config=SimpleNamespace(max_model_len=8192),
    )

    def initialize(self, config, model, encoder_cache, device):
        self.scheduler_config = config.scheduler_config
        self.model_config = config.model_config
        self.max_num_tokens = 4096

    mocker.patch.object(OmniModelState, "__init__", initialize)
    rope = mocker.patch("vllm_omni.model_executor.models.bagel.model_state.RopeState")
    state = init_omni_model_state(cfg, model, None, torch.device("cpu"))

    assert isinstance(state, BagelModelState)
    assert state.rope_state is rope.return_value
    rope.assert_called_once_with(
        num_dims=1, max_num_reqs=3, max_num_tokens=4096, max_model_len=8192, device=torch.device("cpu")
    )


def test_preparation_keeps_static_positions_and_restores_placeholder_ids(mocker):
    state = object.__new__(BagelModelState)
    positions = torch.tensor([[0, 0, 0, 1, 2]])
    embeds = torch.zeros(5, 2)
    base = mocker.patch.object(
        OmniModelState, "prepare_inputs", return_value={"positions": positions, "inputs_embeds": embeds}
    )
    batch = SimpleNamespace(input_ids=torch.tensor([9, 9, 9, 1, 2]), req_ids=[])
    req_states = object()

    inputs = state.prepare_inputs(batch, req_states)

    base.assert_called_once_with(batch, req_states)
    assert inputs["positions"].tolist() == [0, 0, 0, 1, 2]
    assert inputs["positions"].data_ptr() == positions.data_ptr()
    assert inputs["inputs_embeds"] is embeds
    assert inputs["input_ids"] is batch.input_ids


def test_capture_uses_the_same_one_dimensional_position_view(mocker):
    state = object.__new__(BagelModelState)
    positions = torch.zeros(1, 8, dtype=torch.long)
    mocker.patch.object(OmniModelState, "prepare_dummy_inputs", return_value={"positions": positions})
    inputs = state.prepare_dummy_inputs(2, 8)
    assert inputs["positions"].shape == (8,)
    assert inputs["positions"].data_ptr() == positions.data_ptr()


def test_profile_batch_without_images_uses_native_batch_counts(mocker):
    state = object.__new__(BagelModelState)
    state.encoder_cache = EncoderCache()
    batch = InputBatch.make_dummy(2, 8, InputBuffers(2, 8, torch.device("cpu")))
    mocker.patch.object(OmniModelState, "prepare_inputs", return_value={"positions": batch.positions[None]})
    assert state.prepare_inputs(batch, object())["img2img_layouts"] == [None, None]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_cached_edit_and_fresh_edit_keep_their_own_layouts(mocker, cached, reverse):
    from dataclasses import replace

    import numpy as np
    from transformers import BatchFeature
    from vllm.multimodal.inputs import (
        MultiModalFeatureSpec,
        MultiModalFieldConfig,
        MultiModalKwargsItems,
        PlaceholderRange,
    )
    from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

    from vllm_omni.diffusion.models.bagel.image_transforms import vit_size

    state = object.__new__(BagelModelState)
    state.encoder_cache = EncoderCache()
    state._edit_layouts = {}
    state.encoder_runner = object.__new__(EncoderRunner)
    state.encoder_runner.encoder_cache = state.encoder_cache
    state.encoder_runner.enable_timing = False
    # Keep the native input selection and encoder cache. Only the expensive
    # vision inference is replaced; it is irrelevant to layout ownership.
    encode = Mock(side_effect=lambda kwargs: [torch.zeros(1, 2) for _ in kwargs])
    state.encoder_runner.execute_mm_encoder = encode

    def feature(name, height, width):
        nv = (height // 16) * (width // 16) + 2
        vw, vh = vit_size(width, height)
        nt = (vh // 14) * (vw // 14) + 2
        mask = torch.tensor([True] * nv + [False] + [True] * nt)
        data = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature({"pixel_values_img2img": [torch.zeros(3, height, width)]}),
            {"pixel_values_img2img": MultiModalFieldConfig.batched("img2img")},
        )["img2img"][0]
        return MultiModalFeatureSpec(data, "img2img", name, PlaceholderRange(0, nv + 1 + nt, mask))

    a, b = feature("a-image", 1024, 512), feature("b-image", 512, 512)
    if cached:
        state.encoder_cache.add_request("original-a", [a])
        state.execute_mm_encoder({"original-a": [0]})
        state.encoder_cache.remove_request("original-a")
        # The later CFG request may omit the pixel payload on a cache hit.
        a = replace(a, data=None)
    state.encoder_cache.add_request("a", [a])
    state.encoder_cache.add_request("b", [b])
    state.execute_mm_encoder({"a": [0], "b": [0]})
    assert len(encode.call_args.args[0]) == (1 if cached else 2)

    order = [("a", a), ("b", b)]
    if reverse:
        order.reverse()
    counts = np.array([f.mm_position.length + 5 for _, f in order])
    ids = torch.cat([torch.tensor([7] * (n - 5) + [1] * 5) for n in counts])
    positions = torch.cat([torch.arange(n) for n in counts])
    batch = InputBatch.make_dummy(2, len(ids), InputBuffers(2, len(ids), torch.device("cpu")))
    batch.req_ids = [r for r, _ in order]
    batch.input_ids = ids
    batch.num_scheduled_tokens = counts
    req_states = object()
    mocker.patch.object(OmniModelState, "prepare_inputs", side_effect=lambda *_: {"positions": positions[None]})
    inputs = state.prepare_inputs(batch, req_states)

    model = object.__new__(OmniBagelForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._img2img_token_id = 7
    model._ropes_pending = []
    model._ropes_metadata = {}
    result = model._adjust_positions_for_img2img(inputs["positions"], ids, counts, inputs["img2img_layouts"])
    model.flush_pending_metadata(batch.req_ids)
    offset = 0
    for (req_id, f), count in zip(order, counts):
        patches = 2048 if req_id == "a" else 1024
        height = 1024 if req_id == "a" else 512
        assert model._vae_token_mask[offset : offset + count].sum().item() == patches
        assert result[offset : offset + patches + 3].tolist() == [0] * (patches + 3)
        assert result[offset + count - 5 : offset + count].tolist() == [2, 3, 4, 5, 6]
        meta = model.get_kv_transfer_metadata(req_id)
        assert meta == {"ropes": [7], "image_shape": [height, 512], "prefill_position_count": count}
        offset += count

    # A decoded or partially scheduled image no longer needs an encoder
    # layout. Releasing the GPU outputs must also release their metadata.
    state.encoder_cache.free_encoder_cache("a-image")
    state.encoder_cache.free_encoder_cache("b-image")
    state.execute_mm_encoder({})
    assert state._edit_layouts == {}
    batch.num_computed_tokens_np[:] = int(max(counts))
    assert state.prepare_inputs(batch, req_states)["img2img_layouts"] == [None, None]
    batch.num_computed_tokens_np[:] = 0
    batch.num_scheduled_tokens[:] = 1
    assert state.prepare_inputs(batch, req_states)["img2img_layouts"] == [None, None]

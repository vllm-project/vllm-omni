import sys

from vllm.inputs.data import TokensPrompt as _OriginalTokensPrompt
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding as _OriginalMRotaryEmbedding,
)
from vllm.v1.engine import EngineCoreOutput as _OriginalEngineCoreOutput
from vllm.v1.engine import EngineCoreOutputs as _OriginalEngineCoreOutputs
from vllm.v1.engine import EngineCoreRequest as _OriginalEngineCoreRequest
from vllm.v1.request import Request as _OriginalRequest

import vllm_omni.logger  # noqa: F401
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs, OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.layers.mrope import MRotaryEmbedding
from vllm_omni.request import OmniRequest
from vllm_omni.utils import is_npu

for module_name, module in sys.modules.items():
    # only do patch on module of vllm, pass others
    if "vllm" not in module_name:
        continue
    if hasattr(module, "EngineCoreOutput") and module.EngineCoreOutput == _OriginalEngineCoreOutput:
        module.EngineCoreOutput = OmniEngineCoreOutput
    if hasattr(module, "EngineCoreOutputs") and module.EngineCoreOutputs == _OriginalEngineCoreOutputs:
        module.EngineCoreOutputs = OmniEngineCoreOutputs
    if hasattr(module, "TokensPrompt") and module.TokensPrompt == _OriginalTokensPrompt:
        module.TokensPrompt = OmniTokensPrompt
    if hasattr(module, "MRotaryEmbedding") and module.MRotaryEmbedding == _OriginalMRotaryEmbedding:
        module.MRotaryEmbedding = MRotaryEmbedding
    if hasattr(module, "Request") and module.Request == _OriginalRequest:
        module.Request = OmniRequest
    if hasattr(module, "EngineCoreRequest") and module.EngineCoreRequest == _OriginalEngineCoreRequest:
        module.EngineCoreRequest = OmniEngineCoreRequest


# Patch for vllm-ascend prefetch functions bug fix
# Issue: The original functions access forward_context attributes like
# prefetch_mlp_gate_up_proj, prefetch_mlp_down_proj, layer_idx without checking
# if they exist, which causes AttributeError when prefetch_mlp_enabled is not set.
# TODO: Remove this patch after upgrading to vllm-ascend v0.13.0 or later.
# This issue has been fixed in https://github.com/vllm-project/vllm-ascend/pull/5035
if is_npu():
    import torch
    import torch.nn as nn
    from vllm.model_executor.models.qwen2_5_omni_thinker import Qwen2_5_VLImageInputs, Qwen2_5_VLVideoInputs
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    class AscendQwen2_5OmniThinkerForConditionalGeneration(nn.Module):
        def _process_image_input(self, image_input: Qwen2_5_VLImageInputs) -> tuple[torch.Tensor, ...]:
            if image_input["type"] == "image_embeds":
                return image_input["image_embeds"].type(self.visual.dtype)

            grid_thw = image_input["image_grid_thw"]
            assert grid_thw.ndim == 2

            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
            # Split concatenated embeddings for each image item.
            merge_size = self.visual.spatial_merge_size
            sizes = grid_thw.prod(-1) // merge_size // merge_size

            return image_embeds.split(sizes.tolist())

        def _process_video_input(
            self,
            video_input: Qwen2_5_VLVideoInputs,
            video_hashes: list[str] | None = None,
            cached_video_embeds: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if video_input["type"] == "video_embeds":
                return video_input["video_embeds"].type(self.visual.dtype)

            grid_thw = video_input["video_grid_thw"]
            assert grid_thw.ndim == 2

            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            with set_ascend_forward_context(None, self.vllm_config):
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
            # Split concatenated embeddings for each video item.
            merge_size = self.visual.spatial_merge_size
            sizes = grid_thw.prod(-1) // merge_size // merge_size

            return video_embeds.split(sizes.tolist())

    Qwen2_5OmniThinkerForConditionalGeneration._process_image_input = (
        AscendQwen2_5OmniThinkerForConditionalGeneration._process_image_input
    )
    Qwen2_5OmniThinkerForConditionalGeneration._process_video_input = (
        AscendQwen2_5OmniThinkerForConditionalGeneration._process_video_input
    )

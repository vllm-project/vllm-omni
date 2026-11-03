# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-Omni image and audio encoder contract for the upstream CUDA graph manager."""

from collections.abc import Hashable, Mapping
from typing import Any

import torch
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_omni_moe_thinker import _get_feat_extract_output_lengths
from vllm.model_executor.models.utils import scatter_output_slices
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphPathConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from vllm_omni.platforms import current_omni_platform

_FLASH_BACKENDS = {
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.ROCM_AITER_FA,
    AttentionBackendEnum.TRITON_ATTN,
}
# Audio graphs are validated on CUDA with these two backends only.
_AUDIO_BACKENDS = {AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN}
logger = init_logger(__name__)
# Images keep the path name they were captured under before audio existed.
_IMAGE_PATH = "default"
_AUDIO_PATH = "audio"


def _audio_graphs_requested(vllm_config) -> bool:
    """False when the stage's additional_config sets ``audio_encoder_cudagraph: false``."""
    additional = getattr(vllm_config, "additional_config", None)
    value = additional.get("audio_encoder_cudagraph", True) if isinstance(additional, Mapping) else True
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"additional_config.audio_encoder_cudagraph must be true or false, got {value!r}")


def _pad_cu_seqlens(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Unused attention sequences must be empty, not run backwards to zero.
    dst.copy_(src[-1].expand_as(dst))
    dst[: src.shape[0]].copy_(src)


class Qwen3OmniVisionEncoderCudaGraphMixin:
    """Image and audio protocol methods; video retains embed_multimodal.

    Images replay on the ``"default"`` path and audio on the ``"audio"`` path.
    The runner encodes one modality per group, so each replay uses one path.
    The capability attribute is set on eligible model instances after tower
    construction. It is deliberately absent from this mixin's class: the
    runtime protocol checks attribute presence, not the value of a false flag.
    """

    # The eager tower is launch-bound, so a group that needs several replays
    # or per-item eager calls runs as one eager call instead; see
    # SingleReplayEncoderCudaGraphManager.
    encoder_cudagraph_single_replay = True

    def _add_encoder_cudagraph_modality(self, modality: str) -> None:
        modalities = getattr(self, "_encoder_cudagraph_modalities", [])
        if modality not in modalities:
            self._encoder_cudagraph_modalities = [*modalities, modality]
        self.supports_encoder_cudagraph = True

    def _enable_image_encoder_cudagraph(self) -> None:
        # Only the first pipeline rank runs the encoder. Dynamic-scale FP8 ViT
        # attention advances a host-side amax slot per call, which a graph
        # would freeze. The pipeline-group query comes last so a thinker built
        # without a vision tower or distributed state never reaches it.
        if (
            not self.multimodal_config.enable_mm_embeds
            and self.multimodal_config.mm_encoder_attn_dtype is None
            and self.multimodal_config.get_limit_per_prompt("image") > 0
            and getattr(self.visual, "attn_backend", None) in _FLASH_BACKENDS
            and get_pp_group().is_first_rank
        ):
            self._add_encoder_cudagraph_modality("image")

    def _enable_audio_encoder_cudagraph(self, vllm_config) -> None:
        # Same conditions as images. In addition audio stays eager when the
        # encoder runs data parallel (the audio tower itself stays tensor
        # parallel there) and under model runner v2, which builds the upstream
        # manager and never consults encoder_cudagraph_group_eligible; and the
        # largest budget must hold at least one audio chunk. Audio graphs are
        # CUDA only, so ROCm, NPU and XPU take the same eager path. The stage's
        # additional_config can turn audio graphs off while images still replay.
        audio_tower = getattr(self, "audio_tower", None)
        mm = self.multimodal_config
        if mm.enable_mm_embeds or mm.mm_encoder_attn_dtype is not None or mm.get_limit_per_prompt("audio") <= 0:
            return
        reason = None
        if not _audio_graphs_requested(vllm_config):
            reason = "additional_config sets audio_encoder_cudagraph to false"
        elif getattr(vllm_config, "use_v2_model_runner", False):
            reason = "model runner v2 is in use"
        elif mm.mm_encoder_tp_mode == "data":
            reason = "the multimodal encoder runs data parallel"
        elif not current_omni_platform.is_cuda():
            reason = (
                f"audio encoder graphs are validated on CUDA only, this platform is {current_omni_platform.device_name}"
            )
        elif getattr(audio_tower, "attn_backend", None) not in _AUDIO_BACKENDS:
            reason = f"the audio attention backend is {getattr(audio_tower, 'attn_backend', None)}"
        elif self._max_encoder_cudagraph_budget(vllm_config) < self._audio_tokens_per_chunk():
            reason = f"no encoder graph budget holds one {self._audio_tokens_per_chunk()}-token audio chunk"
        if reason is None:
            if get_pp_group().is_first_rank:
                self._add_encoder_cudagraph_modality("audio")
        elif getattr(vllm_config.compilation_config, "cudagraph_mm_encoder", False):
            logger.info("Audio encoder CUDA graphs stay off: %s.", reason)

    def _max_encoder_cudagraph_budget(self, vllm_config) -> int:
        budgets = vllm_config.compilation_config.encoder_cudagraph_token_budgets
        return max(budgets) if budgets else self.get_encoder_cudagraph_budget_range(vllm_config)[1]

    def _audio_tokens_per_chunk(self) -> int:
        tower = self.audio_tower
        return int(tower._get_cnn_output_lengths(tower.n_window * 2))

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        modalities = getattr(self, "_encoder_cudagraph_modalities", ["image"])
        both = len(modalities) > 1
        keys: list[str] = []
        paths: dict[str, EncoderCudaGraphPathConfig] = {}
        padding_logics = {}
        if "image" in modalities:
            keys += ["pixel_values", "rotary_pos_emb_cos", "rotary_pos_emb_sin", "cu_seqlens", "max_seqlen"]
            if self.visual.apply_vit_abs_pos_embed:
                keys.append("pos_embeds")
            paths[_IMAGE_PATH] = EncoderCudaGraphPathConfig(allow_zero_tokens=both)
            padding_logics["cu_seqlens"] = _pad_cu_seqlens
        if "audio" in modalities:
            keys += ["audio_chunks", "audio_valid_index", "audio_cu_seqlens", "audio_max_seqlen"]
            paths[_AUDIO_PATH] = EncoderCudaGraphPathConfig(
                min_token_budget=self._audio_tokens_per_chunk(), allow_zero_tokens=both
            )
            padding_logics["audio_cu_seqlens"] = _pad_cu_seqlens
        # The output width only sizes the data-parallel gather, which audio
        # never uses.
        out_hidden_size = (
            self.visual_dim + self.multiscale_dim if "image" in modalities else self.audio_tower.proj2.output_size
        )
        return EncoderCudaGraphConfig(
            modalities=list(modalities),
            buffer_keys=keys,
            out_hidden_size=out_hidden_size,
            padding_logics=padding_logics,
            paths=paths,
        )

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        return "audio" if "input_audio_features" in mm_kwargs else "image"

    def get_max_frames_per_video(self) -> int:
        return 1

    def get_encoder_cudagraph_budget_range(self, vllm_config) -> tuple[int, int]:
        # Replay pays off while the eager tower is launch-bound: on an A800 a
        # 196-token image replays 32% faster, while 280-400 tokens replay
        # 4-10% slower and 560-784 tokens padded to the 1024 budget 12-51%
        # slower. Every budget also adds captured graph memory. Auto-inferred
        # budgets therefore stop at 256 tokens; larger images run eagerly
        # unless encoder_cudagraph_token_budgets asks for more.
        maximum = min(vllm_config.scheduler_config.max_num_batched_tokens, vllm_config.model_config.max_model_len, 256)
        return min(64, maximum), maximum

    def get_encoder_cudagraph_path_max_budget(self, path: str) -> int | None:
        """Largest budget for one path; None keeps the shared maximum."""
        if path != _AUDIO_PATH:
            return None
        # One attention window, 8 s of audio. Serving Qwen3-Omni on an A800,
        # replay cut TTFT by 20-27% for 1-8 s clips but by 6% for 12 s clips,
        # while the 208- and 256-token graphs grow the audio graph pool from
        # 88 MiB to about 270 MiB, which comes out of the thinker's KV cache.
        tower = self.audio_tower
        return self._audio_tokens_per_chunk() * (tower.n_window_infer // (tower.n_window * 2))

    def get_encoder_cudagraph_path_budgets(self, path: str, user_budgets: list[int]) -> list[int] | None:
        """Budgets for one path from an explicit encoder_cudagraph_token_budgets; None keeps the manager's."""
        if path != _AUDIO_PATH:
            return None
        # Audio graphs fill whole chunks, so each listed budget rounds down to
        # whole chunks and stops at one attention window.
        per_chunk = self._audio_tokens_per_chunk()
        cap = self.get_encoder_cudagraph_path_max_budget(path)
        return sorted({min(budget // per_chunk * per_chunk, cap) for budget in user_budgets if budget >= per_chunk})

    def encoder_cudagraph_group_eligible(self, mm_kwargs: dict[str, Any]) -> bool:
        """Whether a graph reproduces the eager result for this group.

        Graphs pad every audio chunk to full length. Eager pads to the longest
        chunk in the group, and the convolutions read that padding at the last
        valid frames, so a group whose audio items are all shorter than one
        chunk stays eager.
        """
        if self.get_input_modality(mm_kwargs) != "audio":
            return True
        return max(self._audio_lengths(mm_kwargs), default=0) >= self.audio_tower.n_window * 2

    def _audio_lengths(self, mm_kwargs: dict[str, Any]) -> list[int]:
        lengths = mm_kwargs["audio_feature_lengths"]
        return [int(n) for n in (lengths if isinstance(lengths, list) else lengths.tolist())]

    def _image_grid(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
        grid = mm_kwargs["image_grid_thw"]
        return grid if isinstance(grid, list) else grid.tolist()

    def get_encoder_cudagraph_item_specs(self, mm_kwargs: dict[str, Any]) -> list[EncoderItemSpec]:
        if self.get_input_modality(mm_kwargs) == "audio":
            # A graph fills whole chunks: it is sized by the padded token
            # count, while the real count slices the output.
            chunk_size = self.audio_tower.n_window * 2
            per_chunk = self._audio_tokens_per_chunk()
            return [
                EncoderItemSpec(
                    input_size=length,
                    output_tokens=int(_get_feat_extract_output_lengths(length)),
                    path_output_tokens={_AUDIO_PATH: per_chunk * -(-length // chunk_size)},
                )
                for length in self._audio_lengths(mm_kwargs)
            ]
        merge = self.visual.spatial_merge_size
        result = []
        for t, h, w in self._image_grid(mm_kwargs):
            if t != 1 or h <= 0 or w <= 0 or h % merge or w % merge:
                raise ValueError(f"Invalid Qwen3-Omni image grid: {(t, h, w)}")
            result.append(EncoderItemSpec(input_size=h * w, output_tokens=(h // merge) * (w // merge)))
        return result

    def select_encoder_cudagraph_items(self, mm_kwargs: dict[str, Any], indices: list[int]) -> dict[str, Any]:
        if self.get_input_modality(mm_kwargs) == "audio":
            lengths = self._audio_lengths(mm_kwargs)
            features = mm_kwargs["input_audio_features"]
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            selected = {
                "input_audio_features": torch.cat([features[:, offsets[i] : offsets[i + 1]] for i in indices], dim=1)
                if indices
                else features[:, :0],
                "audio_feature_lengths": torch.tensor([lengths[i] for i in indices], dtype=torch.long),
            }
            if (mask := mm_kwargs.get("feature_attention_mask")) is not None:
                selected["feature_attention_mask"] = [mask[i] for i in indices]
            return selected
        grid = self._image_grid(mm_kwargs)
        pixels = mm_kwargs["pixel_values"]
        offsets = [0]
        for spec in self.get_encoder_cudagraph_item_specs(mm_kwargs):
            offsets.append(offsets[-1] + spec.input_size)
        return {
            "pixel_values": torch.cat([pixels[offsets[i] : offsets[i + 1]] for i in indices])
            if indices
            else pixels[:0],
            "image_grid_thw": [grid[i] for i in indices],
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        if axis_keys:
            raise ValueError("Qwen3-Omni encoder graphs use no capture axes")
        if path == _AUDIO_PATH:
            return self._audio_capture_inputs(token_budget, max_batch_size, device, dtype)
        if path != _IMAGE_PATH:
            raise ValueError(f"Unknown Qwen3-Omni encoder graph path: {path!r}")
        visual = self.visual
        merge = visual.spatial_merge_size
        patches = token_budget * merge**2
        # Infer the RoPE layout from one real minimal grid. No large dummy
        # aspect ratio is needed, and no dummy grid indexes past the RoPE cache.
        small = visual.prepare_encoder_metadata([[1, merge, merge]])
        values = {
            "rotary_pos_emb_cos": torch.ones(
                (patches, *small["rotary_pos_emb_cos"].shape[1:]),
                device=device,
                dtype=small["rotary_pos_emb_cos"].dtype,
            ),
            "rotary_pos_emb_sin": torch.zeros(
                (patches, *small["rotary_pos_emb_sin"].shape[1:]),
                device=device,
                dtype=small["rotary_pos_emb_sin"].dtype,
            ),
            "cu_seqlens": torch.full((max_batch_size + 1,), patches, device=device, dtype=torch.int32),
            # This host scalar is baked into the captured attention launch.
            # Any single image in this graph has at most `patches` patches.
            "max_seqlen": torch.tensor(patches, dtype=torch.int32),
        }
        values["cu_seqlens"][0] = 0
        patch_embed = visual.patch_embed
        patch_width = patch_embed.proj.in_channels * patch_embed.temporal_patch_size * patch_embed.patch_size**2
        values["pixel_values"] = torch.zeros((patches, patch_width), device=device, dtype=dtype)
        if visual.apply_vit_abs_pos_embed:
            values["pos_embeds"] = torch.zeros((patches, visual.hidden_size), device=device, dtype=visual.dtype)
        return EncoderCudaGraphCaptureInputs(values=values)

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        if path == _AUDIO_PATH:
            tower = self.audio_tower
            metadata = tower.prepare_encoder_metadata(self._audio_lengths(mm_kwargs))
            features = mm_kwargs["input_audio_features"].to(tower.dtype)
            if max(metadata["chunk_lengths"], default=0) != tower.n_window * 2:
                raise ValueError("audio graph replay needs a full-length chunk; check encoder_cudagraph_group_eligible")
            return EncoderCudaGraphReplayBuffers(
                values={
                    # encoder_cudagraph_group_eligible guarantees a full-length
                    # chunk, so the eager padding width equals the capture's.
                    "audio_chunks": tower.pad_chunks(features, metadata["chunk_lengths"]),
                    "audio_valid_index": metadata["valid_index"],
                    "audio_cu_seqlens": metadata["cu_seqlens"],
                }
            )
        values = self.visual.prepare_encoder_metadata(self._image_grid(mm_kwargs))
        # Replaying a smaller request must not lower the scalar bound retained
        # by the capture. Tensor positions and sequence boundaries do change.
        values.pop("max_seqlen")
        values["pixel_values"] = mm_kwargs["pixel_values"]
        return EncoderCudaGraphReplayBuffers(values=values)

    def encoder_cudagraph_forward(self, inputs: dict[str, torch.Tensor], path: str = "default") -> torch.Tensor:
        if path == _AUDIO_PATH:
            return self.audio_tower.forward_chunks(
                inputs["audio_chunks"],
                {
                    "valid_index": inputs["audio_valid_index"],
                    "cu_seqlens": inputs["audio_cu_seqlens"],
                    "max_seqlen": inputs["audio_max_seqlen"],
                },
            )
        return self.visual(inputs["pixel_values"], None, encoder_metadata=inputs)

    def encoder_eager_forward(self, mm_kwargs: dict[str, Any], path: str = "default") -> torch.Tensor:
        if self.get_input_modality(mm_kwargs) == "audio":
            tower = self.audio_tower
            return tower.forward_with_encoder_metadata(
                mm_kwargs["input_audio_features"].to(tower.dtype),
                tower.prepare_encoder_metadata(self._audio_lengths(mm_kwargs)),
            )
        return self.visual(mm_kwargs["pixel_values"], self._image_grid(mm_kwargs))

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        # A group holds one modality, so exactly one path produced output.
        (output,) = outputs.values()
        scatter_output_slices(output, indices, per_item_out_tokens, dest, clone)

    def _audio_capture_inputs(
        self, token_budget: int, max_batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> EncoderCudaGraphCaptureInputs:
        tower = self.audio_tower
        chunk_size = tower.n_window * 2
        per_chunk = self._audio_tokens_per_chunk()
        num_chunks = token_budget // per_chunk
        window = per_chunk * (tower.n_window_infer // chunk_size)
        # Every item adds at most one partial attention window.
        boundaries = list(range(0, token_budget, window)) + [token_budget]
        cu_seqlens = torch.full((token_budget // window + max_batch_size + 1,), token_budget, dtype=torch.int32)
        cu_seqlens[: len(boundaries)] = torch.tensor(boundaries, dtype=torch.int32)
        return EncoderCudaGraphCaptureInputs(
            values={
                "audio_chunks": torch.zeros((num_chunks, tower.num_mel_bins, chunk_size), device=device, dtype=dtype),
                # Slots past the real tokens gather row 0; scatter drops them.
                "audio_valid_index": torch.arange(token_budget, device=device).clamp_(max=num_chunks * per_chunk - 1),
                "audio_cu_seqlens": cu_seqlens.to(device),
                # This host scalar is baked into the captured attention launch:
                # no attention window is longer than this.
                "audio_max_seqlen": torch.tensor(window, dtype=torch.int32),
            }
        )

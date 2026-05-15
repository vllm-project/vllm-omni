import re

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.layer import Attention

try:
    from dreamid_omni.modules.model import WanLayerNorm, WanRMSNorm
except ImportError:
    raise ImportError("Failed to import from dependency 'dreamid_omni'.")

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.dreamid_omni.wan2_2 import WanModel, rope_apply

logger = init_logger(__name__)


class FusedBlock(nn.Module):
    """Wrapper pairing a video block and audio block for layerwise offloading.

    Registers both blocks as submodules so their parameters are visible to the offload hooks.
    """

    def __init__(
        self,
        vid_block: nn.Module,
        audio_block: nn.Module,
        device: torch.device,
    ):
        super().__init__()
        self.vid_block = vid_block
        self.audio_block = audio_block
        self.device = device

    def _cross_attention_forward(
        self,
        attn: Attention,
        cross_attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        src_ref_lengths=None,
        src_freqs_scaling=None,
        target_ref_lengths=None,
        target_freqs_scaling=None,
    ):
        b, n, d = src_seq.size(0), cross_attn_block.num_heads, cross_attn_block.head_dim
        if hasattr(cross_attn_block, "k_img"):
            q, k, v, k_img, v_img = cross_attn_block.qkv_fn(src_seq, context)
        else:
            q, k, v = cross_attn_block.qkv_fn(src_seq, context)
            k_img = v_img = None

        x = attn(q, k, v)

        if k_img is not None:
            img_x = attn(q, k_img, v_img)
            x = x + img_x

        target_seq = cross_attn_block.pre_attn_norm_fusion(target_seq)
        k_target = cross_attn_block.norm_k_fusion(cross_attn_block.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = cross_attn_block.v_fusion(target_seq).view(b, -1, n, d)

        q = rope_apply(q, src_grid_sizes, src_freqs, ref_lengths=src_ref_lengths, freqs_scaling=src_freqs_scaling)
        k_target = rope_apply(
            k_target,
            target_grid_sizes,
            target_freqs,
            ref_lengths=target_ref_lengths,
            freqs_scaling=target_freqs_scaling,
        )

        target_x = attn(q, k_target, v_target)

        x = x + target_x
        x = x.flatten(2)
        x = cross_attn_block.o(x)
        return x

    def _cross_attention_ffn_forward(
        self,
        attn: Attention,
        attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        src_e,
        src_ref_lengths=None,
        src_freqs_scaling=None,
        target_ref_lengths=None,
        target_freqs_scaling=None,
    ):
        src_seq = src_seq + self._cross_attention_forward(
            attn,
            attn_block.cross_attn,
            attn_block.norm3(src_seq),
            src_grid_sizes=src_grid_sizes,
            src_freqs=src_freqs,
            target_seq=target_seq,
            target_seq_lens=target_seq_lens,
            target_grid_sizes=target_grid_sizes,
            target_freqs=target_freqs,
            context=context,
            context_lens=context_lens,
            src_ref_lengths=src_ref_lengths,
            src_freqs_scaling=src_freqs_scaling,
            target_ref_lengths=target_ref_lengths,
            target_freqs_scaling=target_freqs_scaling,
        )
        y = attn_block.ffn(attn_block.norm2(src_seq).bfloat16() * (1 + src_e[4].squeeze(2)) + src_e[3].squeeze(2))
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            src_seq = src_seq + y * src_e[5].squeeze(2)
        return src_seq

    def forward(
        self,
        vid,
        audio,
        attn: Attention,
        vid_e,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        vid_context,
        vid_context_lens,
        vid_ref_lengths,
        vid_freqs_scaling,
        audio_e,
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
        audio_context,
        audio_context_lens,
        audio_ref_lengths,
        audio_freqs_scaling,
    ):
        vid_block = self.vid_block
        audio_block = self.audio_block

        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], (
            f"{audio_e.shape}, {audio.shape}"
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        # audio self-attention
        audio_y = audio_block.self_attn(
            audio_block.norm1(audio).bfloat16() * (1 + audio_e[1].squeeze(2)) + audio_e[0].squeeze(2),
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            ref_lengths=audio_ref_lengths,
            freqs_scaling=audio_freqs_scaling,
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            audio = audio + audio_y * audio_e[2].squeeze(2)

        ## video modulation
        assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], (
            f"{vid_e.shape}, {vid.shape}"
        )
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)

        # video self-attention
        vid_y = vid_block.self_attn(
            vid_block.norm1(vid).bfloat16() * (1 + vid_e[1].squeeze(2)) + vid_e[0].squeeze(2),
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            ref_lengths=vid_ref_lengths,
        )

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=True):
            vid = vid + vid_y * vid_e[2].squeeze(2)

        og_audio = audio

        # audio cross-attention
        audio = self._cross_attention_ffn_forward(
            attn,
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e,
            src_ref_lengths=audio_ref_lengths,
            src_freqs_scaling=audio_freqs_scaling,
            target_ref_lengths=vid_ref_lengths,
            target_freqs_scaling=None,
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self._cross_attention_ffn_forward(
            attn,
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e,
            src_ref_lengths=vid_ref_lengths,
            src_freqs_scaling=None,
            target_ref_lengths=audio_ref_lengths,
            target_freqs_scaling=audio_freqs_scaling,
        )

        return vid, audio


class FusionModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["fused_blocks"]

    @staticmethod
    def _is_fused_block(name: str, module) -> bool:
        return "fused_blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_fused_block]

    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        has_video = True
        has_audio = True
        self.device = get_local_device()
        if video_config is not None:
            self.video_model = WanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            logger.warning("No video model is provided!")

        if audio_config is not None:
            self.audio_model = WanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            logger.warning("No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.inject_cross_attention_kv_projections()

        self.num_heads = self.video_model.num_heads
        self.head_dim = self.video_model.dim // self.video_model.num_heads
        # Make a single shared instance to pass in at forward time
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

        if has_video and has_audio:
            self.fused_blocks = nn.ModuleList(
                [
                    FusedBlock(
                        self.video_model.blocks[i],
                        self.audio_model.blocks[i],
                        self.device,
                    )
                    for i in range(self.num_blocks)
                ]
            )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Remap checkpoints where blocks are stored under
        `video_model.blocks.N.*` / `audio_model.blocks.N.*` to the current
        `fused_blocks.N.vid_block.*` / `fused_blocks.N.audio_block.*`.
        """
        needs_remap = any(re.match(r"^(video_model|audio_model)\.blocks\.\d+\.", k) for k in state_dict)
        if needs_remap:
            remapped = {}
            for k, v in state_dict.items():
                new_k = re.sub(r"^video_model\.blocks\.(\d+)\.", r"fused_blocks.\1.vid_block.", k)
                new_k = re.sub(r"^audio_model\.blocks\.(\d+)\.", r"fused_blocks.\1.audio_block.", new_k)
                remapped[new_k] = v
            state_dict = remapped

        self._detach_blocks_from_backbones()

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def inject_cross_attention_kv_projections(self):
        for vid_block in self.video_model.blocks:
            vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            )

        for audio_block in self.audio_model.blocks:
            audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()
            )

    def _detach_blocks_from_backbones(self) -> None:
        """Keep offloadable blocks owned only by a single place.

        NOTE: This is a special workaround to support layerwise offloading.
        The model registers the same Wan blocks under both the video/audio
        backbones and `fused_blocks` which is a wrapper for unified blocks
        walking through. However, layerwise offloading will only consider
        `fused_blocks` as offloadable components and will materialize all
        other modules onto device, including the same blocks owned by both
        `fused_blocks` and `video_model` and `audio_model`.
        """
        video_blocks = list(self.video_model.blocks)
        audio_blocks = list(self.audio_model.blocks)
        self.video_model._modules.pop("blocks", None)
        self.audio_model._modules.pop("blocks", None)
        self.video_model.blocks = tuple(video_blocks)
        self.audio_model.blocks = tuple(audio_blocks)

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        ref_ip_lengths=None,
        ref_audio_lengths=None,
        slg_layer=False,
        freqs_scaling=None,
    ):
        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid, t=t, context=vid_context, seq_len=vid_seq_len, ref_lengths=ref_ip_lengths
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio,
            t=t,
            context=audio_context,
            seq_len=audio_seq_len,
            ref_lengths=ref_audio_lengths,
            freqs_scaling=freqs_scaling,
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for fused_block in self.fused_blocks:
            vid, audio = fused_block(vid, audio, self.attn, **kwargs)

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs["grid_sizes"], vid_e)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs["grid_sizes"], audio_e)

        return vid, audio

    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()

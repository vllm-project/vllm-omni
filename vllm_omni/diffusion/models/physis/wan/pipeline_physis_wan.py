# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline, _as_request_list
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func as get_wan22_post_process_func,
)
from vllm_omni.experimental.ar_diffusion.cell_executor import resolve_pp_rank_and_group, run_cell_pipeline
from vllm_omni.experimental.ar_diffusion.cell_schedule import cfg_step_replicas
from vllm_omni.experimental.ar_diffusion.cell_timing import CellTiming

from ..common import build_physis_cell_plan, prepare_video_output, validate_physis_request
from .sampler import WAN21_1_3B, WAN21_14B, WAN22_5B, WanFourStepSampler

PROFILES = {
    "wan21_1p3b": (WAN21_1_3B, (1, 16, 1, 60, 104), 19),
    "wan21_14b": (WAN21_14B, (1, 16, 1, 60, 104), 19),
    "wan22_5b": (WAN22_5B, (1, 48, 1, 30, 52), 29),
}


def _apply_wan_rope(value, cosine, sine):
    even, odd = value.unflatten(-1, (-1, 2)).unbind(-1)
    result = torch.empty_like(value)
    result[..., 0::2] = even * cosine[..., 0::2] - odd * sine[..., 1::2]
    result[..., 1::2] = even * sine[..., 0::2] + odd * cosine[..., 1::2]
    return result


class PhysisStageWanTransformer(StageWanTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for index in range(self.start_layer, self.end_layer):
            block = self._wan.blocks[index]
            for attention in (block.attn1, block.attn2):
                for name in ("norm_q", "norm_k"):
                    original = getattr(attention, name)
                    norm = torch.nn.RMSNorm(
                        original.hidden_size, eps=original.variance_epsilon, elementwise_affine=True
                    )
                    norm.weight = original.weight
                    setattr(attention, name, norm)
            for norm in (block.norm1, block.norm3):
                self._use_fp32_modulation(norm)
        self._use_fp32_modulation(self._wan.norm_out)
        fp32_parts = {"time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"}
        for name, parameter in self._wan.named_parameters():
            if fp32_parts.intersection(name.split(".")):
                parameter.data = parameter.data.float()
        self.sink_pages = {}
        self.capture_sink = False
        self._physis_chunk = 0
        self._physis_layer = 0
        self._physis_latent = None
        self._physis_rope = {}

    @staticmethod
    def _use_fp32_modulation(norm):
        original_forward = norm.forward

        def forward(hidden, scale, shift):
            return original_forward(hidden.float(), scale, shift)

        norm.forward = forward

    def _install_cell_attn_hooks(self):
        owner = self
        for index in range(self.start_layer, self.end_layer):
            attn = self._wan.blocks[index].attn1

            def bind(module):
                def forward(hidden, rotary_emb_arg=None, attn_metadata=None):
                    return owner._self_attention(module, hidden)

                return forward

            attn.forward = bind(attn)

    def prepare_cell(self, encoder_hidden_states, **kwargs):
        super().prepare_cell(encoder_hidden_states, **kwargs)
        self._physis_rope = {}

    def _rotary_emb(self, chunk, latent):
        return None

    def block(self, index, hidden, *, chunk, step, latent, timestep, kv_ctx=None):
        self._physis_chunk = chunk
        self._physis_layer = index
        self._physis_latent = latent
        return super().block(index, hidden, chunk=chunk, step=step, latent=latent, timestep=timestep, kv_ctx=kv_ctx)

    def _tables(self, history_tokens, tokens_per_frame):
        latent = self._physis_latent
        frames = latent.shape[2]
        chunk = self._physis_chunk
        start = 0 if chunk == 0 else chunk + 2
        key = (chunk, history_tokens)
        if key not in self._physis_rope:
            own = self._rope_with_temporal_offset(latent, start_frame=start)
            history_frames = history_tokens // tokens_per_frame
            window_latent = latent.new_empty((1, latent.shape[1], frames + history_frames, *latent.shape[3:]))
            window = self._rope_with_temporal_offset(window_latent, start_frame=start - history_frames)
            self._physis_rope = {key: (own, window)}
        return self._physis_rope[key]

    def _self_attention(self, attn, hidden):
        qkv, _ = attn.to_qkv(hidden)
        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        query = attn.norm_q(query).unflatten(2, (attn.num_heads, attn.head_dim))
        key = attn.norm_k(key).unflatten(2, (attn.num_kv_heads, attn.head_dim))
        value = value.unflatten(2, (attn.num_kv_heads, attn.head_dim))
        ctx = self._cell_kv_ctx
        if self.capture_sink:
            self.sink_pages[self._physis_layer] = (key[0].detach().contiguous(), value[0].detach().contiguous())
        history_tokens = 0
        if ctx is not None:
            ctx.key_pool.index_copy_(0, ctx.video_slots, key[0])
            ctx.value_pool.index_copy_(0, ctx.video_slots, value[0])
            count = ctx.seq_len // ctx.block_size
            blocks = ctx.block_table[0, :count].to(torch.long)
            slots = (
                blocks[:, None] * ctx.block_size + torch.arange(ctx.block_size, device=blocks.device)[None, :]
            ).flatten()
            sink_key, sink_value = self.sink_pages[self._physis_layer]
            key = torch.cat((sink_key, ctx.key_pool.index_select(0, slots)), dim=0).unsqueeze(0)
            value = torch.cat((sink_value, ctx.value_pool.index_select(0, slots)), dim=0).unsqueeze(0)
            history_tokens = key.shape[1] - query.shape[1]
        per_frame = query.shape[1] // self._physis_latent.shape[2]
        own, window = self._tables(history_tokens, per_frame)
        query = _apply_wan_rope(query, *own)
        key = _apply_wan_rope(key, *window)
        output = attn.attn(query, key, value, None).flatten(2, 3).type_as(query)
        return attn.dropout(attn.to_out(output))


class PhysisWanCell(WanCell):
    def __init__(self, transformer, *, sampler, **kwargs):
        super().__init__(transformer, sampler=sampler, **kwargs)

    def _init_noise(self, chunk):
        return self.sampler.initial_noise(
            self.latent_shape, device=self.device, dtype=self.dtype, seed=self.seed, chunk=chunk + 1
        )

    def forward(self, tasks, kv_contexts, *, hidden):
        assert len(tasks) == 1
        req, (chunk, step, block) = tasks[0]
        assert 0 <= step < 4
        self._last_meta = (req, chunk, step)
        live = self._live.setdefault(req, {})
        latent, tokens = self._resolve_latent_hidden(
            req=req, chunk=chunk, step=step, index=0, n_tasks=1, hidden=hidden, live=live
        )
        tf = self.transformer
        timestep = self._timesteps[step]
        if block == tf.start_layer and tf.is_layer_group_first:
            tokens = tf.embed(latent)
        else:
            assert tokens is not None
        tokens = tf.block(
            block,
            tokens,
            chunk=chunk + 1,
            step=step,
            latent=latent,
            timestep=timestep,
            kv_ctx=kv_contexts[0][0],
        )
        if not (block == tf.end_layer - 1 and tf.is_layer_group_last):
            live[chunk] = latent
            return {"latent": latent, "hidden_states": tokens}
        prediction = tf.unembed(tokens, chunk=chunk + 1, step=step, latent=latent, timestep=timestep)
        latent = self.sampler.advance(latent, prediction, step=step, seed=self.seed, chunk=chunk + 1)
        live[chunk] = latent
        if step == 3:
            self.finished.setdefault(req, {})[chunk] = latent.detach()
            if self.on_finished is not None:
                self.on_finished(chunk + 1, latent)
        return {"latent": latent}


class PhysisWanPipeline(CausVidWanPipeline):
    _transformer_cls = PhysisStageWanTransformer

    def __init__(self, *, od_config, prefix=""):
        config = dict(od_config.model_config or {})
        profile_name = config["physis_profile"]
        profile, shape, chunks = PROFILES[profile_name]
        if config.get("causvid_ckpt") or config.get("generator_ckpt"):
            raise ValueError("Physis checkpoints must not receive a CausVid overlay")
        stage_cfg = dict(getattr(od_config, "ar_diffusion_stage_config", None) or {})
        stage_cfg.update(config.get("ar_diffusion_stage_config") or {})
        step_replicas = cfg_step_replicas(stage_cfg)
        pp_world = int(od_config.parallel_config.pipeline_parallel_size)
        if step_replicas != 4 or pp_world < 4 or pp_world % step_replicas:
            raise ValueError("Physis Wan requires step_replicas=4 and a PP world size divisible by 4 (at least 4 GPUs)")
        super().__init__(od_config=od_config, prefix=prefix)
        self.profile = profile
        self.latent_shape = shape
        self.default_chunks = chunks
        self._default_denoise = 4
        self.max_history_chunks = 18
        self.block_size = self.transformer.seq_len_for_latent(shape)
        self.max_chunk_tokens = self.block_size

    def _init_codec(self, model_path):
        device = next(self.transformer.parameters()).device
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        if self.pp_rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(Path(model_path) / "tokenizer", local_files_only=True)
            with torch.device("cpu"):
                self.text_encoder = UMT5EncoderModel.from_pretrained(
                    Path(model_path) / "text_encoder", torch_dtype=torch.bfloat16, local_files_only=True
                ).eval()
            self.vae = (
                DistributedAutoencoderKLWan.from_pretrained(
                    model_path, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True
                )
                .to(device)
                .eval()
            )

    def _encode_prompt(self, prompt, device, dtype):
        _, pp = resolve_pp_rank_and_group()
        if self.pp_rank == 0:
            encoded = super()._encode_prompt(prompt, torch.device("cpu"), dtype).to(device)
        else:
            encoded = torch.empty((1, 512, 4096), device=device, dtype=dtype)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(encoded, src=0)
        return encoded

    def _plan_for(self, req):
        extra = dict(req.sampling_params.extra_args or {})
        if int(extra.get("num_denoise_steps", 4)) != 4:
            raise ValueError("Physis Wan requires its four trained denoising steps")
        if str(extra.get("cell_schedule", "same_step")) != "same_step":
            raise ValueError("Physis Wan requires Same-Step history")
        if int(extra.get("kv_history_chunks", 18)) != 18:
            raise ValueError("Physis Wan requires 18 recent history chunks plus its clean sink")
        chunks = int(extra.get("num_chunks", self.default_chunks))
        if chunks < 2:
            raise ValueError("Use at least two chunks for Physis Wan Cell execution")
        validate_physis_request(req.sampling_params, num_frames=4 * chunks + 5)
        plan = build_physis_cell_plan(
            chunks=chunks,
            step_replicas=self.step_replicas,
            layer_groups=self.layer_groups,
            blocks=self.transformer.num_layers,
            blocks_per_rank=self.transformer.local_num_layers,
        )
        return plan, extra, 4

    def _sink_forward(self, latent, timestep, *, step, pp):
        tf = self.transformer
        group = self.pp_rank % self.layer_groups
        if group == 0:
            hidden = tf.embed(latent)
        else:
            hidden = pp.recv_tensor_dict(src=self.pp_rank - 1)["hidden_states"]
        for block in range(tf.start_layer, tf.end_layer):
            hidden = tf.block(block, hidden, chunk=0, step=step, latent=latent, timestep=timestep, kv_ctx=None)
        if group < self.layer_groups - 1:
            pp.send_tensor_dict({"hidden_states": hidden}, dst=self.pp_rank + 1)
            result = torch.empty_like(latent)
        else:
            result = tf.unembed(hidden, chunk=0, step=step, latent=latent, timestep=timestep)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(result, src=self.layer_groups - 1)
        return result

    def forward(self, req):
        assert self._chunk_ctx is not None
        _, pp = resolve_pp_rank_and_group()
        param = next(self.transformer.parameters())
        device, dtype = param.device, torch.bfloat16
        outputs = []
        for item in _as_request_list(req):
            plan, extra, _ = self._plan_for(item)
            seed = int(extra.get("seed", item.sampling_params.seed or 0))
            request_id = str(item.request_id)
            prompt = self._encode_prompt(self._prompt_text(item.prompt), device, dtype)
            tf = self.transformer
            tf.sink_pages.clear()
            tf.capture_sink = False
            tf.prepare_cell(prompt)
            timing = CellTiming(bool(extra.get("collect_chunk_timing", False)))
            sampler = WanFourStepSampler(self.profile)
            shape = (*self.latent_shape[:2], 3, *self.latent_shape[3:])
            latent = sampler.initial_noise(shape, device=device, dtype=dtype, seed=seed, chunk=0)
            sink_timesteps = [torch.tensor(t, dtype=torch.float32) for t in self.profile.timesteps]
            for step, timestep in enumerate(sink_timesteps):
                prediction = self._sink_forward(latent, timestep, step=step, pp=pp)
                latent = sampler.advance(latent, prediction, step=step, seed=seed, chunk=0)
            sink_latent = latent
            timing.mark(0)
            tf.capture_sink = True
            self._sink_forward(latent, torch.zeros((), dtype=torch.float32), step=4, pp=pp)
            tf.capture_sink = False
            self._chunk_ctx.enqueue(request_id, plan, chunk_tokens=self.block_size)
            adapter = PhysisWanCell(
                tf,
                sampler=sampler,
                prompt_embeds=prompt,
                latent_shape=self.latent_shape,
                seed=seed,
                device=device,
                dtype=dtype,
                on_finished=timing.mark,
            )
            run_cell_pipeline(ctx=self._chunk_ctx, adapter=adapter)
            finished = self._gather_finished(adapter.finished, pp)
            durations = timing.durations(pp)
            if self.pp_rank == 0:
                chunks = finished[request_id]
                ordered = [sink_latent, *(chunks[index] for index in range(plan.schedule.chunks))]
                video = self._decode_latents(torch.cat(ordered, dim=2))
                result = prepare_video_output(self._video_output(video), output_type=item.sampling_params.output_type)
                result.stage_durations = durations
                outputs.append(result)
            else:
                outputs.append(self._placeholder_output(device, dtype))
            tf.sink_pages.clear()
        return outputs

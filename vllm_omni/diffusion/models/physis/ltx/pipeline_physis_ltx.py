# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import torch
import transformers
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_ltx2 import (
    DistributedAutoencoderKLLTX2Video,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import (
    CausVidWanPipeline,
    _as_request_list,
)
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.diffusion.models.physis.common import (
    build_physis_cell_plan,
    prepare_video_output,
    validate_physis_request,
)
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    get_wan22_post_process_func as get_wan22_post_process_func,
)
from vllm_omni.experimental.ar_diffusion.cell_executor import (
    resolve_pp_rank_and_group,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import cfg_step_replicas
from vllm_omni.experimental.ar_diffusion.cell_timing import CellTiming
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec

from .sampler import LTXFourStepSampler
from .transformer import StageLTXTransformer


class PhysisLTXCell(WanCell):
    def _init_noise(self, chunk):
        return self.sampler.initial_noise(self.latent_shape, device=self.device, seed=self.seed, chunk=chunk + 1)

    def activation_spec(self, *, batch, include_hidden):
        spec = super().activation_spec(batch=batch, include_hidden=include_hidden)
        return [(name, shape, torch.bfloat16 if name == "hidden_states" else dtype) for name, shape, dtype in spec]

    def forward(self, tasks, kv_contexts, *, hidden):
        assert len(tasks) == 1
        request, (chunk, step, block) = tasks[0]
        assert 0 <= step < 4
        self._last_meta = request, chunk, step
        live = self._live.setdefault(request, {})
        latent, tokens = self._resolve_latent_hidden(
            req=request,
            chunk=chunk,
            step=step,
            index=0,
            n_tasks=1,
            hidden=hidden,
            live=live,
        )
        tf, timestep = self.transformer, self._timesteps[step]
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
            self.finished.setdefault(request, {})[chunk] = latent.detach()
            if self.on_finished is not None:
                self.on_finished(chunk + 1, latent)
        return {"latent": latent}


class PhysisLTXPipeline(CausVidWanPipeline):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "connectors"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config, prefix=""):
        nn.Module.__init__(self)
        del prefix
        self.od_config = od_config
        config = dict(od_config.model_config or {})
        stage = {
            **dict(getattr(od_config, "ar_diffusion_stage_config", None) or {}),
            **dict(config.get("ar_diffusion_stage_config") or {}),
        }
        self.step_replicas = cfg_step_replicas(stage)
        pp_world = int(od_config.parallel_config.pipeline_parallel_size)
        if self.step_replicas != 4 or pp_world < 4 or pp_world % self.step_replicas:
            raise ValueError("Physis LTX requires step_replicas=4 and a PP world size divisible by 4 (at least 4 GPUs)")
        self.layer_groups = pp_world // self.step_replicas
        if int(od_config.parallel_config.tensor_parallel_size) != 1:
            raise ValueError("Physis LTX Cell execution currently requires tensor_parallel_size=1")
        self.pp_rank, _ = resolve_pp_rank_and_group()
        self.model_path = str(od_config.model)
        root = Path(self.model_path)
        transformer_config = json.loads((root / "transformer" / "config.json").read_text())
        self.transformer = StageLTXTransformer(
            transformer_config,
            layer_groups=self.layer_groups,
            pp_rank=self.pp_rank,
        )
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model_path,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        self.latent_shape = (1, 128, 1, 15, 26)
        self.default_chunks, self.max_history_chunks = 14, 18
        self.block_size = self.max_chunk_tokens = 15 * 26
        self._chunk_ctx = None
        self.tokenizer = self.text_encoder = self.connectors = self.vae = None
        if self.pp_rank == 0:
            index = json.loads((root / "model_index.json").read_text())
            with torch.device("cpu"):
                self.tokenizer = getattr(transformers, index["tokenizer"][1]).from_pretrained(
                    root / "tokenizer",
                    local_files_only=True,
                )
                self.text_encoder = (
                    getattr(transformers, index["text_encoder"][1])
                    .from_pretrained(
                        root / "text_encoder",
                        dtype=torch.bfloat16,
                        local_files_only=True,
                    )
                    .eval()
                    .requires_grad_(False)
                )
                self.connectors = (
                    LTX2TextConnectors.from_pretrained(
                        root,
                        subfolder="connectors",
                        torch_dtype=torch.bfloat16,
                        local_files_only=True,
                    )
                    .eval()
                    .requires_grad_(False)
                )
            device = next(self.transformer.parameters()).device
            self.vae = (
                DistributedAutoencoderKLLTX2Video.from_pretrained(
                    root,
                    subfolder="vae",
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
                .to(device)
                .eval()
                .requires_grad_(False)
            )

    def load_weights(self, weights):
        return self.transformer.load_weights(weights)

    def ar_diffusion_noisy_kv_spec(self):
        return ARDiffusionNoisyKVSpec(
            num_layers=self.transformer.local_num_layers,
            num_kv_heads=self.transformer.num_heads,
            head_size=self.transformer.head_dim,
            block_size=self.block_size,
            max_chunk_tokens=self.max_chunk_tokens,
            max_history_chunks=18,
        )

    def _encode_prompt(self, prompt, device, dtype):
        del dtype
        _, pp = resolve_pp_rank_and_group()
        if self.pp_rank == 0:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            with torch.device("cpu"):
                tokens = self.tokenizer(
                    [prompt.strip()],
                    padding="max_length",
                    max_length=1024,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                hidden = self.text_encoder.base_model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                ).hidden_states
                stacked = torch.stack(hidden, dim=-1).flatten(2, 3).to(torch.bfloat16)
                video, _, mask = self.connectors(stacked, tokens.attention_mask, padding_side="left")
                encoded = video[:, mask[0].bool()].float().to(device)
            shape = torch.tensor(encoded.shape, device=device, dtype=torch.long)
        else:
            shape = torch.empty(3, device=device, dtype=torch.long)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(shape, src=0)
        if self.pp_rank != 0:
            encoded = torch.empty(tuple(shape.tolist()), device=device, dtype=torch.float32)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(encoded, src=0)
        return encoded

    def _plan_for(self, req):
        extra = dict(req.sampling_params.extra_args or {})
        if int(extra.get("num_denoise_steps", 4)) != 4:
            raise ValueError("Physis LTX requires four trained denoising steps")
        if str(extra.get("cell_schedule", "same_step")) != "same_step":
            raise ValueError("Physis LTX requires Same-Step history")
        if int(extra.get("kv_history_chunks", 18)) != 18:
            raise ValueError("Physis LTX requires 18 recent history chunks plus its clean sink")
        chunks = int(extra.get("num_chunks", self.default_chunks))
        if chunks < 2:
            raise ValueError("Use at least two chunks for Physis LTX Cell execution")
        validate_physis_request(req.sampling_params, num_frames=8 * chunks + 9, frame_rate=24.0)
        plan = build_physis_cell_plan(
            chunks=chunks,
            step_replicas=self.step_replicas,
            layer_groups=self.layer_groups,
            blocks=self.transformer.num_layers,
            blocks_per_rank=self.transformer.local_num_layers,
        )
        return plan, extra

    def _sink_forward(self, latent, timestep, *, step, pp):
        tf = self.transformer
        group = self.pp_rank % self.layer_groups
        if group == 0:
            hidden = tf.embed(latent)
        else:
            hidden = pp.recv_tensor_dict(src=self.pp_rank - 1)["hidden_states"]
        for block in range(tf.start_layer, tf.end_layer):
            hidden = tf.block(block, hidden, chunk=0, step=step, latent=latent, timestep=timestep)
        if group < self.layer_groups - 1:
            pp.send_tensor_dict({"hidden_states": hidden}, dst=self.pp_rank + 1)
            prediction = torch.empty_like(latent)
        else:
            prediction = tf.unembed(hidden, chunk=0, step=step, latent=latent, timestep=timestep).float()
        if pp is not None and pp.world_size > 1:
            pp.broadcast(prediction, src=self.layer_groups - 1)
        return prediction

    def _decode_latents(self, latent):
        device = next(self.vae.parameters()).device
        mean = self.vae.latents_mean.float().view(1, -1, 1, 1, 1)
        std = self.vae.latents_std.float().view(1, -1, 1, 1, 1)
        latent = latent.to(device).float() * std / self.vae.config.scaling_factor + mean
        temb = torch.zeros(1, device=device, dtype=torch.bfloat16) if self.vae.config.timestep_conditioning else None
        return self.vae.decode(latent.to(torch.bfloat16), temb, return_dict=False)[0].clamp(-1, 1)

    def forward(self, req):
        assert self._chunk_ctx is not None
        _, pp = resolve_pp_rank_and_group()
        device = next(self.transformer.parameters()).device
        outputs = []
        for item in _as_request_list(req):
            plan, extra = self._plan_for(item)
            seed = int(extra.get("seed", item.sampling_params.seed or 0))
            request_id = str(item.request_id)
            prompt = self._encode_prompt(self._prompt_text(item.prompt), device, torch.float32)
            tf = self.transformer
            tf.sink_pages.clear()
            tf.capture_sink = False
            tf.prepare_cell(prompt)
            timing = CellTiming(bool(extra.get("collect_chunk_timing", False)))
            sampler = LTXFourStepSampler()
            shape = (1, 128, 3, 15, 26)
            latent = sampler.initial_noise(shape, device=device, seed=seed, chunk=0)
            for step, timestep in enumerate(sampler.timesteps):
                prediction = self._sink_forward(
                    latent,
                    torch.tensor(timestep, dtype=torch.float32, device="cpu"),
                    step=step,
                    pp=pp,
                )
                latent = sampler.advance(latent, prediction, step=step, seed=seed, chunk=0)
            sink_latent = latent
            timing.mark(0)
            tf.capture_sink = True
            self._sink_forward(
                latent,
                torch.zeros((), dtype=torch.float32, device="cpu"),
                step=4,
                pp=pp,
            )
            tf.capture_sink = False
            self._chunk_ctx.enqueue(request_id, plan, chunk_tokens=self.block_size)
            adapter = PhysisLTXCell(
                tf,
                sampler=sampler,
                prompt_embeds=prompt,
                latent_shape=self.latent_shape,
                seed=seed,
                device=device,
                dtype=torch.float32,
                on_finished=timing.mark,
            )
            run_cell_pipeline(ctx=self._chunk_ctx, adapter=adapter)
            finished = self._gather_finished(adapter.finished, pp)
            durations = timing.durations(pp)
            if self.pp_rank == 0:
                chunks = finished[request_id]
                latent = torch.cat(
                    [
                        sink_latent,
                        *(chunks[index] for index in range(plan.schedule.chunks)),
                    ],
                    dim=2,
                )
                result = prepare_video_output(
                    self._video_output(self._decode_latents(latent)), output_type=item.sampling_params.output_type
                )
                result.stage_durations = durations
                outputs.append(result)
            else:
                outputs.append(self._placeholder_output(device, torch.bfloat16))
            tf.sink_pages.clear()
        return outputs

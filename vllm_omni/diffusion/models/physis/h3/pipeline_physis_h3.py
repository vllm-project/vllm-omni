# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from dataclasses import replace
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import AutoencoderKLMiniMaxH3
from torch import nn
from transformers import AutoTokenizer

from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline, _as_request_list
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.physis.common import (
    build_physis_cell_plan,
    prepare_video_output,
    validate_physis_request,
)
from vllm_omni.diffusion.models.physis.wan.pipeline_physis_wan import PhysisWanCell
from vllm_omni.diffusion.models.physis.wan.sampler import WAN22_5B, WanFourStepSampler
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import get_wan22_post_process_func as get_wan22_post_process_func
from vllm_omni.experimental.ar_diffusion.cell_executor import resolve_pp_rank_and_group, run_cell_pipeline
from vllm_omni.experimental.ar_diffusion.cell_schedule import cfg_step_replicas
from vllm_omni.experimental.ar_diffusion.cell_timing import CellTiming
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec

from .positions import output_frames
from .transformer import PhysisStageH3Transformer

H3_PROFILE = replace(WAN22_5B, name="minimax_h3", timesteps=tuple(1.0 - sigma for sigma in WAN22_5B.sigmas[:-1]))
LATENT_SHAPE = (1, 24, 1, 30, 52)
SINK_SIGMA = 0.001


class PhysisH3Cell(PhysisWanCell):
    def activation_spec(self, *, batch, include_hidden):
        spec = super().activation_spec(batch=batch, include_hidden=False)
        if include_hidden:
            tf = self.transformer
            tokens = tf.text_tokens + tf.seq_len_for_latent(self.latent_shape)
            spec.append(("hidden_states", (batch, tokens, tf.dim), self.dtype))
        return spec


class PhysisH3Pipeline(CausVidWanPipeline):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae"]

    def __init__(self, *, od_config, prefix=""):
        nn.Module.__init__(self)
        self.od_config = od_config
        self.model_path = str(od_config.model)
        model_cfg = dict(od_config.model_config or {})
        stage_cfg = dict(getattr(od_config, "ar_diffusion_stage_config", None) or {})
        stage_cfg.update(model_cfg.get("ar_diffusion_stage_config") or {})
        self.step_replicas = cfg_step_replicas(stage_cfg)
        parallel = od_config.parallel_config
        if (
            int(parallel.tensor_parallel_size) != 1
            or int(parallel.ulysses_degree) != 1
            or int(parallel.ring_degree) != 1
        ):
            raise ValueError("Physis H3 Cell execution currently requires TP1/SP1")
        if int(parallel.cfg_parallel_size) != 1:
            raise ValueError("Physis H3 is guidance distilled and requires CFG1")
        world = int(parallel.pipeline_parallel_size)
        if self.step_replicas != 4 or world < 4 or world % self.step_replicas:
            raise ValueError("Physis H3 requires step_replicas=4 and a PP world size divisible by 4 (at least 4 GPUs)")
        self.layer_groups = world // self.step_replicas
        self.pp_rank, _ = resolve_pp_rank_and_group()
        self.device = get_local_device()
        config = json.loads((Path(self.model_path) / "transformer" / "config.json").read_text(encoding="utf-8"))
        self.transformer = PhysisStageH3Transformer(
            od_config, config, layer_groups=self.layer_groups, pp_rank=self.pp_rank
        )
        self.latent_shape = LATENT_SHAPE
        self.default_chunks = 48
        self.max_history_chunks = 18
        self.block_size = self.transformer.seq_len_for_latent(self.latent_shape)
        self.max_chunk_tokens = self.block_size
        self._chunk_ctx = None
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        if self.pp_rank == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(Path(self.model_path) / "tokenizer", local_files_only=True)
            # The encoder is retained on CPU, leaving HBM for the DiT and Same-Step KV.
            with torch.device("cpu"):
                self.text_encoder = (
                    MiniMaxH3Qwen3VLEncoder(
                        str(Path(self.model_path) / "text_encoder"),
                        device=torch.device("cpu"),
                        load_model=True,
                        encoder_group=SimpleNamespace(rank_in_group=0, world_size=1),
                    )
                    .eval()
                    .requires_grad_(False)
                )
                self.vae = (
                    AutoencoderKLMiniMaxH3.from_pretrained(
                        Path(self.model_path) / "vae", torch_dtype=torch.float32, local_files_only=True
                    )
                    .eval()
                    .requires_grad_(False)
                )
            self.vae.to(self.device)
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=self.model_path,
                    subfolder="text_encoder",
                    revision=od_config.revision,
                    prefix="text_encoder.",
                    fall_back_to_pt=False,
                )
            )

    def load_weights(self, weights):
        # Source iterators are ordered by weights_sources; split without retaining a checkpoint in RAM.
        for prefix, group in groupby(weights, key=lambda item: item[0].partition(".")[0]):
            entries = ((name.partition(".")[2], tensor) for name, tensor in group)
            if prefix == "transformer":
                self.transformer.load_weights(entries)
            elif prefix == "text_encoder" and self.text_encoder is not None:
                self.text_encoder.load_weights(entries)
            else:
                raise ValueError(f"Unknown Physis H3 checkpoint source {prefix}")
        # Both native components enforce strict completeness themselves; the VAE loaded eagerly.
        return None

    def ar_diffusion_noisy_kv_spec(self):
        return ARDiffusionNoisyKVSpec(
            num_layers=self.transformer.local_num_layers,
            num_kv_heads=self.transformer.num_heads,
            head_size=self.transformer.head_dim,
            block_size=self.block_size,
            max_chunk_tokens=self.block_size,
            max_history_chunks=18,
        )

    def _encode_prompt(self, prompt, device, dtype):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Physis H3 requires a nonempty text prompt")
        _, pp = resolve_pp_rank_and_group()
        if self.pp_rank == 0:
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            encoded = self.text_encoder.encode_ids(torch.tensor(ids, dtype=torch.long, device="cpu"))
            encoded = encoded[None].to(device=device, dtype=dtype)
            length = torch.tensor([encoded.shape[1]], device=device, dtype=torch.long)
        else:
            length = torch.empty(1, device=device, dtype=torch.long)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(length, src=0)
            if self.pp_rank != 0:
                encoded = torch.empty((1, int(length.item()), 5120), device=device, dtype=dtype)
            pp.broadcast(encoded, src=0)
        return encoded

    def _plan_for(self, req):
        extra = dict(req.sampling_params.extra_args or {})
        if int(extra.get("num_denoise_steps", 4)) != 4:
            raise ValueError("Physis H3 requires four trained denoising passes")
        if str(extra.get("cell_schedule", "same_step")) != "same_step":
            raise ValueError("Physis H3 requires Same-Step history")
        if int(extra.get("kv_history_chunks", 18)) != 18:
            raise ValueError("Physis H3 requires 18 recent chunks plus a permanent clean sink")
        chunks = int(extra.get("num_chunks", 48))
        if chunks < 3:
            raise ValueError("H3 whole-sequence VAE decoding requires at least three latent chunks")
        validate_physis_request(req.sampling_params, num_frames=output_frames(chunks))
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
            prediction = torch.empty_like(latent, dtype=torch.float32)
        else:
            prediction = tf.unembed(hidden, chunk=0, step=step, latent=latent, timestep=timestep)
        if pp is not None and pp.world_size > 1:
            pp.broadcast(prediction, src=self.layer_groups - 1)
        return prediction

    def _decode_latents(self, latents):
        device = next(self.vae.parameters()).device
        latents = latents.to(device=device, dtype=torch.float32)
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            decoded = self.vae.decode(latents * std + mean, return_dict=False)[0]
        pixel_mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1)
        pixel_std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1)
        video = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1).mul(2).sub(1)
        if video.shape[2] != output_frames(latents.shape[2]):
            raise ValueError(f"H3 VAE returned {video.shape[2]} frames for {latents.shape[2]} latent chunks")
        return video

    @torch.inference_mode()
    def forward(self, req):
        if self._chunk_ctx is None:
            raise RuntimeError("Physis H3 requires the native AR-Diffusion Cell execution context")
        _, pp = resolve_pp_rank_and_group()
        outputs = []
        for item in _as_request_list(req):
            plan, extra = self._plan_for(item)
            seed = int(extra.get("seed", item.sampling_params.seed or 0))
            request_id = str(item.request_id)
            dtype = torch.bfloat16
            prompt = self._encode_prompt(self._prompt_text(item.prompt), self.device, dtype)
            tf = self.transformer
            tf.sink_pages.clear()
            tf.capture_sink = False
            tf.prepare_cell(prompt)
            timing = CellTiming(bool(extra.get("collect_chunk_timing", False)))
            sampler = WanFourStepSampler(H3_PROFILE)
            latent = sampler.initial_noise(self.latent_shape, device=self.device, dtype=dtype, seed=seed, chunk=0)
            for step, value in enumerate(sampler.timesteps):
                timestep = torch.tensor(value, dtype=torch.float32, device="cpu")
                prediction = self._sink_forward(latent, timestep, step=step, pp=pp)
                latent = sampler.advance(latent, prediction, step=step, seed=seed, chunk=0)
            sink_latent = latent
            timing.mark(0)
            sink_input = sampler.sink_input(latent, sink_sigma=SINK_SIGMA, seed=seed)
            tf.capture_sink = True
            self._sink_forward(
                sink_input, torch.tensor(1.0 - SINK_SIGMA, dtype=torch.float32, device="cpu"), step=4, pp=pp
            )
            tf.capture_sink = False
            self._chunk_ctx.enqueue(request_id, plan, chunk_tokens=self.block_size)
            adapter = PhysisH3Cell(
                tf,
                sampler=sampler,
                prompt_embeds=prompt,
                latent_shape=self.latent_shape,
                seed=seed,
                device=self.device,
                dtype=dtype,
                on_finished=timing.mark,
            )
            run_cell_pipeline(ctx=self._chunk_ctx, adapter=adapter)
            finished = self._gather_finished(adapter.finished, pp)
            durations = timing.durations(pp)
            if self.pp_rank == 0:
                chunks = finished[request_id]
                latents = torch.cat([sink_latent, *(chunks[i] for i in range(plan.schedule.chunks))], dim=2)
                video = self._decode_latents(latents)
                result = prepare_video_output(self._video_output(video), output_type=item.sampling_params.output_type)
                result.stage_durations = durations
                outputs.append(result)
            else:
                outputs.append(self._placeholder_output(self.device, dtype))
            tf.sink_pages.clear()
        return outputs

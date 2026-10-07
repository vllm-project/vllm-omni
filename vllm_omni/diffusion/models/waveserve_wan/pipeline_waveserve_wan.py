# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WaveServe Wan 2.1 1.3B (rectified-flow) chunk pipeline.

Checkpoint: ``Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev``.
DiT weights load through shared ``wan2_2.WanTransformer3DModel`` (standard Wan
layout). Path: text encode → 5D latent noise → vertical Chunk SERIAL/Latest
with real Wan forward + FlowEuler → VAE decode on rank 0.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.waveserve_wan.transformer import StageWanTransformer
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.chunk_executor import (
    ARDiffusionChunkContext,
    ChunkAdapter,
    resolve_pp_rank_and_group,
    run_chunk_pipeline,
)
from vllm_omni.experimental.ar_diffusion.chunk_schedule import (
    ChunkSchedule,
    Ordering,
    build_chunk_plan,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec

logger = init_logger(__name__)

HF_MODEL_ID = "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev"
# Wan 2.1 1.3B geometry.
WAN21_1_3B = {
    "num_layers": 30,
    "dim": 1536,
    "num_heads": 12,
    "ffn_dim": 8960,
    "in_channels": 16,
    "patch_size": (1, 2, 2),
}
_DEFAULT_LATENT_SHAPE = (1, 16, 3, 60, 104)
_DEFAULT_SHIFT = 5.0


def _as_request_list(req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[OmniDiffusionRequest]:
    if isinstance(req, DiffusionRequestBatch):
        return list(req.requests)
    return [req]


class FlowEuler:
    """Rectified-flow Euler matching Diffusers FlowMatchEulerDiscreteScheduler(shift=...).

    Diffusers 0.40 builds the shifted grid from an already-shifted ``sigma_min``
    endpoint (``__init__`` reads ``sigma_min`` from the warped schedule), so the
    linspace endpoint is effectively warped twice. Replicate that faithfully.
    """

    def __init__(self, steps: int, shift: float = 1.0) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        if shift <= 0:
            raise ValueError("shift must be positive")

        def warp(sigma: torch.Tensor) -> torch.Tensor:
            return shift * sigma / (1 + (shift - 1) * sigma)

        # Faithful to FlowMatchEulerDiscreteScheduler: endpoint is warp(1/1000)
        # before linspace, then the whole grid is warped again.
        shifted = warp(torch.linspace(1.0, float(warp(torch.tensor(1.0 / 1000))), steps, dtype=torch.float64))
        self.sigmas = [*shifted.float().tolist(), 0.0]
        self.timesteps = [1000.0 * sigma for sigma in self.sigmas[:-1]]

    def advance(self, prediction: torch.Tensor, sample: torch.Tensor, step: int) -> torch.Tensor:
        delta = self.sigmas[step + 1] - self.sigmas[step]
        return (sample.float() + delta * prediction.float()).to(sample.dtype)


class _LatentChunkAdapter(ChunkAdapter):
    """Real Wan path: 5D latents + FlowEuler along the PP chain.

    Activation payload is always a dict:
    - ``{"latent": ...}`` after stage-last (advanced / clean latent)
    - ``{"latent": ..., "hidden_states": ...}`` after non-last layer groups
      so the next group can RoPE from latent and resume tokens
    """

    def __init__(
        self,
        transformer: StageWanTransformer,
        *,
        sampler: FlowEuler,
        prompt_embeds: torch.Tensor,
        latent_shape: tuple[int, int, int, int, int],
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        on_finished: Callable[[int, torch.Tensor], None] | None = None,
    ) -> None:
        self.transformer = transformer
        self.sampler = sampler
        self.prompt_embeds = prompt_embeds
        self.latent_shape = latent_shape
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.on_finished = on_finished
        self.num_denoise_steps = len(sampler.timesteps)
        # req -> chunk -> latent (after last denoise advance)
        self.finished: dict[str, dict[int, torch.Tensor]] = {}
        # Per-request current latents keyed by chunk while in flight on this rank
        self._live: dict[str, dict[int, torch.Tensor]] = {}

    def _init_noise(self, chunk: int) -> torch.Tensor:
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed * 1_000_003 + chunk * 4096)
        return torch.randn(self.latent_shape, generator=gen, device=self.device, dtype=self.dtype)

    @staticmethod
    def _slice_batch(tensor: torch.Tensor, index: int, n_tasks: int) -> torch.Tensor:
        if tensor.shape[0] == n_tasks:
            return tensor[index : index + 1]
        return tensor

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs: list[dict[str, torch.Tensor]] = []
        for i, (req, (chunk, step)) in enumerate(tasks):
            ctx = kv_contexts[i] if i < len(kv_contexts) else None
            live = self._live.setdefault(req, {})
            inter: IntermediateTensors | None = None
            if hidden is None:
                latent = live.get(chunk)
                if latent is None:
                    latent = self._init_noise(chunk)
            elif isinstance(hidden, torch.Tensor):
                latent = self._slice_batch(hidden, i, len(tasks))
            elif isinstance(hidden, dict):
                if "latent" not in hidden:
                    raise KeyError(f"activation dict missing latent; keys={list(hidden)}")
                latent = self._slice_batch(hidden["latent"], i, len(tasks))
                hs = hidden.get("hidden_states")
                if hs is not None:
                    inter = IntermediateTensors({"hidden_states": self._slice_batch(hs, i, len(tasks))})
            else:
                raise TypeError(f"expected latent activation dict/tensor, got {type(hidden)}")

            if step < self.num_denoise_steps:
                t = torch.tensor(self.sampler.timesteps[step], device=self.device, dtype=torch.float32)
                out = self.transformer.forward_latent_step(
                    latent,
                    timestep=t,
                    encoder_hidden_states=self.prompt_embeds,
                    kv_contexts=ctx,
                    intermediate_tensors=inter,
                )
                if isinstance(out, IntermediateTensors):
                    # Non-last layer group: forward tokens + carry 5D latent.
                    live[chunk] = latent
                    outs.append({"latent": latent, "hidden_states": out["hidden_states"]})
                    continue
                # Stage-last: unpatched pred → FlowEuler advance.
                latent = self.sampler.advance(out, latent, step)
                if step == self.num_denoise_steps - 1:
                    self.finished.setdefault(req, {})[chunk] = latent.detach()
                    if self.on_finished is not None:
                        self.on_finished(chunk, latent)
            else:
                # Clean KV refresh at t=0; latent already final.
                t = torch.zeros((), device=self.device, dtype=torch.float32)
                out = self.transformer.forward_latent_step(
                    latent,
                    timestep=t,
                    encoder_hidden_states=self.prompt_embeds,
                    kv_contexts=ctx,
                    intermediate_tensors=inter,
                )
                if isinstance(out, IntermediateTensors):
                    live[chunk] = latent
                    outs.append({"latent": latent, "hidden_states": out["hidden_states"]})
                    continue
                self.finished.setdefault(req, {})[chunk] = latent.detach()

            live[chunk] = latent
            outs.append({"latent": latent})
        return self._stack_payloads(outs)

    @staticmethod
    def _stack_payloads(outs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if len(outs) == 1:
            return outs[0]
        keys = outs[0].keys()
        return {key: torch.cat([out[key] for out in outs], dim=0) for key in keys}

    def pack_activation(self, output):
        if isinstance(output, dict):
            return {k: v.contiguous() for k, v in output.items() if isinstance(v, torch.Tensor)}
        if isinstance(output, IntermediateTensors):
            return {k: v.contiguous() for k, v in output.tensors.items()}
        if isinstance(output, torch.Tensor):
            return {"latent": output.contiguous()}
        raise TypeError(f"latent adapter packs dict/tensor, got {type(output)}")

    def unpack_activation(self, payload: dict):
        return payload


class WaveServeWanPipeline(nn.Module):
    """Chunk SERIAL/Latest pipeline (AR-Diffusion engine + noisy KV)."""

    supports_request_batch = True
    # Engine startup dummy forces num_inference_steps=2, which conflicts with
    # vertical S=T+1; skip the engine dummy (not a measurement warmup).
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        stage_cfg = getattr(od_config, "ar_diffusion_stage_config", None) or {}
        if not isinstance(stage_cfg, dict):
            stage_cfg = {}
        model_cfg = getattr(od_config, "model_config", None) or {}
        if isinstance(model_cfg, dict):
            stage_cfg = {**stage_cfg, **(model_cfg.get("ar_diffusion_stage_config") or {})}
        self.stage_parallel_size = int(stage_cfg.get("stage_parallel_size", 1) or 1)
        pp_world = int(getattr(getattr(od_config, "parallel_config", None), "pipeline_parallel_size", 1) or 1)
        self.layer_groups = max(1, pp_world // max(1, self.stage_parallel_size))
        if bool(stage_cfg.get("tiny", False)):
            raise ValueError(
                "WaveServeWanPipeline no longer supports tiny=True; use tests/diffusion/ar_diffusion/waveserve_tiny.py"
            )
        geo = dict(WAN21_1_3B)
        self.max_history_chunks = int(stage_cfg.get("max_history_chunks", 6) or 6)
        pp_rank, _pp_group = resolve_pp_rank_and_group()
        self.pp_rank = pp_rank
        model_path = str(getattr(od_config, "model", None) or HF_MODEL_ID)
        self.model_path = model_path
        transformer_config: dict[str, Any] | None = None
        try:
            from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import load_transformer_config

            loaded = load_transformer_config(model_path, local_files_only=True)
            if not loaded:
                loaded = load_transformer_config(model_path, local_files_only=False)
            transformer_config = loaded or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("WaveServe: could not load transformer config from %s: %s", model_path, exc)
        self.transformer = StageWanTransformer(
            num_layers=int(geo["num_layers"]),
            dim=int(geo["dim"]),
            num_heads=int(geo["num_heads"]),
            ffn_dim=int(geo["ffn_dim"]),
            in_channels=int(geo["in_channels"]),
            patch_size=tuple(geo["patch_size"]),
            layer_groups=self.layer_groups,
            pp_rank=pp_rank,
            transformer_config=transformer_config,
        )
        # Without this the Diffusers loader's get_all_weights() is empty, so the
        # DiT stays at init (zeros / garbage NaN biases) and every video is NaN.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self._init_codec(model_path)
        # Prefer a page size that divides the default WaveServe latent token count.
        default_tokens = self.transformer.seq_len_for_latent(_DEFAULT_LATENT_SHAPE)
        self.block_size = default_tokens
        self.max_chunk_tokens = default_tokens
        self._chunk_ctx: ARDiffusionChunkContext | None = None
        self._stream_decode_group = None
        logger.info(
            "WaveServe Wan pipeline: S=%d G=%d layers=%d local=[%d, %d) codec=%s model=%s",
            self.stage_parallel_size,
            self.layer_groups,
            self.transformer.num_layers,
            self.transformer.start_layer,
            self.transformer.end_layer,
            self.vae is not None,
            model_path,
        )

    def _init_codec(self, model_path: str) -> None:
        """Load UMT5 + Wan VAE. Prefer rank 0 for VAE decode ownership."""
        from transformers import AutoTokenizer, UMT5EncoderModel

        from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
            DistributedAutoencoderKLWan,
        )
        from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch

        dtype = torch.bfloat16
        try:
            param = next(self.transformer.parameters())
            dtype = param.dtype
            device = param.device
        except StopIteration:
            device = torch.device("cpu")

        # Text encoder on every DiT rank (needed each denoise step).
        # VAE decode ownership is rank 0 only — skip allocating it elsewhere.
        prefetch = ("tokenizer", "text_encoder", "vae") if self.pp_rank == 0 else ("tokenizer", "text_encoder")
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            model_path,
            subfolder="tokenizer",
            prefetch_list=prefetch,
            local_files_only=False,
        )
        self.text_encoder = (
            from_pretrained_with_prefetch(
                UMT5EncoderModel.from_pretrained,
                model_path,
                subfolder="text_encoder",
                prefetch_list=prefetch,
                local_files_only=False,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        if self.pp_rank != 0:
            self.vae = None
            return
        self.vae = (
            from_pretrained_with_prefetch(
                DistributedAutoencoderKLWan.from_pretrained,
                model_path,
                subfolder="vae",
                prefetch_list=prefetch,
                local_files_only=False,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        for p in self.vae.parameters():
            p.requires_grad_(False)

    def ar_diffusion_noisy_kv_spec(self) -> ARDiffusionNoisyKVSpec:
        local_layers = max(1, self.transformer.local_num_layers)
        return ARDiffusionNoisyKVSpec(
            num_layers=local_layers,
            num_kv_heads=self.transformer.num_heads,
            head_size=self.transformer.head_dim,
            block_size=self.block_size,
            max_chunk_tokens=self.max_chunk_tokens,
            max_history_chunks=max(1, self.max_history_chunks),
        )

    @contextmanager
    def bind_ar_diffusion_chunk_context(self, ctx: ARDiffusionChunkContext) -> Iterator[None]:
        prev = self._chunk_ctx
        self._chunk_ctx = ctx
        try:
            yield
        finally:
            self._chunk_ctx = prev

    def load_weights(self, weights):
        """Load Wan DiT weights via ``wan2_2.WanTransformer3DModel.load_weights``."""
        return self.transformer.load_weights(weights)

    def _encode_prompt(self, prompt: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self.tokenizer is not None and self.text_encoder is not None
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        max_length = 512
        text = prompt_clean(prompt)
        tokens = self.tokenizer(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        hidden = self.text_encoder(ids, mask).last_hidden_state.to(dtype=dtype)
        length = int(mask.gt(0).sum().item())
        return torch.cat([hidden[:, :length], hidden.new_zeros(1, max_length - length, hidden.shape[2])], dim=1)

    def _plan_for(self, req: OmniDiffusionRequest):
        extra = (req.sampling_params.extra_args or {}) if req.sampling_params is not None else {}
        chunks = int(extra.get("num_chunks", extra.get("chunks", 2)))
        explicit_denoise = (
            extra.get("num_inference_steps")
            or extra.get("num_denoise_steps")
            or getattr(req.sampling_params, "num_inference_steps", None)
        )
        denoise = int(explicit_denoise or 4)
        history = int(extra.get("kv_history_chunks", 0) or 0)
        if self.stage_parallel_size > 1 and history < 1:
            history = self.max_history_chunks
        schedule_name = str(extra.get("chunk_schedule", "serial")).strip().lower()
        if schedule_name in ("", "serial"):
            ordering = Ordering.SERIAL
        elif schedule_name == "latest":
            ordering = Ordering.INTERLEAVED
        else:
            raise ValueError(f"chunk_schedule must be 'serial' or 'latest', got {schedule_name!r}")
        stages = self.stage_parallel_size
        if stages not in (1, denoise + 1):
            if explicit_denoise:
                raise ValueError(f"WaveServe stages must be 1 or num_denoise_steps+1 ({denoise + 1}), got {stages}")
            denoise = stages - 1
        schedule = ChunkSchedule(
            chunks=chunks,
            num_denoise_steps=denoise,
            stages=stages,
            layer_groups=self.layer_groups,
            ordering=ordering,
            kv_history_chunks=history,
        )
        return build_chunk_plan(schedule), extra, denoise

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        assert self.vae is not None
        # all_gather_object can resurface peer-rank latents on another local
        # cuda index (e.g. cuda:1 under CUDA_VISIBLE_DEVICES=0,3); pin to VAE.
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=self.vae.dtype)
        mean = torch.tensor(self.vae.config.latents_mean, device=vae_device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = 1.0 / torch.tensor(self.vae.config.latents_std, device=vae_device, dtype=latents.dtype).view(
            1, -1, 1, 1, 1
        )
        latents = latents / std + mean
        video = self.vae.decode(latents, return_dict=False)[0]
        return video.clamp(-1, 1)

    def _video_output(self, video: torch.Tensor) -> DiffusionOutput:
        if not self.od_config.video_output_transport.enable_device_postprocess:
            return DiffusionOutput(output=video)
        return DiffusionOutput(
            media=DiffusionMediaOutput(
                video=VideoMediaOutput(
                    tensor=video,
                    spec=VideoTensorSpec(
                        layout=VideoTensorLayout.BCTHW,
                        encoding=VideoTensorEncoding.NORMALIZED_FLOAT,
                        value_range=VideoValueRange.NEGATIVE_ONE_TO_ONE,
                    ),
                )
            )
        )

    def _gather_finished(
        self, finished: dict[str, dict[int, torch.Tensor]], pp_group: Any | None
    ) -> dict[str, dict[int, torch.Tensor]]:
        """Union finished-chunk latents across PP ranks onto every rank (object list)."""
        if not dist.is_initialized() or pp_group is None or getattr(pp_group, "world_size", 1) <= 1:
            return finished
        world = int(pp_group.world_size)
        group = getattr(pp_group, "device_group", None) or pp_group
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, finished, group=group)
        merged: dict[str, dict[int, torch.Tensor]] = {}
        # Prefer a stable local device for any CPU/foreign-cuda tensors after
        # object gather (peer ranks may serialize a different cuda index).
        local_device = next(self.transformer.parameters()).device
        for part in gathered:
            if not part:
                continue
            for req_id, chunks in part.items():
                bucket = merged.setdefault(req_id, {})
                for chunk, latent in chunks.items():
                    if isinstance(latent, torch.Tensor):
                        bucket[chunk] = latent.to(local_device, non_blocking=True)
                    else:
                        bucket[chunk] = latent
        return merged

    def _decode_latent_chunks(self, chunks: Iterator[torch.Tensor], latent_frames: int) -> torch.Tensor:
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        assert self.vae is not None
        vae = self.vae
        param = next(vae.parameters())
        mean = torch.tensor(vae.config.latents_mean, device=param.device, dtype=param.dtype).view(1, -1, 1, 1, 1)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=param.device, dtype=param.dtype).view(
            1, -1, 1, 1, 1
        )
        frames_per_latent = 2 ** sum(bool(flag) for flag in vae.decoder.temperal_upsample)
        output = None
        offset = 0
        vae.clear_cache()
        try:
            with vae._execution_context():
                for latent in chunks:
                    hidden = vae.post_quant_conv(latent / inv_std + mean)
                    for index in range(hidden.shape[2]):
                        vae._conv_idx = [0]
                        pixels = vae.decoder(
                            hidden[:, :, index : index + 1],
                            feat_cache=vae._feat_map,
                            feat_idx=vae._conv_idx,
                            first_chunk=offset == 0,
                        )
                        if vae.config.patch_size is not None:
                            pixels = unpatchify(pixels, patch_size=vae.config.patch_size)
                        if output is None:
                            frames = pixels.shape[2] + (latent_frames - 1) * frames_per_latent
                            output = pixels.new_empty((pixels.shape[0], pixels.shape[1], frames, *pixels.shape[-2:]))
                        end = offset + pixels.shape[2]
                        torch.clamp(pixels, min=-1.0, max=1.0, out=output[:, :, offset:end])
                        offset = end
            assert output is not None and offset == output.shape[2]
            return output
        finally:
            vae.clear_cache()

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[DiffusionOutput]:
        ctx = self._chunk_ctx
        if ctx is None:
            raise RuntimeError("WaveServeWanPipeline requires bind_ar_diffusion_chunk_context")
        requests = _as_request_list(req)
        pp_rank, pp_group = resolve_pp_rank_and_group()

        param = next(self.transformer.parameters())
        device, dtype = param.device, param.dtype
        outputs: list[DiffusionOutput] = []
        for item in requests:
            plan, extra, denoise = self._plan_for(item)
            req_id = str(getattr(item, "request_id", None) or extra.get("request_id") or "")
            if not req_id:
                raise ValueError("WaveServe chunk path requires a non-empty request_id")
            shape = tuple(int(x) for x in (extra.get("latent_shape") or _DEFAULT_LATENT_SHAPE))
            if len(shape) != 5:
                raise ValueError(f"latent_shape must be B,C,T,H,W; got {shape}")
            shift = float(extra.get("shift", _DEFAULT_SHIFT))
            seed = int(extra.get("seed", getattr(item.sampling_params, "seed", 0) or 0))
            chunk_tokens = self.transformer.seq_len_for_latent(shape)
            # Resize noisy KV paging to this request's token count.
            self.block_size = chunk_tokens
            self.max_chunk_tokens = chunk_tokens
            ctx.enqueue(req_id, plan, chunk_tokens=chunk_tokens)

            prompt = item.prompt if isinstance(item.prompt, str) else str(item.prompt or "")
            prompt_embeds = self._encode_prompt(prompt, device, dtype)
            sampler = FlowEuler(denoise, shift=shift)
            stream_decode = bool(extra.get("stream_decode", False))
            decode_buffers = []
            recv_works = []
            send_works = []
            on_finished = None
            if stream_decode:
                source_rank = denoise * self.layer_groups - 1
                if self.stage_parallel_size != denoise + 1 or source_rank == 0 or pp_group is None:
                    raise ValueError("stream_decode requires a remote final denoise rank in S=T+1 topology")
                if self.vae is not None and (self.vae.use_tiling or self.vae.is_distributed_enabled()):
                    raise ValueError("stream_decode requires non-tiled single-owner VAE")
                source = pp_group.ranks[source_rank]
                owner = pp_group.ranks[0]
                if self._stream_decode_group is None:
                    # CPU 潜变量通信不被 PP0 的设备同步阻塞；不混用激活/KV 顺序。
                    self._stream_decode_group = dist.new_group(ranks=[owner, source], backend="gloo")
                if pp_rank == 0:
                    decode_buffers = [
                        torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
                        for _ in range(plan.schedule.chunks)
                    ]
                    recv_works = [
                        dist.irecv(buffer, src=source, group=self._stream_decode_group) for buffer in decode_buffers
                    ]
                elif pp_rank == source_rank:

                    def on_finished(chunk: int, latent: torch.Tensor) -> None:
                        assert chunk == len(send_works)
                        host = latent.to(device="cpu").contiguous()
                        send_works.append((host, dist.isend(host, dst=owner, group=self._stream_decode_group)))

            adapter = _LatentChunkAdapter(
                self.transformer,
                sampler=sampler,
                prompt_embeds=prompt_embeds,
                latent_shape=shape,  # type: ignore[arg-type]
                seed=seed,
                device=device,
                dtype=dtype,
                on_finished=on_finished,
            )
            run_chunk_pipeline(ctx=ctx, adapter=adapter)
            if stream_decode:
                for _latent, work in send_works:
                    work.wait()
                if pp_rank == 0:

                    def ready_chunks() -> Iterator[torch.Tensor]:
                        for buffer, work in zip(decode_buffers, recv_works):
                            work.wait()
                            yield buffer.to(device=device, non_blocking=True)

                    video = self._decode_latent_chunks(ready_chunks(), plan.schedule.chunks * shape[2])
                    outputs.append(self._video_output(video))
                else:
                    outputs.append(DiffusionOutput(output=torch.zeros(1, 3, 8, 8, 8, device=device, dtype=dtype)))
                continue
            finished = self._gather_finished(adapter.finished, pp_group)
            chunk_map = finished.get(req_id, {})
            if pp_rank == 0 and chunk_map:
                ordered = [chunk_map[c] for c in sorted(chunk_map)]
                # Concatenate along time for multi-chunk video.
                latents = torch.cat(ordered, dim=2)
                video = self._decode_latents(latents)
                outputs.append(self._video_output(video))
            else:
                # Non-owner ranks still return a typed placeholder for the engine.
                outputs.append(DiffusionOutput(output=torch.zeros(1, 3, 8, 8, 8, device=device, dtype=dtype)))
        return outputs

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan 2.1 1.3B on the Cell / Noisy KV path.

Wires Wan codec + ``StageWanTransformer`` onto ``ARDiffusionEngine``, with
CausVid defaults (shift=8, T=3) and optional student ``model.pt`` overlay.

* Codec base: Diffusers ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``
* Student DiT: ``tianweiy/CausVid`` → ``autoregressive_checkpoint/model.pt``
* Shared infra: ``experimental.ar_diffusion`` (schedule / executor / KV / FlowEuler)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.logger import init_logger

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
from vllm_omni.diffusion.models.causvid_wan.async_stream_decode import start_async_stream_decode
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.diffusion.models.causvid_wan.weights import (
    iter_causvid_diffusers_weights,
    load_causvid_generator_state,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.cell_executor import (
    ARDiffusionCellContext,
    resolve_pp_rank_and_group,
    run_cell_pipeline,
)
from vllm_omni.experimental.ar_diffusion.cell_schedule import (
    CellSchedule,
    Ordering,
    build_cell_plan,
)
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec
from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

logger = init_logger(__name__)

HF_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HF_CAUSVID_REPO = "tianweiy/CausVid"
# CausVid causal DMD: shift=8, three noisy steps + clean@0 via S=T+1.
CAUSVID_DEFAULT_SHIFT = 8.0
CAUSVID_DEFAULT_DENOISE_STEPS = 3
WAN21_1_3B = {
    "num_layers": 30,
    "dim": 1536,
    "num_heads": 12,
    "ffn_dim": 8960,
    "in_channels": 16,
    "patch_size": (1, 2, 2),
}
_DEFAULT_LATENT_SHAPE = (1, 16, 3, 60, 104)

__all__ = [
    "HF_MODEL_ID",
    "HF_CAUSVID_REPO",
    "CAUSVID_DEFAULT_SHIFT",
    "CAUSVID_DEFAULT_DENOISE_STEPS",
    "FlowEuler",
    "WanCell",
    "CausVidWanPipeline",
]


def _as_request_list(req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[OmniDiffusionRequest]:
    if isinstance(req, DiffusionRequestBatch):
        return list(req.requests)
    return [req]


class CausVidWanPipeline(nn.Module):
    """CausVid Wan on Cell Self-Forcing/Latest-KV (AR-Diffusion + noisy KV)."""

    supports_request_batch = True
    # Skip engine dummy: num_inference_steps=2 breaks vertical S=T+1.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.od_config = od_config
        stage_cfg = getattr(od_config, "ar_diffusion_stage_config", None) or {}
        if not isinstance(stage_cfg, dict):
            stage_cfg = {}
        model_cfg = getattr(od_config, "model_config", None) or {}
        if not isinstance(model_cfg, dict):
            model_cfg = {}
        if model_cfg:
            stage_cfg = {**stage_cfg, **(model_cfg.get("ar_diffusion_stage_config") or {})}
        self._default_shift = float(model_cfg.get("shift", CAUSVID_DEFAULT_SHIFT))
        self._default_denoise = int(model_cfg.get("num_denoise_steps", CAUSVID_DEFAULT_DENOISE_STEPS))
        # Defer student overlay until after Diffusers ``load_weights`` (weights_sources).
        ckpt = model_cfg.get("causvid_ckpt") or model_cfg.get("generator_ckpt")
        self._causvid_ckpt = str(ckpt) if ckpt else None
        self._causvid_dit_loaded = False
        self.stage_parallel_size = int(stage_cfg.get("stage_parallel_size", 1) or 1)
        pp_world = int(getattr(getattr(od_config, "parallel_config", None), "pipeline_parallel_size", 1) or 1)
        self.layer_groups = max(1, pp_world // max(1, self.stage_parallel_size))
        if bool(stage_cfg.get("tiny", False)):
            raise ValueError("CausVidWanPipeline no longer supports tiny=True")
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
            logger.warning("CausVidWan: could not load transformer config from %s: %s", model_path, exc)
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
        # Required so Diffusers loader populates DiT weights (else NaN video).
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
        default_tokens = self.transformer.seq_len_for_latent(_DEFAULT_LATENT_SHAPE)
        self.block_size = default_tokens
        self.max_chunk_tokens = default_tokens
        self._chunk_ctx: ARDiffusionCellContext | None = None
        self._stream_decode_group = None
        self._streaming_decoder: WanStreamingDecoder | None = None
        logger.info(
            "CausVidWan pipeline: S=%d G=%d layers=%d local=[%d, %d) codec=%s model=%s causvid_ckpt=%s",
            self.stage_parallel_size,
            self.layer_groups,
            self.transformer.num_layers,
            self.transformer.start_layer,
            self.transformer.end_layer,
            self.vae is not None,
            model_path,
            self._causvid_ckpt,
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

        # Text encoder on every DiT rank; VAE decode ownership is rank 0 only.
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
    def bind_ar_diffusion_cell_context(self, ctx: ARDiffusionCellContext) -> Iterator[None]:
        prev = self._chunk_ctx
        self._chunk_ctx = ctx
        try:
            yield
        finally:
            self._chunk_ctx = prev

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load Diffusers DiT, then overlay CausVid student if ``causvid_ckpt`` is set."""
        loaded = self.transformer.load_weights(weights)
        if self._causvid_ckpt and not self._causvid_dit_loaded:
            self._load_causvid_dit(self._causvid_ckpt)
            self._causvid_dit_loaded = True
        return loaded

    def _load_causvid_dit(self, ckpt_path: str) -> None:
        """Overlay CausVid generator weights onto the Diffusers DiT."""
        logger.info("CausVid: loading generator DiT from %s", ckpt_path)
        raw = load_causvid_generator_state(ckpt_path)
        mapped = list(iter_causvid_diffusers_weights(raw))
        if not mapped:
            raise RuntimeError(f"CausVid checkpoint produced no remapped DiT tensors: {ckpt_path}")
        prefixed = ((f"transformer.{name}", tensor) for name, tensor in mapped)
        result = self.transformer.load_weights(prefixed)
        logger.info(
            "CausVid: loaded %d remapped DiT tensors from %s (loader_result=%s)",
            len(mapped),
            ckpt_path,
            result,
        )

    @staticmethod
    def _prompt_text(prompt: object) -> str:
        """Extract text from a plain string or serving ``OmniTextPrompt`` dict."""
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, dict):
            text = prompt.get("prompt")
            if isinstance(text, str):
                return text
            raise ValueError("Prompt dict must include a string 'prompt' field.")
        raise TypeError(f"Unsupported prompt type: {type(prompt)!r}")

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
        sp = req.sampling_params
        extra = dict(sp.extra_args or {}) if sp is not None else {}
        if "shift" not in extra:
            extra["shift"] = self._default_shift
        chunks = int(extra.get("num_chunks", extra.get("chunks", 2)))
        # User-provided steps only (do not treat model defaults as explicit).
        explicit_denoise = extra.get("num_inference_steps")
        if explicit_denoise is None:
            explicit_denoise = extra.get("num_denoise_steps")
        if explicit_denoise is None:
            explicit_denoise = getattr(sp, "num_inference_steps", None) if sp is not None else None
        denoise = self._default_denoise if explicit_denoise is None else int(explicit_denoise)
        if denoise < 1:
            raise ValueError(f"num_denoise_steps must be positive, got {denoise}")
        history = int(extra.get("kv_history_chunks", 0) or 0)
        if self.stage_parallel_size > 1 and history < 1:
            history = self.max_history_chunks
        schedule_name = str(extra.get("cell_schedule", "self_forcing")).strip().lower().replace("-", "_")
        if schedule_name in ("", "self_forcing"):
            ordering = Ordering.SELF_FORCING
        elif schedule_name == "latest_kv":
            ordering = Ordering.LATEST_KV
        elif schedule_name == "same_step":
            ordering = Ordering.SAME_STEP
        else:
            raise ValueError(
                f"cell_schedule must be 'self_forcing', 'latest_kv', or 'same_step', got {schedule_name!r}"
            )
        stages = self.stage_parallel_size
        if stages not in (1, denoise + 1):
            if explicit_denoise is not None:
                raise ValueError(f"CausVidWan stages must be 1 or num_denoise_steps+1 ({denoise + 1}), got {stages}")
            denoise = stages - 1
        extra["num_denoise_steps"] = denoise
        if sp is not None:
            sp.extra_args = extra
        num_blocks = int(getattr(self.transformer, "num_layers", 0) or 0)
        local_blocks = int(getattr(self.transformer, "local_num_layers", 0) or 0)
        schedule = CellSchedule(
            chunks=chunks,
            num_denoise_steps=denoise,
            stages=stages,
            layer_groups=self.layer_groups,
            ordering=ordering,
            kv_history_chunks=history,
            blocks=num_blocks if num_blocks > 0 else None,
            blocks_per_rank=local_blocks if local_blocks > 0 else None,
        )
        return build_cell_plan(schedule), extra, denoise

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

    def _placeholder_output(self, device: torch.device, dtype: torch.dtype) -> DiffusionOutput:
        return DiffusionOutput(output=torch.zeros(1, 3, 8, 8, 8, device=device, dtype=dtype))

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

    def _get_streaming_decoder(self) -> WanStreamingDecoder | None:
        """Reuse experimental ``WanStreamingDecoder`` on the local VAE (rank0)."""
        if self.vae is None:
            return None
        if self._streaming_decoder is None:
            try:
                self._streaming_decoder = WanStreamingDecoder(self.vae)
            except TypeError as exc:
                logger.warning("WanStreamingDecoder unavailable for CausVid VAE: %s", exc)
                return None
        return self._streaming_decoder

    def _ensure_stream_decode_group(self, pp_group: Any, source_rank: int) -> Any:
        owner = pp_group.ranks[0]
        source = pp_group.ranks[source_rank]
        if self._stream_decode_group is None:
            self._stream_decode_group = dist.new_group(ranks=[owner, source], backend="gloo")
        return self._stream_decode_group

    @staticmethod
    def _maybe_timeline(
        on_finished: Callable[[int, torch.Tensor], None] | None,
        timeline_file: Any,
        *,
        source_rank: int | None,
        pp_rank: int,
    ) -> Callable[[int, torch.Tensor], None] | None:
        if timeline_file is None or source_rank is None or pp_rank != source_rank:
            return on_finished
        import time as _time

        prev = on_finished

        def _wrapped(chunk: int, latent: torch.Tensor) -> None:
            if latent.is_cuda:
                torch.accelerator.synchronize(latent.device)
            with open(str(timeline_file), "a", encoding="utf-8") as fh:
                fh.write(f"{int(chunk)}\t{_time.time()}\n")
            if prev is not None:
                prev(chunk, latent)

        return _wrapped

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[DiffusionOutput]:
        ctx = self._chunk_ctx
        if ctx is None:
            raise RuntimeError("CausVidWanPipeline requires bind_ar_diffusion_cell_context")
        requests = _as_request_list(req)
        pp_rank, pp_group = resolve_pp_rank_and_group()

        param = next(self.transformer.parameters())
        device, dtype = param.device, param.dtype
        outputs: list[DiffusionOutput] = []
        for item in requests:
            plan, extra, denoise = self._plan_for(item)
            req_id = str(getattr(item, "request_id", None) or extra.get("request_id") or "")
            if not req_id:
                raise ValueError("CausVidWan path requires a non-empty request_id")
            shape = tuple(int(x) for x in (extra.get("latent_shape") or _DEFAULT_LATENT_SHAPE))
            if len(shape) != 5:
                raise ValueError(f"latent_shape must be B,C,T,H,W; got {shape}")
            shift = float(extra.get("shift", self._default_shift))
            seed = int(extra.get("seed", getattr(item.sampling_params, "seed", 0) or 0))
            chunk_tokens = self.transformer.seq_len_for_latent(shape)
            # The KV pool is fixed at load time; requests cannot resize it by changing pipeline fields.
            spec = ctx.kv.cache.spec
            if chunk_tokens != spec.max_chunk_tokens:
                raise ValueError(
                    f"latent_shape token count {chunk_tokens} does not match preallocated NoisyKV "
                    f"max_chunk_tokens={spec.max_chunk_tokens}"
                )
            ctx.enqueue(req_id, plan, chunk_tokens=chunk_tokens)

            prompt = self._prompt_text(item.prompt)
            prompt_embeds = self._encode_prompt(prompt, device, dtype)
            sampler = FlowEuler(denoise, shift=shift)
            stream_decode = bool(extra.get("stream_decode", False))
            source_rank = denoise * self.layer_groups - 1 if self.stage_parallel_size == denoise + 1 else None
            stream_job = None
            on_finished = None
            if stream_decode:
                stream_group = None
                if source_rank is not None and source_rank != 0 and pp_group is not None:
                    stream_group = self._ensure_stream_decode_group(pp_group, source_rank)
                stream_job = start_async_stream_decode(
                    vae=self.vae,
                    decoder=self._get_streaming_decoder(),
                    session_id=req_id,
                    n_chunks=plan.schedule.chunks,
                    latent_shape=shape,
                    dtype=dtype,
                    pp_rank=pp_rank,
                    source_rank=source_rank,
                    pp_group=pp_group,
                    stream_group=stream_group,
                )
                on_finished = stream_job.on_finished
            on_finished = self._maybe_timeline(
                on_finished, extra.get("chunk_timeline_file"), source_rank=source_rank, pp_rank=pp_rank
            )

            adapter = WanCell(
                self.transformer,
                sampler=sampler,
                prompt_embeds=prompt_embeds,
                latent_shape=shape,  # type: ignore[arg-type]
                seed=seed,
                device=device,
                dtype=dtype,
                on_finished=on_finished,
            )
            # DiT tick loop; stream_job decode/send threads overlap on the side.
            run_cell_pipeline(ctx=ctx, adapter=adapter)
            if stream_job is not None:
                video = stream_job.join()
                if video is not None:
                    outputs.append(self._video_output(video))
                else:
                    outputs.append(self._placeholder_output(device, dtype))
                continue
            finished = self._gather_finished(adapter.finished, pp_group)
            chunk_map = finished.get(req_id, {})
            if pp_rank == 0 and chunk_map:
                ordered = [chunk_map[c] for c in sorted(chunk_map)]
                latents = torch.cat(ordered, dim=2)
                video = self._decode_latents(latents)
                outputs.append(self._video_output(video))
            else:
                outputs.append(self._placeholder_output(device, dtype))
        return outputs

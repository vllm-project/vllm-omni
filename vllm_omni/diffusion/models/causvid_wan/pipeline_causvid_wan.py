# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan 2.1 1.3B on the Cell / Noisy PP path.

Wires Wan codec + ``StageWanTransformer`` onto ``ARDiffusionEngine``, with
CausVid defaults (shift=8, D=3, S=D+1) and optional student ``model.pt`` overlay.
CausVidWan supports clean-history and Same-Step policies with their declared topology.

* Codec base: Diffusers ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``
* Student DiT: ``tianweiy/CausVid`` → ``autoregressive_checkpoint/model.pt``
* Shared infra: ``experimental.ar_diffusion`` (schedule / executor / KV / FlowEuler)
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable, Iterable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any, ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellRunSpec,
    CellTopology,
    resolve_pp_rank_and_group,
    run_cell_pipeline,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    CellSchedule,
    Ordering,
    build_cell_plan,
    cfg_step_replicas,
    expected_prior_cells,
    rank_of_block,
)
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
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import (
    ARDiffusionNoisyKVSpec,
    NoisyKVState,
    clean_history_cells,
)
from vllm_omni.experimental.ar_diffusion.request_binding import kv_request_binding
from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

logger = init_logger(__name__)


def _record_chunk_timeline(
    chunk: int,
    latent: torch.Tensor,
    *,
    timeline_file: Any,
    previous_callback: Callable[[int, torch.Tensor], None] | None,
) -> None:
    """Append one chunk finish time, then run the callback this wrapper replaced."""
    if latent.is_cuda:
        torch.accelerator.synchronize(latent.device)
    with open(str(timeline_file), "a", encoding="utf-8") as fh:
        fh.write(f"{int(chunk)}\t{time.time()}\n")
    if previous_callback is not None:
        previous_callback(chunk, latent)


def _chunk_timeline_callback(
    on_finished: Callable[[int, torch.Tensor], None] | None,
    timeline_file: Any,
    *,
    finish_rank: int,
    pp_rank: int,
) -> Callable[[int, torch.Tensor], None] | None:
    """Bind chunk timeline recording on the last-noisy producer rank."""
    if timeline_file is None or pp_rank != finish_rank:
        return on_finished
    return partial(_record_chunk_timeline, timeline_file=timeline_file, previous_callback=on_finished)


HF_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HF_CAUSVID_REPO = "tianweiy/CausVid"
# CausVid causal DMD: shift=8, D=3; schedule default derives from topology
# (S=1 → self_forcing layer PP, S>1 → latest_kv vertical with clean).
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
    # Skip engine dummy: num_inference_steps=2 breaks vertical S=D+1.
    dummy_run_num_frames: ClassVar[int] = 0
    _transformer_cls: ClassVar[type[StageWanTransformer]] = StageWanTransformer

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
        self.step_replicas = cfg_step_replicas(stage_cfg)
        pp_world = int(getattr(getattr(od_config, "parallel_config", None), "pipeline_parallel_size", 1) or 1)
        self.layer_groups = max(1, pp_world // max(1, self.step_replicas))
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
        self.transformer = self._transformer_cls(
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
        self._noisy_kv: NoisyKVState | None = None
        self._chunk_ctx: ARDiffusionCellContext | None = None
        self._stream_decode_group = None
        self._streaming_decoder: WanStreamingDecoder | None = None
        logger.info(
            "CausVidWan pipeline: S=%d G=%d layers=%d local=[%d, %d) codec=%s model=%s causvid_ckpt=%s",
            self.step_replicas,
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
        # The diagonal orderings retain one version per chunk (the clean KV), so the
        # window size is also the version count. Same-Step keeps one version per
        # denoise step plus chunk 0's clean sink, so the retained set is wider and
        # the shared pool has to be sized for it at load time.
        versions_per_chunk = 1
        if self.step_replicas > 1 and self.step_replicas == self._default_denoise:
            versions_per_chunk = self._default_denoise + 1
        return ARDiffusionNoisyKVSpec(
            num_layers=local_layers,
            num_kv_heads=self.transformer.num_heads,
            head_size=self.transformer.head_dim,
            block_size=self.block_size,
            max_chunk_tokens=self.max_chunk_tokens,
            max_history_chunks=max(1, self.max_history_chunks),
            max_history_versions=max(1, max(1, self.max_history_chunks) * versions_per_chunk),
        )

    @contextmanager
    def bind_ar_diffusion_noisy_kv(self, state: NoisyKVState) -> Iterator[None]:
        """Expose runner-owned Noisy KV for one request invocation."""
        prev = getattr(self, "_noisy_kv", None)
        self._noisy_kv = state
        try:
            yield
        finally:
            self._noisy_kv = prev

    def _cell_context(self) -> ARDiffusionCellContext:
        """Build the Cell executor state from the bound KV pool and this pipeline's topology.

        ``max_batch_size`` comes from the runner-allocated pool so admission cannot
        disagree with the capacity ``_preallocate_noisy_kv`` sized.
        """
        state = getattr(self, "_noisy_kv", None)
        if state is None:
            raise RuntimeError(
                "CausVidWanPipeline requires Noisy KV bound by the AR-Diffusion runner before the request forward"
            )
        rank, pp_group = resolve_pp_rank_and_group()
        return ARDiffusionCellContext(
            spec=CellRunSpec(
                topology=CellTopology(step_replicas=self.step_replicas, layer_groups=self.layer_groups),
                max_batch_size=state.cache.max_batch_size,
                rank=rank,
                pp_group=pp_group,
            ),
            kv=state,
        )

    def _execution_context(self) -> ARDiffusionCellContext:
        """Cell plan state for this request forward.

        Serving binds Noisy KV and the pipeline calls ``run_cell_pipeline``.
        A preinstalled ``_chunk_ctx`` covers tests that do not go through the runner.
        """
        if getattr(self, "_noisy_kv", None) is not None:
            return self._cell_context()
        installed = getattr(self, "_chunk_ctx", None)
        if installed is not None:
            return installed
        raise RuntimeError(
            "CausVidWanPipeline requires Noisy KV bound by the AR-Diffusion runner before the request forward"
        )

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

    def _plan_for(
        self,
        req: OmniDiffusionRequest,
        *,
        chunk_base: int = 0,
        same_step_session: bool = False,
    ):
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
        if "kv_history_chunks" in extra:
            history = int(extra.get("kv_history_chunks") or 0)
        else:
            # Default the history window from model max_history_chunks (H is
            # free — an explicit 0 is respected on every topology).
            history = int(self.max_history_chunks or 0)
        # Topology-derived default: each CausVid topology maps to exactly one
        # ordering (S=1 → self_forcing layer PP; S>1 → latest_kv vertical).
        default_schedule = "self_forcing" if self.step_replicas == 1 else "latest_kv"
        schedule_name = str(extra.get("cell_schedule", default_schedule)).strip().lower().replace("-", "_")
        if schedule_name in ("", "self_forcing"):
            ordering = Ordering.SELF_FORCING
        elif schedule_name == "latest_kv":
            ordering = Ordering.LATEST_KV
        elif schedule_name == "same_step":
            ordering = Ordering.SAME_STEP
        else:
            raise ValueError(
                f"CausVidWan cell_schedule must be 'self_forcing', 'latest_kv' or 'same_step', got {schedule_name!r}"
            )
        step_replicas = self.step_replicas
        if ordering is Ordering.SELF_FORCING and step_replicas > 1:
            # Self-Forcing is single-replica (paper semantics, layer PP);
            # vertical step replicas are the Latest-KV topology.
            raise ValueError(
                f"self_forcing requires step_replicas=1 (layer PP), got {step_replicas}; "
                "vertical step replicas use cell_schedule: latest_kv"
            )
        if ordering is Ordering.LATEST_KV and step_replicas == 1:
            # Latest-KV is the vertical clean-history topology (S=D+1);
            # the S=1 layer-PP schedule is self_forcing.
            raise ValueError(
                f"latest_kv requires step_replicas=D+1 ({denoise + 1}) for clean history, "
                "got 1; step_replicas=1 layer PP uses cell_schedule: self_forcing"
            )
        # Same-Step is the S=D vertical placement (one replica per denoise step);
        # Latest-KV is S=D+1 (one extra clean replica).
        if ordering is Ordering.SAME_STEP:
            if step_replicas != denoise:
                raise ValueError(
                    f"same_step requires step_replicas=D ({denoise}) so every denoise step owns a "
                    f"replica, got {step_replicas}"
                )
        elif step_replicas not in (1, denoise + 1):
            if explicit_denoise is not None:
                raise ValueError(f"step_replicas must be 1 or D+1 ({denoise + 1}), got {step_replicas}")
            # S>=2 here (S=1 is always in the set above); infer D = S-1 → S = D+1.
            denoise = step_replicas - 1
        extra["num_denoise_steps"] = denoise
        if step_replicas == 1 and self.layer_groups > 1 and bool(extra.get("stream_decode", False)):
            # S=1 finish rank is G-1, but only pp_rank 0 owns the VAE; with
            # G>1 the local path would crash non-VAE ranks and there is no
            # remote source_rank on layer PP. S=1 with G=1 (single rank) is
            # the valid local streaming decode and stays allowed.
            raise ValueError(
                "stream_decode requires a vertical topology or a single-rank layer PP; "
                "S=1 with layer_groups>1 has no VAE on its finish rank (omit stream_decode)"
            )
        if sp is not None:
            sp.extra_args = extra
        num_blocks = int(getattr(self.transformer, "num_layers", 0) or 0)
        local_blocks = int(getattr(self.transformer, "local_num_layers", 0) or 0)
        schedule = CellSchedule(
            chunks=chunks,
            num_denoise_steps=denoise,
            step_replicas=step_replicas,
            layer_groups=self.layer_groups,
            ordering=ordering,
            kv_history_chunks=history,
            blocks=num_blocks if num_blocks > 0 else None,
            blocks_per_rank=local_blocks if local_blocks > 0 else None,
            chunk_base=chunk_base,
            same_step_session=same_step_session,
        )
        # Logical prior (not local residency) drives plan sources + cross-rank handoffs.
        prior = expected_prior_cells(schedule)
        return build_cell_plan(schedule, prior_history=prior), extra, denoise

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
    def _noisy_request_lifecycle(
        req: OmniDiffusionRequest,
    ) -> tuple[str, bool, bool, bool]:
        """Compatibility projection of the runner/model shared KV binding."""
        binding = kv_request_binding(req)
        return binding.owner, binding.persist, binding.reset, binding.close

    def admit_cell_request(
        self, req: OmniDiffusionRequest, *, decode_tag: int = 0
    ) -> tuple[Any, str, bool, Any | None]:
        """Admit CellPlan and build WanCell under the bound context.

        Returns ``(adapter, kv_id, persist, stream_job)``.
        """
        ctx = self._chunk_ctx
        if ctx is None:
            raise RuntimeError("CausVidWanPipeline requires a Cell execution context for this request")
        pp_rank, pp_group = resolve_pp_rank_and_group()
        param = next(self.transformer.parameters())
        device, dtype = param.device, param.dtype

        binding = kv_request_binding(req)
        kv_id = binding.owner
        if not kv_id:
            raise ValueError("CausVidWan path requires a non-empty request_id or session_id")
        persist = binding.persist
        plan, extra, denoise = self._plan_for(req, chunk_base=binding.chunk_base, same_step_session=persist)
        schedule = plan.schedule
        if persist:
            ctx.kv.cache.bind_session_structure(kv_id, schedule.structural_key())
            # Fail-closed only for cells this rank owns; remotes arrive via prep handoff.
            logical = expected_prior_cells(schedule)
            missing_owned = [
                cell
                for cell in logical
                if rank_of_block(cell[1], cell[2], schedule) == pp_rank
                and not ctx.kv.cache.pool.has((kv_id, cell[0], cell[1], cell[2]))
            ]
            if missing_owned:
                raise ValueError(
                    f"session continuation chunk_base={binding.chunk_base} for {kv_id!r} missing owned "
                    f"prior cells {sorted(missing_owned)!r} on rank {pp_rank}; "
                    "restart with chunk_base=0 or fix retain/release"
                )
        shape = tuple(int(x) for x in (extra.get("latent_shape") or _DEFAULT_LATENT_SHAPE))
        if len(shape) != 5 or shape[0] != 1 or any(dim <= 0 for dim in shape):
            raise ValueError(f"latent_shape must be positive 1,C,T,H,W; got {shape}")
        shift = float(extra.get("shift", self._default_shift))
        seed = int(extra.get("seed", getattr(req.sampling_params, "seed", 0) or 0))
        chunk_tokens = self.transformer.seq_len_for_latent(shape)
        # The KV pool is fixed at load time; requests cannot resize it by changing pipeline fields.
        spec = ctx.kv.cache.spec
        if chunk_tokens != spec.max_chunk_tokens:
            raise ValueError(
                f"latent_shape token count {chunk_tokens} does not match preallocated NoisyKV "
                f"max_chunk_tokens={spec.max_chunk_tokens}"
            )
        num_blocks = int(schedule.num_blocks)
        # Protect only the post-request keep set. Prior sources still needed mid-request
        # are covered by plan last-use and are reclaimed after their last consume.
        retain = clean_history_cells(schedule, num_blocks=num_blocks) if persist else set()
        ctx.enqueue(kv_id, plan, chunk_tokens=chunk_tokens, retain=retain)

        prompt = self._prompt_text(req.prompt)
        prompt_embeds = self._encode_prompt(prompt, device, dtype)
        sampler = FlowEuler(denoise, shift=shift)
        stream_decode = bool(extra.get("stream_decode", False))
        # Rank that finishes the last noisy step (WanCell on_finished).
        # S=1: one layer-PP replica → trailing group G-1.
        # Vertical S∈{D,D+1}: last noisy step replica → D·G-1.
        if self.step_replicas > 1 and self.step_replicas in (denoise, denoise + 1):
            finish_rank = denoise * self.layer_groups - 1
        else:
            finish_rank = self.layer_groups - 1
        # Remote stream_decode only for vertical; S=1 stays local (_start_local).
        source_rank = finish_rank if self.step_replicas > 1 else None
        # Members may share one model call only when this key agrees. Same-Step
        # schedules must stay serial: each denoise step reads a different
        # history version, so packing two of them would mix histories. Remote
        # stream decodes also stay serial because their sends are untagged on
        # one shared Gloo group and cannot be demultiplexed per member.
        if plan.schedule.ordering is Ordering.SAME_STEP:
            serial_key = ("serial", kv_id)
        elif stream_decode and source_rank is not None:
            serial_key = ("remote_stream", kv_id)
        else:
            serial_key = None
        adapter, stream_job = self._start_stream_decode_and_wan_cell(
            kv_id=kv_id,
            n_chunks=plan.schedule.chunks,
            latent_shape=shape,
            dtype=dtype,
            device=device,
            seed=seed,
            prompt_embeds=prompt_embeds,
            sampler=sampler,
            stream_decode=stream_decode,
            finish_rank=finish_rank,
            source_rank=source_rank,
            pp_rank=pp_rank,
            pp_group=pp_group,
            timeline_file=extra.get("chunk_timeline_file"),
            decode_tag=decode_tag,
            serial_key=serial_key,
        )
        return adapter, kv_id, persist, stream_job

    def _start_stream_decode_and_wan_cell(
        self,
        *,
        kv_id: str,
        n_chunks: int,
        latent_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        seed: int,
        prompt_embeds: torch.Tensor,
        sampler: Any,
        stream_decode: bool,
        finish_rank: int,
        source_rank: int | None,
        pp_rank: int,
        pp_group: Any,
        timeline_file: Any,
        decode_tag: int = 0,
        serial_key: Hashable | None = None,
    ) -> tuple[Any, Any | None]:
        """Start stream_decode (if any) then build WanCell; cancel the job on build failure.

        Called only after fallible admit setup (encode / residency / enqueue) so those
        failures cannot leak a decode or sender thread.
        """
        stream_job = None
        on_finished = None
        if stream_decode:
            stream_group = None
            if source_rank is not None and source_rank != 0 and pp_group is not None:
                stream_group = self._ensure_stream_decode_group(pp_group, source_rank)
            stream_job = start_async_stream_decode(
                vae=self.vae,
                decoder=self._get_streaming_decoder(),
                session_id=kv_id,
                n_chunks=n_chunks,
                latent_shape=latent_shape,
                dtype=dtype,
                pp_rank=pp_rank,
                source_rank=source_rank,
                pp_group=pp_group,
                stream_group=stream_group,
                decode_tag=decode_tag,
            )
            on_finished = stream_job.on_finished
        on_finished = _chunk_timeline_callback(on_finished, timeline_file, finish_rank=finish_rank, pp_rank=pp_rank)
        try:
            adapter = WanCell(
                self.transformer,
                sampler=sampler,
                prompt_embeds=prompt_embeds,
                latent_shape=latent_shape,  # type: ignore[arg-type]
                seed=seed,
                device=device,
                dtype=dtype,
                on_finished=on_finished,
                serial_key=serial_key,
            )
        except Exception:
            if stream_job is not None:
                stream_job.cancel()
            raise
        return adapter, stream_job

    def emit_cell_output(self, adapter: Any, kv_id: str, stream_job: Any | None = None) -> DiffusionOutput:
        """Decode / gather after Cell plans finished."""
        param = next(self.transformer.parameters())
        device, dtype = param.device, param.dtype
        pp_rank, pp_group = resolve_pp_rank_and_group()
        if stream_job is not None:
            video = stream_job.join()
            if video is not None:
                return self._video_output(video)
            return self._placeholder_output(device, dtype)
        finished = self._gather_finished(adapter.finished, pp_group)
        chunk_map = finished.get(kv_id, {})
        if pp_rank == 0 and chunk_map:
            ordered = [chunk_map[c] for c in sorted(chunk_map)]
            latents = torch.cat(ordered, dim=2)
            video = self._decode_latents(latents)
            return self._video_output(video)
        return self._placeholder_output(device, dtype)

    def forward(self, req: OmniDiffusionRequest | DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Mainline request entry: plan, run the Cell executor, then emit.

        Compatible requests admitted into the same Cell context share one
        ``run_cell_pipeline`` so same-cell members can compute together. The
        executor groups members through ``CellAdapter.compatibility_key``; that
        key keeps Same-Step requests serial and isolates concurrent remote
        decodes. Session reset and release stay on the runner.
        """
        ctx = self._execution_context()
        previous = getattr(self, "_chunk_ctx", None)
        self._chunk_ctx = ctx
        stream_jobs: list[Any | None] = []
        try:
            members: dict[str, WanCell] = {}
            # Stable wave-local member identity, shared by sender and VAE owner.
            for decode_tag, item in enumerate(_as_request_list(req)):
                adapter, kv_id, _persist, stream_job = self.admit_cell_request(item, decode_tag=decode_tag)
                # Record before type checks so a later admit/type failure still cancels.
                stream_jobs.append(stream_job)
                if not isinstance(adapter, WanCell):
                    raise TypeError(f"expected WanCell adapter, got {type(adapter).__name__}")
                if kv_id in members:
                    raise ValueError(f"duplicate cell-batch member id {kv_id!r}")
                members[kv_id] = adapter
            # Batch organization lives in run_cell_pipeline; Pipeline only supplies members.
            run_cell_pipeline(ctx=ctx, members=members)
            return [
                self.emit_cell_output(adapter, kv_id, stream_job)
                for (kv_id, adapter), stream_job in zip(members.items(), stream_jobs, strict=True)
            ]
        except Exception:
            for stream_job in stream_jobs:
                if stream_job is not None:
                    stream_job.cancel()
            raise
        finally:
            self._chunk_ctx = previous

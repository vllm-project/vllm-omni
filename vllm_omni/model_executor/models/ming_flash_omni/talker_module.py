# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/dit.py
#
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/modules.py
# Ported from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/talker_module/cfm.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Partial of the following source code
# is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import logging
from collections.abc import Callable
from contextlib import nullcontext
from functools import cached_property
from queue import Queue

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase, Qwen2Config, Qwen2Model, StaticCache
from vllm.logger import init_logger
from vllm.platforms import current_platform
from x_transformers.x_transformers import RotaryEmbedding

from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.common.ming.cfm_solver import (
    apply_sway_sampling,
    get_epss_timesteps,
    integrate_cfm_steps,
)
from vllm_omni.model_executor.models.common.ming.dit_blocks import CondEmbedder, DiTBlock, FinalLayer
from vllm_omni.model_executor.models.model_local_kv import (
    ModelLocalKVScope,
    ModelLocalKVSpec,
    RowDriver,
    spec_from_hf_config,
)

from .talker_llm_graph import Qwen2DecodeGraph

logger = init_logger(__name__)


class DiT(nn.Module):
    """Diffusion model with a Transformer backbone for audio latent generation."""

    def __init__(
        self,
        in_channels: int = 64,
        hidden_size: int = 1024,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        llm_cond_dim: int = 896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.t_embedder = DiTTimestepEmbedding(hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.c_embedder = CondEmbedder(llm_cond_dim, hidden_size)
        if "spk_dim" in kwargs:
            self.spk_embedder = nn.Linear(kwargs["spk_dim"], hidden_size)
        else:
            self.spk_embedder = None
        self.hidden_size = hidden_size

        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = torch.cat([latent_history, x], dim=1)
        x = self.x_embedder(x)
        t = self.t_embedder(t).unsqueeze(1)
        c = self.c_embedder(c)
        y = t + c
        if spk_emb is None:
            assert self.spk_embedder is None
            x = torch.cat([y, x], dim=1)
        else:
            x = torch.cat([self.spk_embedder(spk_emb), y, x], dim=1)
        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])

        for block in self.blocks:
            x = block(x, None, rope)
        x = self.final_layer(x)
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        latent_history: torch.Tensor,
        spk_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance (doubles batch for CFG)."""
        x = torch.cat([x, x], dim=0)
        latent_history = torch.cat([latent_history, latent_history], dim=0)
        fake_latent = torch.zeros_like(c)
        c = torch.cat([c, fake_latent], dim=0)
        if t.ndim == 0:
            t = t.repeat(x.shape[0])
        if spk_emb is not None:
            spk_emb = torch.cat([spk_emb, spk_emb], dim=0)
        model_out = self.forward(x, t, c, latent_history, spk_emb)
        return model_out[:, -x.shape[1] :, :]


class CFM(nn.Module):
    """Conditional Flow Matching module for audio latent generation."""

    def __init__(self, model: nn.Module, steps: int = 10, sway_sampling_coef: float | None = -1.0):
        """
        Args:
            model: DiT used for the velocity prediction.
            steps: number of integration steps per sample call.
            sway_sampling_coef: coefficient used to skew the integration
                grid towards low-noise timesteps. Defaults to -1.0 which
                packs more steps near t=0, where prediction error is highest.
                Set to `None` to use the linear grid as-is.
        """
        super().__init__()
        self.model = model
        self.steps = steps
        self.sway_sampling_coef = sway_sampling_coef

    @torch.no_grad()
    def sample(
        self,
        llm_cond: torch.Tensor,
        lat_cond: torch.Tensor,
        y0: torch.Tensor,
        t: torch.Tensor,
        sde_args: torch.Tensor,
        sde_rnd: torch.Tensor,
        forward_with_cfg: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor
        ]
        | None = None,
    ):
        """Sample audio latent via ODE/SDE integration with CFG.

        Args:
            llm_cond: LLM hidden state (B, 1, hidden_size)
            lat_cond: latent history (B, his_patch_size, latent_dim)
            y0: initial noise (B, patch_size, latent_dim)
            t: timesteps from get_epss_timesteps
            sde_args: [cfg_strength, sigma, temperature]
            sde_rnd: random noise for SDE steps (steps, B, patch_size, latent_dim)
            forward_with_cfg: replacement for ``self.model.forward_with_cfg``,
                such as a compiled one
        """
        forward_with_cfg = forward_with_cfg or self.model.forward_with_cfg

        def fn(fn_t, x):
            pred_cfg = forward_with_cfg(x, fn_t, llm_cond, lat_cond, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * sde_args[0]

        t = apply_sway_sampling(t, self.sway_sampling_coef)
        return integrate_cfm_steps(fn, y0, t, sde_args, sde_rnd, self.steps)


class CFMGraphExecutor:
    """CUDA graph-accelerated executor for CFM + Aggregator + StopHead pipeline."""

    def __init__(self, config, cfm, aggregator, stop_head: nn.Linear):
        self.config = config
        self.cfm = cfm
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.initialized = False

        self.last_hidden_state_placeholder = None
        self.his_lat_placeholder = None
        self.randn_like_placeholder = None
        self.t_placeholder = None
        self.sde_args_placeholder = None
        self.sde_rnd_placeholder = None
        self.gen_lat_placeholder = None
        self.inputs_embeds_placeholder = None
        self.stop_out_placeholder = None
        self.graph = None
        self._last_sde_args: tuple[float, float, float] | None = None

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bat_size, his_patch_size, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.config.patch_size, z_dim), device=input_tensor.device, dtype=input_tensor.dtype
        )
        sde_rnd = torch.randn(
            (self.config.steps, *randn_tensor.shape), device=input_tensor.device, dtype=input_tensor.dtype
        )

        if not self.initialized:
            self._initialize_graph(input_tensor, his_lat, randn_tensor, sde_rnd)

        self.last_hidden_state_placeholder.copy_(input_tensor)
        self.his_lat_placeholder.copy_(his_lat)
        self.randn_like_placeholder.copy_(randn_tensor)
        # t_placeholder holds the constant EPSS schedule built at capture;
        # rebuilding it from a Python list each step is a blocking H2D copy.
        sde_args = (cfg_strength, sigma, temperature)
        if sde_args != self._last_sde_args:
            self.sde_args_placeholder[0] = cfg_strength
            self.sde_args_placeholder[1] = sigma
            self.sde_args_placeholder[2] = temperature
            self._last_sde_args = sde_args
        self.sde_rnd_placeholder.copy_(sde_rnd)

        self.graph.replay()

        gen_lat = torch.empty_like(self.gen_lat_placeholder)
        gen_lat.copy_(self.gen_lat_placeholder)

        inputs_embeds = torch.empty_like(self.inputs_embeds_placeholder)
        inputs_embeds.copy_(self.inputs_embeds_placeholder)

        stop_out = torch.empty_like(self.stop_out_placeholder)
        stop_out.copy_(self.stop_out_placeholder)

        return gen_lat, inputs_embeds, stop_out

    def _initialize_graph(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        randn_tensor: torch.Tensor,
        sde_rnd: torch.Tensor,
    ) -> None:
        self.last_hidden_state_placeholder = torch.empty_like(input_tensor)
        self.his_lat_placeholder = torch.empty_like(his_lat)
        self.randn_like_placeholder = torch.empty_like(randn_tensor)
        self.t_placeholder = get_epss_timesteps(self.config.steps, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_args_placeholder = torch.empty(3, device=input_tensor.device, dtype=input_tensor.dtype)
        self.sde_rnd_placeholder = torch.empty_like(sde_rnd)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=current_platform.get_global_graph_pool()):
            self.gen_lat_placeholder = self.cfm.sample(
                self.last_hidden_state_placeholder,
                self.his_lat_placeholder,
                self.randn_like_placeholder,
                self.t_placeholder,
                self.sde_args_placeholder,
                self.sde_rnd_placeholder,
            )
            self.inputs_embeds_placeholder = self.aggregator(self.gen_lat_placeholder)
            self.stop_out_placeholder = self.stop_head(self.last_hidden_state_placeholder[:, -1, :]).softmax(dim=-1)

        self.initialized = True


class CFMGraphExecutorPool:
    """Thread-safe pool of CFMGraphExecutors for concurrent inference."""

    def __init__(self, config, cfm, aggregator, stop_head: nn.Linear, pool_size: int = 1):
        self.pool = Queue(maxsize=pool_size)

        for _ in range(pool_size):
            executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)
            self.pool.put(executor)

    def acquire(self) -> CFMGraphExecutor:
        return self.pool.get()

    def release(self, executor: CFMGraphExecutor) -> None:
        self.pool.put(executor)

    def execute(
        self,
        input_tensor: torch.Tensor,
        his_lat: torch.Tensor,
        cfg_strength: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        executor = self.acquire()
        try:
            return executor.execute(
                input_tensor, his_lat, cfg_strength=cfg_strength, sigma=sigma, temperature=temperature
            )
        finally:
            self.release(executor)


########################################################################
# Audio Postprocess
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
########################################################################


@torch.no_grad()
def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform via linear interpolation (no torchaudio dep).

    Args:
        waveform: Tensor shaped ``(..., num_samples)``.
        orig_sr: Source sample rate (Hz); must be > 0.
        target_sr: Target sample rate (Hz); must be > 0.

    Raises:
        ValueError: If sample rates are non-positive, the waveform is empty,
            or the resampled length would round to zero.
    """
    if orig_sr <= 0:
        raise ValueError(f"orig_sr must be positive, got {orig_sr}")
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")
    if waveform.numel() == 0 or waveform.shape[-1] == 0:
        raise ValueError("waveform must contain at least one sample")
    if orig_sr == target_sr:
        return waveform

    ratio = target_sr / orig_sr
    new_len = int(waveform.shape[-1] * ratio)
    if new_len <= 0:
        raise ValueError(
            f"resampled waveform would be empty for input length {waveform.shape[-1]}, "
            f"orig_sr={orig_sr}, target_sr={target_sr}"
        )
    return torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0)


def trim_trailing_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    sil_th: float = 1e-3,
    tail_silence_s: float = 0.3,
) -> torch.Tensor:
    """Apply native Ming's final-chunk silence holder to a full waveform.

    Works on 2-D ``(channels, samples)`` or 3-D ``(batch, channels, samples)``
    tensors. Any other shape is returned unchanged.
    """
    if waveform.numel() == 0:
        return waveform

    original_dim = waveform.dim()
    if original_dim == 3:
        speech = waveform[:, 0, :]
    elif original_dim == 2:
        speech = waveform
    else:
        return waveform

    trimmed, _ = silence_holder(speech, sample_rate, last_chunk=True, sil_th=sil_th, last_sil=tail_silence_s)

    if original_dim == 3:
        return trimmed.unsqueeze(1)
    return trimmed


def silence_holder(
    speech: torch.Tensor,
    sample_rate: int,
    sil_cache: dict | None = None,
    last_chunk: bool = True,
    sil_th: float = 1e-3,
    last_sil: float = 0.3,
) -> tuple[torch.Tensor, dict]:
    """Ming-style streaming silence holder.

    Used during streaming VAE decode to defer emission of silent regions
    until a non-silent frame arrives (or the stream ends). ``sil_cache``
    is carried across chunks and updated in place.
    """
    if speech.numel() == 0:
        return speech, sil_cache or {"holder": [], "buffer": []}

    frame_step = int(sample_rate * 0.1)
    frame_size = int(sample_rate * 0.1)
    if sil_cache is None:
        sil_cache = {"holder": [], "buffer": []}

    if sil_cache["buffer"]:
        speech = torch.cat([*sil_cache["buffer"], speech], dim=-1)
        sil_cache["buffer"] = []

    if speech.shape[-1] < frame_size:
        sil_cache["buffer"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    num_frame = (speech.shape[-1] - frame_size) // frame_step + 1
    cur_len = (num_frame - 1) * frame_step + frame_size
    if speech.shape[-1] > cur_len:
        sil_cache["buffer"].append(speech[..., cur_len:])
        speech = speech[..., :cur_len]

    spe_frames = speech.unfold(-1, frame_size, frame_step)
    scores = spe_frames.abs().mean(dim=-1)
    scores = scores.mean(dim=list(range(scores.dim() - 1)))
    idx = scores.shape[0] - 1
    while idx >= 0 and scores[idx] <= sil_th:
        idx -= 1

    if idx < 0:
        sil_cache["holder"].append(speech)
        if last_chunk:
            speech = torch.cat(sil_cache["holder"] + sil_cache["buffer"], dim=-1)
            return speech[..., : int(last_sil * sample_rate)], sil_cache
        return torch.zeros((*speech.shape[:-1], 0), device=speech.device, dtype=speech.dtype), sil_cache

    non_sil_len = idx * frame_step + frame_size
    if last_chunk:
        non_sil_len += int(last_sil * sample_rate)
    non_sil_len = min(non_sil_len, speech.shape[-1])
    speech_out = torch.cat([*sil_cache["holder"], speech[..., :non_sil_len]], dim=-1)
    sil_cache["holder"] = []
    if non_sil_len < speech.shape[-1]:
        sil_cache["holder"].append(speech[..., non_sil_len:])
    return speech_out, sil_cache


########################################################################
# Prompt Builder
# Adapted from:
# https://github.com/inclusionAI/Ming/blob/e58533db227031990c5a6864dcf5f08fb53ed0d2/modeling_bailing_talker.py
########################################################################

_MUSIC_TAGS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


def _looks_like_music_prompt(text: str) -> bool:
    return all(tag in text for tag in _MUSIC_TAGS)


def build_tts_input(
    *,
    tokenizer: PreTrainedTokenizerBase,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    text: str,
    prompt: str,
    spk_emb: list[torch.Tensor] | None = None,
    instruction: str | None = None,
    prompt_text: str | None = None,
    prompt_wav_emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, input_ids) for one TTS segment.

    Args:
        tokenizer: HF tokenizer
        embed_tokens: The LLM's input-embedding module
        device: Device to place the returned tensors on.
        dtype: dtype for the returned `inputs_embeds`.
        text: Text to synthesize.
        prompt: System-level instruction prompt prepended to the user turn.
        spk_emb: Optional list of speaker embeddings already projected into
            LLM hidden dim; each is injected at a `<|vision_start|>` slot.
        instruction: Optional free-form instruction
        prompt_text: Reference text for zero-shot voice cloning.
        prompt_wav_emb: Reference-wav embeddings to inject.
    """
    spk_emb_prompt: list[int] = []
    if spk_emb is not None:
        for i in range(len(spk_emb)):
            spk_emb_prompt.extend(
                tokenizer.encode(f"  speaker_{i + 1}:")
                + tokenizer.encode("<|vision_start|>")
                + tokenizer.encode("<|vision_pad|>")
                + tokenizer.encode("<|vision_end|>\n")
            )

    instruction_prompt: list[int] = []
    if instruction is not None:
        instruction_prompt = tokenizer.encode(instruction) + tokenizer.encode("<|im_end|>")

    prompt_text_token: list[int] = []
    prompt_latent_token: list[int] = []
    if prompt_wav_emb is not None and prompt_text is not None:
        prompt_text_token = tokenizer.encode(prompt_text)
        prompt_latent_token = tokenizer.encode("<audioPatch>") * prompt_wav_emb.size(1)

    prompt2 = [] if _looks_like_music_prompt(text) else tokenizer.encode(" Text input:\n")

    input_part = (
        tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
        + tokenizer.encode("<|im_start|>user\n")
        + tokenizer.encode(prompt)
        + spk_emb_prompt
        + prompt2
        + prompt_text_token
        + tokenizer.encode(text)
        + tokenizer.encode("<|im_end|>\n")
        + tokenizer.encode("<|im_start|>assistant\n")
        + instruction_prompt
        + tokenizer.encode("<audio>")
        + prompt_latent_token
    )

    input_ids = torch.tensor(input_part, dtype=torch.long, device=device).unsqueeze(0)
    inputs_embeds = embed_tokens(input_ids).to(device=device, dtype=dtype)

    # inject speaker embeddings
    if spk_emb is not None:
        spk_token_id = tokenizer.encode("<|vision_start|>")
        assert len(spk_token_id) == 1, "<|vision_start|> must tokenize to a single id"
        # Positions from the host token list: no device round trip.
        spk_indices = [i for i, token in enumerate(input_part) if token == spk_token_id[0]]
        assert len(spk_indices) > 0, "expected at least one <|vision_start|> slot"
        for i, se in enumerate(spk_emb):
            inputs_embeds[0, spk_indices[i] + 1] = se

    # inject prompt-wav embeddings after <audio>
    if prompt_wav_emb is not None and prompt_text is not None:
        audio_token_id = tokenizer.encode("<audio>")
        assert len(audio_token_id) == 1, "<audio> must tokenize to a single id"
        audio_indices = [i for i, token in enumerate(input_part) if token == audio_token_id[0]]
        assert len(audio_indices) > 0, "expected at least one <audio> slot"
        start = audio_indices[0] + 1
        inputs_embeds[0, start : start + prompt_wav_emb.size(1), :] = prompt_wav_emb[0]

    return inputs_embeds, input_ids


########################################################################
# Audio Generator
########################################################################


class MingAudioGenerator:
    """Generator driving prefill -> AR decode -> VAE decode
    for a single TTS request. The generator is stateless across requests.
    """

    _STATIC_CACHE_LEN = 2048
    """Positions the talker's StaticCache is preallocated for.

    Read by both ``_init_kv_cache`` and ``model_local_kv_specs`` so the
    declaration cannot drift from the allocation.
    """

    def __init__(
        self,
        config,
        llm_config: Qwen2Config,
        model: Qwen2Model,
        cfm: CFM,
        aggregator: Aggregator,
        stop_head: torch.nn.Module,
        audio_vae: AudioVAE | None,
        patch_size: int,
        his_patch_size: int,
        latent_dim: int,
        cfg_strength: float,
        use_cuda_graphs: bool,
        enable_llm_cuda_graph: bool = False,
    ) -> None:
        self._config = config
        self._llm_config = llm_config
        self._model = model
        self._cfm = cfm
        self._aggregator = aggregator
        self._stop_head = stop_head
        self._audio_vae = audio_vae

        self.patch_size = patch_size
        self.his_patch_size = his_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength

        self._use_cuda_graphs = use_cuda_graphs
        self._enable_llm_cuda_graph = enable_llm_cuda_graph and use_cuda_graphs
        self._max_cache_len = int(getattr(config, "max_cache_len", self._STATIC_CACHE_LEN))
        self._llm_graph = (
            Qwen2DecodeGraph(self._model, max_cache_len=self._max_cache_len) if self._enable_llm_cuda_graph else None
        )

    def model_local_kv_specs(self) -> list[ModelLocalKVSpec]:
        """Declare the talker's preallocated decode cache.

        This is the case that a static table cannot hold. ``self._llm_config``
        is a ``Qwen2Config`` resolved from the checkpoint at load time, so
        layers, kv-heads and head-dim do not exist anywhere in this repo, and
        the dtype is whatever the weights loaded as. Reading them off the same
        objects ``_init_kv_cache`` builds from is the only way to get a real
        number.

        Unlike the other three this is not a ceiling that a short utterance
        stays under. transformers v5 initializes ``StaticLayer`` lazily, but
        the first write to a layer allocates all ``max_cache_len`` positions at
        once rather than growing, so a prefill brings the full extent into
        existence on step 0 of every generation.

        Width is fixed at one row, not ``max_num_seqs``. ``forward`` takes only
        ``runtime_additional_information[0]`` and runs the whole AR loop inline
        before returning, so one worker holds one cache no matter how many
        sequences the scheduler admits. Declaring this per-sequence -- which an
        earlier revision did, by deriving the multiplier from scope -- reported
        ``max_num_seqs`` times the real figure.

        With LLM graphs enabled, a request lock serializes the entire AR loop
        and the persistent cache is reset before and after each text segment.
        Without LLM graphs, each invocation owns a separate request-local cache.
        """
        max_cache_len = getattr(self, "_max_cache_len", self._STATIC_CACHE_LEN)
        resident = getattr(self, "_enable_llm_cuda_graph", False)
        return [
            spec_from_hf_config(
                self._llm_config,
                name="talker_llm_static_cache",
                dtype=next(self._model.parameters()).dtype,
                physical_capacity_positions=max_cache_len,
                capacity_source=f"talker config max_cache_len={max_cache_len}",
                scope=ModelLocalKVScope.MODEL if resident else ModelLocalKVScope.REQUEST,
                rows=RowDriver.FIXED,
                rows_fixed=1,
                rows_reason="one active Talker request per worker, including background streaming generation",
                allocation_note=(
                    "resident static cache reset between text segments"
                    if resident
                    else "full extent on first write per layer, not grown; rebuilt per text segment"
                ),
            )
        ]

    @cached_property
    def _sampler_pool(self) -> CFMGraphExecutorPool | None:
        device = next(self._model.parameters()).device
        if self._use_cuda_graphs and device.type == "cuda":
            return CFMGraphExecutorPool(self._config, self._cfm, self._aggregator, self._stop_head, pool_size=1)
        return None

    def duration_capped_steps(self, text_len: int, requested_max_steps: int) -> int:
        """Apply the original Ming duration heuristic as a cap on decode steps."""
        if self._audio_vae is None:
            return requested_max_steps

        # Transformers >=5.x may expose these config values as 0-d tensors.
        sample_rate = float(self._audio_vae.config.sample_rate)
        vae_patch_size = float(getattr(self._audio_vae.config, "patch_size", 4))
        hop_size = float(getattr(self._audio_vae.decoder, "hop_length", 320))
        seconds_per_step = (self.patch_size * vae_patch_size * hop_size) / sample_rate
        if seconds_per_step <= 0:
            return requested_max_steps

        max_duration_s = max(2.0, float(text_len) * (5818.0 / 16000.0))
        max_steps_by_duration = max(1, int(max_duration_s / seconds_per_step))
        return min(requested_max_steps, max_steps_by_duration)

    @torch.no_grad()
    def generate_latents(
        self,
        inputs_embeds: torch.Tensor,
        *,
        prompt_wav_lat: torch.Tensor | None = None,
        min_new_token: int = 3,
        max_steps: int = 1000,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
        use_static_cache: bool = True,
        stop_threshold: float = 0.5,
        on_latent=None,
    ) -> list[torch.Tensor]:
        """Autoregressive LLM + CFM sampling loop"""
        graph = self._llm_graph if use_static_cache else None
        with graph.request(inputs_embeds) if graph is not None else nullcontext():
            if cfg is None:
                cfg = self.cfg_strength
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype

            his_lat = self._init_his_lat(prompt_wav_lat, device, dtype)
            past_key_values, max_cache_len = self._init_kv_cache(use_static_cache, device, dtype)
            prefill_len = inputs_embeds.shape[1]
            if prefill_len >= max_cache_len:
                raise ValueError(f"TTS prompt has {prefill_len} positions, exceeding cache limit {max_cache_len}")
            all_latents: list[torch.Tensor] = []
            num_steps = min(max_steps, max_cache_len - prefill_len)
            # On CUDA, launch the next LLM step before reading each stop probability.
            overlap = device.type == "cuda"
            stop_host = torch.empty((1,), dtype=torch.float32, pin_memory=True) if overlap else None
            last_hs = None

            for step in range(num_steps):
                if last_hs is None:
                    last_hs = self.llm_step(
                        inputs_embeds,
                        step=step,
                        past_key_values=past_key_values,
                        use_static_cache=use_static_cache,
                        position=prefill_len + step - 1,
                    )
                step_hs = last_hs
                gen_lat, inputs_embeds, stop_out = self.cfm_sample_step(
                    step_hs, his_lat, cfg=cfg, sigma=sigma, temperature=temperature
                )
                his_lat = self._update_his_lat(his_lat, gen_lat)
                all_latents.append(gen_lat)

                latent_ready = None
                last_hs = None
                if overlap:
                    # Queue the next LLM step before reading the stop
                    # probability, so its launch overlaps the flow head on the
                    # GPU. The step draws no random numbers; if this latent
                    # stops the segment, its output is discarded.
                    stop_host.copy_(stop_out[0, 1:2].float(), non_blocking=True)
                    latent_ready = torch.cuda.Event()
                    latent_ready.record()
                    if step + 1 < num_steps:
                        last_hs = self.llm_step(
                            inputs_embeds,
                            step=step + 1,
                            past_key_values=past_key_values,
                            use_static_cache=use_static_cache,
                            position=prefill_len + step,
                        )
                    latent_ready.synchronize()
                    stop_prob = float(stop_host[0])
                else:
                    stop_prob = stop_out.cpu()[0, 1].item()

                if logger.isEnabledFor(logging.DEBUG):
                    if step % 50 == 0 or step < 5:
                        logger.debug(
                            "step=%d stop_prob=%.4f hs_norm=%.4f lat_norm=%.4f emb_norm=%.4f",
                            step,
                            stop_prob,
                            step_hs.float().norm().item(),
                            gen_lat.float().norm().item(),
                            inputs_embeds.float().norm().item(),
                        )

                stopped = step > min_new_token and stop_prob > stop_threshold
                last = stopped or step + 1 == num_steps
                if on_latent is not None:
                    if latent_ready is None:
                        on_latent(gen_lat, last)
                    else:
                        on_latent(gen_lat, last, latent_ready)
                if stopped:
                    logger.debug("Stopping at step %d with stop_prob=%.4f", step, stop_prob)
                    break

            return all_latents

    def cfm_sample_step(
        self,
        last_hidden_state: torch.Tensor,
        his_lat: torch.Tensor,
        *,
        cfg: float | None = None,
        sigma: float = 0.25,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one CFM sampling step.

        This is the CFM one-shot sampling step with CUDA-graph fast path.
        """
        if cfg is None:
            cfg = self.cfg_strength

        if self._sampler_pool is not None:
            return self._sampler_pool.execute(last_hidden_state, his_lat, cfg, sigma, temperature)

        bat_size, _, z_dim = his_lat.shape
        randn_tensor = torch.randn(
            (bat_size, self.patch_size, z_dim),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        t = get_epss_timesteps(self._config.steps, device=last_hidden_state.device, dtype=last_hidden_state.dtype)
        sde_rnd = torch.randn(
            (self._config.steps, *randn_tensor.shape),
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        sde_args = torch.tensor(
            [cfg, sigma, temperature],
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )

        gen_lat = self._cfm.sample(last_hidden_state, his_lat, randn_tensor, t, sde_args, sde_rnd)
        inputs_embeds = self._aggregator(gen_lat)
        stop_out = self._stop_head(last_hidden_state[:, -1, :]).softmax(dim=-1)

        return gen_lat, inputs_embeds, stop_out

    def decode_to_waveform(self, latents: list[torch.Tensor], stream_decode: bool = True) -> torch.Tensor:
        """Decode accumulated latents to waveform via AudioVAE."""
        if self._audio_vae is None:
            raise RuntimeError("AudioVAE not loaded. Cannot decode audio latents to waveform.")

        if stream_decode:
            return self._stream_decode(latents)

        all_lat = torch.cat(latents, dim=1)
        waveform, _, _ = self._audio_vae.decode(
            all_lat, use_cache=False, stream_state=(None, None, None), last_chunk=True
        )
        return waveform

    def llm_step(
        self,
        inputs_embeds: torch.Tensor,
        *,
        step: int,
        past_key_values: StaticCache | None,
        use_static_cache: bool,
        position: int | None = None,
    ) -> torch.Tensor:
        graph = self._llm_graph if use_static_cache else None
        if step > 0 and graph is not None and past_key_values is graph.cache and position is not None:
            output = graph.decode(inputs_embeds, position=position)
            if output is not None:
                return output[:, -1:, :]
        if step == 0 or not use_static_cache:
            outputs = self._model(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
            )
        else:
            # ``position`` is the cached length; reading it back from the
            # cache would synchronize the host every step.
            past_seen_tokens = position if position is not None else int(past_key_values.get_seq_length())
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            outputs = self._model(
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                cache_position=cache_position,
            )
        return outputs.last_hidden_state[:, -1:, :]

    def _init_his_lat(
        self, prompt_wav_lat: torch.Tensor | None, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        his_lat = torch.zeros(1, self.his_patch_size, self.latent_dim, device=device, dtype=dtype)
        if prompt_wav_lat is not None:
            start_index = self.his_patch_size - prompt_wav_lat.size(1)
            if start_index < 0:
                his_lat[:] = prompt_wav_lat[:, -start_index:, :]
            else:
                his_lat[:, start_index:, :] = prompt_wav_lat
        return his_lat

    def _init_kv_cache(
        self, use_static_cache: bool, device: torch.device, dtype: torch.dtype
    ) -> tuple[StaticCache | None, int]:
        max_cache_len = self._max_cache_len
        if not use_static_cache:
            return None, max_cache_len
        if self._llm_graph is not None and self._llm_graph.active:
            return self._llm_graph.cache, max_cache_len
        cache = StaticCache(
            config=self._llm_config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        return cache, max_cache_len

    def _update_his_lat(self, his_lat: torch.Tensor, gen_lat: torch.Tensor) -> torch.Tensor:
        if self.his_patch_size == self.patch_size:
            return gen_lat
        if self.his_patch_size > self.patch_size:
            return torch.cat([his_lat[:, self.patch_size - self.his_patch_size :], gen_lat], dim=1)
        raise NotImplementedError(f"his_patch_size ({self.his_patch_size}) < patch_size ({self.patch_size})")

    # VAE streaming decode
    def _stream_decode(self, latents: list[torch.Tensor]) -> torch.Tensor:
        """Decode new latents with request-local VAE and overlap-add state.

        The upsampler holds the first chunk until the next chunk supplies
        its right context. Only the final call flushes that pending chunk
        and the ISTFT overlap. This still returns one accumulated waveform;
        it does not enable streaming transport to the client.
        """
        past_key_values = None
        stream_state = (None, None, None)
        wav_chunks: list[torch.Tensor] = []

        for i, lat in enumerate(latents):
            last_chunk = i == (len(latents) - 1)
            speech, stream_state, past_key_values = self._audio_vae.decode(
                lat,
                past_key_values=past_key_values,
                use_cache=True,
                stream_state=stream_state,
                last_chunk=last_chunk,
            )
            if speech.numel() > 0:
                wav_chunks.append(speech.detach().float())

        if not wav_chunks:
            device = next(self._model.parameters()).device
            dtype = next(self._model.parameters()).dtype
            return torch.zeros((1, 1, 0), device=device, dtype=dtype)
        return torch.cat(wav_chunks, dim=-1)

    # Post-decode helper
    def trim_trailing_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        if self._audio_vae is None:
            return waveform
        return trim_trailing_silence(waveform, int(self._audio_vae.config.sample_rate))

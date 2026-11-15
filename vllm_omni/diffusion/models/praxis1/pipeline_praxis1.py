# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Native Praxis-1 pipeline: shared Wan2.2 VAE, Qwen3.5 text encoder and robot episodes.

One worker holds one model instance and an independent observation history
per explicitly named robot session. Requests carry a ``robot_obs`` mapping and
a ``session_id`` in ``sampling_params.extra_args``; the worker returns absolute
Cartesian pose/gripper targets as a structured ``actions`` payload.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context as vllm_forward_context
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import OmniAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.praxis1 import processing as single_arm
from vllm_omni.diffusion.models.praxis1.configuration_praxis1 import load_config
from vllm_omni.diffusion.models.praxis1.layers import ATTENTION_IMPLS
from vllm_omni.diffusion.models.praxis1.praxis1_transformer import Praxis1Transformer
from vllm_omni.diffusion.models.praxis1.processing import NormStats
from vllm_omni.diffusion.models.utils import create_transformers_model_with_vllm_linears
from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_extras.praxis1 import DEFAULT_MAX_NUM_STEPS, PraxisSessionOptions, SamplerSettings, decode_prompt

if TYPE_CHECKING:
    from vllm_omni.outputs.output_metadata import DiffusionPayload

DEFAULT_TEXT_CACHE_MAX_BYTES = 64 * 1024**2
DEFAULT_TEXT_CACHE_MAX_ENTRIES = 128
TEXT_HIDDEN_LAYER = -3
TEXT_PAD_TO_MULTIPLE_OF = 32


class Praxis1VAE(nn.Module):
    """The shared Wan2.2 TI2V autoencoder with the policy's latent contract.

    The policy latent is the normalized posterior mean ``(1, 48, T, h, w)``
    with 2x2 spatial patches folded into channels: ``(1, 192, T, h / 2, w / 2)``.
    """

    mean: Tensor
    inv_std: Tensor

    def __init__(self, path: str | Path, device: torch.device) -> None:
        super().__init__()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        self.model = OmniAutoencoderKLWan.from_wan22_state_dict(state)
        self.model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        config = self.model.config
        self.register_buffer("mean", torch.tensor(config.latents_mean, device=device, dtype=torch.float32))
        self.register_buffer(
            "inv_std", torch.tensor(config.latents_std, device=device, dtype=torch.float32).reciprocal()
        )

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> nn.Module:
        mean, inv_std = self.mean, self.inv_std
        super()._apply(fn, recurse)
        # The normalization constants stay float32; move them only.
        self.mean = mean.to(device=self.mean.device)
        self.inv_std = inv_std.to(device=self.inv_std.device)
        return self

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """Encode ``(1, 3, T, H, W)`` frames in ``[0, 255]`` to the bfloat16 policy latent."""
        x = ((frames.float() / 255) * 2 - 1).to(device=self.mean.device, dtype=torch.bfloat16)
        z = self.model.encode(x).latent_dist.mode().float()
        z = ((z - self.mean.view(1, -1, 1, 1, 1)) * self.inv_std.view(1, -1, 1, 1, 1)).bfloat16()
        b, c, t, h, w = z.shape
        return z.view(b, c, t, h // 2, 2, w // 2, 2).permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 4, t, h // 2, w // 2)

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """Decode a policy latent ``(1, 192, T, h, w)`` to uint8 frames ``(T', 3, H, W)``."""
        b, c, t, h, w = latent.shape
        z = (
            latent.bfloat16()
            .view(b, c // 4, 2, 2, t, h, w)
            .permute(0, 1, 4, 5, 2, 6, 3)
            .reshape(b, c // 4, t, h * 2, w * 2)
        )
        z = z / self.inv_std.view(1, -1, 1, 1, 1) + self.mean.view(1, -1, 1, 1, 1)
        frames = self.model.decode(z.bfloat16()).sample.float()
        frames = ((frames[0].permute(1, 0, 2, 3) + 1) * 0.5).clamp(0, 1)
        return (frames * 255).to(torch.uint8)


class Praxis1TextEncoder(nn.Module):
    """Qwen3.5 as a frozen text encoder: one hidden state per prompt token."""

    def __init__(self, path: str | Path, device: torch.device, max_prompt_tokens: int = 512) -> None:
        super().__init__()
        self.max_prompt_tokens = max_prompt_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        with set_default_torch_dtype(torch.bfloat16), device:
            self.model = create_transformers_model_with_vllm_linears(
                AutoModel, config, None, dtype=torch.bfloat16, device=device, prefix="text_encoder.model"
            )
        weights = safetensors_weights_iterator(sorted(str(f) for f in Path(path).glob("*.safetensors")), False)
        loaded = AutoWeightsLoader(self.model).load_weights(
            weights, mapper=WeightsMapper(orig_to_new_prefix={"model.": "", "lm_head.": None, "mtp.": None})
        )
        if missing := set(dict(self.model.named_parameters())) - loaded:
            raise ValueError(f"Missing Praxis-1 text encoder tensors: {sorted(missing)}")
        self.model.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> list[Tensor]:
        """Return one ``(1, S_i, C)`` bfloat16 tensor per prompt, padding removed."""
        batch = self.tokenizer(
            [single_arm.text_encoder_prompt(decode_prompt(p)) for p in prompts],
            padding=True,
            truncation=True,
            max_length=(self.max_prompt_tokens // TEXT_PAD_TO_MULTIPLE_OF + 1) * TEXT_PAD_TO_MULTIPLE_OF,
            add_special_tokens=True,
            pad_to_multiple_of=TEXT_PAD_TO_MULTIPLE_OF,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if batch["attention_mask"].sum(-1).max().item() > self.max_prompt_tokens:
            raise ValueError(f"Praxis-1 task exceeds max_prompt_tokens={self.max_prompt_tokens}")
        batch = batch.to(self.device)
        hidden = self.model(**batch, output_hidden_states=True).hidden_states[TEXT_HIDDEN_LAYER]
        return [hidden[i : i + 1, :n] for i, n in enumerate(batch["attention_mask"].sum(-1).tolist())]


@dataclass
class StepResult:
    """Outputs of one policy call."""

    commands: list[dict]
    action_rows_norm: np.ndarray
    action_rel: np.ndarray
    pred_video_latent: Tensor
    depth: int
    seed: int
    cleans: list[Tensor] = field(default_factory=list)


class Session:
    """The context state of one robot episode."""

    transport_owned = False

    def __init__(
        self,
        pipeline: Praxis1Pipeline,
        prompt: str = "",
        seed: int = 0,
        max_context_chunks: int = 1,
        control_mode: str = "droid_impedance",
        exec_horizon: int | None = 24,
        sampler: SamplerSettings | None = None,
    ) -> None:
        self.pipe = pipeline
        self.sampler = sampler or pipeline.sampler
        self.seed = int(seed)
        self.last_used = time.monotonic()
        config = pipeline.transformer.config
        self.state = single_arm.ServeSession(
            encode=pipeline.vae.encode,
            stats=pipeline.stats,
            device=pipeline.device,
            prompt=prompt,
            max_context_chunks=max_context_chunks,
            fps=float(config["chunking"].get("grid_hz", single_arm.FPS)),
            dims=config["normalization"].get("q99_normed_dims"),
            control_mode=control_mode,
            exec_horizon=exec_horizon,
            n_chunks=int(config["chunking"].get("n_chunks", single_arm.N_CHUNKS)),
            tile_size=(config["video"]["tile_height"], config["video"]["tile_width"]),
            aspect_tol=config["video"].get("aspect_tol", 0.03),
        )

    @torch.no_grad()
    def step(self, obs: dict) -> StepResult:
        """Run one policy call on ``obs`` and return the commands of the chunk."""
        state = self.state
        inputs = state.prepare(obs)
        video, actions = self._sample(inputs)
        commands, rows, rel = state.finish(inputs, actions)
        return StepResult(
            commands=commands,
            action_rows_norm=rows,
            action_rel=rel,
            pred_video_latent=video,
            depth=inputs.depth,
            seed=self.seed + inputs.chunk_index,
            cleans=inputs.cleans,
        )

    def _sample(self, inputs: single_arm.ChunkInputs) -> tuple[Tensor, Tensor]:
        """Sample one chunk. Return the video latent and actions."""
        text = self.pipe.encode_text(inputs.prompt)
        uncond = self.pipe.encode_uncond_text() if self.sampler.guided else None
        return self.pipe.transformer.sample_window_joint(
            text=text,
            text_uncond=uncond,
            anchor=inputs.anchor,
            cleans=inputs.cleans,
            proprio_all=inputs.proprio_all,
            seed=self.seed + inputs.chunk_index,
            settings=self.sampler,
        )

    @torch.no_grad()
    def decode_video(self, result: StepResult) -> Tensor:
        """Decode the predicted video of one call: one frame per command row."""
        latent = result.pred_video_latent
        if result.cleans:
            latent = torch.cat([*result.cleans, latent.to(result.cleans[0].device, result.cleans[0].dtype)], dim=2)
        frames = self.pipe.vae.decode(latent)
        return frames[-len(result.action_rows_norm) :][: len(result.commands)]


class Praxis1Pipeline(nn.Module):
    """One model instance, independent context per explicitly named robot session.

    Native video/action transformer, shared Wan2.2 VAE and a Transformers text
    encoder. The checkpoint streams through Omni's component weight loader.
    Robot sessions retain observation history, never model instances.
    """

    # Intercept the normal startup dummy request with model-valid observations.
    dummy_run_num_frames: ClassVar[int] = 1

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        if od_config.parallel_config.world_size != 1:
            raise ValueError("Praxis-1 currently requires one GPU (all parallel degrees must be 1).")
        if resolve_offload_strategy(od_config) is not OffloadStrategy.NONE:
            raise ValueError("Praxis-1 does not support Omni CPU offloading.")
        if od_config.quantization_config is not None:
            raise ValueError("Praxis-1 does not support Omni quantization.")
        if od_config.enable_sleep_mode or od_config.lora_path or od_config.step_execution:
            raise ValueError("Praxis-1 does not support sleep mode, LoRA or step execution.")
        if od_config.dtype != torch.bfloat16:
            raise ValueError("Praxis-1 requires dtype=bfloat16 for its checkpoint tensors.")
        self.device = get_local_device()

        config = dict(od_config.model_config)
        handshake = config.pop("policy_server_config", {})
        allowed = {
            "vae_weights",
            "text_encoder",
            "attention_impl",
            "max_sessions",
            "max_num_steps",
            "max_prompt_tokens",
            "session_idle_timeout",
            "warmup",
            "text_cache_max_bytes",
            "text_cache_max_entries",
        } | PraxisSessionOptions.__dataclass_fields__.keys()
        if unknown := config.keys() - allowed:
            raise ValueError(f"Unknown Praxis-1 model_config keys: {sorted(unknown)}")
        checkpoint = Path(od_config.model or "")
        if not (checkpoint / "config.json").is_file():
            raise ValueError("Praxis-1 requires a local checkpoint directory containing config.json.")
        # Companion paths default to files alongside the native checkpoint.
        vae_weights = Path(config.get("vae_weights") or checkpoint / "Wan2.2_VAE.pth")
        text_encoder = Path(config.get("text_encoder") or checkpoint / "text_encoder")
        if not vae_weights.is_file():
            raise ValueError(f"Praxis-1 VAE weights not found: {vae_weights}. Set model_config.vae_weights.")
        if not (text_encoder / "config.json").is_file():
            raise ValueError(f"Praxis-1 text encoder not found: {text_encoder}. Set model_config.text_encoder.")
        attention_impl = config.get("attention_impl", "flex")
        if attention_impl not in ATTENTION_IMPLS:
            raise ValueError(f"attention_impl must be one of {ATTENTION_IMPLS}")
        self.text_cache_max_bytes = config.get("text_cache_max_bytes", DEFAULT_TEXT_CACHE_MAX_BYTES)
        self.text_cache_max_entries = config.get("text_cache_max_entries", DEFAULT_TEXT_CACHE_MAX_ENTRIES)
        for name in ("text_cache_max_bytes", "text_cache_max_entries"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        max_prompt_tokens = config.get("max_prompt_tokens", 512)
        if type(max_prompt_tokens) is not int or max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        self.max_num_steps = config.get("max_num_steps", DEFAULT_MAX_NUM_STEPS)
        self.session_idle_timeout = config.get("session_idle_timeout", 300.0)
        self.startup_warmup = config.get("warmup", True)
        if type(self.max_num_steps) is not int or self.max_num_steps < 1:
            raise ValueError("max_num_steps must be a positive integer")
        if type(self.session_idle_timeout) not in (int, float) or not 0 < self.session_idle_timeout < float("inf"):
            raise ValueError("session_idle_timeout must be finite and positive")
        if type(self.startup_warmup) is not bool:
            raise ValueError("warmup must be Boolean")
        self.sampler = PraxisSessionOptions(
            **{k: config[k] for k in PraxisSessionOptions.__dataclass_fields__ if k in config}
        )
        if self.sampler.num_steps > self.max_num_steps:
            raise ValueError("Configured num_steps exceeds max_num_steps")
        self.max_sessions = config.get("max_sessions", 4)
        if type(self.max_sessions) is not int or self.max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if self.sampler.exec_horizon is None:
            raise ValueError("Deployment exec_horizon must be an integer in [1, 24]")
        native_config = load_config(checkpoint)
        self.embodiment = native_config["embodiment"]
        video_config = native_config["video"]
        # The OpenPI/Positronic handshake must advertise what this worker does.
        # Deployment settings may be omitted; checkpoint geometry and session
        # flags are required once a handshake is declared.
        required = {
            "image_resolution": list(single_arm.fit_resize_size(video_config["tile_width"])),
            "tile_size": [video_config["tile_height"], video_config["tile_width"]],
            "fps": native_config["chunking"]["grid_hz"],
            "video_antialias": True,
            "needs_session_id": True,
            "supports_close_session": True,
        }
        optional = {
            "action_horizon": self.sampler.exec_horizon,
            "first_action_horizon": min(21, self.sampler.exec_horizon),
            "control_mode": self.sampler.control_mode,
        }
        for key, value in {**optional, **required}.items():
            if key not in handshake and (key in optional or not handshake):
                continue
            actual = handshake.get(key)
            if actual != value or (isinstance(value, bool) and type(actual) is not bool):
                raise ValueError(f"policy_server_config.{key} must match the Praxis-1 worker ({value!r}).")
        self.vllm_config = get_current_vllm_config()
        with torch.device(self.device), set_default_torch_dtype(torch.bfloat16):
            self.transformer = Praxis1Transformer.from_config(native_config, attention_impl)
            self.transformer.eval().requires_grad_(False)
        with torch.device("cpu"), set_default_torch_dtype(torch.float32):
            self.vae = Praxis1VAE(vae_weights, self.device)
            self.text_encoder = Praxis1TextEncoder(text_encoder, self.device, max_prompt_tokens)
        norm = native_config["normalization"]
        self.stats = NormStats(
            **{
                key: np.asarray(norm[key], dtype=np.float32)
                for key in ("action_q01", "action_q99", "proprio_mean", "proprio_std")
            }
        )
        self.blank_caption = native_config["text_encoder"]["blank_caption"]
        self._text_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._text_cache_bytes = 0
        self._sessions: dict[str, Session] = {}
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=str(checkpoint),
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
                allow_patterns_overrides=["model-*.safetensors", "model.safetensors"],
            )
        ]

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        return {"transformer." + key for key in self.transformer.load_weights(weights)}

    def encode_text(self, prompt: str) -> Tensor:
        """Reuse conditioning within a bounded worker-local LRU cache.

        Count backing storage because the encoder returns a trimmed view of
        padded states. Oversized embeddings are used without being retained.
        """
        if prompt in self._text_cache:
            self._text_cache.move_to_end(prompt)
            return self._text_cache[prompt]
        embedding = self.text_encoder.encode([prompt])[0]
        size = embedding.untyped_storage().nbytes()
        if self.text_cache_max_entries and size <= self.text_cache_max_bytes:
            while self._text_cache and (
                len(self._text_cache) >= self.text_cache_max_entries
                or self._text_cache_bytes + size > self.text_cache_max_bytes
            ):
                self._text_cache_bytes -= self._text_cache.popitem(last=False)[1].untyped_storage().nbytes()
            self._text_cache[prompt] = embedding
            self._text_cache_bytes += size
        return embedding

    def encode_uncond_text(self) -> Tensor:
        return self.encode_text(self.blank_caption)

    def reset(self, session_id: str | None = None) -> None:
        """Release one episode, or all episodes for a worker-local reset."""
        if session_id is None:
            self._sessions.clear()
            self._text_cache.clear()
            self._text_cache_bytes = 0
        else:
            self._sessions.pop(session_id, None)

    def drop_session_state(self, session_id: str) -> None:
        """Expose the existing rollout cleanup hook without invoking the model."""
        self.reset(session_id)

    def _empty_actions(self) -> np.ndarray:
        return np.empty((0, 13), dtype=np.float32)

    def _warmup(self) -> DiffusionOutput:
        """Compile the configured context depths before the server is ready."""
        if self.startup_warmup:
            video = self.transformer.config["video"]
            h, w = video["tile_height"], video["tile_width"]
            obs = {
                **{f"image.{v}": np.zeros((25, h, w, 3), dtype=np.uint8) for v in ("exterior", "wrist")},
                "robot_state.ee_pose": np.tile([0.4, 0, 0.3, 1, 0, 0, 0], (25, 1)),
                "grip": np.full((25, 1), 0.5, dtype=np.float32),
                "task": "Move the gripper forward.",
                "descriptor": "praxis.sim",
            }
            session = Session(
                self, max_context_chunks=self.sampler.max_context_chunks, control_mode=self.sampler.control_mode
            )
            try:
                for _ in range(self.sampler.max_context_chunks + 1):
                    session.step(obs)
            finally:
                self.reset()
        return DiffusionOutput(output={"payload": {"actions": self._empty_actions()}})

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        with torch.device("cpu"), set_default_torch_dtype(torch.float32), vllm_forward_context(None, self.vllm_config):
            return self._forward(req)

    def _forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if req.is_dummy_run():
            return self._warmup()
        if req.num_reqs != 1:
            raise ValueError("Praxis-1 processes one observation at a time; set max_num_seqs=1.")
        params = req.sampling_params
        extra = params.extra_args or {}
        session_id = extra.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Praxis-1 requires a non-empty session_id to isolate robot episodes.")
        if extra.get("close_session"):
            self.reset(session_id)
            return DiffusionOutput(output={"payload": {"actions": self._empty_actions()}})
        obs = extra.get("robot_obs")
        if not isinstance(obs, Mapping):
            raise ValueError("Praxis-1 expects sampling_params.extra_args['robot_obs'] to be a mapping.")
        observation = dict(obs)
        for key in ("task", "prompt"):
            if key in observation:
                observation[key] = decode_prompt(observation[key])
        options = self.sampler
        if "praxis_session_options" in extra:
            overrides = extra["praxis_session_options"]
            if not isinstance(overrides, Mapping):
                raise ValueError("praxis_session_options must be a mapping of session option overrides.")
            if unknown := overrides.keys() - PraxisSessionOptions.__dataclass_fields__.keys():
                raise ValueError(f"Unknown praxis_session_options: {sorted(unknown)}")
            options = replace(self.sampler, **overrides)
        steps = params.num_inference_steps if params.num_inference_steps is not None else options.num_steps
        if type(steps) is not int or not 1 <= steps <= self.max_num_steps:
            raise ValueError(f"Praxis-1 num_inference_steps must be in [1, {self.max_num_steps}].")
        now = time.monotonic()
        for expired, state in list(self._sessions.items()):
            if state.transport_owned is not True and now - state.last_used >= self.session_idle_timeout:
                self.reset(expired)
        # OpenPI uses prompt; native Praxis uses task. Preserve native task if
        # both are supplied, and let Praxis validate images, poses and rig frame.
        if "task" not in observation and "prompt" in observation:
            observation["task"] = observation["prompt"]
        if extra.get("reset"):
            self.reset(session_id)
        available = session_id in self._sessions or len(self._sessions) < self.max_sessions
        if not available and not extra.get("reserve_session"):
            raise ValueError("Praxis-1 session limit reached. End an episode before opening another.")
        if session_id not in self._sessions and available:
            self._sessions[session_id] = Session(
                self,
                prompt=observation.get("task", ""),
                seed=params.seed if params.seed is not None else 0,
                max_context_chunks=0 if options.stateless else options.max_context_chunks,
                exec_horizon=options.exec_horizon,
                control_mode=options.control_mode,
            )
        if extra.get("reserve_session"):
            if available:
                self._sessions[session_id].transport_owned = True
            return DiffusionOutput(
                output={
                    "payload": {
                        "actions": self._empty_actions(),
                        "trajectory": {"session_available": available},
                    }
                }
            )
        session = self._sessions[session_id]
        if options.stateless:
            session.state.restart()
        # The seed belongs to the episode: the native session adds its chunk
        # index. Per-request random seeds must not restart that sequence.
        session.sampler = replace(
            self.sampler,
            num_steps=steps,
            **{k: getattr(options, k) for k in ("cfg_scale", "cfg_video_scale", "t_shift")},
        )
        succeeded = False
        try:
            result = session.step(observation)
            session.last_used = time.monotonic()
            commands = result.commands
            actions = np.asarray(
                [np.concatenate((cmd["robot_command"]["pose"], [cmd["target_grip"]])) for cmd in commands],
                dtype=np.float32,
            )
            if actions.ndim != 2 or actions.shape[1] != 13 or not np.isfinite(actions).all():
                raise ValueError("Praxis-1 returned invalid Cartesian pose/gripper actions.")
            payload: DiffusionPayload = {"actions": actions}
            trajectory: DiffusionPayload = {}
            # Diagnostics are opt-in: transferring video latents/frames for each
            # robot call adds synchronization and host transfer to the control path.
            if extra.get("return_diagnostics"):
                trajectory = {
                    "commands": commands,
                    "action_rows_norm": result.action_rows_norm,
                    "action_rel": result.action_rel,
                    "pred_video_latent": result.pred_video_latent.cpu(),
                    "depth": result.depth,
                    "seed": result.seed,
                }
            if extra.get("return_commands"):
                trajectory["commands"] = commands
            if extra.get("decode_video"):
                # Keep joint video/action diagnostics in the structured payload;
                # the image/video endpoints are not the robot policy protocol.
                trajectory["pred_frames"] = session.decode_video(result).cpu()
            if trajectory:
                payload["trajectory"] = trajectory
            succeeded = True
            return DiffusionOutput(output={"payload": payload})
        finally:
            # A failed step may have mutated the native context. Never reuse it.
            if not succeeded or extra.get("end_session") or observation.get("end_session"):
                self.reset(session_id)

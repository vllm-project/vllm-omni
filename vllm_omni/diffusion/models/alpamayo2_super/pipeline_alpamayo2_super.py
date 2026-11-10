# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Alpamayo 2 Super expert stage: flow-matching trajectory sampling over the reasoning KV.

The reasoning stage (``Alpamayo2SuperPolicyForConditionalGeneration``) hands over
each request's KV prefix, up to and including ``<|traj_future_start|>``, with
the prefix length and that token's M-RoPE position as KV-transfer metadata.
This pipeline samples one trajectory per request and runs every request of a
request batch together: prefixes are right-padded to a common length and the
padding is masked out.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

import torch
from torch import nn
from transformers import DynamicCache
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.alpamayo2_super.action_expert import Alpamayo2SuperActionExpert
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig

# The (64, 3) ego positions, under the output key robot policies use for their actions.
ACTIONS_KEY = "actions"

_WEIGHT_PREFIX = "expert."
_DEFAULT_STEPS = 10


def _expert_shards(model: str, revision: str | None) -> list[str | None]:
    """Checkpoint shards holding ``expert.*`` weights, so the VLM-only shards are not read or downloaded.

    The expert and the VLM share a checkpoint, and the expert sits in its last
    shards. ``model`` is a local directory or a Hugging Face repository;
    ``None`` (no index) reads every shard.
    """
    index = get_hf_file_to_dict("model.safetensors.index.json", model, revision)
    if index is None:
        return [None]
    return sorted({file for name, file in index["weight_map"].items() if name.startswith(_WEIGHT_PREFIX)})


class Alpamayo2SuperPipeline(nn.Module):
    """Flow-matching trajectory expert conditioned on a transferred VLM KV prefix.

    Each request carries the ego history in
    ``sampling_params.extra_args["robot_obs"]`` (``ego_history_xyz`` ``(16, 3)``,
    ``ego_history_rot`` ``(16, 3, 3)``); ``seed`` fixes the expert's noise and
    ``num_inference_steps`` (default 10) sets the Euler steps.
    """

    supports_request_batch: ClassVar[bool] = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        config = Alpamayo2SuperConfig.from_pretrained(od_config.model, revision=od_config.revision)
        # The prompt encodes the history as T - 1 steps of 3 coordinates.
        self.history_steps = config.tokens_per_history_traj // 3 + 1
        # Built in the serving dtype (the deploy config's, BF16 for "auto").
        with set_default_torch_dtype(od_config.dtype):
            self.expert = Alpamayo2SuperActionExpert(config.expert_config)
        # One source per shard: with explicit patterns the loader reads only the first one that matches.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=None if shard is None else [shard],
            )
            for shard in _expert_shards(od_config.model, od_config.revision)
        ]
        self.to(self.device)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if not name.startswith(_WEIGHT_PREFIX):
                continue  # VLM weights in a shard shared with the expert
            params[name].data.copy_(weight)
            loaded.add(name)
        return loaded

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        groups: dict[int, list[int]] = {}
        histories: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, request in enumerate(req.requests):
            params = request.sampling_params
            if params.past_key_values is None:
                if request.is_dummy_run():
                    # Engine warm-up and profiling requests carry no KV prefix.
                    outputs[index] = DiffusionOutput(output={ACTIONS_KEY: torch.zeros(64, 3)})
                else:
                    outputs[index] = DiffusionOutput(error="Alpamayo 2 Super expert stage received no KV prefix.")
                continue
            metadata = params.kv_metadata or {}
            if "prefix_len" not in metadata:
                outputs[index] = DiffusionOutput(
                    error="Reasoning ended without <|traj_future_start|> (for example at max_tokens); no trajectory."
                )
                continue
            prefix_len = int(metadata["prefix_len"])
            if int(metadata.get("seq_len", prefix_len)) != prefix_len:
                outputs[index] = DiffusionOutput(
                    error=f"KV transfer covers {metadata.get('seq_len')} tokens, expected {prefix_len}."
                )
                continue
            try:
                histories[index] = self._history(params.extra_args)
            except ValueError as exc:
                outputs[index] = DiffusionOutput(error=str(exc))
                continue
            steps = params.num_inference_steps or _DEFAULT_STEPS
            if steps < 1:
                outputs[index] = DiffusionOutput(error=f"num_inference_steps must be >= 1, got {steps}")
                continue
            groups.setdefault(steps, []).append(index)
        for steps, indices in groups.items():
            trajectories = self._sample(
                [req.requests[index] for index in indices], [histories[index] for index in indices], steps
            )
            for index, xyz in zip(indices, trajectories, strict=True):
                outputs[index] = DiffusionOutput(output={ACTIONS_KEY: xyz})
        # The received prefixes are freed now but stay reserved by this process's
        # caching allocator. The reasoning stage shares the GPU and stages each KV
        # handoff outside its memory budget, so hand the memory back.
        torch.accelerator.empty_cache()
        return outputs

    def _batched_prefix(self, requests: list[OmniDiffusionRequest]) -> tuple[DynamicCache, torch.Tensor, int]:
        """Right-pad the requests' prefixes into one cache; returns it, the lengths and the padded length."""
        # forward() has checked that each transfer covers exactly its prefix.
        lengths = [int(request.sampling_params.kv_metadata["prefix_len"]) for request in requests]
        padded_len = max(lengths)
        received = [request.sampling_params.past_key_values for request in requests]
        cache = DynamicCache()
        for layer in range(len(received[0].key_cache)):
            # Received per layer as [seq_len, kv_heads, head_dim]; the expert's
            # Hugging Face attention takes [batch, kv_heads, seq_len, head_dim].
            heads, head_dim = received[0].key_cache[layer].shape[1:]
            dtype = received[0].key_cache[layer].dtype
            key = torch.zeros(len(requests), heads, padded_len, head_dim, dtype=dtype, device=self.device)
            value = torch.zeros_like(key)
            for row, (kv, length) in enumerate(zip(received, lengths, strict=True)):
                key[row, :, :length] = kv.key_cache[layer][:length].transpose(0, 1)
                value[row, :, :length] = kv.value_cache[layer][:length].transpose(0, 1)
                kv.key_cache[layer] = kv.value_cache[layer] = None  # release as we copy
            cache.update(key, value, layer)
        return cache, torch.tensor(lengths, device=self.device), padded_len

    def _sample(
        self,
        requests: list[OmniDiffusionRequest],
        histories: list[tuple[torch.Tensor, torch.Tensor]],
        num_steps: int,
    ) -> list[torch.Tensor]:
        """Flow-matching (Euler) sampling of one trajectory per request."""
        expert = self.expert
        device = self.device
        batch = len(requests)
        num_waypoints, action_dim = expert.action_space.get_action_space_dims()
        prefix, lengths, padded_len = self._batched_prefix(requests)

        trigger = torch.tensor(
            [request.sampling_params.kv_metadata["trigger_position"] for request in requests], device=device
        )  # [batch, 3]
        position_ids = (trigger.T[:, :, None] + 1 + torch.arange(num_waypoints, device=device)).long()
        dtype = expert.action_out_proj.weight.dtype
        # Waypoints attend to their own prefix and to each other; padding is masked.
        attention_mask = torch.zeros((batch, 1, num_waypoints, padded_len + num_waypoints), device=device, dtype=dtype)
        padding = torch.arange(padded_len, device=device)[None, :] >= lengths[:, None]
        attention_mask[:, 0, :, :padded_len].masked_fill_(padding[:, None, :], torch.finfo(dtype).min)

        # Per-request noise, so a request's trajectory does not depend on its batch.
        action = torch.cat(
            [
                torch.randn(1, num_waypoints, action_dim, device=device, generator=self._generator(request))
                for request in requests
            ]
        )
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        for i in range(num_steps):
            hidden = expert.expert(
                inputs_embeds=expert.action_in_proj(action, time_steps[i].expand(batch, 1, 1)),
                position_ids=position_ids,
                past_key_values=prefix,
                attention_mask=attention_mask,
                use_cache=True,
                is_causal=not expert.non_causal,
            ).last_hidden_state
            prefix.crop(padded_len)
            velocity = expert.action_out_proj(hidden).view(batch, num_waypoints, action_dim)
            # A (1, 1, 1) step keeps the update in float32, as in the reference.
            action = action + (time_steps[i + 1] - time_steps[i]).view(1, 1, 1) * velocity

        history_xyz = torch.stack([xyz for xyz, _ in histories])
        history_rot = torch.stack([rot for _, rot in histories])
        trajectory_xyz, _ = expert.action_space.action_to_traj(action, history_xyz, history_rot)
        return list(trajectory_xyz.cpu().unbind(0))

    def _generator(self, request: OmniDiffusionRequest) -> torch.Generator:
        params = request.sampling_params
        if isinstance(params.generator, torch.Generator):
            return params.generator
        if isinstance(params.generator, list) and params.generator:
            return params.generator[0]  # one request per OmniDiffusionRequest
        generator = torch.Generator(device=self.device)
        # The seed fixes the expert's noise; without one, draw a fresh random seed.
        if params.seed is None:
            generator.seed()
        else:
            generator.manual_seed(int(params.seed))
        return generator

    def _history(self, extra_args: Mapping[str, Any] | None) -> tuple[torch.Tensor, torch.Tensor]:
        """The request's ego history as ``(T, 3)`` positions and ``(T, 3, 3)`` rotations.

        Raises ``ValueError`` for a missing or malformed history, so that only
        this request fails.
        """
        observation = (extra_args or {}).get("robot_obs")
        if not isinstance(observation, Mapping):
            raise ValueError("Alpamayo 2 Super expert stage needs extra_args['robot_obs'].")
        histories = []
        for key, shape in (
            ("ego_history_xyz", (self.history_steps, 3)),
            ("ego_history_rot", (self.history_steps, 3, 3)),
        ):
            try:
                history = torch.as_tensor(observation[key], dtype=torch.float32)
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"robot_obs[{key!r}] is missing or not numeric.") from exc
            # Leading batch dims of size 1 are dropped.
            while history.dim() > len(shape) and history.shape[0] == 1:
                history = history[0]
            if tuple(history.shape) != shape:
                raise ValueError(f"robot_obs[{key!r}] must have shape {shape}, got {tuple(history.shape)}.")
            if not torch.isfinite(history).all():
                raise ValueError(f"robot_obs[{key!r}] has non-finite values.")
            histories.append(history.to(self.device))
        return histories[0], histories[1]

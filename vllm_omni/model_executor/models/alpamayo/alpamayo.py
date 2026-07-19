# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alpamayo single-model class for vLLM-Omni.

``Alpamayo15ForConditionalGeneration`` is the SGLang-style single model: a vLLM
``Qwen3VLForConditionalGeneration`` subclass that **also** holds the
flow-matching ``expert`` (a vLLM ``Qwen3Model``) and the action heads
(``action_in_proj`` / ``action_out_proj`` / ``action_space``). Mirrors the
sglang reference ``AlpamayoR1`` (``python/sglang/srt/models/alpamayo_r1.py``),
not vllm-omni's BAGEL two-stage pattern — Alpamayo is "VLM + flow-matching
head", not two distinct stages.

Responsibilities:
- Discrete trajectory-token logit masking; checkpoint weight routing.
- ``expert`` + action heads alongside the VLM, with full ``load_weights``
  coverage for the Alpamayo-1.5 checkpoint.
- Bidirectional expert + KV sharing with the VLM via
  ``Attention(attn_type=ENCODER_ONLY, kv_sharing_target_layer_name=…)``.
- Trigger-token flow-matching dispatch inside ``forward``.
"""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
from vllm.model_executor.models.utils import maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention
from vllm_omni.model_executor.models.alpamayo.action_space import (
    PerWaypointActionInProjV2,
    UnicycleAccelCurvatureActionSpace,
)
from vllm_omni.model_executor.models.alpamayo.processor import (
    AlpamayoDummyInputsBuilder,
    AlpamayoMultiModalProcessor,
    AlpamayoProcessingInfo,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

# Default discrete-trajectory-token layout (overridden from hf_config when present).
_DEFAULT_TRAJ_TOKEN_START_IDX = 151669
_DEFAULT_TRAJ_VOCAB_SIZE = 4000

# Guard rails for the request-supplied ``robot_obs`` (arrives via vllm_xargs ->
# sampling_params.extra_args, possibly as a JSON string). These bound the work
# the model path does on untrusted input: an oversized or malformed payload is
# rejected early (logged + skipped) instead of reaching tokenization / tensor
# construction. A real ego history is a few hundred 3-vectors, well under these.
_ROBOT_OBS_MAX_JSON_BYTES = 1 << 20  # 1 MiB cap on the JSON string form
_ROBOT_OBS_MAX_ELEMS = 200_000  # cap on element count of each history array


def _validate_robot_obs(robot_obs: object) -> dict | None:
    """Return a well-formed ``robot_obs`` dict, or ``None`` if it fails validation.

    Checks (fail early, never raise into the model path):
      - is a dict carrying both ``ego_history_xyz`` and ``ego_history_rot``;
      - each array is list/ndarray/tensor with a bounded element count.
    """
    if not isinstance(robot_obs, dict):
        logger.warning("Alpamayo15: robot_obs is not a dict (got %s); skipping fusion.", type(robot_obs).__name__)
        return None
    hx = robot_obs.get("ego_history_xyz")
    hr = robot_obs.get("ego_history_rot")
    if hx is None or hr is None:
        logger.warning("Alpamayo15: robot_obs missing ego_history_xyz/ego_history_rot; skipping fusion.")
        return None
    for key, val in (("ego_history_xyz", hx), ("ego_history_rot", hr)):
        if not isinstance(val, (list, np.ndarray, torch.Tensor)):
            logger.warning(
                "Alpamayo15: robot_obs['%s'] has unsupported type %s; skipping fusion.", key, type(val).__name__
            )
            return None
        try:
            n_elems = int(np.asarray(val).size) if not isinstance(val, torch.Tensor) else int(val.numel())
        except Exception as e:
            logger.warning("Alpamayo15: robot_obs['%s'] is not array-like (%s); skipping fusion.", key, e)
            return None
        if n_elems > _ROBOT_OBS_MAX_ELEMS:
            logger.warning(
                "Alpamayo15: robot_obs['%s'] has %d elements (> %d cap); skipping fusion.",
                key,
                n_elems,
                _ROBOT_OBS_MAX_ELEMS,
            )
            return None
    return robot_obs


@MULTIMODAL_REGISTRY.register_processor(
    AlpamayoMultiModalProcessor,
    info=AlpamayoProcessingInfo,
    dummy_inputs=AlpamayoDummyInputsBuilder,
)
class Alpamayo15ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Alpamayo Stage-0 autoregressive model (Qwen3-VL backbone).

    Registered with the Alpamayo multimodal processor: it reuses Qwen3-VL's
    image preprocess and injects the discrete trajectory + special tokens into the
    tokenizer.
    """

    # vLLM's mm-input path drops input_ids by default (passes only inputs_embeds);
    # Alpamayo needs the raw token ids in forward() to detect the trajectory
    # trigger token, so opt into the runner's raw-token path.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Tells the worker's extract_multimodal_outputs to unwrap OmniOutput
        # returned by forward() and route the multimodal_outputs dict to the
        # engine output processor (-> client multimodal_output[...]).
        self.have_multimodal_outputs = True
        config = vllm_config.model_config.hf_config

        # --- Flow-matching expert + action heads (sglang single-model pattern).
        # The expert is a Qwen3 text model sized from text_config + expert_cfg overlay.
        # It is constructed here so the checkpoint expert.* tensors load into the same
        # nn.Module that holds the VLM; the bidirectional / KV-sharing forward semantics
        # are wired below.
        expert_text_config = copy.deepcopy(config.text_config)
        for k, v in (getattr(config, "expert_cfg", None) or {}).items():
            setattr(expert_text_config, k, v)
        # Build the expert as a stock vLLM Qwen3Model — same shape as sglang's
        # ``self.expert = Qwen3Model(expert_config, ...)``. Setting
        # ``is_causal=False`` makes vLLM's Qwen3DecoderLayer route every attention
        # through ``EncoderOnlyAttention``, whose ``get_kv_cache_spec()`` returns
        # ``None`` — so the engine allocates ZERO paged KV slots for the expert
        # (~36 layers' worth of GPU memory saved). We never call
        # ``self.expert.forward`` anyway; flow matching reads weights directly
        # off the layers and runs FA3 in ``_run_flow_matching_inline``
        # (see "Why manual" docstring there).
        expert_text_config.is_causal = False
        self.expert = Qwen3Model(
            vllm_config=vllm_config.with_hf_config(expert_text_config),
            prefix=maybe_prefix(prefix, "expert"),
        )
        # Expert consumes pre-projected action embeddings (no token embedding).
        if getattr(self.expert, "embed_tokens", None) is not None:
            self.expert.embed_tokens = None
        # Per-layer VLM-side cache name for manual prefix-KV reads (with stock
        # Qwen3Model we derive it from this model's prefix directly; the engine
        # registers each VLM attention under this same name).
        vlm_layer_prefix = maybe_prefix(prefix, "language_model.model.layers")
        n_expert_layers = len(self.expert.layers)
        self._vlm_layer_names = tuple(f"{vlm_layer_prefix}.{i}.self_attn.attn" for i in range(n_expert_layers))

        # Per-layer attention kernels for the expert. Stock Qwen3's
        # EncoderOnlyAttention doesn't read paged KV, and vLLM has no
        # "extend with prefix" API, so the expert path bypasses
        # ``layer.self_attn.attn`` and computes its own attention over
        # ``cat[VLM_prefix_K/V ; action_K/V]``. ``DiffusionAttention`` is the
        # right wrapper: it auto-selects FA3 (or SDPA fallback) per platform,
        # supports causal=False + GQA natively, and carries zero learnable
        # parameters (pure dispatch shim) so weight loading is unaffected.
        sample_attn = self.expert.layers[0].self_attn
        self.expert_attns = nn.ModuleList(
            [
                DiffusionAttention(
                    num_heads=sample_attn.num_heads,
                    head_size=sample_attn.head_dim,
                    causal=False,
                    softmax_scale=sample_attn.scaling,
                    num_kv_heads=sample_attn.num_kv_heads,
                    prefix=maybe_prefix(prefix, f"expert_attns.{i}"),
                )
                for i in range(n_expert_layers)
            ]
        )

        traj_tok_cfg = config.traj_tokenizer_cfg
        action_space_cfg = traj_tok_cfg["action_space_cfg"]
        self.n_waypoints = int(action_space_cfg["n_waypoints"])
        self.action_dim = len(traj_tok_cfg["dims_max"])
        aip = config.action_in_proj_cfg
        # Alpamayo-1.5 stores Fourier `freqs` as non-persistent (recomputed); R1 is
        # persistent. Default to R1 unless the config asks otherwise.
        fourier_persistent = bool(getattr(config, "fourier_persistent", True))
        self.action_in_proj = PerWaypointActionInProjV2(
            in_dims=[self.n_waypoints, self.action_dim],
            out_dim=expert_text_config.hidden_size,
            hidden_size=aip["hidden_size"],
            num_enc_layers=aip["num_enc_layers"],
            max_freq=aip["max_freq"],
            num_fourier_feats=aip["num_fourier_feats"],
            fourier_persistent=fourier_persistent,
        )
        self.action_out_proj = nn.Linear(expert_text_config.hidden_size, self.action_dim)

        action_space_kwargs = {
            k: v for k, v in action_space_cfg.items() if k not in ("_target_", "_recursive_", "n_waypoints")
        }
        self.action_space = UnicycleAccelCurvatureActionSpace(
            n_waypoints=self.n_waypoints,
            **action_space_kwargs,
        )
        diffusion_cfg = getattr(config, "diffusion_cfg", None) or {}
        self.num_inference_steps = int(diffusion_cfg.get("num_inference_steps", 10))

        # Discrete trajectory tokens are input-only -> mask them at generation.
        self.traj_token_start_idx = int(getattr(config, "traj_token_start_idx", _DEFAULT_TRAJ_TOKEN_START_IDX))
        self.traj_vocab_size = int(getattr(config, "traj_vocab_size", _DEFAULT_TRAJ_VOCAB_SIZE))
        self.traj_mask_start = self.traj_token_start_idx
        self.traj_mask_end = self.traj_token_start_idx + self.traj_vocab_size

        # History-fusion plumbing: each request's prompt arrives with
        # ``tokens_per_history_traj`` (=48) copies of ``<|traj_history|>``
        # as placeholders; ``_fuse_history_inplace`` (called from forward())
        # substitutes them with delta-encoded ego_history tokens read from
        # ``sampling_params.extra_args["robot_obs"]``.
        from vllm_omni.model_executor.models.alpamayo.action_space import (
            DeltaTrajectoryTokenizer,
        )

        self._hist_traj_tokenizer = DeltaTrajectoryTokenizer()
        self._history_placeholder_id = int((getattr(config, "traj_token_ids", None) or {}).get("history", 155684))
        self._extra_args_per_req: list[dict | None] | None = None

        # Trajectory trigger / stop tokens (default to the Alpamayo special ids;
        # overridden from config.traj_token_ids when present).
        traj_token_ids = getattr(config, "traj_token_ids", None) or {}
        self.traj_future_start_token_id = int(traj_token_ids.get("future_start", 155681))
        # <|im_end|> doubles as the trajectory force-stop token.
        eos = getattr(getattr(config, "text_config", None), "eos_token_id", 151645)
        self.traj_force_stop_token_id = int(eos if eos is not None else 151645)

        logger.info(
            "Alpamayo15: traj-token mask [%d, %d), future_start=%d, stop=%d",
            self.traj_mask_start,
            self.traj_mask_end,
            self.traj_future_start_token_id,
            self.traj_force_stop_token_id,
        )

    # ------------------------------------------------------------------ #
    # Trigger detection + action-token mRoPE positions (pure helpers;
    # forward-time dispatch is wired in forward() below).
    # ------------------------------------------------------------------ #
    @staticmethod
    def find_trigger_indices(
        input_ids: torch.Tensor,
        traj_future_start_token_id: int,
        traj_force_stop_token_id: int,
        has_history_traj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the batch indices whose ``input_ids`` is the flow-matching trigger.

        Mirrors SGLang ``alpamayo_r1.py::forward`` decode branch: the trigger fires
        when the trigger token is the *input* to the current decode step (i.e. it
        was generated in the previous step). Two trigger conditions:
          1. ``input_ids[i] == traj_future_start_token_id``
          2. ``has_history_traj[i] and input_ids[i] == traj_force_stop_token_id``

        ``input_ids`` is ``(batch,)`` (one-token-per-request in decode mode).
        Returns a 1-D ``LongTensor`` of triggered batch indices (possibly empty).
        """
        flat = input_ids.view(-1)
        cond_a = flat == int(traj_future_start_token_id)
        if has_history_traj is None:
            cond = cond_a
        else:
            cond_b = (flat == int(traj_force_stop_token_id)) & has_history_traj.view(-1)
            cond = cond_a | cond_b
        return torch.nonzero(cond, as_tuple=False).view(-1)

    @staticmethod
    def action_token_positions(
        current_mrope_positions: torch.Tensor,
        active_indices: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Compute mRoPE positions for the action tokens of triggered requests.

        ``current_mrope_positions``: ``(3, total_batch)`` — the mRoPE positions
        the engine fed into this decode step (one per active request).
        ``active_indices``: ``(bstar,)`` triggered batch indices.
        Returns ``(3, bstar * n_waypoints)`` — for each triggered request, the
        action tokens occupy positions ``cur+1, cur+2, ..., cur+n_waypoints``
        (mirrors SGLang ``_run_flow_matching`` positions_list logic).
        """
        device = current_mrope_positions.device
        offsets = torch.arange(1, n_waypoints + 1, device=device)
        out = []
        for idx in active_indices.tolist():
            cur = current_mrope_positions[:, idx]  # (3,)
            out.append(cur.unsqueeze(1) + offsets.unsqueeze(0))  # (3, n_waypoints)
        if not out:
            return torch.empty(3, 0, dtype=current_mrope_positions.dtype, device=device)
        return torch.cat(out, dim=1)

    @staticmethod
    def _apply_traj_logit_mask(logits: torch.Tensor | None, start: int, end: int) -> torch.Tensor | None:
        """Mask the discrete-trajectory-token logit range to ``-inf``.

        ``logits`` has shape ``(num_tokens, vocab_size)``. Out-of-range bounds
        are clamped so a wrong-sized vocab never raises.
        """
        if logits is None:
            return None
        vocab = logits.shape[-1]
        lo = max(0, min(start, vocab))
        hi = max(0, min(end, vocab))
        if hi > lo:
            logits[..., lo:hi] = float("-inf")
        return logits

    # ------------------------------------------------------------------ #
    # Server-side history fusion: substitute <|traj_history|> placeholders
    # in the prompt with delta-encoded ego_history tokens from extra_args.
    #
    # Runs inside forward() (before the VLM backbone), NOT via a model-runner
    # hook: the per-request ``sampling_params.extra_args["robot_obs"]`` arrives
    # through the engine's generic ``sampling_extra_args`` forward kwarg
    # (enabled by ``has_sampling_extra_args`` in the deploy config). Per-request
    # token boundaries come from the attention metadata's ``query_start_loc``.
    # ------------------------------------------------------------------ #
    def _fuse_history_inplace(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        sampling_extra_args: list[dict | None] | None,
    ) -> None:
        if input_ids is None or not sampling_extra_args:
            return
        import json

        from vllm.forward_context import get_forward_context

        from vllm_omni.model_executor.models.alpamayo.processing import (
            tokenize_history_trajectory,
        )

        # Per-request query boundaries in the flat token stream. Decode steps
        # carry no placeholders, so their slices simply find no mask hits.
        try:
            meta = get_forward_context().attn_metadata[self._vlm_layer_names[0]]
            qsl = [int(v) for v in meta.query_start_loc.tolist()]
        except Exception:
            qsl = [0, int(input_ids.shape[0])]

        for i in range(len(qsl) - 1):
            ea = sampling_extra_args[i] if i < len(sampling_extra_args) else None
            if not ea:
                continue
            # ``robot_obs`` carries ego_history as a dict (Python clients) or a
            # JSON string (HTTP clients, whose vllm_xargs values must be flat).
            # Validate size + schema before consuming it in the model path.
            robot_obs = ea.get("robot_obs")
            if isinstance(robot_obs, str):
                if len(robot_obs) > _ROBOT_OBS_MAX_JSON_BYTES:
                    logger.warning(
                        "Alpamayo15: robot_obs JSON is %d bytes (> %d cap); skipping fusion.",
                        len(robot_obs),
                        _ROBOT_OBS_MAX_JSON_BYTES,
                    )
                    continue
                try:
                    robot_obs = json.loads(robot_obs)
                except Exception as je:
                    logger.warning("Alpamayo15: bad robot_obs JSON string: %s", je)
                    continue
            if not robot_obs:
                continue
            robot_obs = _validate_robot_obs(robot_obs)
            if robot_obs is None:
                continue
            hx = robot_obs["ego_history_xyz"]
            hr = robot_obs["ego_history_rot"]

            lo, hi = qsl[i], qsl[i + 1]
            req_slice = input_ids[lo:hi]
            mask = req_slice == self._history_placeholder_id
            n_placeholders = int(mask.sum().item())
            if n_placeholders == 0:
                continue

            hx_t = torch.as_tensor(hx) if isinstance(hx, (list, np.ndarray)) else hx
            hr_t = torch.as_tensor(hr) if isinstance(hr, (list, np.ndarray)) else hr
            while hx_t.ndim < 4:
                hx_t = hx_t.unsqueeze(0)
            while hr_t.ndim < 5:
                hr_t = hr_t.unsqueeze(0)
            delta_ids = (
                tokenize_history_trajectory(
                    self._hist_traj_tokenizer,
                    {"ego_history_xyz": hx_t.float().cpu(), "ego_history_rot": hr_t.float().cpu()},
                    start_idx=self.traj_token_start_idx,
                )
                .flatten()
                .to(device=req_slice.device, dtype=req_slice.dtype)
            )
            if delta_ids.numel() != n_placeholders:
                logger.warning(
                    "Alpamayo15: history fusion size mismatch — %d placeholders vs %d delta tokens; skipping.",
                    n_placeholders,
                    delta_ids.numel(),
                )
                continue

            req_slice[mask] = delta_ids
            # ALSO mutate inputs_embeds at placeholder positions: vLLM computes
            # inputs_embeds from input_ids before forward, so they reflect the
            # placeholder vectors, not the delta tokens. Forward attends over
            # inputs_embeds; without this swap the model attends to "unknown
            # history" and ADE regresses ~3× (client-fuse vs server-fuse A/B:
            # 0.43m vs 1.20m on clip 030c760c).
            if inputs_embeds is not None:
                emb_slice = inputs_embeds[lo:hi]
                new_embeds = self.language_model.model.embed_tokens(delta_ids)
                emb_slice[mask] = new_embeds.to(emb_slice.dtype)
            logger.info("Alpamayo15: history fusion done (%d delta tokens)", n_placeholders)

    # ------------------------------------------------------------------ #
    # Forward override — AR decode + trigger detection + inline flow matching.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """Standard Qwen3-VL AR forward + inline flow matching when a request
        emits the trajectory-trigger token in its last decode step.

        sglang-style single-pass: after the VLM forward writes its KV cache,
        ``find_trigger_indices`` checks the current input_ids for the trigger
        token. When fired, ``_run_flow_matching_inline`` runs the flow-matching
        Euler loop entirely inside this same forward(), reading the VLM's
        paged KV via the forward-context's no_compile_layers registry and
        running FA3 over [cached_prefix; action_tokens].
        """
        # Per-request ``sampling_params.extra_args`` arrives via the engine's
        # generic ``sampling_extra_args`` forward kwarg (has_sampling_extra_args,
        # set from the deploy config's default extra_args) — no model-runner
        # hook. Pop it so the base VLM forward doesn't see the extra kwarg.
        sampling_extra_args = kwargs.pop("sampling_extra_args", None)
        self._extra_args_per_req = sampling_extra_args
        # Fuse ego-history into <|traj_history|> placeholders before the VLM.
        self._fuse_history_inplace(input_ids, inputs_embeds, sampling_extra_args)

        hidden_states = super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        # Default to an empty dict (not None): the worker's
        # extract_multimodal_outputs warns "not a dict" for non-Mapping values,
        # which would fire on every CoT decode step (no trigger yet). An empty
        # Mapping is silently skipped — only the trigger step carries actions.
        multimodal_outputs: dict = {}
        if input_ids is not None and input_ids.dim() == 1 and input_ids.numel() > 0:
            triggered = self.find_trigger_indices(
                input_ids,
                self.traj_future_start_token_id,
                self.traj_force_stop_token_id,
                has_history_traj=None,
            )
            self._pending_trigger_indices = triggered
            if triggered.numel() > 0:
                logger.info(
                    "Alpamayo15: trajectory trigger fired for %d request(s)",
                    int(triggered.numel()),
                )
                try:
                    actions = self._run_flow_matching_inline(triggered, positions)
                    # Surface sampled trajectory to the engine output processor
                    # so the client receives it under multimodal_output["actions"].
                    # Shape: (n_samples, n_waypoints=64, action_dim=2), fp32.
                    multimodal_outputs = {"actions": actions.detach().cpu()}
                except Exception as e:
                    logger.warning(
                        "Alpamayo15: flow matching inline call failed: %s: %s",
                        type(e).__name__,
                        e,
                    )
                    self._last_fm_error = (type(e).__name__, str(e))
        else:
            self._pending_trigger_indices = None
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    @torch.no_grad()
    def _read_prefix_kv(
        self, vlm_layer_name: str, block_table_row: torch.Tensor, prefix_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read K/V for one request's prefix from vLLM's paged cache.

        vLLM layout: ``cache shape = (2, num_blocks, block_size, num_kv_heads,
        head_dim)`` where index 0 is K, 1 is V. Gathers the request's blocks via
        the block_table row, flattens to a single sequence, slices to prefix_len.
        Returns ``(prefix_K, prefix_V)`` each shaped
        ``(prefix_len, num_kv_heads, head_dim)``.
        """
        from vllm.forward_context import get_forward_context

        target = get_forward_context().no_compile_layers[vlm_layer_name]
        cache = target.kv_cache
        if isinstance(cache, (list, tuple)):
            cache = cache[0]
        # cache: (2, num_blocks, block_size, num_kv_heads, head_dim)
        k_cache = cache[0]
        v_cache = cache[1]
        blocks_needed = (prefix_len + k_cache.shape[1] - 1) // k_cache.shape[1]
        bt = block_table_row[:blocks_needed].long()
        # gather blocks -> (blocks_needed, block_size, num_kv_heads, head_dim) -> flatten
        prefix_k = k_cache[bt].reshape(-1, k_cache.shape[-2], k_cache.shape[-1])[:prefix_len]
        prefix_v = v_cache[bt].reshape(-1, v_cache.shape[-2], v_cache.shape[-1])[:prefix_len]
        return prefix_k, prefix_v

    @torch.no_grad()
    def _expert_layer_manual_forward(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        action_positions: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One expert decoder layer with diffusion-side FlashAttention over
        [prefix_K; action_K].

        Bypasses ``layer.self_attn.attn`` (an EncoderOnlyAttention that can't
        read paged KV) and runs attention through
        ``self.expert_attns[layer_idx]`` — vllm-omni's existing
        ``DiffusionAttention`` wrapper that auto-selects FA3 / SDPA per
        platform and supports causal=False + GQA natively. All other
        weight-bearing ops (qkv_proj / qk_norm / rotary_emb / o_proj /
        layernorms / MLP) are stock Qwen3 modules.

        ``hidden`` shape: ``(b, n_diff, hidden_size)``.
        ``prefix_k/v`` shape: ``(prefix_len, num_kv_heads, head_dim)`` (one req).
        Returns ``(new_hidden, new_residual)`` post-layer.
        """
        layer = self.expert.layers[layer_idx]
        attn = layer.self_attn
        b, n, _ = hidden.shape

        # input layernorm + residual handling (mirrors Qwen3DecoderLayer.forward)
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)

        # qkv_proj + qk-norm + rotary (mirrors Qwen3Attention.forward up to attn call)
        flat = hidden.view(b * n, -1)
        qkv, _ = attn.qkv_proj(flat)
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
        head_dim = attn.head_dim
        q = attn.q_norm(q.view(b * n, -1, head_dim)).view(b * n, -1)
        k = attn.k_norm(k.view(b * n, -1, head_dim)).view(b * n, -1)
        # rotary_emb expects mRoPE positions as (3, seq) for Qwen3-VL.
        q, k = attn.rotary_emb(action_positions, q, k)

        # Reshape to (b, seqlen, heads, dim) for DiffusionAttention.
        q = q.view(b, n, attn.num_heads, head_dim)
        k_act = k.view(b, n, attn.num_kv_heads, head_dim)
        v_act = v.view(b, n, attn.num_kv_heads, head_dim)
        # Broadcast (prefix_len, kv_h, d) -> (b, prefix_len, kv_h, d) and concat.
        pk = prefix_k.unsqueeze(0).expand(b, -1, -1, -1).to(q.dtype)
        pv = prefix_v.unsqueeze(0).expand(b, -1, -1, -1).to(q.dtype)
        k_full = torch.cat([pk, k_act], dim=1)  # (b, prefix+n, kv_h, d)
        v_full = torch.cat([pv, v_act], dim=1)

        # DiffusionAttention dispatches to the configured backend (FA3 on H200,
        # SDPA fallback elsewhere); causal=False fixed at __init__ time.
        attn_out = self.expert_attns[layer_idx](q, k_full, v_full)  # (b, n, h, d)
        attn_out = attn_out.reshape(b * n, attn.num_heads * head_dim)
        out_proj, _ = attn.o_proj(attn_out)
        hidden = out_proj.view(b, n, -1)

        # post_attention_layernorm + MLP + residual (Qwen3 ordering)
        hidden, residual = layer.post_attention_layernorm(hidden, residual)
        hidden = layer.mlp(hidden)
        return hidden, residual

    @torch.no_grad()
    def _run_flow_matching_inline(
        self,
        triggered: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Manual-SDPA flow matching that bypasses vLLM's Attention layers.

        Why manual: vLLM 0.21 has no kernel that does sglang's
        ``extend_attention_fwd`` semantics (Q from action embeds, K/V =
        cat[cached_prefix; current_action]). ``kv_sharing_target_layer_name``
        substitutes the cache rather than augmenting it. Without the concat,
        the action tokens never see each other and the trajectory diverges
        (early waypoints correct, FDE ~26m vs standalone ~1.5m).

        This implementation per-layer reads the VLM's cached prefix K/V via
        ``forward_context.no_compile_layers[vlm_layer_name].kv_cache``, computes
        the expert's q/k/v from action embeds, concatenates, and runs
        ``torch.nn.functional.scaled_dot_product_attention`` bidirectionally.
        Slower than a fused kernel but numerically correct.
        """
        from vllm.forward_context import get_forward_context

        device = positions.device
        n_diff = self.n_waypoints
        # bstar = real flow-matching batch dim = (number of triggered requests
        # in this forward) × (per-request trajectory samples).
        #
        # The per-request sample count is read from
        # ``sampling_params.extra_args["n_samples"]`` (default 1). All
        # triggered requests in this batch are assumed to agree on n_samples
        # — for minADE@N the typical caller is a single request asking for
        # N trajectories; cross-request batching with mixed n_samples isn't
        # supported yet.
        n_trigs = int(triggered.numel())
        n_samples = 1
        if self._extra_args_per_req:
            first_idx = int(triggered[0].item())
            ea = self._extra_args_per_req[first_idx] if first_idx < len(self._extra_args_per_req) else None
            if isinstance(ea, dict):
                ns = ea.get("n_samples")
                if isinstance(ns, str):
                    # vllm_xargs flat-primitive constraint: accept stringified int.
                    try:
                        ns = int(ns)
                    except Exception:
                        ns = None
                if isinstance(ns, int) and ns > 0:
                    n_samples = ns
        bstar = n_trigs * n_samples

        # mRoPE positions for action tokens: per-triggered positions tiled
        # across n_samples — layout per-req then per-sample: [req0_s0, req0_s1, ..., req0_sN, req1_s0, ...].
        cur_mrope = positions if positions.dim() == 2 else positions.view(1, -1).expand(3, -1)
        per_req_pos = self.action_token_positions(cur_mrope, triggered, n_diff)  # (3, n_trigs * n_diff)
        if n_samples > 1:
            per_req_pos = per_req_pos.view(3, n_trigs, n_diff)
            per_req_pos = per_req_pos.unsqueeze(2).expand(3, n_trigs, n_samples, n_diff)
            action_pos = per_req_pos.reshape(3, n_trigs * n_samples * n_diff)
        else:
            action_pos = per_req_pos

        # Look up per-request block_table + prefix length for cache reads.
        ctx = get_forward_context()
        # attn_metadata is per-layer dict; any layer's metadata has the same
        # block_table and seq_lens. Pick the first VLM layer's metadata.
        n_layers = len(self.expert.layers)
        vlm_layer_names = self._vlm_layer_names
        any_meta = ctx.attn_metadata[vlm_layer_names[0]]
        # block_table: (batch, max_blocks_per_req); seq_lens: (batch,).
        # Per-backend field name varies (FlashAttention uses .block_table;
        # CommonAttentionMetadata uses .block_table_tensor).
        block_table_full = getattr(any_meta, "block_table", None)
        if block_table_full is None:
            block_table_full = any_meta.block_table_tensor
        seq_lens = any_meta.seq_lens
        # Take the first triggered request's prefix as representative. This is
        # correct when all triggered sequences share the same prompt prefix
        # (the ``sampling_params.n=N`` case — N siblings of one request) since
        # vLLM's prefix caching makes their block_tables point at the same
        # underlying KV blocks. For BATCHING ACROSS DIFFERENT prompts with
        # potentially different prefix lengths, this would need varlen
        # attention with per-row cu_seqlens; not yet supported (would also
        # need per-request output routing in OmniOutput.multimodal_outputs).
        req_idx = int(triggered[0].item())
        block_table_row = block_table_full[req_idx]
        # prefix_len = current seq length (already includes the trigger token,
        # which super().forward just wrote to the cache)
        prefix_len = int(seq_lens[req_idx].item())

        # Pre-read all layers' prefix K/V once (constant across Euler steps).
        prefix_kvs = []
        for li in range(n_layers):
            pk, pv = self._read_prefix_kv(vlm_layer_names[li], block_table_row, prefix_len)
            prefix_kvs.append((pk, pv))

        # Initial noise + Euler loop. Seed-pin for reproducibility / parity.
        expert_dtype = next(self.expert.parameters()).dtype
        seed = int(os.environ.get("ALPAMAYO_FM_SEED", "0"))
        g = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(bstar, n_diff, self.action_dim, device=device, dtype=torch.float32, generator=g)
        time_steps = torch.linspace(0.0, 1.0, self.num_inference_steps + 1, device=device)
        for step_i in range(self.num_inference_steps):
            dt = time_steps[step_i + 1] - time_steps[step_i]
            t = time_steps[step_i].view(1, 1, 1).expand(bstar, 1, 1)
            hidden = self.action_in_proj(x.to(expert_dtype), t.to(expert_dtype))
            if hidden.dim() == 2:
                hidden = hidden.view(bstar, n_diff, -1)
            residual = None
            for li in range(n_layers):
                pk, pv = prefix_kvs[li]
                hidden, residual = self._expert_layer_manual_forward(
                    li,
                    hidden,
                    residual,
                    action_pos,
                    pk,
                    pv,
                )
            # final norm + action_out_proj
            hidden, _ = self.expert.norm(hidden, residual)
            pred = self.action_out_proj(hidden.to(expert_dtype)).to(torch.float32)
            pred = pred.view(bstar, n_diff, self.action_dim)
            x = x + dt * pred

        self._last_sampled_actions = x
        logger.info(
            "Alpamayo15: flow matching completed — sampled actions shape %s",
            tuple(x.shape),
        )
        return x

    def get_mrope_input_positions(self, input_tokens, *, mm_features=None, **kwargs):
        """vllm-omni 0.20 passes extra kwargs (hf_config, image_grid_thw, ...)
        that vLLM 0.21's Qwen3VL doesn't accept; swallow them so the omni model
        runner's calling convention works."""
        return super().get_mrope_input_positions(input_tokens, mm_features=mm_features)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        logits = self._apply_traj_logit_mask(logits, self.traj_mask_start, self.traj_mask_end)
        # sglang-equivalent force-stop: after the trigger fired in forward(),
        # the next decode token must be <|im_end|> so AR halts immediately
        # (otherwise the model keeps generating and re-fires the trigger,
        # producing duplicate FM runs / corrupting multimodal_output).
        # Mirrors sglang alpamayo_r1.py:1083-1086.
        triggered = self._pending_trigger_indices
        if (
            triggered is not None
            and triggered.numel() > 0
            and logits is not None
            and triggered.max().item() < logits.shape[0]
        ):
            logits[triggered, :] = float("-inf")
            logits[triggered, self.traj_force_stop_token_id] = 0.0
        return logits

    def _load_expert_weights(self, expert_weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Manually load Qwen3Model expert weights with q/k/v -> qkv_proj and
        gate/up -> gate_up_proj stacking. Mirrors sglang's _load_expert_weights."""
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.expert.named_parameters())
        loaded: set[str] = set()
        for name, tensor in expert_weights:
            # The expert eats pre-projected embeds; embed_tokens is dropped.
            if name.startswith("embed_tokens.") or "rotary_emb.inv_freq" in name:
                continue
            matched = False
            for tgt, src, shard_id in stacked:
                if src in name:
                    mapped = name.replace(src, tgt)
                    if mapped in params_dict:
                        p = params_dict[mapped]
                        loader = getattr(p, "weight_loader", default_weight_loader)
                        loader(p, tensor, shard_id)
                        loaded.add(mapped)
                        matched = True
                    break
            if matched:
                continue
            if name in params_dict:
                p = params_dict[name]
                loader = getattr(p, "weight_loader", default_weight_loader)
                loader(p, tensor)
                loaded.add(name)
        return loaded

    @staticmethod
    def _bucket_checkpoint(
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Partition the Alpamayo checkpoint into per-submodule buckets.

        Returns a dict with keys ``vlm`` / ``expert`` / ``action_in_proj`` /
        ``action_out_proj`` / ``action_space`` (each maps to a list of
        ``(stripped_name, tensor)``). Names without a known prefix fall through
        to ``vlm`` for back-compat.
        """
        buckets: dict[str, list[tuple[str, torch.Tensor]]] = {
            "vlm": [],
            "expert": [],
            "action_in_proj": [],
            "action_out_proj": [],
            "action_space": [],
        }
        for name, tensor in weights:
            for prefix in ("vlm.", "expert.", "action_in_proj.", "action_out_proj.", "action_space."):
                if name.startswith(prefix):
                    buckets[prefix[:-1]].append((name[len(prefix) :], tensor))
                    break
            else:
                buckets["vlm"].append((name, tensor))
        return buckets

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the full Alpamayo checkpoint into vlm + expert + action heads.

        VLM goes through ``Qwen3VLForConditionalGeneration.load_weights`` (which
        uses ``AutoWeightsLoader`` + ``hf_to_vllm_mapper``). Expert goes through
        ``AutoWeightsLoader`` on ``self.expert`` (handles q/k/v -> qkv_proj and
        gate/up -> gate_up_proj stacking automatically). Action heads load via
        plain ``load_state_dict``. The action space carries no checkpoint tensors
        (its statistics come from the config); ``self.action_space`` is kept in
        fp32 for the precision-sensitive least-squares solves.
        """
        buckets = self._bucket_checkpoint(weights)

        # 1) VLM
        loaded_vlm = super().load_weights(buckets["vlm"])

        # 2) Expert (Qwen3Model with vLLM-stacked qkv/gate_up). AutoWeightsLoader
        # alone doesn't know to stack q/k/v -> qkv_proj here (packed_modules_mapping
        # lives on Qwen3ForCausalLM, not Qwen3Model), so do it manually — same
        # pattern as sglang's _load_expert_weights.
        loaded_expert = self._load_expert_weights(buckets["expert"])

        # 3) Action heads
        ain_sd = dict(buckets["action_in_proj"])
        # FourierEncoderV2.freqs is non-persistent for 1.5 (recomputed in __init__);
        # tolerate its absence.
        res_in = self.action_in_proj.load_state_dict(ain_sd, strict=False)
        missing_in = [m for m in res_in.missing_keys if "freqs" not in m]
        if missing_in or res_in.unexpected_keys:
            raise RuntimeError(
                f"action_in_proj load mismatch: missing={missing_in[:5]}, unexpected={res_in.unexpected_keys[:5]}"
            )
        self.action_out_proj.load_state_dict(dict(buckets["action_out_proj"]), strict=True)

        # 4) Action space: kept in fp32; its tensors are config-derived buffers
        # (accel_mean/std, curvature_mean/std). The released checkpoint usually
        # does not ship them as tensors; if it does, load them.
        self.action_space.float()
        if buckets["action_space"]:
            self.action_space.load_state_dict(dict(buckets["action_space"]), strict=False)

        # Match the standalone HF path: cast action projections to the expert's
        # dtype (bf16) AFTER loading. The expert runs in bf16; the action heads
        # were ckpt-loaded in their native dtype (mixed). Without this we hit
        # "mat1 and mat2 must have the same dtype" inside the Euler loop.
        if len(loaded_expert) > 0:
            try:
                expert_dtype = next(self.expert.parameters()).dtype
                self.action_in_proj = self.action_in_proj.to(dtype=expert_dtype)
                self.action_out_proj = self.action_out_proj.to(dtype=expert_dtype)
            except StopIteration:
                pass

        # The returned set must match the model's nn.Module parameter names
        # exactly (vLLM cross-checks against named_parameters). super() already
        # returned correctly-mapped names ("language_model.model.layers.*",
        # "visual.*", etc.); the expert / action submodules live at "expert.*",
        # "action_in_proj.*", "action_out_proj.*" on this nn.Module.
        loaded: set[str] = set()
        loaded.update(loaded_vlm)
        loaded.update(f"expert.{n}" for n in loaded_expert)
        loaded.update(f"action_in_proj.{k}" for k in ain_sd)
        loaded.update(f"action_out_proj.{name}" for name, _ in buckets["action_out_proj"])
        logger.info(
            "Alpamayo15 load: vlm=%d expert=%d action_in=%d action_out=%d",
            len(loaded_vlm),
            len(loaded_expert),
            len(buckets["action_in_proj"]),
            len(buckets["action_out_proj"]),
        )
        return loaded


__all__ = ["Alpamayo15ForConditionalGeneration"]

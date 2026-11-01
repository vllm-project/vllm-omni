# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speech-vocabulary DSpark checkpoints and exact RAS target probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SPEECH_VOCAB_SIZE = 6561
STAGE_VOCAB_SIZE = SPEECH_VOCAB_SIZE + 200
UNIFIED_SPEECH_OFFSET = 151924
MASK_TOKEN_ID = STAGE_VOCAB_SIZE - 1
UNIFIED_MASK_TOKEN_ID = UNIFIED_SPEECH_OFFSET + STAGE_VOCAB_SIZE


def verification_history(
    all_token_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    request_indices: torch.Tensor,
    positions: torch.Tensor,
    input_ids: torch.Tensor,
    local_positions: torch.Tensor,
    win_size: int,
) -> torch.Tensor:
    """Recent generated history including each verification row's draft prefix."""
    rows = request_indices.long()
    delta = torch.arange(win_size, device=positions.device)
    offsets = positions.long().unsqueeze(1) - delta
    history = all_token_ids[rows.unsqueeze(1), offsets.clamp(0, all_token_ids.shape[1] - 1)]
    # The persistent token table contains committed output. Drafts live in the
    # current packed input instead, including the anchor at local position 0.
    prefix_rows = torch.arange(positions.numel(), device=positions.device).unsqueeze(1) - delta
    in_prefix = delta.unsqueeze(0) <= local_positions.unsqueeze(1)
    prefix = input_ids[prefix_rows.clamp_min(0)]
    history = torch.where(in_prefix, prefix, history)
    valid = (offsets >= prompt_lengths[rows].unsqueeze(1)) & (offsets >= 0)
    return history.masked_fill(~valid, -1)


def ras_probabilities(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    recent_tokens: torch.Tensor,
    *,
    win_size: int = 10,
    tau_r: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Marginalize the ordinary sampler's initial draw and one RAS redraw.

    ``recent_tokens`` contains stage-vocabulary generated tokens only, with -1
    for absent history, validated by ``ras_verification_logits`` before compile.
    The nucleus uses full-distribution top-p before the top-k cap. A repeated
    first draw is replaced from the full tempered distribution excluding that
    token, rather than from the nucleus. Temperature-zero rows remain greedy.
    """
    scores = logits.float() / temperatures.clamp_min(eps).unsqueeze(1)
    full = scores.softmax(dim=-1)
    sorted_probs, sorted_ids = full.sort(dim=-1, descending=True, stable=True)
    ranks = torch.arange(logits.shape[-1], device=logits.device)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_ps.unsqueeze(1)
    keep |= ranks.unsqueeze(0) == 0
    keep &= (top_ks.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_ks.unsqueeze(1))
    nucleus = sorted_probs.masked_fill(~keep, 0)
    nucleus = nucleus / nucleus.sum(dim=-1, keepdim=True)
    nucleus = torch.zeros_like(nucleus).scatter(1, sorted_ids, nucleus)

    if win_size > 0 and recent_tokens.shape[1] > 0:
        history = recent_tokens[:, -win_size:]
        valid = (history >= 0) & (history < logits.shape[-1])
        counts = torch.zeros_like(full, dtype=torch.int32).scatter_add(
            1, history.clamp(0, logits.shape[-1] - 1).long(), valid.to(torch.int32)
        )
        rejected = (counts >= win_size * tau_r) & valid.any(dim=-1, keepdim=True)
        # Sum in fp64: 1 - p(j) loses the remaining mass when a logit dominates.
        probability = full.double()
        initial = nucleus.double()
        remaining = probability.sum(dim=-1, keepdim=True) - probability
        dominant = probability.argmax(dim=-1, keepdim=True)
        other_mass = probability.scatter(1, dominant, 0).sum(dim=-1, keepdim=True)
        remaining = remaining.scatter(1, dominant, other_mass)
        redraw_weight = torch.where(rejected & (remaining > 0), initial / remaining.clamp_min(1e-300), 0.0)
        unchanged = initial.masked_fill(rejected & (remaining > 0), 0)
        result = unchanged + probability * (redraw_weight.sum(dim=-1, keepdim=True) - redraw_weight)
        result = result.clamp_min(0)
        result = (result / result.sum(dim=-1, keepdim=True)).to(full.dtype)
    else:
        result = nucleus

    greedy = torch.zeros_like(full).scatter(1, logits.argmax(dim=-1, keepdim=True), 1)
    return torch.where((temperatures < eps).unsqueeze(1), greedy, result)


def ras_verification_logits(
    sampler,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    seq_lens_upper_bound_np: np.ndarray,
    skip_top_k_top_p: bool = False,
    *,
    default_top_p: float,
    default_top_k: int,
    win_size: int,
    tau_r: float,
) -> torch.Tensor:
    """Prepare the actual RAS distribution for vLLM's rejection sampler."""
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState

    if skip_top_k_top_p:
        raise ValueError("CosyVoice3 RAS verification requires the complete sampling policy")
    ctx = LogitsContext(
        expanded_idx_mapping=expanded_idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        expanded_local_pos=expanded_local_pos,
        input_ids=input_ids,
        pos=pos,
        seq_lens_upper_bound_np=seq_lens_upper_bound_np,
    )
    logits = logits.float().clone()
    for processor in sampler.logits_processors:
        if not isinstance(processor, PenaltiesState):
            logits = processor.apply(logits, ctx)

    states = sampler.sampling_states
    rows = expanded_idx_mapping.long()
    top_k, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    if top_k is None:
        top_k = torch.full_like(rows, default_top_k, dtype=torch.int32)
    if top_p is None:
        top_p = logits.new_full((logits.shape[0],), default_top_p)
    history = verification_history(
        sampler.req_states.all_token_ids.gpu,
        sampler.req_states.prompt_len.gpu,
        expanded_idx_mapping,
        pos,
        input_ids,
        expanded_local_pos,
        win_size,
    )
    # Keep the check outside compiled math: Inductor can elide _assert_async.
    torch._assert_async(
        ((history == -1) | ((history >= 0) & (history < logits.shape[-1]))).all(),
        "CosyVoice3 RAS history must contain stage speech-token IDs or -1, not unified-vocabulary IDs",
    )
    probabilities = sampler._cosyvoice3_ras_probabilities(
        logits, states.temperature.gpu[rows], top_k, top_p, history, win_size=win_size, tau_r=tau_r
    )
    # RejectionSampler expects logits after temperature and support shaping.
    return probabilities.log()


def remap_vocabulary_weights(weights: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Keep reachable draft columns and map unified input ids to speech ids."""
    offsets = weights["d2t"].long()
    stage_ids = offsets + torch.arange(offsets.numel(), device=offsets.device) - UNIFIED_SPEECH_OFFSET
    reachable = (stage_ids >= 0) & (stage_ids < MASK_TOKEN_ID)
    selected = stage_ids[reachable]
    if selected.numel() == 0:
        raise ValueError("DSpark checkpoint has no reachable CosyVoice3 speech ids")
    if selected.unique().numel() != selected.numel():
        raise ValueError("DSpark d2t contains duplicate speech ids")

    rows = slice(UNIFIED_SPEECH_OFFSET, UNIFIED_SPEECH_OFFSET + STAGE_VOCAB_SIZE)
    embedding = weights["embed_tokens.weight"][rows].clone()
    if embedding.shape[0] != STAGE_VOCAB_SIZE:
        raise ValueError("DSpark input embedding does not cover the speech vocabulary")
    embedding[MASK_TOKEN_ID].copy_(weights["embed_tokens.weight"][UNIFIED_MASK_TOKEN_ID])
    result = {
        "embed_tokens.weight": embedding,
        "lm_head.weight": weights["lm_head.weight"][reachable].contiguous(),
        "markov_head.markov_w1.weight": weights["markov_head.markov_w1.weight"][rows].clone(),
        "markov_head.markov_w2.weight": weights["markov_head.markov_w2.weight"][reachable].contiguous(),
        "d2t": (selected - torch.arange(selected.numel(), device=selected.device)).contiguous(),
    }
    return result, selected


def converted_config(config: dict, draft_vocab_size: int) -> dict:
    if config.get("speculators_model_type") != "dspark" or config.get("markov_head_type") != "vanilla":
        raise ValueError("Expected a vanilla-Markov speculators DSpark checkpoint")
    if config.get("mask_token_id") != UNIFIED_MASK_TOKEN_ID:
        raise ValueError("Checkpoint mask id does not match the CosyVoice3 unified vocabulary")
    aux_layers = config["aux_hidden_state_layer_ids"]
    if aux_layers != [2, 12, 21]:
        raise ValueError("Expected CosyVoice3 DSpark auxiliary layers [2, 12, 21]")
    result = dict(config["transformer_layer_config"])
    result.update(
        architectures=["Qwen3DSparkModel"],
        vocab_size=STAGE_VOCAB_SIZE,
        draft_vocab_size=draft_vocab_size,
        mask_token_id=MASK_TOKEN_ID,
        markov_rank=config["markov_rank"],
        markov_head_type="vanilla",
        target_hidden_size=result["hidden_size"],
        use_aux_hidden_state=True,
        sample_from_anchor=False,
        dspark_bonus_anchor=True,
        block_size=config["block_size"] - 1,
        enable_confidence_head=False,
        tie_word_embeddings=False,
        eagle_aux_hidden_state_layer_ids=aux_layers,
        target_layer_ids=[layer - 1 for layer in aux_layers],
        dflash_config={"target_layer_ids": [layer - 1 for layer in aux_layers]},
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint(source: Path, destination: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint directory: {destination}")
    source_config = source / "config.json"
    config = json.loads(source_config.read_text())
    # This published checkpoint is a single shard. Reject unknown layouts.
    checkpoint = source / "model.safetensors"
    with safe_open(checkpoint, framework="pt", device="cpu") as stream:
        weights = {name.removeprefix("model."): stream.get_tensor(name) for name in stream.keys()}
    vocabulary, selected = remap_vocabulary_weights(weights)
    ignored = {*vocabulary, "t2d"}
    converted = {
        name: value.contiguous()
        for name, value in weights.items()
        if name not in ignored and not name.startswith("confidence_head.")
    }
    converted.update(vocabulary)
    destination.mkdir(parents=True)
    save_file(converted, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(converted_config(config, selected.numel()), indent=2) + "\n")
    receipt = {
        "source_config_sha256": _sha256(source_config),
        "source_weights_sha256": _sha256(checkpoint),
        "weights_sha256": _sha256(destination / "model.safetensors"),
        "config_sha256": _sha256(destination / "config.json"),
        "reachable_draft_columns": selected.numel(),
        "input_vocabulary": STAGE_VOCAB_SIZE,
        "mask_token_id": MASK_TOKEN_ID,
        "unified_speech_offset": UNIFIED_SPEECH_OFFSET,
    }
    (destination / "conversion-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)

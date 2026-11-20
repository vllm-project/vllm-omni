# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small fixtures for the real Qwen3 thinker embedding method and its oracle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

SCENARIOS = ("text", "empty", "audio", "image", "video", "mixed", "interleaved", "vision_no_deepstack")


@dataclass(frozen=True)
class ModelShape:
    vocab_size: int = 152064
    hidden_size: int = 2048
    deepstack_levels: int = 3
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151675

    def __post_init__(self):
        if min(self.vocab_size, self.hidden_size, self.deepstack_levels) <= 0:
            raise ValueError("vocab_size, hidden_size and deepstack_levels must be positive")
        ids = (self.image_token_id, self.video_token_id, self.audio_token_id)
        if len(set(ids)) != 3 or not all(0 < token < self.vocab_size for token in ids):
            raise ValueError("Provide three distinct, positive, in-vocabulary modality IDs; use oov=True to test OOV")

    def token_ids(self, oov=False):
        if oov:
            return dict(zip(("image", "video", "audio"), range(self.vocab_size, self.vocab_size + 3)))
        return {name: getattr(self, f"{name}_token_id") for name in ("image", "video", "audio")}


@dataclass
class Feature:
    modality: str
    positions: torch.Tensor
    main: torch.Tensor
    deepstack: torch.Tensor | None
    combined: torch.Tensor


@dataclass
class EmbeddingCase:
    name: str
    input_ids: torch.Tensor
    is_multimodal: torch.Tensor | None
    features: tuple[Feature, ...]
    fingerprint: str

    def fresh_embeddings(self):
        # The thinker replaces list entries with main-scale views.
        return None if self.name == "text" else [feature.combined for feature in self.features]


def _tensor_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def build_case(
    name,
    num_tokens,
    num_mm_tokens,
    shape,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    mask_device="cpu",
    seed=42,
    oov=False,
):
    """Build known feature placements; no model merge is used to define them."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    if num_tokens <= 0 or not 0 <= num_mm_tokens < num_tokens:
        raise ValueError("Require N > 0 and 0 <= M < N")
    if (name in {"text", "empty"}) != (num_mm_tokens == 0):
        raise ValueError("Only text/empty scenarios accept M=0")
    if name in {"mixed", "interleaved"} and (num_mm_tokens < 4 or num_mm_tokens % 2):
        raise ValueError("Mixed/interleaved scenarios require an even M >= 4")
    token_ids = shape.token_ids(oov)
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, min(shape.vocab_size, *token_ids.values()), (num_tokens,), generator=generator)
    mask = torch.zeros(num_tokens, dtype=torch.bool)
    positions = torch.arange((num_tokens - num_mm_tokens) // 2, (num_tokens - num_mm_tokens) // 2 + num_mm_tokens)
    groups = []
    if name == "mixed":
        groups = [("audio", positions[: num_mm_tokens // 2]), ("image", positions[num_mm_tokens // 2 :])]
    elif name == "interleaved":
        groups = [("video", positions[::2]), ("audio", positions[1::2])]
    elif num_mm_tokens:
        groups = [("image" if name == "vision_no_deepstack" else name, positions)]
    features = []
    for modality, indexes in groups:
        ids[indexes] = token_ids[modality]
        mask[indexes] = True
        rows = torch.arange(len(indexes), dtype=torch.float32)[:, None].remainder(31) / 32
        cols = torch.arange(shape.hidden_size, dtype=torch.float32)[None, :].remainder(7) / 16
        main_cpu = rows + cols + {"audio": 2, "image": 4, "video": 6}[modality]
        main = main_cpu.to(device=device, dtype=dtype)
        deepstack = None
        parts = [main]
        if modality != "audio" and name != "vision_no_deepstack":
            deepstack = torch.stack(
                [(main_cpu + 16 + 4 * level).to(device=device, dtype=dtype) for level in range(shape.deepstack_levels)]
            )
            parts.extend(deepstack.unbind(0))
        combined = torch.cat(parts, dim=-1)
        # set_mm_embedding_modality simply attaches this attribute. Tag only
        # after cat/to: views and copies do not preserve arbitrary attributes.
        combined.modality = modality
        features.append(Feature(modality, indexes, main, deepstack, combined))
    digest = hashlib.sha256(json.dumps({"name": name, "shape": asdict(shape), "seed": seed, "oov": oov}).encode())
    digest.update(_tensor_bytes(ids))
    digest.update(_tensor_bytes(mask))
    for feature in features:
        digest.update(feature.modality.encode())
        digest.update(_tensor_bytes(feature.combined))
    return EmbeddingCase(
        name,
        ids.to(device=device),
        None if name == "text" else mask.to(device=mask_device),
        tuple(features),
        digest.hexdigest(),
    )


class LanguageModelAdapter(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding

    def embed_input_ids(self, input_ids):
        return self.embedding(input_ids)

    def forward(self, input_ids):
        return self.embed_input_ids(input_ids)


def make_thinker(
    shape,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    seed=42,
    backend="torch",
    deepstack=True,
    buffer_capacity=32,
    oov=False,
):
    """Skip checkpoint construction, preserving the real class and its MRO."""
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    if buffer_capacity <= 0:
        raise ValueError("buffer_capacity must be positive")
    device = torch.device(device)
    if backend == "torch":
        embedding = nn.Embedding(shape.vocab_size, shape.hidden_size, device=device, dtype=dtype)
    elif backend == "vllm":
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

        with torch.device(device):
            embedding = VocabParallelEmbedding(shape.vocab_size, shape.hidden_size, params_dtype=dtype, disable_tp=True)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")
    with torch.no_grad():
        embedding.weight.uniform_(-0.5, 0.5, generator=torch.Generator(device=device).manual_seed(seed))
    thinker = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=shape.hidden_size))
    for modality, token_id in shape.token_ids(oov).items():
        setattr(thinker.config, f"{modality}_token_id", token_id)
    thinker.visual = SimpleNamespace(
        deepstack_visual_indexes=list(range(shape.deepstack_levels)) if deepstack else None
    )
    thinker.language_model = LanguageModelAdapter(embedding)
    thinker._has_oov_mm_tokens = oov
    thinker.use_deepstack = deepstack
    thinker.deepstack_num_level = shape.deepstack_levels if deepstack else 0
    thinker.deepstack_input_embeds = [
        torch.zeros(buffer_capacity, shape.hidden_size, device=device, dtype=dtype)
        for _ in range(thinker.deepstack_num_level)
    ]
    thinker.deepstack_input_embeds_num_tokens = 0
    return thinker.eval()


@torch.inference_mode()
def expected_outputs(model, case):
    """Independent oracle: direct lookup and explicit positions, not merge helpers."""
    ids = case.input_ids
    if model._has_oov_mm_tokens and case.is_multimodal is not None:
        ids = ids.masked_fill(case.is_multimodal.to(ids.device), 0)
    main = F.embedding(ids, model.language_model.embedding.weight)
    has_deepstack = any(feature.deepstack is not None for feature in case.features)
    deepstack = main.new_zeros((model.deepstack_num_level, *main.shape)) if has_deepstack else None
    for feature in case.features:
        indexes = feature.positions.to(main.device)
        main[indexes] = feature.main
        if feature.deepstack is not None:
            deepstack[:, indexes, :] = feature.deepstack
    return main, deepstack


@torch.inference_mode()
def assert_case_output(model, case, result):
    expected, deepstack = expected_outputs(model, case)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if deepstack is None:
        assert model.deepstack_input_embeds_num_tokens == 0, "Unexpected stale DeepStack buffer"
    else:
        assert model.deepstack_input_embeds_num_tokens == case.input_ids.numel()
        actual = torch.stack([buffer[: case.input_ids.numel()] for buffer in model.deepstack_input_embeds])
        torch.testing.assert_close(actual, deepstack, rtol=0, atol=0)

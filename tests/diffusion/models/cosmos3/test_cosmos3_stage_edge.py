# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end walk of the Cosmos3 reasoner -> generator stage edge, on CPU.

Wires the real chain -- reasoner forward, post-process, serde, the
``reasoner2generator`` bridge, generator forward -- with only the transformer
stubbed. The tokenizer stub derives ids from every resolved conditioning value,
so fingerprint agreement between the stages is under test, not assumed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_disagg import (
    Cosmos3GeneratorPipeline,
    Cosmos3ReasonerPipeline,
    get_cosmos3_reasoner_post_process_func,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_META_KEY as META_KEY,
)
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerde
from vllm_omni.model_executor.stage_input_processors.cosmos3 import reasoner2generator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 4
KV_HEADS = 2
HEAD_DIM = 8
PAD_TO = 6


class _StubUndTower(nn.Module):
    """Per-layer K/V whose values encode the token ids they came from.

    Marking the tensors is what lets the generator side assert it replayed *this*
    branch's K/V rather than merely something of the right shape.

    Each head is offset by its index so a rank that replays the wrong KV-head range
    is distinguishable too. ``rope_only`` is the shape the real tower takes on a
    stage that does not own it: the mRoPE embedding stays, the blocks are absent.
    """

    def __init__(self, num_layers: int, *, rope_only: bool = False) -> None:
        super().__init__()
        self.rotary_emb = nn.Identity()
        self.num_layers = num_layers
        self.layers = nn.ModuleList() if rope_only else nn.ModuleList(nn.Linear(1, 1) for _ in range(num_layers))

    def forward(self, text_ids: torch.Tensor, freqs: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del freqs
        batch, seq = text_ids.shape
        marker = float(text_ids.sum().item())
        heads = torch.arange(KV_HEADS, dtype=torch.float32).reshape(1, 1, KV_HEADS, 1)
        return [
            (
                (heads + (marker + layer)).expand(batch, seq, KV_HEADS, HEAD_DIM).contiguous(),
                (heads - (marker + layer)).expand(batch, seq, KV_HEADS, HEAD_DIM).contiguous(),
            )
            for layer in range(self.num_layers)
        ]


class _StubCrossAttention(nn.Module):
    def __init__(self, num_kv_heads: int, num_kv_heads_local: int, head_dim: int) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim


class _StubGenBlock(nn.Module):
    def __init__(self, num_kv_heads: int, num_kv_heads_local: int, head_dim: int) -> None:
        super().__init__()
        self.cross_attention = _StubCrossAttention(num_kv_heads, num_kv_heads_local, head_dim)


class _StubTransformer(nn.Module):
    """``owned_towers`` reproduces what the real transformer does with the
    pipeline's ``cosmos3_owned_towers``: the unowned tower is never constructed."""

    def __init__(
        self,
        *,
        owned_towers: tuple[str, ...],
        num_kv_heads: int = KV_HEADS,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.owned_towers = tuple(owned_towers)
        self.language_model = _StubUndTower(NUM_LAYERS, rope_only="reasoner" not in self.owned_towers)
        self.gen_layers = nn.ModuleList(
            _StubGenBlock(num_kv_heads, num_kv_heads_local, head_dim)
            for _ in range(NUM_LAYERS if "generator" in self.owned_towers else 0)
        )
        self.proj_in = nn.Linear(HEAD_DIM, HEAD_DIM)
        self.num_hidden_layers = NUM_LAYERS

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        return h, w, 0, 0

    def _compute_rope_freqs(self, *args: Any, **kwargs: Any) -> tuple[str, str]:
        return "freqs_und", "freqs_gen"

    @contextlib.contextmanager
    def _offload_context(self, name: str):
        del name
        yield


def _tokenize(prompt: str, negative_prompt: str, *, height, width, max_sequence_length, use_system_prompt, frame_rate):
    """Ids that depend on every value the real formatter/tokenizer depends on.

    Padded past the real length so the reasoner's trim to the true token count is
    exercised too, and the mask is what tells it where to cut.
    """

    def _encode(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        seed = (
            sum(ord(c) for c in text),
            int(height),
            int(width),
            int(max_sequence_length),
            int(use_system_prompt),
            int(frame_rate),
        )
        real = [1 + (sum(seed) + i) % 97 for i in range(3)]
        ids = torch.tensor([real + [0] * (PAD_TO - len(real))], dtype=torch.long)
        mask = torch.zeros(1, PAD_TO, dtype=torch.long)
        mask[:, : len(real)] = 1
        return ids, mask

    cond_ids, cond_mask = _encode(prompt)
    uncond_ids, uncond_mask = _encode(f"negative::{negative_prompt}")
    return cond_ids, cond_mask, uncond_ids, uncond_mask


def _install_tokenizer(pipeline) -> None:
    def _format_and_tokenize_prompts(
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        frame_rate: float,
        height: int,
        width: int,
        max_sequence_length: int,
        sp: Any,
        use_system_prompt: bool = False,
        is_t2i: bool = False,
    ):
        del num_frames, sp, is_t2i
        return _tokenize(
            prompt,
            negative_prompt,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            use_system_prompt=use_system_prompt,
            frame_rate=frame_rate,
        )

    pipeline._format_and_tokenize_prompts = _format_and_tokenize_prompts


def _make_stage(cls: type[Any], **transformer_kwargs: Any) -> Any:
    pipeline: Any = object.__new__(cls)
    nn.Module.__init__(pipeline)
    # Each stage builds only the tower it declares ownership of, exactly as the
    # real transformer does with cosmos3_owned_towers.
    transformer_kwargs.setdefault("owned_towers", cls.cosmos3_owned_towers)
    pipeline.transformer = _StubTransformer(**transformer_kwargs)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor_spatial = 8
    pipeline.is_edge_model = False
    pipeline.is_distilled_model = False
    _install_tokenizer(pipeline)
    if cls is Cosmos3GeneratorPipeline:
        # What the generator's __init__ does: swap in the replay stub.
        pipeline._bind_owned_tower()
    return pipeline


def _sampling_params(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "height": 1024,
        "width": 1024,
        "guidance_scale": None,
        "guidance_scale_provided": False,
        "max_sequence_length": 512,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _request(prompt: dict[str, Any], sp: Any) -> SimpleNamespace:
    return SimpleNamespace(prompts=[prompt], sampling_params=sp)


def _reasoner_request_output(envelope: dict[str, Any]) -> SimpleNamespace:
    """What the output formatter hands the next stage: payload under ``trajectory``."""
    return SimpleNamespace(
        multimodal_output=envelope["payload"],
        outputs=[SimpleNamespace(multimodal_output=None)],
    )


def _across_the_wire(prompt: dict[str, Any]) -> dict[str, Any]:
    """Round-trip the bridge's prompt through the connector serde."""
    serde = OmniSerde()
    forwarded = dict(prompt)
    forwarded["extra"] = serde.deserialize(serde.serialize(prompt["extra"]))
    return forwarded


def _run_stage_edge(
    reasoner, generator, *, prompt_text="a red car", negative_prompt="blurry", reasoner_sp=None, generator_sp=None
):
    """Drive the whole chain and return what the generator replayed."""
    reasoner_sp = reasoner_sp if reasoner_sp is not None else _sampling_params()
    generator_sp = generator_sp if generator_sp is not None else _sampling_params()

    stage0_prompt = {"prompt": prompt_text, "negative_prompt": negative_prompt, "modalities": ["image"]}
    payload = reasoner.forward(_request(stage0_prompt, reasoner_sp)).output
    envelope = get_cosmos3_reasoner_post_process_func(od_config=None)(payload)
    bridged = reasoner2generator([_reasoner_request_output(envelope)], prompt=stage0_prompt)
    assert bridged is not None
    stage1_prompt = _across_the_wire(bridged)

    replayed: list[list[tuple[torch.Tensor, torch.Tensor]]] = []

    def _seam(self, req):
        """Stand in for Cosmos3VFMTransformer.forward at the tower boundary.

        Resolves the conditioning the way the stock T2I path does, tokenizes with
        this stage's own tokenizer, then calls the UND tower -- which on this stage
        is the replay stub.
        """
        sp = req.sampling_params
        height, width = self._resolve_t2i_geometry(sp)
        max_sequence_length, use_system_prompt, frame_rate = self._resolve_text_encode_params(
            sp, default_use_system_prompt=False
        )
        cond_ids, _cond_mask, uncond_ids, _uncond_mask = self._format_and_tokenize_prompts(
            req.prompts[0].get("prompt", ""),
            req.prompts[0].get("negative_prompt") or "",
            1,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=True,
        )
        for ids in (cond_ids, uncond_ids):
            replayed.append(self.transformer.language_model(ids, "freqs_und"))
        return "image"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Cosmos3OmniDiffusersPipeline, "forward", _seam)
        result = generator.forward(_request(stage1_prompt, generator_sp))

    return SimpleNamespace(result=result, replayed=replayed, payload=payload, bridged=bridged)


class TestStageEdge:
    def test_a_request_survives_the_whole_chain_and_replays_both_branches(self):
        """The headline invariant: K/V computed on stage 0 is what stage 1 uses."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)

        run = _run_stage_edge(reasoner, generator)

        assert run.result == "image"
        # Guidance defaults above 1 for T2I, so both CFG branches were encoded and
        # both were replayed -- and from different table entries.
        table = run.payload[KV_KEY]
        assert len(table) == 2
        assert len(run.replayed) == 2
        cond_first_layer = run.replayed[0][0][0]
        uncond_first_layer = run.replayed[1][0][0]
        assert not torch.equal(cond_first_layer, uncond_first_layer)

        for branch in run.replayed:
            assert len(branch) == NUM_LAYERS
            for k, v in branch:
                # Trimmed to the real token count (3), not the padded length.
                assert k.shape == (1, 3, KV_HEADS, HEAD_DIM)
                assert v.shape == (1, 3, KV_HEADS, HEAD_DIM)

    def test_a_stage_that_resolves_conditioning_differently_misses(self):
        """The failure mode the fingerprint exists to catch.

        Whatever route gets the two towers tokenizing differently -- a per-stage
        ``sampling_constraints``, a future config path -- it must fail loudly, not
        silently produce an image conditioned on the wrong prompt.
        """
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)

        with pytest.raises(RuntimeError, match="no reasoner K/V for this prompt branch"):
            _run_stage_edge(
                reasoner,
                generator,
                reasoner_sp=_sampling_params(max_sequence_length=512),
                generator_sp=_sampling_params(max_sequence_length=256),
            )

    def test_a_generator_at_a_larger_tp_size_replays_its_own_head_shard(self):
        """The two stages need not run the same tensor_parallel_size.

        The reasoner gathers its heads before the payload leaves the tower, so a
        generator sharded more finely simply takes the range its own
        cross-attention owns -- here rank 0's half of the reasoner's full head set.
        """
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        # Half the local KV heads out of the same full count is what TP 2 against the
        # reasoner's TP 1 looks like.
        local_heads = KV_HEADS // 2
        generator = _make_stage(Cosmos3GeneratorPipeline, num_kv_heads_local=local_heads)

        run = _run_stage_edge(reasoner, generator)

        assert run.result == "image"
        # The wire still carried every head, TP-independently.
        assert run.payload[META_KEY]["num_kv_heads"] == KV_HEADS
        emitted = run.payload[KV_KEY]
        for branch in run.replayed:
            for k, v in branch:
                assert k.shape == (1, 3, local_heads, HEAD_DIM)
            # And what it replayed is this rank's slice of one emitted entry.
            matches = [
                entry
                for entry in emitted.values()
                if all(
                    torch.equal(k, ek[..., :local_heads, :]) and torch.equal(v, ev[..., :local_heads, :])
                    for (k, v), (ek, ev) in zip(branch, entry, strict=True)
                )
            ]
            assert len(matches) == 1

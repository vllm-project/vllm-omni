# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end walk of the Cosmos3 reasoner -> generator stage edge, on CPU.

The other Cosmos3 disagg tests exercise one component at a time with hand-built
inputs. This file wires the *real* chain together --

    Cosmos3ReasonerPipeline.forward
      -> get_cosmos3_reasoner_post_process_func   (payload envelope)
      -> OmniSerde                                (the wire)
      -> reasoner2generator                       (the stage bridge)
      -> Cosmos3GeneratorPipeline.forward         (install + replay + clear)

-- with only the 31 B transformer itself stubbed out. It is what can be verified
without two H200s: every seam between the towers is real code, and a payload that
survives all of it is one the deployed pipeline would replay.

The tokenizer stub deliberately *derives* its token ids from the resolved
conditioning values (prompt, geometry, max_sequence_length, use_system_prompt), so
the fingerprint agreement between the two stages is a property under test rather
than an artifact of returning constant ids. That is what makes
``test_a_stage_that_resolves_conditioning_differently_misses`` meaningful: change
one value on one stage and the replay lookup really does miss.
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


# =============================================================================
# Stub transformer -- everything below the tower boundary
# =============================================================================


class _StubUndTower(nn.Module):
    """Per-layer K/V whose values encode the token ids they came from.

    Marking the tensors is what lets the generator side assert it replayed *this*
    branch's K/V rather than merely something of the right shape.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(num_layers))

    def forward(self, text_ids: torch.Tensor, freqs: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del freqs
        batch, seq = text_ids.shape
        marker = float(text_ids.sum().item())
        return [
            (
                torch.full((batch, seq, KV_HEADS, HEAD_DIM), marker + layer),
                torch.full((batch, seq, KV_HEADS, HEAD_DIM), -(marker + layer)),
            )
            for layer in range(len(self.layers))
        ]


class _StubCrossAttention(nn.Module):
    def __init__(self, num_kv_heads_local: int, head_dim: int) -> None:
        super().__init__()
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim


class _StubGenBlock(nn.Module):
    def __init__(self, num_kv_heads_local: int, head_dim: int) -> None:
        super().__init__()
        self.cross_attention = _StubCrossAttention(num_kv_heads_local, head_dim)


class _StubTransformer(nn.Module):
    def __init__(self, *, num_kv_heads_local: int = KV_HEADS, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.language_model = _StubUndTower(NUM_LAYERS)
        self.gen_layers = nn.ModuleList(_StubGenBlock(num_kv_heads_local, head_dim) for _ in range(NUM_LAYERS))
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


# =============================================================================
# Helpers
# =============================================================================


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


def _make_stage(cls: type, **transformer_kwargs: Any) -> Any:
    pipeline: Any = object.__new__(cls)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _StubTransformer(**transformer_kwargs)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor_spatial = 8
    pipeline.is_edge_model = False
    pipeline.is_distilled_model = False
    _install_tokenizer(pipeline)
    # Real _drop_unused_tower: the reasoner loses gen_layers, the generator's UND
    # tower becomes the replay stub with the layout its cross-attention implies.
    pipeline._drop_unused_tower()
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


# =============================================================================
# Tests
# =============================================================================


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

    def test_the_replayed_values_are_the_reasoners_own_tensors(self):
        """Not just the right shape: the same numbers, after the serde hop."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)

        run = _run_stage_edge(reasoner, generator)

        emitted = run.payload[KV_KEY]
        # Match each replayed branch back to the entry it came from by value.
        for branch in run.replayed:
            matches = [
                entry
                for entry in emitted.values()
                if all(
                    torch.equal(k, ek) and torch.equal(v, ev) for (k, v), (ek, ev) in zip(branch, entry, strict=True)
                )
            ]
            assert len(matches) == 1

    def test_the_payload_is_dropped_once_the_request_finishes(self):
        """A stale table would let a later request replay this one's conditioning."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)

        _run_stage_edge(reasoner, generator)

        assert generator.transformer.language_model._table == {}

    def test_two_requests_in_a_row_each_replay_their_own_conditioning(self):
        """The stub is long-lived pipeline state; the payload belongs to a request."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)

        first = _run_stage_edge(reasoner, generator, prompt_text="a red car")
        second = _run_stage_edge(reasoner, generator, prompt_text="a blue bicycle")

        assert not torch.equal(first.replayed[0][0][0], second.replayed[0][0][0])

    def test_the_bridge_forwards_no_geometry_and_the_generator_still_agrees(self):
        """The bridge carries prompt + modalities + K/V and nothing else.

        Geometry and tokenization settings ride in sampling params, which reach
        every stage independently, so dropping them from the prompt dict cannot
        desynchronize the two tokenizations.
        """
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        generator = _make_stage(Cosmos3GeneratorPipeline)
        sp = _sampling_params(height=512, width=768, max_sequence_length=256)

        run = _run_stage_edge(reasoner, generator, reasoner_sp=sp, generator_sp=sp)

        assert set(run.bridged) == {"prompt", "negative_prompt", "modalities", "extra"}
        assert run.result == "image"
        # The reasoner recorded the resolution it used, for diagnostics only.
        assert (run.payload[META_KEY]["height"], run.payload[META_KEY]["width"]) == (512, 768)
        assert run.payload[META_KEY]["max_sequence_length"] == 256

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

    def test_a_generator_at_a_different_tp_size_is_rejected_by_name(self):
        """Same checkpoint, mismatched tensor_parallel_size: the operator must be
        told which knob disagrees, not handed a bare shape error."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)
        # Half the local KV heads is what TP 2 against the reasoner's TP 1 looks like.
        generator = _make_stage(Cosmos3GeneratorPipeline, num_kv_heads_local=KV_HEADS // 2)

        with pytest.raises(RuntimeError, match="disagree on the UND K/V layout"):
            _run_stage_edge(reasoner, generator)

    def test_the_generator_cannot_run_without_a_reasoner(self):
        """Stage 1 is not a standalone pipeline; the error has to say so."""
        generator = _make_stage(Cosmos3GeneratorPipeline)

        with pytest.raises(ValueError, match="cannot run standalone"):
            generator.forward(_request({"prompt": "x", "modalities": ["image"]}, _sampling_params()))

    def test_the_reasoner_refuses_a_video_request(self):
        """Only T2I is split, and the reasoner is where that is enforced."""
        reasoner = _make_stage(Cosmos3ReasonerPipeline)

        with pytest.raises(ValueError, match="text-to-image only"):
            reasoner.forward(_request({"prompt": "x", "modalities": ["video"]}, _sampling_params()))

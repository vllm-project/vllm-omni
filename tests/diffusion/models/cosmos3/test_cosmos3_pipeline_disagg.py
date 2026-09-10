# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the tower-split Cosmos3 pipelines (reasoner / generator).

Both towers are stubbed: what is under test is the split itself -- which tower
each stage owns, the typed conditioning contract the reasoner hands off, how the
generator's TP rank slices its own KV-head shard out of it, and the fingerprint
keying that keeps the two CFG branches apart.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3_disagg as disagg
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_disagg import (
    Cosmos3GeneratorPipeline,
    Cosmos3ReasonerPipeline,
    Cosmos3TextConditioning,
    _gather_kv_heads,
    _ReplayLanguageModel,
    _require_unowned_absent,
    fingerprint_text_ids,
    get_cosmos3_reasoner_post_process_func,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_META_KEY as META_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_SCHEMA,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4


# =============================================================================
# Stubs
# =============================================================================


class StubUndTower(nn.Module):
    """Stands in for ``Cosmos3LanguageModel``: returns marked per-layer K/V.

    ``rope_only`` mirrors the real tower on a stage that does not own it: the
    mRoPE embedding is kept (the GEN pathway needs it every step) and the block
    container is empty.
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        *,
        rope_only: bool = False,
        num_kv_heads_local: int = KV_HEADS,
    ) -> None:
        super().__init__()
        self.rope_only = rope_only
        self.rotary_emb = nn.Identity()
        self.num_layers = num_layers
        self.num_kv_heads_local = num_kv_heads_local
        self.layers = nn.ModuleList() if rope_only else nn.ModuleList(nn.Linear(1, 1) for _ in range(num_layers))
        self.calls: list[tuple[torch.Tensor, Any]] = []

    def forward(self, text_ids: torch.Tensor, freqs: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.calls.append((text_ids.clone(), freqs))
        batch, seq = text_ids.shape
        # [B, S, H_kv_local, D] -- TP-local, as the real tower produces. The trim
        # in the reasoner slices dim 1 (sequence), the gather concatenates dim -2.
        return [
            (
                torch.full((batch, seq, self.num_kv_heads_local, HEAD_DIM), float(i)),
                torch.full((batch, seq, self.num_kv_heads_local, HEAD_DIM), float(i) + 100),
            )
            for i in range(self.num_layers)
        ]


class StubCrossAttention(nn.Module):
    """Stands in for ``Cosmos3CrossAttention``: the consumer of the replayed K/V.

    The real one resolves ``num_kv_heads // tp_size`` at construction time and the
    generator reads both the full head count and its own local slice straight off
    it, so the stub has to carry the same three attributes.
    """

    def __init__(
        self,
        num_kv_heads: int = KV_HEADS,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim


class StubGenBlock(nn.Module):
    """Stands in for ``Cosmos3GenDecoderLayer``."""

    def __init__(
        self,
        num_kv_heads: int = KV_HEADS,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.cross_attention = StubCrossAttention(num_kv_heads, num_kv_heads_local, head_dim)
        self.mlp = nn.Linear(1, 1)


class StubTowerTransformer(nn.Module):
    """The parts of ``Cosmos3VFMTransformer`` the two tower pipelines touch.

    ``owned_towers`` reproduces what the real transformer does with the pipeline's
    ``cosmos3_owned_towers``: the unowned tower's blocks are never built.
    """

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        *,
        owned_towers: tuple[str, ...] = ("reasoner", "generator"),
        fsdp: bool = False,
        num_kv_heads: int = KV_HEADS,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.owned_towers = tuple(owned_towers)
        self.owns_reasoner = "reasoner" in self.owned_towers
        self.owns_generator = "generator" in self.owned_towers
        self.language_model = StubUndTower(
            num_layers,
            rope_only=not self.owns_reasoner,
            num_kv_heads_local=num_kv_heads_local,
        )
        self.gen_layers = nn.ModuleList(
            StubGenBlock(num_kv_heads, num_kv_heads_local, head_dim)
            for _ in range(num_layers if self.owns_generator else 0)
        )
        self.proj_in = nn.Linear(HEAD_DIM, HEAD_DIM)
        self.num_hidden_layers = num_layers
        self.rope_calls: list[dict[str, Any]] = []
        self.offload_contexts: list[str] = []
        self.shard_events: list[str] = []
        if fsdp:
            # FSDP2 adds these at runtime; the reasoner probes with hasattr.
            self.unshard = lambda: self.shard_events.append("unshard")
            self.reshard = lambda: self.shard_events.append("reshard")

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        return h, w, 0, 0

    def _compute_rope_freqs(
        self,
        text_mask: torch.Tensor,
        t: int,
        hp: int,
        wp: int,
        _unused: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[str, str]:
        self.rope_calls.append({"t": t, "hp": hp, "wp": wp, "dtype": dtype, "mask": text_mask.clone()})
        return "freqs_und", "freqs_gen"

    @contextlib.contextmanager
    def _offload_context(self, name: str):
        self.offload_contexts.append(name)
        yield


def _ids(*values: int) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.long)


def _mask(real_len: int, total_len: int) -> torch.Tensor:
    mask = torch.zeros(1, total_len, dtype=torch.long)
    mask[:, :real_len] = 1
    return mask


def _sampling_params(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "height": None,
        "width": None,
        "guidance_scale": None,
        "guidance_scale_provided": False,
        "max_sequence_length": None,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def make_reasoner():
    def _make(
        *,
        num_layers: int = NUM_LAYERS,
        fsdp: bool = False,
        real_len: int = 2,
        total_len: int = 4,
        owned_towers: tuple[str, ...] = ("reasoner",),
        num_kv_heads_local: int = KV_HEADS,
    ):
        pipeline = object.__new__(Cosmos3ReasonerPipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = StubTowerTransformer(
            num_layers,
            owned_towers=owned_towers,
            fsdp=fsdp,
            num_kv_heads_local=num_kv_heads_local,
        )
        pipeline.device = torch.device("cpu")
        pipeline.vae_scale_factor_spatial = 8
        pipeline.is_edge_model = False
        pipeline.is_distilled_model = False

        tokenize_calls: list[dict[str, Any]] = []

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
            tokenize_calls.append(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "max_sequence_length": max_sequence_length,
                    "use_system_prompt": use_system_prompt,
                    "is_t2i": is_t2i,
                }
            )
            return (
                _ids(11, 12, 0, 0),
                _mask(real_len, total_len),
                _ids(21, 22, 0, 0),
                _mask(real_len, total_len),
            )

        pipeline._format_and_tokenize_prompts = _format_and_tokenize_prompts
        pipeline.tokenize_calls = tokenize_calls
        return pipeline

    return _make


@pytest.fixture
def make_generator():
    def _make(
        *,
        num_layers: int = NUM_LAYERS,
        num_kv_heads: int = KV_HEADS,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
        owned_towers: tuple[str, ...] = ("generator",),
    ):
        pipeline = object.__new__(Cosmos3GeneratorPipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = StubTowerTransformer(
            num_layers,
            owned_towers=owned_towers,
            num_kv_heads=num_kv_heads,
            num_kv_heads_local=num_kv_heads_local,
            head_dim=head_dim,
        )
        pipeline.device = torch.device("cpu")
        return pipeline

    return _make


def _entry(
    num_layers: int = NUM_LAYERS,
    seq: int = 2,
    fill: float = 0.0,
    num_kv_heads: int = KV_HEADS,
    head_dim: int = HEAD_DIM,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One branch of a payload, shaped [B, S_und, num_kv_heads, D] per layer.

    ``num_kv_heads`` is the *full*, gathered head count: the wire format carries
    every head regardless of either stage's TP size.
    """
    return [
        (
            torch.full((1, seq, num_kv_heads, head_dim), fill),
            torch.full((1, seq, num_kv_heads, head_dim), fill),
        )
        for _ in range(num_layers)
    ]


def _per_head_entry(num_layers: int = NUM_LAYERS, seq: int = 2, num_kv_heads: int = KV_HEADS) -> list[Any]:
    """A branch whose every head holds a distinguishable value.

    Head ``h`` of layer ``l`` is filled with ``100 * l + h``, so a rank that
    replays the wrong head range cannot pass by accident.
    """
    heads = torch.arange(num_kv_heads, dtype=torch.float32).reshape(1, 1, num_kv_heads, 1)
    return [
        (
            (heads + 100 * layer).expand(1, seq, num_kv_heads, HEAD_DIM).contiguous(),
            (heads + 100 * layer + 0.5).expand(1, seq, num_kv_heads, HEAD_DIM).contiguous(),
        )
        for layer in range(num_layers)
    ]


def _conditioning(branches: dict[str, Any], **overrides: Any) -> Cosmos3TextConditioning:
    values: dict[str, Any] = {
        "branches": branches,
        "num_layers": NUM_LAYERS,
        "num_kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 4096,
        "use_system_prompt": False,
        "reasoner_tp_size": 1,
        "payload_mib": 1.0,
    }
    values.update(overrides)
    return Cosmos3TextConditioning(**values)


# =============================================================================
# fingerprint_text_ids
# =============================================================================


class TestFingerprint:
    def test_stable_for_equal_ids(self):
        assert fingerprint_text_ids(_ids(1, 2, 3)) == fingerprint_text_ids(_ids(1, 2, 3))

    def test_independent_of_dtype_and_shape(self):
        """Both stages must agree even if one hands over int32 or a flat tensor."""
        expected = fingerprint_text_ids(_ids(1, 2, 3))

        assert fingerprint_text_ids(torch.tensor([[1, 2, 3]], dtype=torch.int32)) == expected
        assert fingerprint_text_ids(torch.tensor([1, 2, 3], dtype=torch.int64)) == expected

    @pytest.mark.parametrize("other", [(1, 2, 4), (1, 2), (2, 1, 3), (1, 2, 3, 0)])
    def test_sensitive_to_content(self, other: tuple[int, ...]):
        """CFG branches differ only in their token stream; padding counts too."""
        assert fingerprint_text_ids(_ids(*other)) != fingerprint_text_ids(_ids(1, 2, 3))

    def test_is_a_short_hex_digest(self):
        digest = fingerprint_text_ids(_ids(1, 2, 3))

        assert len(digest) == 32
        assert all(char in "0123456789abcdef" for char in digest)


# =============================================================================
# Tower ownership
# =============================================================================


class TestTowerOwnership:
    """Each stage must *construct* only its own tower, not prune one afterwards.

    The pipeline is built inside the target device's context, so a tower allocated
    in ``__init__`` is allocated on the card: constructing both and dropping one
    still pays the full ~120 GiB peak and fails to start on an 80 GB card.
    """

    def test_each_stage_declares_the_one_tower_it_owns(self):
        assert Cosmos3ReasonerPipeline.cosmos3_owned_towers == ("reasoner",)
        assert Cosmos3GeneratorPipeline.cosmos3_owned_towers == ("generator",)

    def test_the_colocated_pipeline_owns_both(self):
        """``None`` means "no restriction", which is what the shared base must say."""
        assert Cosmos3OmniDiffusersPipeline.cosmos3_owned_towers is None

    def test_require_unowned_absent_accepts_an_empty_container(self):
        _require_unowned_absent(nn.ModuleList(), "test")

    def test_require_unowned_absent_rejects_a_built_tower(self):
        """A non-empty container means the ownership request never reached the
        transformer, so this stage is holding two towers' worth of weights."""
        blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(4))

        with pytest.raises(RuntimeError, match="built 4 test block"):
            _require_unowned_absent(blocks, "test")

    def test_reasoner_binding_accepts_a_reasoner_only_transformer(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline._bind_owned_tower()

        assert len(pipeline.transformer.gen_layers) == 0
        assert len(pipeline.transformer.language_model.layers) == NUM_LAYERS

    def test_reasoner_binding_rejects_a_transformer_that_built_both(self, make_reasoner):
        pipeline = make_reasoner(owned_towers=("reasoner", "generator"))

        with pytest.raises(RuntimeError, match="GEN \\(generator\\) block"):
            pipeline._bind_owned_tower()

    def test_generator_binding_rejects_a_transformer_that_built_both(self, make_generator):
        pipeline = make_generator(owned_towers=("reasoner", "generator"))

        with pytest.raises(RuntimeError, match="UND \\(reasoner\\) block"):
            pipeline._bind_owned_tower()


# =============================================================================
# Cosmos3TextConditioning -- the typed contract on the edge
# =============================================================================


class TestTextConditioningContract:
    def test_declares_the_code_owned_schema_identifier(self):
        """The generator refuses anything that does not declare exactly this."""
        conditioning = _conditioning({"fp": _entry()})

        assert conditioning.schema == COSMOS3_UND_SCHEMA == "cosmos3.text_conditioning/v1"
        assert conditioning.to_payload()[META_KEY]["schema"] == COSMOS3_UND_SCHEMA

    def test_round_trips_through_the_wire_shape(self):
        branches = {"fp": _entry()}
        original = _conditioning(branches, height=512, width=768, reasoner_tp_size=4)

        restored = Cosmos3TextConditioning.from_payload(original.to_payload())

        for field in Cosmos3TextConditioning._WIRE_FIELDS + ("schema",):
            assert getattr(restored, field) == getattr(original, field)
        # The tensors are shared, not copied -- the payload is a view of the same
        # K/V, which is what keeps the round trip free.
        assert restored.branches.keys() == original.branches.keys()
        assert restored.branches["fp"][0][0] is original.branches["fp"][0][0]

    def test_wire_shape_is_the_two_payload_keys(self):
        """Kept a plain dict of tensors: the connector serde handles those, and the
        typed object is the contract rather than the encoding."""
        branches = {"fp": _entry()}
        payload = _conditioning(branches).to_payload()

        assert set(payload) == {KV_KEY, META_KEY}
        assert payload[KV_KEY].keys() == branches.keys()
        assert payload[META_KEY]["num_branches"] == 1

    def test_the_payload_does_not_hand_out_the_live_branch_mapping(self):
        """``frozen=True`` does not freeze what the fields point at.

        Handing out the validated dict itself would let a caller add a branch after
        ``__post_init__`` ran -- past every shape check, straight into replay.
        """
        conditioning = _conditioning({"fp": _entry()})

        payload = conditioning.to_payload()
        payload[KV_KEY]["injected"] = _entry(num_layers=NUM_LAYERS + 3)

        assert set(conditioning.branches) == {"fp"}
        assert conditioning.num_branches == 1

    def test_round_trips_after_tuples_decay_to_lists(self):
        """Stage serializers turn the K/V tuples into 2-lists on the way across."""
        payload = _conditioning({"fp": _entry()}).to_payload()
        payload[KV_KEY] = {key: [list(pair) for pair in entry] for key, entry in payload[KV_KEY].items()}

        restored = Cosmos3TextConditioning.from_payload(payload)

        assert restored.num_branches == 1

    def test_rejects_a_payload_declaring_another_schema(self):
        payload = _conditioning({"fp": _entry()}).to_payload()
        payload[META_KEY]["schema"] = "cosmos3.text_conditioning/v2"

        with pytest.raises(ValueError, match="declares schema 'cosmos3.text_conditioning/v2'"):
            Cosmos3TextConditioning.from_payload(payload)

    def test_rejects_a_payload_with_no_metadata(self):
        with pytest.raises(ValueError, match="carries no .* metadata"):
            Cosmos3TextConditioning.from_payload({KV_KEY: {"fp": _entry()}})

    @pytest.mark.parametrize("field", ["num_layers", "num_kv_heads", "head_dim", "reasoner_tp_size"])
    def test_rejects_metadata_missing_a_layout_field(self, field: str):
        """Every layout field is required: the schema declares a complete layout or
        the payload is not this schema."""
        payload = _conditioning({"fp": _entry()}).to_payload()
        del payload[META_KEY][field]

        with pytest.raises(ValueError, match=f"missing {field}"):
            Cosmos3TextConditioning.from_payload(payload)

    @pytest.mark.parametrize("branches", [{}, None, "not-a-dict"])
    def test_rejects_a_payload_with_no_branches(self, branches: Any):
        payload = _conditioning({"fp": _entry()}).to_payload()
        payload[KV_KEY] = branches

        with pytest.raises(ValueError, match="no .* branches"):
            Cosmos3TextConditioning.from_payload(payload)

    def test_rejects_a_declared_layer_count_the_tensors_do_not_carry(self):
        with pytest.raises(ValueError, match="has 1 layer.*declares num_layers=3"):
            _conditioning({"fp": _entry(num_layers=1)})

    def test_rejects_a_declared_head_count_the_tensors_do_not_carry(self):
        with pytest.raises(ValueError, match="contract declares"):
            _conditioning({"fp": _entry(num_kv_heads=KV_HEADS * 2)})

    def test_rejects_a_declared_head_dim_the_tensors_do_not_carry(self):
        with pytest.raises(ValueError, match="contract declares"):
            _conditioning({"fp": _entry(head_dim=HEAD_DIM + 1)})

    def test_rejects_a_non_4d_tensor(self):
        entry = [(torch.zeros(1, 2), torch.zeros(1, 2))] * NUM_LAYERS

        with pytest.raises(ValueError, match="expected 4"):
            _conditioning({"fp": entry})

    @pytest.mark.parametrize("bad", ["not-a-tensor", None, 7, [1, 2, 3]])
    def test_rejects_a_member_that_is_not_a_tensor(self, bad: Any):
        """A contract error naming the member, not an ``AttributeError`` from inside
        validation. The wire decodes the ``(K, V)`` *pairing* as a list, so a member
        that is itself a list is exactly the corruption worth naming."""
        entry = [(torch.zeros(1, 2, KV_HEADS, HEAD_DIM), bad)] * NUM_LAYERS

        with pytest.raises(ValueError, match="not a tensor"):
            _conditioning({"fp": entry})

    def test_rejects_branches_that_disagree_on_batch_size(self):
        """All branches replay into one GEN forward, so a mixed batch size cannot be
        conditioned on -- and would otherwise broadcast or fail inside attention."""
        branches = {
            "cond": _entry(),
            "uncond": [
                (torch.zeros(3, 2, KV_HEADS, HEAD_DIM), torch.zeros(3, 2, KV_HEADS, HEAD_DIM))
                for _ in range(NUM_LAYERS)
            ],
        }

        with pytest.raises(ValueError, match="batch size 3, but the rest of the payload carries 1"):
            _conditioning(branches)

    def test_a_uniform_batch_size_above_one_is_accepted(self):
        """The check is consistency, not a hard-coded 1: nothing here should stand in
        the way of a future batched reasoner."""
        entry = [(torch.zeros(2, 2, KV_HEADS, HEAD_DIM), torch.zeros(2, 2, KV_HEADS, HEAD_DIM))] * NUM_LAYERS

        conditioning = _conditioning({"fp": entry})

        assert conditioning.num_branches == 1

    def test_rejects_layers_of_a_branch_that_cover_different_token_counts(self):
        """Every layer of a branch comes from one UND forward over one prompt, so a
        varying S_und is corruption. Branch *lengths* legitimately differ from each
        other -- each is trimmed to its own real prompt length -- which is why this
        is compared within a branch and not across the payload."""
        entry = [*_entry(seq=2)]
        entry[1] = (torch.zeros(1, 5, KV_HEADS, HEAD_DIM), torch.zeros(1, 5, KV_HEADS, HEAD_DIM))

        with pytest.raises(ValueError, match="covers 5 UND token.*rest of the branch covers 2"):
            _conditioning({"fp": entry})

    def test_branches_of_different_lengths_are_accepted(self):
        """The other half of the check above: two prompts of different real lengths
        are the normal CFG case, not an error."""
        conditioning = _conditioning({"cond": _entry(seq=5), "uncond": _entry(seq=2)})

        assert conditioning.conditioning_length == 5

    def test_rejects_a_payload_of_mixed_dtypes(self):
        """The replay casts to the GEN dtype on the way to the device, so a stray
        dtype would be silently normalized there. It means the tensors did not
        survive the stage edge intact, which is worth failing on."""
        entry = [*_entry()]
        entry[1] = (
            torch.zeros(1, 2, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
            torch.zeros(1, 2, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16),
        )

        with pytest.raises(ValueError, match="torch.bfloat16, but the rest of the payload is torch.float32"):
            _conditioning({"fp": entry})

    def test_rejects_k_and_v_of_different_shapes(self):
        """There is no K/V-pair check any more, and none is needed: every dim of
        *both* members is validated, so the pair cannot disagree without one of them
        having already been rejected -- here on the token count that differs."""
        entry = [(torch.zeros(1, 2, KV_HEADS, HEAD_DIM), torch.zeros(1, 3, KV_HEADS, HEAD_DIM))] * NUM_LAYERS

        with pytest.raises(ValueError, match="covers 3 UND token"):
            _conditioning({"fp": entry})

    @pytest.mark.parametrize("field", ["num_layers", "num_kv_heads", "head_dim"])
    def test_rejects_a_non_positive_layout_field(self, field: str):
        """``num_layers=0`` matches an empty branch list, so it passes every other
        check vacuously and only surfaces later as a bare ``max()`` error from
        ``conditioning_length`` -- or as a replay of nothing at all."""
        with pytest.raises(ValueError, match="must be positive"):
            _conditioning({"fp": []} if field == "num_layers" else {"fp": _entry()}, **{field: 0})

    def test_rejects_no_branches(self):
        with pytest.raises(ValueError, match="no prompt branches"):
            _conditioning({})

    def test_reports_the_longest_conditioning_length(self):
        conditioning = _conditioning({"a": _entry(seq=2), "b": _entry(seq=5)})

        assert conditioning.conditioning_length == 5
        assert conditioning.num_branches == 2


# =============================================================================
# TP sharding on the edge
# =============================================================================


class TestKvHeadGather:
    """UND K/V is born TP-sharded and only rank 0's stage output escapes, so the
    reasoner must gather the head dimension before the payload leaves the tower."""

    def test_is_a_no_op_at_tp_1(self, monkeypatch):
        """No collective is entered, so this stays callable in a single process."""
        monkeypatch.setattr(
            disagg,
            "tensor_model_parallel_all_gather",
            lambda *_a, **_k: pytest.fail("no collective at TP 1"),
        )
        tensor = torch.zeros(1, 2, KV_HEADS, HEAD_DIM)

        assert _gather_kv_heads(tensor) is tensor

    def test_gathers_the_head_dimension(self, monkeypatch):
        """dim=-2 is the KV-head axis of [B, S, H_kv, D], which is the axis
        ``ColumnParallelLinear`` sharded."""
        seen: list[dict[str, Any]] = []

        def _fake_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
            seen.append({"dim": dim, "contiguous": tensor.is_contiguous()})
            return torch.cat([tensor, tensor], dim=dim)

        monkeypatch.setattr(disagg, "_tp_world_size", lambda: 2)
        monkeypatch.setattr(disagg, "tensor_model_parallel_all_gather", _fake_all_gather)

        gathered = _gather_kv_heads(torch.zeros(1, 2, KV_HEADS, HEAD_DIM))

        assert gathered.shape == (1, 2, KV_HEADS * 2, HEAD_DIM)
        assert seen == [{"dim": -2, "contiguous": True}]

    def test_reasoner_ships_the_full_head_set(self, make_reasoner, monkeypatch):
        """Simulated TP 2: the tower produces KV_HEADS local heads per rank and the
        payload declares -- and carries -- twice that."""
        monkeypatch.setattr(disagg, "_tp_world_size", lambda: 2)
        monkeypatch.setattr(
            disagg,
            "tensor_model_parallel_all_gather",
            lambda tensor, dim: torch.cat([tensor, tensor + 1], dim=dim),
        )
        pipeline = make_reasoner()

        conditioning = pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert conditioning.num_kv_heads == KV_HEADS * 2
        assert conditioning.reasoner_tp_size == 2
        for entry in conditioning.branches.values():
            for k, _v in entry:
                assert k.shape == (1, 2, KV_HEADS * 2, HEAD_DIM)


class TestMultiRankShardIdentity:
    """The finding this gather exists for: with TP-local shards on the wire, every
    generator rank receives rank 0's heads and silently conditions on them.

    Two *real* generator pipelines stand in for the two generator TP ranks: each is
    bound by ``Cosmos3GeneratorPipeline._bind_owned_tower`` with this process
    reporting that rank, and loaded through ``install_text_conditioning`` off the
    same wire payload. Nothing here passes ``kv_head_offset`` by hand -- that is the
    whole point. The offset is derived by the code under test, so reverting it to a
    rank-agnostic handoff (every rank taking rank 0's heads) makes these fail.

    Every head of the payload holds a distinguishable value, so a rank replaying
    the wrong range cannot pass by coincidence.
    """

    FULL_HEADS = 4
    LOCAL_HEADS = 2

    def _rank_pipeline(self, rank: int, make_generator, monkeypatch):
        """A generator stage that believes it is ``rank`` of a 2-rank TP group."""
        monkeypatch.setattr(disagg, "_tp_rank", lambda: rank)
        monkeypatch.setattr(disagg, "_tp_world_size", lambda: 2)
        pipeline = make_generator(num_kv_heads=self.FULL_HEADS, num_kv_heads_local=self.LOCAL_HEADS)
        pipeline._bind_owned_tower()
        return pipeline

    def _replay(self, rank: int, text_ids: torch.Tensor, conditioning: Cosmos3TextConditioning, *args):
        """Install through the real payload path and replay, at ``rank``."""
        pipeline = self._rank_pipeline(rank, *args)
        pipeline.install_text_conditioning(conditioning.to_payload())
        return pipeline.transformer.language_model(text_ids, freqs=None)

    def _payload(self, text_ids: torch.Tensor, branch=None):
        branch = branch if branch is not None else _per_head_entry(num_kv_heads=self.FULL_HEADS)
        return _conditioning(
            {fingerprint_text_ids(text_ids): branch},
            num_kv_heads=self.FULL_HEADS,
        )

    def test_each_rank_replays_its_own_contiguous_head_range(self, make_generator, monkeypatch):
        text_ids = _ids(11, 12)
        conditioning = self._payload(text_ids)

        for rank in (0, 1):
            replayed = self._replay(rank, text_ids, conditioning, make_generator, monkeypatch)

            assert len(replayed) == NUM_LAYERS
            for layer, (k, v) in enumerate(replayed):
                assert k.shape == (1, 2, self.LOCAL_HEADS, HEAD_DIM)
                expected = [100 * layer + rank * self.LOCAL_HEADS + h for h in range(self.LOCAL_HEADS)]
                assert k[0, 0, :, 0].tolist() == expected
                assert v[0, 0, :, 0].tolist() == [value + 0.5 for value in expected]

    def test_the_two_ranks_do_not_receive_the_same_heads(self, make_generator, monkeypatch):
        """Fails outright under the rank-0-only handoff this replaces."""
        text_ids = _ids(11, 12)
        conditioning = self._payload(text_ids)

        rank0 = self._replay(0, text_ids, conditioning, make_generator, monkeypatch)
        rank1 = self._replay(1, text_ids, conditioning, make_generator, monkeypatch)

        for (k0, _v0), (k1, _v1) in zip(rank0, rank1):
            assert not torch.equal(k0, k1)

    def test_the_two_ranks_together_reconstruct_the_full_head_set(self, make_generator, monkeypatch):
        """Concatenating the shards in rank order returns the gathered payload, which
        is what makes the split numerically faithful to the co-located tower."""
        text_ids = _ids(11, 12)
        branch = _per_head_entry(num_kv_heads=self.FULL_HEADS)
        conditioning = self._payload(text_ids, branch)

        rank0 = self._replay(0, text_ids, conditioning, make_generator, monkeypatch)
        rank1 = self._replay(1, text_ids, conditioning, make_generator, monkeypatch)

        for (k_full, _v_full), (k0, _v0), (k1, _v1) in zip(branch, rank0, rank1):
            torch.testing.assert_close(torch.cat([k0, k1], dim=-2), k_full)

    def test_no_rank_reads_the_offset_from_anywhere_but_its_tp_rank(self, make_generator, monkeypatch):
        """The offset must come from the process's TP rank, not from the payload or
        the config -- the two stages' TP sizes are independent, so nothing on the
        wire can tell this rank which heads are its own.

        Stated as a property over the whole group rather than one rank's arithmetic:
        the ranks' ranges must partition ``[0, FULL_HEADS)`` exactly. A handoff that
        gave every rank the same shard, or that derived the offset from
        ``reasoner_tp_size`` (1 here, while this stage runs 2), fails this.
        """
        offsets = []
        for rank in (0, 1):
            pipeline = self._rank_pipeline(rank, make_generator, monkeypatch)
            stub = pipeline.transformer.language_model
            offsets.append((stub.kv_head_offset, stub.kv_head_offset + stub.num_kv_heads_local))

        assert offsets == [(0, 2), (2, 4)]
        covered = [head for start, stop in offsets for head in range(start, stop)]
        assert sorted(covered) == list(range(self.FULL_HEADS))
        assert len(covered) == len(set(covered))  # no head replayed twice

    def test_a_tp_1_generator_takes_every_head(self):
        """The payload is TP-independent, so the two stages need not match."""
        text_ids = _ids(11, 12)
        conditioning = _conditioning(
            {fingerprint_text_ids(text_ids): _per_head_entry(num_kv_heads=self.FULL_HEADS)},
            num_kv_heads=self.FULL_HEADS,
        )
        stub = _ReplayLanguageModel(
            NUM_LAYERS,
            nn.Identity(),
            num_kv_heads=self.FULL_HEADS,
            num_kv_heads_local=self.FULL_HEADS,
            kv_head_offset=0,
            head_dim=HEAD_DIM,
        )
        stub.install(conditioning)

        k, _v = stub(text_ids, freqs=None)[0]

        assert k.shape == (1, 2, self.FULL_HEADS, HEAD_DIM)
        assert k[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


# =============================================================================
# _ReplayLanguageModel
# =============================================================================


class TestReplayLanguageModel:
    def _stub(self, num_layers: int = NUM_LAYERS) -> _ReplayLanguageModel:
        return _ReplayLanguageModel(
            num_layers,
            nn.Identity(),
            num_kv_heads=KV_HEADS,
            num_kv_heads_local=KV_HEADS,
            kv_head_offset=0,
            head_dim=HEAD_DIM,
        )

    def test_attribute_surface_the_transformer_relies_on(self):
        """``_compute_rope_freqs`` needs rotary_emb; offload needs ``layers``."""
        rotary = nn.Identity()
        stub = _ReplayLanguageModel(
            NUM_LAYERS,
            rotary,
            num_kv_heads=KV_HEADS,
            num_kv_heads_local=KV_HEADS,
            kv_head_offset=0,
            head_dim=HEAD_DIM,
        )

        assert stub.rotary_emb is rotary
        assert isinstance(stub.layers, nn.ModuleList)
        assert len(stub.layers) == 0
        assert _ReplayLanguageModel._layerwise_offload_blocks_attrs == ["layers"]
        assert isinstance(stub, nn.Module)

    def test_carries_no_parameters(self):
        """No UND weights on the generator stage -- that is the point of the split."""
        assert list(self._stub().parameters()) == []
        assert self._stub().state_dict() == {}

    def test_replays_the_installed_branch(self):
        text_ids = _ids(11, 12)
        stub = self._stub()
        stub.install(_conditioning({fingerprint_text_ids(text_ids): _entry()}), dtype=torch.bfloat16)

        replayed = stub(text_ids, freqs=("und", "gen"))

        assert len(replayed) == NUM_LAYERS
        for k, v in replayed:
            assert k.shape == (1, 2, KV_HEADS, HEAD_DIM)
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16

    def test_accepts_lists_instead_of_tuples(self):
        """Stage serializers turn the K/V tuples into lists on the way across."""
        text_ids = _ids(11, 12)
        entry = [list(pair) for pair in _entry()]
        stub = self._stub()
        stub.install(_conditioning({fingerprint_text_ids(text_ids): entry}))

        assert len(stub(text_ids, freqs=None)) == NUM_LAYERS

    def test_keeps_cfg_branches_apart(self):
        cond, uncond = _ids(11, 12), _ids(21, 22)
        stub = self._stub()
        stub.install(
            _conditioning(
                {
                    fingerprint_text_ids(cond): _entry(fill=0.0),
                    fingerprint_text_ids(uncond): _entry(fill=1.0),
                }
            )
        )

        assert stub(cond, freqs=None)[0][0].sum() == 0
        assert stub(uncond, freqs=None)[0][0].sum() == 2 * KV_HEADS * HEAD_DIM

    def test_fingerprint_miss_raises_loudly(self):
        """A tokenization divergence must not silently produce a wrong image."""
        stub = self._stub()
        stub.install(_conditioning({fingerprint_text_ids(_ids(11, 12)): _entry()}))

        with pytest.raises(RuntimeError, match="no reasoner K/V for this prompt"):
            stub(_ids(99, 98), freqs=None)

    def test_the_miss_reports_what_the_reasoner_resolved(self):
        """A bare fingerprint mismatch tells an operator nothing actionable.

        The settings that feed the fingerprint are deliberately *not* re-validated
        against this stage -- the fingerprint is the check -- so the error has to
        carry the reasoner's own values, which is the other half of the comparison.
        """
        stub = self._stub()
        stub.install(
            _conditioning(
                {fingerprint_text_ids(_ids(11, 12)): _entry()},
                height=512,
                width=768,
                max_sequence_length=256,
                use_system_prompt=True,
            )
        )

        with pytest.raises(RuntimeError, match="reasoner resolved height=512, width=768") as excinfo:
            stub(_ids(99, 98), freqs=None)
        assert "max_sequence_length=256" in str(excinfo.value)
        assert "use_system_prompt=True" in str(excinfo.value)

    def test_empty_table_raises(self):
        with pytest.raises(RuntimeError, match="no reasoner K/V"):
            self._stub()(_ids(11, 12), freqs=None)

    def test_install_rejects_a_layer_count_this_stage_does_not_run(self):
        """Checked at install time, once per request, rather than once per step."""
        stub = self._stub(num_layers=NUM_LAYERS + 1)

        with pytest.raises(RuntimeError, match="num_layers=3 from reasoner, 4 here"):
            stub.install(_conditioning({fingerprint_text_ids(_ids(11, 12)): _entry()}))

    def test_install_rejects_a_head_count_this_stage_does_not_consume(self):
        """A cross-stage checkpoint or config mismatch -- *not* a TP mismatch, which
        the gathered payload makes irrelevant. The message says so, and names both
        stages' TP sizes because that is the context an operator needs."""
        stub = self._stub()
        conditioning = _conditioning(
            {fingerprint_text_ids(_ids(11, 12)): _entry(num_kv_heads=KV_HEADS * 2)},
            num_kv_heads=KV_HEADS * 2,
            reasoner_tp_size=2,
        )

        with pytest.raises(RuntimeError, match="disagree on the UND K/V layout") as excinfo:
            stub.install(conditioning)

        message = str(excinfo.value)
        assert f"num_kv_heads={KV_HEADS * 2} from reasoner, {KV_HEADS} here" in message
        assert "TP-independent" in message
        assert "tensor_parallel_size=2" in message

    def test_install_rejects_a_wrong_head_dim(self):
        stub = self._stub()
        conditioning = _conditioning(
            {fingerprint_text_ids(_ids(11, 12)): _entry(head_dim=HEAD_DIM + 1)},
            head_dim=HEAD_DIM + 1,
        )

        with pytest.raises(RuntimeError, match="head_dim=5 from reasoner, 4 here"):
            stub.install(conditioning)

    def test_clear_drops_the_payload(self):
        """A payload belongs to one request; the stub outlives it."""
        text_ids = _ids(11, 12)
        stub = self._stub()
        stub.install(_conditioning({fingerprint_text_ids(text_ids): _entry()}), dtype=torch.bfloat16)

        stub.clear()

        assert stub._table == {}
        assert stub._dtype is None
        with pytest.raises(RuntimeError, match="no reasoner K/V"):
            stub(text_ids, freqs=None)


# =============================================================================
# Reasoner stage
# =============================================================================


class TestReasonerPipeline:
    def test_is_a_cosmos3_pipeline_with_warmup_disabled(self):
        """Neither tower can serve the engine's synthetic warmup request."""
        assert issubclass(Cosmos3ReasonerPipeline, Cosmos3OmniDiffusersPipeline)
        assert Cosmos3ReasonerPipeline.dummy_run_num_frames == 0

    def test_payload_is_keyed_by_fingerprint_and_trimmed(self, make_reasoner):
        pipeline = make_reasoner(real_len=2, total_len=4)

        # Guidance off, so the contract holds the conditional branch alone.
        conditioning = pipeline.encode_text_conditioning(
            "a red car",
            "",
            _sampling_params(guidance_scale=1.0, guidance_scale_provided=True),
        )

        branches = conditioning.branches
        assert set(branches) == {fingerprint_text_ids(_ids(11, 12, 0, 0))}
        entry = branches[fingerprint_text_ids(_ids(11, 12, 0, 0))]
        assert len(entry) == NUM_LAYERS
        for k, v in entry:
            # Trimmed to the real text length, not the padded one.
            assert k.shape == (1, 2, KV_HEADS, HEAD_DIM)
            assert v.shape == (1, 2, KV_HEADS, HEAD_DIM)
            assert k.device.type == "cpu"
            assert k.is_contiguous()

    def test_contract_reports_what_the_generator_must_reproduce(self, make_reasoner):
        pipeline = make_reasoner()

        conditioning = pipeline.encode_text_conditioning(
            "a red car",
            "",
            _sampling_params(height=512, width=768, max_sequence_length=256),
        )

        assert conditioning.height == 512
        assert conditioning.width == 768
        assert conditioning.max_sequence_length == 256
        assert conditioning.use_system_prompt is False
        # The T2I default guidance scale is > 1, so both CFG branches are encoded.
        assert conditioning.num_branches == 2
        # Reported for logging only, rounded to 0.1 MiB -- a stub payload floors to 0.0.
        assert isinstance(conditioning.payload_mib, float)

    def test_contract_reports_the_kv_layout_read_off_the_tensors(self, make_reasoner):
        """The generator compares these against its own cross-attention to name a
        stage-configuration mismatch. Read from the emitted tensors rather than the
        config, so the contract cannot describe a payload this stage did not send."""
        pipeline = make_reasoner()

        conditioning = pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert conditioning.num_layers == NUM_LAYERS
        assert conditioning.num_kv_heads == KV_HEADS
        assert conditioning.head_dim == HEAD_DIM
        # No TP group in a single-process test, which is TP 1 by definition.
        assert conditioning.reasoner_tp_size == 1

    def test_oversized_payload_warns_but_still_ships(self, make_reasoner, monkeypatch, caplog):
        """An oversized payload is correct, just expensive -- so warn, do not raise."""
        monkeypatch.setattr(disagg, "COSMOS3_UND_PAYLOAD_WARN_MIB", 0.0)
        pipeline = make_reasoner()

        with caplog.at_level("WARNING"):
            conditioning = pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert conditioning.branches
        assert "max_sequence_length" in caplog.text

    def test_normal_payload_does_not_warn(self, make_reasoner, caplog):
        pipeline = make_reasoner()

        with caplog.at_level("WARNING"):
            pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert "K/V payload is" not in caplog.text

    def test_geometry_and_tokenizer_settings_reach_the_tokenizer(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_text_conditioning("a red car", "blurry", _sampling_params(height=512, width=768))

        call = pipeline.tokenize_calls[0]
        assert call["prompt"] == "a red car"
        assert call["negative_prompt"] == "blurry"
        assert call["is_t2i"] is True
        assert call["num_frames"] == 1
        assert (call["height"], call["width"]) == (512, 768)

    def test_gen_latent_geometry_is_passed_to_rope(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_text_conditioning("a red car", "", _sampling_params(height=512, width=768))

        rope = pipeline.transformer.rope_calls[0]
        assert rope["t"] == 1
        assert (rope["hp"], rope["wp"]) == (512 // 8, 768 // 8)

    def test_encodes_both_branches_when_guidance_is_active(self, make_reasoner):
        pipeline = make_reasoner()

        conditioning = pipeline.encode_text_conditioning(
            "a red car",
            "blurry",
            _sampling_params(guidance_scale=7.0, guidance_scale_provided=True),
        )

        assert conditioning.num_branches == 2
        assert set(conditioning.branches) == {
            fingerprint_text_ids(_ids(11, 12, 0, 0)),
            fingerprint_text_ids(_ids(21, 22, 0, 0)),
        }

    def test_skips_the_unconditional_branch_without_guidance(self, make_reasoner):
        """Saves a full UND forward and halves the payload."""
        pipeline = make_reasoner()

        conditioning = pipeline.encode_text_conditioning(
            "a red car",
            "blurry",
            _sampling_params(guidance_scale=1.0, guidance_scale_provided=True),
        )

        assert conditioning.num_branches == 1
        assert len(pipeline.transformer.language_model.calls) == 1

    def test_runs_the_und_tower_inside_its_offload_context(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        # One entry per CFG branch: every tower call is wrapped.
        assert pipeline.transformer.offload_contexts == ["reasoner", "reasoner"]

    def test_unshards_around_the_direct_tower_call(self, make_reasoner):
        """Calling language_model directly bypasses FSDP2's root pre-forward hook."""
        pipeline = make_reasoner(fsdp=True)

        pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert pipeline.transformer.shard_events == ["unshard", "reshard"]

    def test_reshards_even_when_the_tower_raises(self, make_reasoner):
        pipeline = make_reasoner(fsdp=True)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("tower exploded")

        pipeline.transformer.language_model.forward = _boom

        with pytest.raises(RuntimeError, match="tower exploded"):
            pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert pipeline.transformer.shard_events == ["unshard", "reshard"]

    def test_forward_emits_the_handoff_payload(self, make_reasoner):
        pipeline = make_reasoner()
        req = SimpleNamespace(
            prompts=[{"prompt": "a red car", "modalities": ["image"], "negative_prompt": "blurry"}],
            sampling_params=_sampling_params(),
        )

        output = pipeline.forward(req)

        assert set(output.output) == {KV_KEY, META_KEY}
        assert output.output[META_KEY]["schema"] == COSMOS3_UND_SCHEMA
        assert pipeline.tokenize_calls[0]["negative_prompt"] == "blurry"

    def test_forward_output_is_installable_on_the_generator(self, make_reasoner, make_generator):
        """The two ends of the contract, end to end and in one process."""
        reasoner = make_reasoner()
        generator = make_generator()
        generator._bind_owned_tower()
        req = SimpleNamespace(
            prompts=[{"prompt": "a red car", "modalities": ["image"]}],
            sampling_params=_sampling_params(),
        )

        payload = reasoner.forward(req).output
        installed = generator.install_text_conditioning(payload)

        assert installed.num_branches == 2
        assert len(generator.transformer.language_model._table) == 2

    @pytest.mark.parametrize(
        "prompts",
        [
            # A bare string carries no modalities, which stock Cosmos3 reads as video.
            ["a red car"],
            [{"prompt": "a red car", "modalities": ["video"]}],
        ],
    )
    def test_forward_rejects_non_t2i_requests(self, make_reasoner, prompts):
        pipeline = make_reasoner()
        req = SimpleNamespace(prompts=prompts, sampling_params=_sampling_params())

        with pytest.raises(ValueError, match="text-to-image only"):
            pipeline.forward(req)


# =============================================================================
# Generator stage
# =============================================================================


class TestGeneratorPipeline:
    def test_is_a_cosmos3_pipeline_with_warmup_disabled(self):
        assert issubclass(Cosmos3GeneratorPipeline, Cosmos3OmniDiffusersPipeline)
        assert Cosmos3GeneratorPipeline.dummy_run_num_frames == 0

    def test_binds_the_replay_stub_over_the_rope_only_tower(self, make_generator):
        pipeline = make_generator()
        original = pipeline.transformer.language_model
        rotary = original.rotary_emb

        pipeline._bind_owned_tower()

        stub = pipeline.transformer.language_model
        assert isinstance(stub, _ReplayLanguageModel)
        assert stub.num_hidden_layers == NUM_LAYERS
        # The real GEN mRoPE frequencies are built from this, every step.
        assert stub.rotary_emb is rotary
        # No UND blocks were ever constructed on this stage.
        assert len(original.layers) == 0
        # The GEN tower is the one this stage actually runs.
        assert len(pipeline.transformer.gen_layers) == NUM_LAYERS

    def test_stub_takes_its_expected_layout_from_the_consuming_cross_attention(self, make_generator):
        """Read from the module that will receive the tensors, not recomputed from
        the config and TP size, so it cannot disagree with the consumer."""
        pipeline = make_generator(num_kv_heads=8, num_kv_heads_local=4, head_dim=64)

        pipeline._bind_owned_tower()

        stub = pipeline.transformer.language_model
        assert (stub.num_kv_heads, stub.num_kv_heads_local, stub.head_dim) == (8, 4, 64)

    def test_stub_head_offset_follows_this_process_tp_rank(self, make_generator, monkeypatch):
        """The slice each rank takes is ``[rank * H_local, (rank + 1) * H_local)`` --
        the contiguous range ``ColumnParallelLinear`` assigned to it."""
        monkeypatch.setattr(disagg, "_tp_rank", lambda: 1)
        pipeline = make_generator(num_kv_heads=8, num_kv_heads_local=4)

        pipeline._bind_owned_tower()

        assert pipeline.transformer.language_model.kv_head_offset == 4

    def test_head_offset_is_zero_without_a_tp_group(self, make_generator):
        """A single-process stage is rank 0 of a TP-1 group, so it takes every head."""
        pipeline = make_generator()

        pipeline._bind_owned_tower()

        assert pipeline.transformer.language_model.kv_head_offset == 0

    def test_a_stage_without_its_own_tower_is_caught(self, make_generator):
        """Without GEN blocks there is no cross-attention to describe the layout and
        nothing to replay into -- the stage did not build the tower it owns."""
        pipeline = make_generator()
        del pipeline.transformer.gen_layers[:]

        with pytest.raises(RuntimeError, match="no GEN blocks"):
            pipeline._bind_owned_tower()

    @pytest.mark.parametrize(
        ("num_kv_heads", "num_kv_heads_local", "rank"),
        [
            # A stage whose TP size exceeds the KV-head count: num_kv_heads // tp is
            # 0, so every rank would install an empty head range.
            (2, 0, 0),
            # A TP size that does not divide the head count: rank 2's range runs off
            # the end of the gathered set, and Python slicing would clamp it.
            (3, 2, 1),
            (4, 3, 1),
        ],
    )
    def test_a_head_range_that_is_not_a_valid_shard_is_rejected_at_bind_time(
        self, make_generator, monkeypatch, num_kv_heads: int, num_kv_heads_local: int, rank: int
    ):
        """Slicing clamps instead of raising, so an impossible range has to be caught.

        Left to itself, ``install`` would hand cross-attention fewer heads than it
        expects -- or none -- and the failure would surface much later as a shape
        error inside attention, or not at all. Fail here, naming the TP size.
        """
        monkeypatch.setattr(disagg, "_tp_rank", lambda: rank)
        monkeypatch.setattr(disagg, "_tp_world_size", lambda: 2)
        pipeline = make_generator(num_kv_heads=num_kv_heads, num_kv_heads_local=num_kv_heads_local)

        with pytest.raises(ValueError, match="not a valid shard"):
            pipeline._bind_owned_tower()

    def test_a_valid_shard_at_the_end_of_the_head_set_is_accepted(self, make_generator, monkeypatch):
        """The boundary case the check must not reject: the last rank's range ends
        exactly at the head count."""
        monkeypatch.setattr(disagg, "_tp_rank", lambda: 3)
        monkeypatch.setattr(disagg, "_tp_world_size", lambda: 4)
        pipeline = make_generator(num_kv_heads=8, num_kv_heads_local=2)

        pipeline._bind_owned_tower()

        stub = pipeline.transformer.language_model
        assert (stub.kv_head_offset, stub.kv_head_offset + stub.num_kv_heads_local) == (6, 8)

    def test_forward_clears_the_payload_when_the_denoise_loop_finishes(self, make_generator, monkeypatch):
        """The payload belongs to one request but the stub is long-lived pipeline
        state, so a table left installed could be replayed by a later request."""
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        text_ids = _ids(11, 12)
        payload = _conditioning({fingerprint_text_ids(text_ids): _entry()}).to_payload()
        req = SimpleNamespace(prompts=[{"prompt": "x", "extra": payload}], sampling_params=None)

        installed: list[list[str]] = []

        def _record_then_return_image(self, _req):
            """Snapshot the replay table from inside the denoise loop."""
            installed.append(sorted(self.transformer.language_model._table))
            return "image"

        monkeypatch.setattr(Cosmos3OmniDiffusersPipeline, "forward", _record_then_return_image)

        assert pipeline.forward(req) == "image"
        # Installed for the duration of the denoise loop, gone afterwards.
        assert installed == [[fingerprint_text_ids(text_ids)]]
        assert pipeline.transformer.language_model._table == {}

    def test_forward_clears_the_payload_even_when_denoising_raises(self, make_generator, monkeypatch):
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        payload = _conditioning({fingerprint_text_ids(_ids(11, 12)): _entry()}).to_payload()
        req = SimpleNamespace(prompts=[{"prompt": "x", "extra": payload}], sampling_params=None)

        def _boom(self, _req):
            raise RuntimeError("denoise exploded")

        monkeypatch.setattr(Cosmos3OmniDiffusersPipeline, "forward", _boom)

        with pytest.raises(RuntimeError, match="denoise exploded"):
            pipeline.forward(req)

        assert pipeline.transformer.language_model._table == {}

    def test_extracts_the_payload_from_the_prompt(self):
        payload = {KV_KEY: {"fp": []}, META_KEY: {"height": 1024}}
        req = SimpleNamespace(prompts=[{"prompt": "x", "extra": payload}], sampling_params=None)

        assert Cosmos3GeneratorPipeline._extract_und_payload(req) is payload

    def test_extracts_the_payload_from_sampling_params(self):
        """Fallback that mirrors GLM-Image's DiT stage; handy for direct driving."""
        payload: dict[str, Any] = {KV_KEY: {"fp": []}}
        req = SimpleNamespace(prompts=[{"prompt": "x"}], sampling_params=SimpleNamespace(extra_args=payload))

        assert Cosmos3GeneratorPipeline._extract_und_payload(req) is payload

    @pytest.mark.parametrize(
        "req",
        [
            SimpleNamespace(prompts=[{"prompt": "x"}], sampling_params=SimpleNamespace(extra_args={})),
            SimpleNamespace(prompts=[{"prompt": "x", "extra": {}}], sampling_params=None),
            SimpleNamespace(prompts=["x"], sampling_params=None),
            SimpleNamespace(prompts=[], sampling_params=None),
        ],
    )
    def test_missing_payload_raises(self, req):
        """This stage cannot run standalone: it has no UND weights to fall back on."""
        with pytest.raises(ValueError, match="without reasoner K/V"):
            Cosmos3GeneratorPipeline._extract_und_payload(req)

    def test_forward_rejects_a_request_without_kv(self, make_generator):
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        req = SimpleNamespace(prompts=[{"prompt": "x"}], sampling_params=SimpleNamespace(extra_args={}))

        with pytest.raises(ValueError, match="without reasoner K/V"):
            pipeline.forward(req)

    def test_install_loads_this_rank_shard_of_the_payload(self, make_generator):
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        text_ids = _ids(11, 12)
        payload = _conditioning({fingerprint_text_ids(text_ids): _entry()}).to_payload()

        installed = pipeline.install_text_conditioning(payload)

        stub = pipeline.transformer.language_model
        assert installed.num_branches == 1
        assert set(stub._table) == {fingerprint_text_ids(text_ids)}
        assert stub._dtype == pipeline.transformer.proj_in.weight.dtype
        assert len(stub(text_ids, freqs=None)) == NUM_LAYERS

    def test_install_rejects_a_layout_this_stage_does_not_consume(self, make_generator):
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        payload = _conditioning(
            {fingerprint_text_ids(_ids(11, 12)): _entry(head_dim=HEAD_DIM + 1)},
            head_dim=HEAD_DIM + 1,
        ).to_payload()

        with pytest.raises(RuntimeError, match="disagree on the UND K/V layout"):
            pipeline.install_text_conditioning(payload)

    def test_install_rejects_a_payload_without_the_declared_schema(self, make_generator):
        """A future layout change is a named mismatch on the first request rather
        than a wrong image."""
        pipeline = make_generator()
        pipeline._bind_owned_tower()

        with pytest.raises(ValueError, match="carries no .* metadata"):
            pipeline.install_text_conditioning({KV_KEY: {"fp": _entry()}})

    def test_install_rejects_an_empty_payload(self, make_generator):
        pipeline = make_generator()
        pipeline._bind_owned_tower()

        with pytest.raises(ValueError, match="no .* branches"):
            pipeline.install_text_conditioning({KV_KEY: {}})

    def test_install_requires_the_replay_stub(self, make_generator):
        """Guards against installing K/V into a pipeline built by the wrong class."""
        pipeline = make_generator()  # _bind_owned_tower deliberately not called
        payload = _conditioning({fingerprint_text_ids(_ids(11, 12)): _entry()}).to_payload()

        with pytest.raises(RuntimeError, match="not running the replay UND stub"):
            pipeline.install_text_conditioning(payload)


# =============================================================================
# Reasoner postprocessor
# =============================================================================


class TestReasonerPostProcess:
    @pytest.fixture
    def post_process(self):
        return get_cosmos3_reasoner_post_process_func(od_config=None)

    def test_parks_the_kv_under_the_trajectory_payload_key(self, post_process):
        """``trajectory`` is the one payload key the output formatter copies through."""
        table = {"fp": [(torch.zeros(1), torch.zeros(1))]}
        meta = {"height": 1024, "payload_mib": 3.0, "schema": COSMOS3_UND_SCHEMA}

        result = post_process({KV_KEY: table, META_KEY: meta})

        assert result["payload"] == {"trajectory": {KV_KEY: table, META_KEY: meta}}
        assert result["metadata"] == {"cosmos3_und": meta}
        # A copy, so downstream metadata validation cannot mutate the payload.
        assert result["metadata"]["cosmos3_und"] is not meta

    def test_reserved_trajectory_subkeys_are_left_alone(self, post_process):
        result = post_process({KV_KEY: {"fp": []}, META_KEY: {}})

        assert set(result["payload"]["trajectory"]) == {KV_KEY, META_KEY}

    def test_tolerates_missing_meta(self, post_process):
        result = post_process({KV_KEY: {"fp": []}})

        assert result["metadata"] == {"cosmos3_und": {}}

    def test_latent_output_type_passes_through(self, post_process):
        sentinel = object()

        assert post_process(sentinel, output_type="latent") is sentinel

    @pytest.mark.parametrize("output", [None, "an image", {"images": []}])
    def test_rejects_anything_that_is_not_a_kv_payload(self, post_process, output):
        with pytest.raises(ValueError, match=f"dict payload containing {KV_KEY!r}"):
            post_process(output)


# =============================================================================
# Registry wiring
# =============================================================================


class TestDisaggRegistryWiring:
    @pytest.mark.parametrize("arch", ["Cosmos3ReasonerPipeline", "Cosmos3GeneratorPipeline"])
    def test_pipeline_class_is_resolvable(self, arch: str):
        from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, DiffusionModelRegistry

        assert _DIFFUSION_MODELS[arch] == ("cosmos3", "pipeline_cosmos3_disagg", arch)
        assert DiffusionModelRegistry._try_load_model_cls(arch).__name__ == arch

    @pytest.mark.parametrize("arch", ["Cosmos3ReasonerPipeline", "Cosmos3GeneratorPipeline"])
    def test_process_funcs_resolve_from_the_mapped_module(self, arch: str):
        """``_load_process_func`` looks them up in the module ``_DIFFUSION_MODELS``
        names, which is why the stock funcs are re-exported there."""
        from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3_disagg as module
        from vllm_omni.diffusion.registry import (
            _DIFFUSION_IR_OP_PRIORITY_FUNCS,
            _DIFFUSION_POST_PROCESS_FUNCS,
            _DIFFUSION_PRE_PROCESS_FUNCS,
        )

        for table in (
            _DIFFUSION_PRE_PROCESS_FUNCS,
            _DIFFUSION_POST_PROCESS_FUNCS,
            _DIFFUSION_IR_OP_PRIORITY_FUNCS,
        ):
            assert callable(getattr(module, table[arch]))

    def test_reasoner_has_its_own_postprocessor(self):
        from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS

        # The reasoner emits K/V, not pixels; the generator's output is an image.
        assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3ReasonerPipeline"] == "get_cosmos3_reasoner_post_process_func"
        assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3GeneratorPipeline"] == "get_cosmos3_post_process_func"

    @pytest.mark.parametrize("arch", ["Cosmos3ReasonerPipeline", "Cosmos3GeneratorPipeline"])
    def test_openai_extra_body_specs_are_inherited(self, arch: str):
        """The endpoint sees a per-tower class name but the same Cosmos3 params."""
        from vllm_omni.model_extras.registry import _EXTRA_SPECS

        assert _EXTRA_SPECS[arch] == _EXTRA_SPECS["Cosmos3OmniDiffusersPipeline"]

    def test_cache_dit_enabler_covers_the_denoising_stage_only(self):
        from vllm_omni.diffusion.cache.cachedit import CUSTOM_DIT_ENABLERS

        assert CUSTOM_DIT_ENABLERS["Cosmos3GeneratorPipeline"] is CUSTOM_DIT_ENABLERS["Cosmos3OmniDiffusersPipeline"]
        # The reasoner has no gen_layers and no denoising steps to cache.
        assert "Cosmos3ReasonerPipeline" not in CUSTOM_DIT_ENABLERS


# =============================================================================
# Shared text-conditioning resolvers
# =============================================================================


class TestSharedConditioningResolvers:
    """The two stages must resolve the tokenizer's inputs from one implementation.

    The generator finds its replayed K/V by fingerprinting the token ids it
    tokenizes itself, so anything feeding the tokenizer -- geometry,
    ``max_sequence_length``, ``use_system_prompt`` -- has to come out the same on
    both stages for the same sampling params. A divergence here does not produce a
    slightly different image; it produces a replay miss and a failed request.
    These tests pin the single implementation in place rather than re-testing its
    behaviour, which ``TestReasonerPipeline`` already covers end to end.
    """

    @pytest.mark.parametrize(
        "helper",
        ["_resolve_t2i_geometry", "_resolve_text_encode_params", "_resolve_guidance_scale"],
    )
    @pytest.mark.parametrize("stage", [Cosmos3ReasonerPipeline, Cosmos3GeneratorPipeline])
    def test_neither_tower_overrides_the_shared_resolvers(self, stage: type, helper: str):
        """An override on one tower is the failure mode this whole class exists for."""
        assert getattr(stage, helper) is getattr(Cosmos3OmniDiffusersPipeline, helper)

    @pytest.mark.parametrize("helper", ["_resolve_t2i_geometry", "_resolve_text_encode_params"])
    def test_the_colocated_t2i_path_still_routes_through_them(self, helper: str):
        """The resolvers are only shared while the stock ``forward`` uses them too.

        Inlining the defaults back into ``forward`` would leave the reasoner
        resolving them alone -- the co-located path would keep working, so nothing
        else here would fail. Exercising the real ``forward`` needs a loaded 31B
        checkpoint, so this reads the source instead.
        """
        import inspect

        source = inspect.getsource(Cosmos3OmniDiffusersPipeline.forward)

        assert f"self.{helper}(" in source

    def test_both_stages_resolve_identical_tokenizer_inputs(self, make_reasoner, make_generator):
        """Same sampling params in, same values out, across a spread of inputs.

        Bound to instances of both stage classes, because the geometry default
        depends on ``self.is_edge_model``.
        """
        reasoner = make_reasoner()
        generator = make_generator()
        generator.is_edge_model = False

        for sp in (
            _sampling_params(),
            _sampling_params(height=512, width=768),
            _sampling_params(max_sequence_length=256),
            _sampling_params(extra_args={"use_system_prompt": True}),
            _sampling_params(extra_args={"max_sequence_length": 128}),
        ):
            assert reasoner._resolve_t2i_geometry(sp) == generator._resolve_t2i_geometry(sp)
            assert reasoner._resolve_text_encode_params(
                sp, default_use_system_prompt=False
            ) == generator._resolve_text_encode_params(sp, default_use_system_prompt=False)

    def test_the_reasoner_encodes_at_the_resolved_values(self, make_reasoner):
        """Closes the loop: the values the resolver returns are the ones that reach
        the tokenizer, so the parity above is parity of what actually gets hashed."""
        pipeline = make_reasoner()
        sp = _sampling_params(height=512, width=768, extra_args={"max_sequence_length": 128})

        conditioning = pipeline.encode_text_conditioning("a red car", "", sp)

        height, width = pipeline._resolve_t2i_geometry(sp)
        max_sequence_length, use_system_prompt, _ = pipeline._resolve_text_encode_params(
            sp, default_use_system_prompt=False
        )
        call = pipeline.tokenize_calls[0]
        assert (call["height"], call["width"]) == (height, width)
        assert call["max_sequence_length"] == max_sequence_length
        assert call["use_system_prompt"] is use_system_prompt
        # And the contract declares the same resolution, so a mismatch reported by
        # the generator names the values the reasoner really used.
        assert conditioning.height == height
        assert conditioning.max_sequence_length == max_sequence_length

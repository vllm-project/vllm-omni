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
    _ReplayLanguageModel,
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

    @pytest.mark.parametrize("other", [(1, 2, 4), (1, 2), (2, 1, 3), (1, 2, 3, 0)])
    def test_sensitive_to_content(self, other: tuple[int, ...]):
        """CFG branches differ only in their token stream; padding counts too."""
        assert fingerprint_text_ids(_ids(*other)) != fingerprint_text_ids(_ids(1, 2, 3))


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


# =============================================================================
# VAE ownership
# =============================================================================


@pytest.fixture
def stub_pipeline_loaders(monkeypatch: pytest.MonkeyPatch):
    """Stub every loader the real ``Cosmos3OmniDiffusersPipeline.__init__`` calls,
    recording whether the VAE was loaded with weights or only had its config read."""
    from vllm_omni.diffusion.models.cosmos3 import pipeline_cosmos3

    vae_calls: list[str] = []

    class _StubVAE:
        config = SimpleNamespace(scale_factor_temporal=4, scale_factor_spatial=8)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            vae_calls.append("from_pretrained")
            return cls()

        @classmethod
        def load_config(cls, *args, **kwargs):
            vae_calls.append("load_config")
            return {"scale_factor_temporal": 4, "scale_factor_spatial": 8}

        def to(self, *args, **kwargs):
            return self

    class _StubScheduler:
        @classmethod
        def load_config(cls, *args, **kwargs):
            return {"_class_name": "UniPCMultistepScheduler"}

        @classmethod
        def from_config(cls, *args, **kwargs):
            return SimpleNamespace(config=SimpleNamespace())

    monkeypatch.setattr(pipeline_cosmos3, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr(pipeline_cosmos3, "DistributedAutoencoderKLWan", _StubVAE)
    monkeypatch.setattr(pipeline_cosmos3, "FlowUniPCMultistepScheduler", _StubScheduler)
    monkeypatch.setattr(pipeline_cosmos3, "VideoProcessor", lambda *a, **k: None)
    monkeypatch.setattr(pipeline_cosmos3, "get_local_device", lambda: torch.device("cpu"))
    return vae_calls


def _tiny_od_config() -> SimpleNamespace:
    return SimpleNamespace(
        enable_cpu_offload=False,
        enable_diffusion_pipeline_profiler=False,
        enable_session_state_manager=False,
        model="/nonexistent/model/path",
        dtype=torch.float32,
        flow_shift=None,
        quantization_config=None,
        custom_pipeline_args={},
        model_config={},
        tf_model_config={
            "hidden_size": 8,
            "num_hidden_layers": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 32,
            "latent_patch_size": 1,
            "latent_channel": 2,
            "rope_scaling": {"mrope_section": [1, 1, 0]},
        },
        parallel_config=SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1),
    )


class TestVaeOwnership:
    """Only the generator decodes, so only the generator loads VAE weights."""

    def test_reasoner_reads_only_the_vae_config(self, stub_pipeline_loaders):
        pipeline = Cosmos3ReasonerPipeline(od_config=_tiny_od_config())

        assert stub_pipeline_loaders == ["load_config"]
        assert pipeline.vae is None
        # The scale factors still come from the checkpoint's VAE config.
        assert pipeline.vae_scale_factor_temporal == 4
        assert pipeline.vae_scale_factor_spatial == 8


# =============================================================================
# Cosmos3TextConditioning -- the typed contract on the edge
# =============================================================================


class TestTextConditioningContract:
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

    def test_rejects_a_payload_declaring_another_schema(self):
        payload = _conditioning({"fp": _entry()}).to_payload()
        payload[META_KEY]["schema"] = "cosmos3.text_conditioning/v2"

        with pytest.raises(ValueError, match="declares schema 'cosmos3.text_conditioning/v2'"):
            Cosmos3TextConditioning.from_payload(payload)

    def test_rejects_a_declared_head_count_the_tensors_do_not_carry(self):
        with pytest.raises(ValueError, match="contract declares"):
            _conditioning({"fp": _entry(num_kv_heads=KV_HEADS * 2)})


# =============================================================================
# TP sharding on the edge
# =============================================================================


class TestKvHeadGather:
    """UND K/V is born TP-sharded and only rank 0's stage output escapes, so the
    reasoner must gather the head dimension before the payload leaves the tower."""

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

    def test_install_rejects_a_head_count_this_stage_does_not_consume(self):
        """A cross-stage checkpoint mismatch; the gathered payload makes TP irrelevant."""
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

    def test_reshards_even_when_the_tower_raises(self, make_reasoner):
        pipeline = make_reasoner(fsdp=True)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("tower exploded")

        pipeline.transformer.language_model.forward = _boom

        with pytest.raises(RuntimeError, match="tower exploded"):
            pipeline.encode_text_conditioning("a red car", "", _sampling_params())

        assert pipeline.transformer.shard_events == ["unshard", "reshard"]

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

    def test_forward_rejects_a_request_without_kv(self, make_generator):
        pipeline = make_generator()
        pipeline._bind_owned_tower()
        req = SimpleNamespace(prompts=[{"prompt": "x"}], sampling_params=SimpleNamespace(extra_args={}))

        with pytest.raises(ValueError, match="without reasoner K/V"):
            pipeline.forward(req)


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


# =============================================================================
# Registry wiring
# =============================================================================


class TestDisaggRegistryWiring:
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

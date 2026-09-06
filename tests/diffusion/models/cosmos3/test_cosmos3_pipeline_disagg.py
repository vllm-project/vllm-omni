# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the tower-split Cosmos3 pipelines (reasoner / generator).

Both towers are stubbed: what is under test is the split itself -- which tower
gets dropped, what the reasoner hands off, how the generator replays it, and the
fingerprint keying that keeps the two CFG branches apart.
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
    _drop_blocks,
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

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

NUM_LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4


# =============================================================================
# Stubs
# =============================================================================


class StubUndTower(nn.Module):
    """Stands in for ``Cosmos3LanguageModel``: returns marked per-layer K/V."""

    def __init__(self, num_layers: int = NUM_LAYERS) -> None:
        super().__init__()
        self.rotary_emb = nn.Identity()
        self.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(num_layers))
        self.calls: list[tuple[torch.Tensor, Any]] = []

    def forward(self, text_ids: torch.Tensor, freqs: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.calls.append((text_ids.clone(), freqs))
        batch, seq = text_ids.shape
        # [B, S, H, D] -- the trim in the reasoner slices dim 1 (sequence).
        return [
            (
                torch.full((batch, seq, KV_HEADS, HEAD_DIM), float(i)),
                torch.full((batch, seq, KV_HEADS, HEAD_DIM), float(i) + 100),
            )
            for i in range(len(self.layers))
        ]


class StubCrossAttention(nn.Module):
    """Stands in for ``Cosmos3CrossAttention``: the consumer of the replayed K/V.

    The real one resolves ``num_kv_heads // tp_size`` at construction time and the
    generator reads the expected replay shape straight off it, so the stub has to
    carry the same two attributes.
    """

    def __init__(self, num_kv_heads_local: int = KV_HEADS, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim


class StubGenBlock(nn.Module):
    """Stands in for ``Cosmos3GenDecoderLayer``."""

    def __init__(self, num_kv_heads_local: int = KV_HEADS, head_dim: int = HEAD_DIM) -> None:
        super().__init__()
        self.cross_attention = StubCrossAttention(num_kv_heads_local, head_dim)
        self.mlp = nn.Linear(1, 1)


class StubTowerTransformer(nn.Module):
    """The parts of ``Cosmos3VFMTransformer`` the two tower pipelines touch."""

    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        *,
        fsdp: bool = False,
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ) -> None:
        super().__init__()
        self.language_model = StubUndTower(num_layers)
        self.gen_layers = nn.ModuleList(StubGenBlock(num_kv_heads_local, head_dim) for _ in range(num_layers))
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
    def _make(*, num_layers: int = NUM_LAYERS, fsdp: bool = False, real_len: int = 2, total_len: int = 4):
        pipeline = object.__new__(Cosmos3ReasonerPipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = StubTowerTransformer(num_layers, fsdp=fsdp)
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
        num_kv_heads_local: int = KV_HEADS,
        head_dim: int = HEAD_DIM,
    ):
        pipeline = object.__new__(Cosmos3GeneratorPipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = StubTowerTransformer(
            num_layers,
            num_kv_heads_local=num_kv_heads_local,
            head_dim=head_dim,
        )
        pipeline.device = torch.device("cpu")
        return pipeline

    return _make


def _entry(num_layers: int = NUM_LAYERS, seq: int = 2, fill: float = 0.0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One branch of a reasoner payload, shaped [B, S_und, H_kv_local, D] per layer."""
    return [
        (
            torch.full((1, seq, KV_HEADS, HEAD_DIM), fill),
            torch.full((1, seq, KV_HEADS, HEAD_DIM), fill),
        )
        for _ in range(num_layers)
    ]


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
# _drop_blocks
# =============================================================================


class TestDropBlocks:
    def test_empties_the_container_in_place(self):
        """The container object is kept: offload rings and cache-dit introspect it."""
        blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(4))

        _drop_blocks(blocks, "test")

        assert len(blocks) == 0
        assert isinstance(blocks, nn.ModuleList)
        # Emptied, not detached: the parameters are gone from the module tree.
        assert list(blocks.parameters()) == []

    def test_empty_container_is_a_no_op(self):
        blocks = nn.ModuleList()

        _drop_blocks(blocks, "test")

        assert len(blocks) == 0


# =============================================================================
# _ReplayLanguageModel
# =============================================================================


class TestReplayLanguageModel:
    def _stub(self, num_layers: int = NUM_LAYERS) -> _ReplayLanguageModel:
        return _ReplayLanguageModel(
            num_layers,
            nn.Identity(),
            num_kv_heads_local=KV_HEADS,
            head_dim=HEAD_DIM,
        )

    def test_attribute_surface_the_transformer_relies_on(self):
        """``_compute_rope_freqs`` needs rotary_emb; offload needs ``layers``."""
        rotary = nn.Identity()
        stub = _ReplayLanguageModel(NUM_LAYERS, rotary, num_kv_heads_local=KV_HEADS, head_dim=HEAD_DIM)

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
        entry = _entry()
        stub = self._stub()
        stub.install({fingerprint_text_ids(text_ids): entry}, dtype=torch.bfloat16)

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
        stub.install({fingerprint_text_ids(text_ids): entry})

        assert len(stub(text_ids, freqs=None)) == NUM_LAYERS

    def test_keeps_cfg_branches_apart(self):
        cond, uncond = _ids(11, 12), _ids(21, 22)
        table = {
            fingerprint_text_ids(cond): _entry(fill=0.0),
            fingerprint_text_ids(uncond): _entry(fill=1.0),
        }
        stub = self._stub()
        stub.install(table)

        assert stub(cond, freqs=None)[0][0].sum() == 0
        assert stub(uncond, freqs=None)[0][0].sum() == 2 * KV_HEADS * HEAD_DIM

    def test_fingerprint_miss_raises_loudly(self):
        """A tokenization divergence must not silently produce a wrong image."""
        stub = self._stub()
        stub.install({fingerprint_text_ids(_ids(11, 12)): _entry()})

        with pytest.raises(RuntimeError, match="no reasoner K/V for this prompt"):
            stub(_ids(99, 98), freqs=None)

    def test_empty_table_raises(self):
        with pytest.raises(RuntimeError, match="no reasoner K/V"):
            self._stub()(_ids(11, 12), freqs=None)

    def test_install_rejects_a_layer_count_mismatch(self):
        """Checked at install time, once per request, rather than once per step."""
        stub = self._stub()

        with pytest.raises(RuntimeError, match="has 1 layer.*generator expects 3"):
            stub.install({fingerprint_text_ids(_ids(11, 12)): _entry(num_layers=1)})

    def test_install_rejects_a_kv_head_count_from_a_different_tp_size(self):
        """A payload sharded at another TP size is the failure mode the shape check
        exists for: unchecked it is either a shape error deep inside attention or,
        worse, silently wrong conditioning."""
        stub = self._stub()
        entry = [(torch.zeros(1, 2, KV_HEADS * 2, HEAD_DIM), torch.zeros(1, 2, KV_HEADS * 2, HEAD_DIM))] * NUM_LAYERS

        with pytest.raises(RuntimeError, match="TP-sharded"):
            stub.install({fingerprint_text_ids(_ids(11, 12)): entry})

    def test_install_rejects_a_wrong_head_dim(self):
        stub = self._stub()
        entry = [(torch.zeros(1, 2, KV_HEADS, HEAD_DIM + 1), torch.zeros(1, 2, KV_HEADS, HEAD_DIM + 1))] * NUM_LAYERS

        with pytest.raises(RuntimeError, match="this generator stage consumes"):
            stub.install({fingerprint_text_ids(_ids(11, 12)): entry})

    def test_install_rejects_a_non_4d_tensor(self):
        stub = self._stub()
        entry = [(torch.zeros(1, 2), torch.zeros(1, 2))] * NUM_LAYERS

        with pytest.raises(RuntimeError, match="expected 4"):
            stub.install({fingerprint_text_ids(_ids(11, 12)): entry})

    def test_install_rejects_k_and_v_of_different_shapes(self):
        stub = self._stub()
        entry = [(torch.zeros(1, 2, KV_HEADS, HEAD_DIM), torch.zeros(1, 3, KV_HEADS, HEAD_DIM))] * NUM_LAYERS

        with pytest.raises(RuntimeError, match="disagree on shape"):
            stub.install({fingerprint_text_ids(_ids(11, 12)): entry})

    def test_clear_drops_the_payload(self):
        """A payload belongs to one request; the stub outlives it."""
        text_ids = _ids(11, 12)
        stub = self._stub()
        stub.install({fingerprint_text_ids(text_ids): _entry()}, dtype=torch.bfloat16)

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

    def test_drops_only_the_gen_tower(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline._drop_unused_tower()

        assert len(pipeline.transformer.gen_layers) == 0
        assert len(pipeline.transformer.language_model.layers) == NUM_LAYERS

    def test_payload_is_keyed_by_fingerprint_and_trimmed(self, make_reasoner):
        pipeline = make_reasoner(real_len=2, total_len=4)

        # Guidance off, so the table holds the conditional branch alone.
        payload = pipeline.encode_prompt_to_kv(
            "a red car",
            "",
            _sampling_params(guidance_scale=1.0, guidance_scale_provided=True),
        )

        table = payload[KV_KEY]
        assert set(table) == {fingerprint_text_ids(_ids(11, 12, 0, 0))}
        entry = table[fingerprint_text_ids(_ids(11, 12, 0, 0))]
        assert len(entry) == NUM_LAYERS
        for k, v in entry:
            # Trimmed to the real text length, not the padded one.
            assert k.shape == (1, 2, KV_HEADS, HEAD_DIM)
            assert v.shape == (1, 2, KV_HEADS, HEAD_DIM)
            assert k.device.type == "cpu"
            assert k.is_contiguous()

    def test_meta_reports_what_the_generator_must_reproduce(self, make_reasoner):
        pipeline = make_reasoner()

        payload = pipeline.encode_prompt_to_kv(
            "a red car",
            "",
            _sampling_params(height=512, width=768, max_sequence_length=256),
        )

        meta = payload[META_KEY]
        assert meta["height"] == 512
        assert meta["width"] == 768
        assert meta["max_sequence_length"] == 256
        assert meta["use_system_prompt"] is False
        # The T2I default guidance scale is > 1, so both CFG branches are encoded.
        assert meta["num_branches"] == 2
        # Reported for logging only, rounded to 0.1 MiB -- a stub payload floors to 0.0.
        assert isinstance(meta["payload_mib"], float)

    def test_meta_reports_the_kv_layout_read_off_the_tensors(self, make_reasoner):
        """The generator compares these against its own cross-attention to name a
        stage-configuration mismatch. Read from the emitted tensors rather than the
        config, so the metadata cannot describe a payload this stage did not send."""
        pipeline = make_reasoner()

        payload = pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        meta = payload[META_KEY]
        assert meta["num_layers"] == NUM_LAYERS
        assert meta["num_kv_heads_local"] == KV_HEADS
        assert meta["head_dim"] == HEAD_DIM
        # No TP group in a single-process test, which is TP 1 by definition.
        assert meta["tp_size"] == 1

    def test_oversized_payload_warns_but_still_ships(self, make_reasoner, monkeypatch, caplog):
        """An oversized payload is correct, just expensive -- so warn, do not raise."""
        monkeypatch.setattr(disagg, "COSMOS3_UND_PAYLOAD_WARN_MIB", 0.0)
        pipeline = make_reasoner()

        with caplog.at_level("WARNING"):
            payload = pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        assert payload[KV_KEY]
        assert "max_sequence_length" in caplog.text

    def test_normal_payload_does_not_warn(self, make_reasoner, caplog):
        pipeline = make_reasoner()

        with caplog.at_level("WARNING"):
            pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        assert "K/V payload is" not in caplog.text

    def test_geometry_and_tokenizer_settings_reach_the_tokenizer(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_prompt_to_kv("a red car", "blurry", _sampling_params(height=512, width=768))

        call = pipeline.tokenize_calls[0]
        assert call["prompt"] == "a red car"
        assert call["negative_prompt"] == "blurry"
        assert call["is_t2i"] is True
        assert call["num_frames"] == 1
        assert (call["height"], call["width"]) == (512, 768)

    def test_gen_latent_geometry_is_passed_to_rope(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_prompt_to_kv("a red car", "", _sampling_params(height=512, width=768))

        rope = pipeline.transformer.rope_calls[0]
        assert rope["t"] == 1
        assert (rope["hp"], rope["wp"]) == (512 // 8, 768 // 8)

    def test_encodes_both_branches_when_guidance_is_active(self, make_reasoner):
        pipeline = make_reasoner()

        payload = pipeline.encode_prompt_to_kv(
            "a red car",
            "blurry",
            _sampling_params(guidance_scale=7.0, guidance_scale_provided=True),
        )

        assert payload[META_KEY]["num_branches"] == 2
        assert set(payload[KV_KEY]) == {
            fingerprint_text_ids(_ids(11, 12, 0, 0)),
            fingerprint_text_ids(_ids(21, 22, 0, 0)),
        }

    def test_skips_the_unconditional_branch_without_guidance(self, make_reasoner):
        """Saves a full UND forward and halves the payload."""
        pipeline = make_reasoner()

        payload = pipeline.encode_prompt_to_kv(
            "a red car",
            "blurry",
            _sampling_params(guidance_scale=1.0, guidance_scale_provided=True),
        )

        assert payload[META_KEY]["num_branches"] == 1
        assert len(pipeline.transformer.language_model.calls) == 1

    def test_runs_the_und_tower_inside_its_offload_context(self, make_reasoner):
        pipeline = make_reasoner()

        pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        # One entry per CFG branch: every tower call is wrapped.
        assert pipeline.transformer.offload_contexts == ["reasoner", "reasoner"]

    def test_unshards_around_the_direct_tower_call(self, make_reasoner):
        """Calling language_model directly bypasses FSDP2's root pre-forward hook."""
        pipeline = make_reasoner(fsdp=True)

        pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        assert pipeline.transformer.shard_events == ["unshard", "reshard"]

    def test_reshards_even_when_the_tower_raises(self, make_reasoner):
        pipeline = make_reasoner(fsdp=True)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("tower exploded")

        pipeline.transformer.language_model.forward = _boom

        with pytest.raises(RuntimeError, match="tower exploded"):
            pipeline.encode_prompt_to_kv("a red car", "", _sampling_params())

        assert pipeline.transformer.shard_events == ["unshard", "reshard"]

    def test_forward_emits_the_handoff_payload(self, make_reasoner):
        pipeline = make_reasoner()
        req = SimpleNamespace(
            prompts=[{"prompt": "a red car", "modalities": ["image"], "negative_prompt": "blurry"}],
            sampling_params=_sampling_params(),
        )

        output = pipeline.forward(req)

        assert set(output.output) == {KV_KEY, META_KEY}
        assert pipeline.tokenize_calls[0]["negative_prompt"] == "blurry"

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

    def test_swaps_the_und_tower_for_the_replay_stub(self, make_generator):
        pipeline = make_generator()
        original = pipeline.transformer.language_model
        rotary = original.rotary_emb

        pipeline._drop_unused_tower()

        stub = pipeline.transformer.language_model
        assert isinstance(stub, _ReplayLanguageModel)
        assert stub.num_hidden_layers == NUM_LAYERS
        # The real GEN mRoPE frequencies are built from this, every step.
        assert stub.rotary_emb is rotary
        assert len(original.layers) == 0
        # The GEN tower is the one this stage actually runs.
        assert len(pipeline.transformer.gen_layers) == NUM_LAYERS

    def test_stub_takes_its_expected_layout_from_the_consuming_cross_attention(self, make_generator):
        """Read from the module that will receive the tensors, not recomputed from
        the config and TP size, so it cannot disagree with the consumer."""
        pipeline = make_generator(num_kv_heads_local=4, head_dim=64)

        pipeline._drop_unused_tower()

        stub = pipeline.transformer.language_model
        assert (stub.num_kv_heads_local, stub.head_dim) == (4, 64)

    def test_dropping_the_wrong_tower_is_caught(self, make_generator):
        """Without GEN blocks there is no cross-attention to describe the layout and
        nothing to replay into -- a bug in this class, so say so."""
        pipeline = make_generator()
        del pipeline.transformer.gen_layers[:]

        with pytest.raises(RuntimeError, match="no GEN blocks"):
            pipeline._drop_unused_tower()

    def test_forward_clears_the_payload_when_the_denoise_loop_finishes(self, make_generator, monkeypatch):
        """The payload belongs to one request but the stub is long-lived pipeline
        state, so a table left installed could be replayed by a later request."""
        pipeline = make_generator()
        pipeline._drop_unused_tower()
        table = {fingerprint_text_ids(_ids(11, 12)): _entry()}
        req = SimpleNamespace(prompts=[{"prompt": "x", "extra": {KV_KEY: table}}], sampling_params=None)

        installed: list[dict[str, Any]] = []

        def _record_then_return_image(self, _req):
            """Snapshot the replay table from inside the denoise loop."""
            installed.append(dict(self.transformer.language_model._table))
            return "image"

        monkeypatch.setattr(Cosmos3OmniDiffusersPipeline, "forward", _record_then_return_image)

        assert pipeline.forward(req) == "image"
        # Installed for the duration of the denoise loop, gone afterwards.
        assert installed == [table]
        assert pipeline.transformer.language_model._table == {}

    def test_forward_clears_the_payload_even_when_denoising_raises(self, make_generator, monkeypatch):
        pipeline = make_generator()
        pipeline._drop_unused_tower()
        table = {fingerprint_text_ids(_ids(11, 12)): _entry()}
        req = SimpleNamespace(prompts=[{"prompt": "x", "extra": {KV_KEY: table}}], sampling_params=None)

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
        pipeline._drop_unused_tower()
        req = SimpleNamespace(prompts=[{"prompt": "x"}], sampling_params=SimpleNamespace(extra_args={}))

        with pytest.raises(ValueError, match="without reasoner K/V"):
            pipeline.forward(req)

    def test_install_und_kv_loads_the_table(self, make_generator):
        pipeline = make_generator()
        pipeline._drop_unused_tower()
        text_ids = _ids(11, 12)
        table = {fingerprint_text_ids(text_ids): _entry()}

        pipeline.install_und_kv({KV_KEY: table, META_KEY: {"payload_mib": 1.0}})

        stub = pipeline.transformer.language_model
        assert stub._table is table
        assert stub._dtype == pipeline.transformer.proj_in.weight.dtype
        assert len(stub(text_ids, freqs=None)) == NUM_LAYERS

    def test_install_und_kv_rejects_a_layout_the_reasoner_declares_differently(self, make_generator):
        """The shape check in ``install`` already makes replay safe; this check is
        what makes a TP-size mistake diagnosable, since no shape reveals TP size."""
        pipeline = make_generator()
        pipeline._drop_unused_tower()
        table = {fingerprint_text_ids(_ids(11, 12)): _entry()}
        meta = {"num_kv_heads_local": KV_HEADS * 2, "tp_size": 2, "head_dim": HEAD_DIM}

        with pytest.raises(RuntimeError, match="disagree on the UND K/V layout") as excinfo:
            pipeline.install_und_kv({KV_KEY: table, META_KEY: meta})

        message = str(excinfo.value)
        assert f"num_kv_heads_local={KV_HEADS * 2} from reasoner, {KV_HEADS} here" in message
        assert "tensor_parallel_size=2" in message

    def test_install_und_kv_accepts_a_payload_that_declares_no_layout(self, make_generator):
        """An older reasoner, or a hand-built payload in a test: ``install`` still
        validates the tensors themselves."""
        pipeline = make_generator()
        pipeline._drop_unused_tower()
        table = {fingerprint_text_ids(_ids(11, 12)): _entry()}

        pipeline.install_und_kv({KV_KEY: table, META_KEY: {"payload_mib": 1.0}})

        assert pipeline.transformer.language_model._table is table

    def test_install_und_kv_rejects_an_empty_payload(self, make_generator):
        pipeline = make_generator()
        pipeline._drop_unused_tower()

        with pytest.raises(ValueError, match="empty reasoner K/V payload"):
            pipeline.install_und_kv({KV_KEY: {}})

    def test_install_und_kv_requires_the_replay_stub(self, make_generator):
        """Guards against installing K/V into a pipeline built by the wrong class."""
        pipeline = make_generator()  # _drop_unused_tower deliberately not called

        with pytest.raises(RuntimeError, match="not running the replay UND stub"):
            pipeline.install_und_kv({KV_KEY: {"fp": []}})


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
        meta = {"height": 1024, "payload_mib": 3.0}

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

        payload = pipeline.encode_prompt_to_kv("a red car", "", sp)

        height, width = pipeline._resolve_t2i_geometry(sp)
        max_sequence_length, use_system_prompt, _ = pipeline._resolve_text_encode_params(
            sp, default_use_system_prompt=False
        )
        call = pipeline.tokenize_calls[0]
        assert (call["height"], call["width"]) == (height, width)
        assert call["max_sequence_length"] == max_sequence_length
        assert call["use_system_prompt"] is use_system_prompt
        # And the metadata describes the same resolution, so a mismatch reported by
        # the generator names the values the reasoner really used.
        assert payload[META_KEY]["height"] == height
        assert payload[META_KEY]["max_sequence_length"] == max_sequence_length

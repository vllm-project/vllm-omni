# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for IndexTTS 2 & 2.5 MRv2 data plane, Stage-1 concurrency, and scoped TF32."""

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.indextts2_talker import (
    IndexTTS2TalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.indextts2.pipeline import (
    INDEXTTS2_PIPELINE,
    INDEXTTS25_PIPELINE,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.length_regulator import (
    InterpolateRegulator,
)
from vllm_omni.model_executor.models.indextts2.utils.maskgct.quantize import ResidualVQ
from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    _FULL_PAYLOAD_REPLACE_KEYS,
    _build_s2mel_additional_information,
    talker2s2mel_full_payload,
    talker2s2mel_token_only,
)
from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 16
GPT_DIM = 12
CODEBOOK = 32


class _FakeS2Mel:
    def __init__(self):
        self.gpt_layer = nn.Linear(GPT_DIM, HIDDEN)
        self.models = {
            "length_regulator": InterpolateRegulator(
                channels=24, sampling_ratios=(1, 1), in_channels=HIDDEN, groups=1
            ).eval()
        }

    def forward_gpt(self, latent):
        return self.gpt_layer(latent)


def _make_decoder() -> IndexTTS2S2MelDecoder:
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    nn.Module.__init__(decoder)
    decoder.use_gpt_latent = True
    decoder.semantic_codec_type = "repcodec"
    decoder.config = SimpleNamespace(gpt={"stop_mel_token": 8193}, s2mel_allow_tf32=True)
    decoder.s2mel = _FakeS2Mel()
    return decoder


def _make_codec() -> SimpleNamespace:
    quantizer = ResidualVQ(
        input_dim=HIDDEN,
        num_quantizers=1,
        codebook_size=CODEBOOK,
        codebook_dim=8,
        quantizer_dropout=0.0,
    ).eval()
    return SimpleNamespace(quantizer=quantizer)


def _stage1_payload(length: int, seed: int, ref_len: int = 5) -> dict:
    gen = torch.Generator().manual_seed(seed)
    mel = torch.randint(0, CODEBOOK, (length,), generator=gen)
    latent = torch.randn(length, GPT_DIM, generator=gen)
    meta = {
        "S_ref": torch.randn(1, ref_len, HIDDEN, generator=gen),
        "ref_mel": torch.randn(1, 80, ref_len, generator=gen),
        "style": torch.randn(1, 192, generator=gen),
    }
    return _build_s2mel_additional_information(mel, latent, meta, use_gpt_latent=True, context="test")


@torch.no_grad()
def test_four_concurrent_stage1_requests_build_semantic_embedding_and_regulate_lengths():
    """Verify B=4 concurrent Stage-1 requests do not hit semantic=(1, T, D) vs latent=(4, T, D)
    mismatch in ResidualVQ.vq2emb and regulate lengths per row without padding contamination.
    """
    torch.manual_seed(0)
    decoder = _make_decoder()
    codec = _make_codec()
    regulator = decoder.s2mel.models["length_regulator"]
    infos = [
        _stage1_payload(27, seed=1, ref_len=8),
        _stage1_payload(19, seed=2, ref_len=5),
        _stage1_payload(23, seed=3, ref_len=7),
        _stage1_payload(15, seed=4, ref_len=6),
    ]

    decoder._validate_payload_policy(infos)
    decoder._validate_payload_contract(infos)
    (
        mel_codes,
        latent,
        _,
        code_lens,
        s_ref,
        _,
        _,
        ref_lengths,
        s_ref_lengths,
    ) = decoder._build_batched_inputs(infos, device=torch.device("cpu"), model_dtype=torch.float32)

    assert mel_codes.shape == (4, 27)
    assert latent.shape == (4, 27, GPT_DIM)
    assert s_ref_lengths == [8, 5, 7, 6]

    batched = decoder._build_semantic_embedding(codec=codec, mel_codes=mel_codes, latent=latent, dtype=torch.float32)
    assert batched.shape == (4, 27, HIDDEN)

    targets = decoder._target_lengths(
        code_lens, semantic_time_scale=1, mel_code_to_frame_ratio=1.72, duration_factors=[1.0] * 4
    )
    cond = decoder._regulate_lengths(batched, code_lens, targets)
    prompt = decoder._regulate_lengths(s_ref, s_ref_lengths, ref_lengths)

    for row, (info, length, target) in enumerate(zip(infos, code_lens, targets)):
        single_sem = decoder._build_semantic_embedding(
            codec=codec, mel_codes=info["mel_codes"], latent=info["latent"], dtype=torch.float32
        )
        torch.testing.assert_close(batched[row, :length], single_sem[0])
        single_cond = regulator(single_sem, ylens=torch.tensor([target]), n_quantizers=3, f0=None)[0]
        torch.testing.assert_close(cond[row, :target], single_cond[0])
        assert not cond[row, target:].any()
        single_prompt = regulator(info["S_ref"], ylens=torch.tensor([ref_lengths[row]]), n_quantizers=3, f0=None)[0]
        torch.testing.assert_close(prompt[row, : ref_lengths[row]], single_prompt[0])


def test_mrv2_postprocess_before_make_omni_output_preserves_step_latent_alignment():
    """In MRv2, run_postprocess executes in execute_model BEFORE sample_tokens calls
    make_omni_output. Verify _preprocess_decode snapshots step_latent so MRv2 and V1
    emit identical (mel_code, latent) pairs on every decode step.
    """
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.use_gpt_latent = True
    talker.model_dim = HIDDEN
    talker.mel_embedding = nn.Embedding(64, HIDDEN)
    talker.mel_pos_embedding = SimpleNamespace(emb=nn.Embedding(64, HIDDEN))

    prefill_hidden = torch.randn(5, HIDDEN)
    decode_hiddens = [torch.randn(1, HIDDEN) for _ in range(3)]
    sampled_tokens = [torch.tensor([11]), torch.tensor([22]), torch.tensor([33])]

    def _run_sequence(mrv2_order: bool):
        buf: dict = {
            "meta": {
                "mel_start_offset": 4,
                "prefill_mel_index": 4,
                "S_ref": torch.ones(1, 4, HIDDEN),
                "ref_mel": torch.ones(1, 80, 4),
                "style": torch.ones(1, 192),
            }
        }
        # Step 0: prefill
        if mrv2_order:
            upd_post = talker.postprocess(prefill_hidden, **buf)
            buf["meta"].update(upd_post.get("meta", {}))
            _ = talker.make_omni_output(prefill_hidden, model_intermediate_buffer=[buf])
        else:
            _ = talker.make_omni_output(prefill_hidden, model_intermediate_buffer=[buf])
            upd_post = talker.postprocess(prefill_hidden, **buf)
            buf["meta"].update(upd_post.get("meta", {}))

        emitted_codes = []
        emitted_latents = []
        for tok, h_step in zip(sampled_tokens, decode_hiddens):
            _, _, upd_pre = talker.preprocess(tok, None, **buf)
            buf.setdefault("codes", {}).update(upd_pre.get("codes", {}))
            buf["meta"].update(upd_pre.get("meta", {}))
            if mrv2_order:
                # MRv2 runs postprocess before make_omni_output
                upd_post = talker.postprocess(h_step, **buf)
                buf["meta"].update(upd_post.get("meta", {}))
                out = talker.make_omni_output(h_step, model_intermediate_buffer=[buf])
            else:
                # V1 runs make_omni_output before postprocess
                out = talker.make_omni_output(h_step, model_intermediate_buffer=[buf])
                upd_post = talker.postprocess(h_step, **buf)
                buf["meta"].update(upd_post.get("meta", {}))
            emitted_codes.append(out.multimodal_outputs["codes"]["mel"][0].clone())
            emitted_latents.append(out.multimodal_outputs["hidden_states"]["latent"][0].clone())
        return torch.cat(emitted_codes, dim=0), torch.cat(emitted_latents, dim=0)

    v1_codes, v1_latents = _run_sequence(mrv2_order=False)
    mrv2_codes, mrv2_latents = _run_sequence(mrv2_order=True)
    torch.testing.assert_close(mrv2_codes, v1_codes)
    torch.testing.assert_close(mrv2_latents, v1_latents)
    # Specifically, decode step 0 must pair token 11 with prefill_hidden[4:5], not decode_hiddens[0]!
    torch.testing.assert_close(mrv2_latents[0:1], prefill_hidden[4:5])


def _make_mrv2_data_plane_with_indextts():
    sent_packets: list[tuple[str, dict]] = []
    cleaned_reqs: list[str] = []

    class _FakeConnector:
        config: dict[str, Any] = {"extra": {"delivery_timeout_s": 1.0}}

        def put(self, from_stage, to_stage, put_key, data):
            sent_packets.append((put_key, data))
            return True, 1, None

    model_config = SimpleNamespace(
        stage_id=0,
        model_stage="talker",
        async_chunk=False,
        worker_type="ar",
        custom_process_next_stage_input_func=(
            "vllm_omni.model_executor.stage_input_processors.indextts2.talker2s2mel_full_payload"
        ),
        stage_connector_config=None,
    )
    vllm_config = SimpleNamespace(model_config=model_config, speculative_config=None)

    def _init_output_worker_no_thread(self, *, max_pending_batches: int) -> None:
        self._native_output_error_lock = threading.Lock()
        self._native_output_error = None
        self._connector_send_error_sink = self._record_output_error
        self._native_output_closed = False
        self._native_output_worker = None

    with (
        patch.object(OmniRunnerDataPlane, "_create_connector", lambda self, _mc: None),
        patch.object(OmniRunnerDataPlane, "_start_output_worker", _init_output_worker_no_thread),
    ):
        plane = OmniRunnerDataPlane(vllm_config, model_config)

    plane._omni_connector = _FakeConnector()
    plane._next_stage_id = "1"
    plane.is_data_transfer_rank = lambda: True
    plane._return_stage_payload = lambda *_a, **_kw: False
    plane._enqueue_chunk_payload = lambda req, payload, **_kw: (
        sent_packets.append((req.external_req_id, payload)) is None,
        None,
    )
    plane.cleanup_finished_request = cleaned_reqs.append
    plane.get_omni_connector_output = lambda: None
    return plane, sent_packets, cleaned_reqs


def test_mrv2_abort_and_publish_output_with_indextts_full_payload():
    """Verify OmniRunnerDataPlane + talker2s2mel_full_payload:
    1. publish_output clears output.multimodal_outputs on full-payload stages so 0-d meta.use_gpt_latent
       scalars are never sent to the API server;
    2. aborting before first output and after partial output raises no exception, sends no partial
       synthesis payload, emits one terminal marker, cleans up request state, and ignores late output;
    3. Both default indextts2.yaml and indextts2_5.yaml deploy configs declare MRv2 on Stage 0 and
       Stage 1 with supports_native_mrv2_data_plane=True, unchunked prefill on Stage 0, and
       opt-in s2mel_allow_tf32=False by default.
    """
    plane, sent_packets, cleaned_reqs = _make_mrv2_data_plane_with_indextts()

    # 1. publish_output clears multimodal_outputs for non-async_chunk full-payload producer
    plane.enqueue_outputs = lambda **_kw: None
    shared_payload = [{"meta.use_gpt_latent": True}]
    runner_out = OmniModelRunnerOutput(
        req_ids=["r0"],
        sampled_token_ids=[[10]],
        pooler_output=shared_payload,
        inter_stage_outputs=shared_payload,
        multimodal_outputs=[{"meta.use_gpt_latent": torch.tensor(True)}],
    )
    plane.publish_output(runner_out)
    assert runner_out.pooler_output is None
    assert runner_out.inter_stage_outputs is None
    assert runner_out.multimodal_outputs is None

    # Non-IndexTTS full-payload processors without _suppress_client_multimodal_outputs
    # preserve multimodal_outputs after publish_output.
    plane._custom_process_func = lambda *_a, **_kw: {}
    preserved_mm = [{"audio": torch.zeros(4)}]
    runner_out_other = OmniModelRunnerOutput(
        req_ids=["r0"],
        sampled_token_ids=[[10]],
        pooler_output=shared_payload,
        inter_stage_outputs=shared_payload,
        multimodal_outputs=preserved_mm,
    )
    plane.publish_output(runner_out_other)
    assert runner_out_other.multimodal_outputs is preserved_mm
    plane._custom_process_func = talker2s2mel_full_payload

    # 2a. Abort BEFORE first output
    req_before = SimpleNamespace(
        req_id="req-before",
        external_req_id="ext-before",
        prompt_token_ids=[1, 2, 3],
        num_computed_tokens=3,
        resumable=False,
        additional_information=None,
        sampling_params=SimpleNamespace(stop_token_ids=[8193]),
    )
    plane.register_request(req_before)
    assert plane.abort_requests({"req-before"}) == 1
    assert len(sent_packets) == 1
    ext_id, terminal_payload = sent_packets[0]
    assert ext_id == "ext-before"
    assert set(terminal_payload.keys()) == {"meta"}
    assert bool(terminal_payload["meta"]["finished"].item()) is True
    assert "mel_codes" not in terminal_payload
    assert "req-before" in cleaned_reqs
    assert "req-before" not in plane._native_requests
    assert "req-before" not in plane._pending_full_payload_send

    # Late output arriving after abort is ignored
    assert (
        plane.complete_outputs(
            req_ids=["req-before"],
            inter_stage_outputs=[{"codes.mel": torch.tensor([[42]]), "meta.use_gpt_latent": True}],
            sampled_token_ids=[[42]],
        )
        == 0
    )
    assert len(sent_packets) == 1

    # 2b. Abort AFTER partial output
    req_partial = SimpleNamespace(
        req_id="req-partial",
        external_req_id="ext-partial",
        prompt_token_ids=[1, 2, 3],
        num_computed_tokens=3,
        resumable=False,
        additional_information=None,
        sampling_params=SimpleNamespace(stop_token_ids=[8193]),
    )
    plane.register_request(req_partial)
    plane.reserve_outputs(["req-partial"])
    partial_step = {
        "codes.mel": torch.tensor([[15]]),
        "hidden_states.latent": torch.randn(1, GPT_DIM),
        "meta.use_gpt_latent": True,
        "meta.S_ref": torch.randn(1, 5, HIDDEN),
        "meta.ref_mel": torch.randn(1, 80, 5),
        "meta.style": torch.randn(1, 192),
    }
    assert (
        plane.complete_outputs(
            req_ids=["req-partial"],
            inter_stage_outputs=[partial_step],
            sampled_token_ids=[[15]],
        )
        == 0
    )
    assert "req-partial" in plane._pending_full_payload_send
    assert len(sent_packets) == 1

    plane.reserve_outputs(["req-partial"])
    assert plane.abort_requests({"req-partial"}) == 1
    assert len(sent_packets) == 2
    ext_id_2, terminal_payload_2 = sent_packets[1]
    assert ext_id_2 == "ext-partial"
    assert set(terminal_payload_2.keys()) == {"meta"}
    assert bool(terminal_payload_2["meta"]["finished"].item()) is True
    assert "mel_codes" not in terminal_payload_2
    assert "req-partial" not in plane._pending_full_payload_send
    assert "req-partial" not in plane._native_requests

    # Late in-flight output after abort is ignored
    assert (
        plane.complete_outputs(
            req_ids=["req-partial"],
            inter_stage_outputs=[partial_step],
            sampled_token_ids=[[8193]],
        )
        == 0
    )
    assert len(sent_packets) == 2
    assert talker2s2mel_token_only([SimpleNamespace(finished=True, aborted=True)]) == []

    # Non-aborted malformed payload still raises ValueError
    with pytest.raises(ValueError, match="IndexTTS Stage 0 payload must be a dict"):
        talker2s2mel_full_payload(None, None, SimpleNamespace(external_req_id="ext-live", aborted=False))

    # 3. Verify both IndexTTS pipelines and default MRv2 deploy configs
    for pipeline in (INDEXTTS2_PIPELINE, INDEXTTS25_PIPELINE):
        assert pipeline.stages[0].supports_native_mrv2_data_plane is True
        assert pipeline.stages[0].custom_process_next_stage_input_func.endswith("talker2s2mel_full_payload")
        assert pipeline.stages[1].supports_native_mrv2_data_plane is True
        assert pipeline.stages[1].sync_process_input_func.endswith("talker2s2mel_token_only")

    deploy_dir = Path(__file__).parents[4] / "vllm_omni" / "deploy"
    for deploy_name in ("indextts2.yaml", "indextts2_5.yaml"):
        cfg = yaml.safe_load((deploy_dir / deploy_name).read_text(encoding="utf-8"))
        stage0 = next(s for s in cfg["stages"] if s["stage_id"] == 0)
        stage1 = next(s for s in cfg["stages"] if s["stage_id"] == 1)
        assert cfg["model_runner"] == "v2"
        assert stage0.get("model_runner", cfg["model_runner"]) == "v2"
        assert stage1.get("model_runner", cfg["model_runner"]) == "v2"
        assert stage0["enable_prefix_caching"] is False
        assert stage0["enable_chunked_prefill"] is False
        assert stage0["max_num_batched_tokens"] >= stage0["max_model_len"]
        assert "tokenizer" not in stage0
        assert "tokenizer" not in stage1
        assert stage1["hf_overrides"]["s2mel_allow_tf32"] is False

    v2_cfg = yaml.safe_load((deploy_dir / "indextts2.yaml").read_text(encoding="utf-8"))
    v2_s0 = next(s for s in v2_cfg["stages"] if s["stage_id"] == 0)
    v2_s1 = next(s for s in v2_cfg["stages"] if s["stage_id"] == 1)
    assert v2_s0["max_num_seqs"] == 16
    assert v2_s1["max_num_seqs"] == 16
    assert v2_s1["hf_overrides"]["s2mel_cfm_batch_size"] == 16


def test_s2mel_tf32_context_restores_backend_flags():
    decoder = _make_decoder()
    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with decoder._maybe_s2mel_tf32_context(torch.device("cuda")):
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn


def test_d2h_eliminations_in_pooler_snapshot_and_emotion_lookup():
    from vllm_omni.model_executor.stage_input_processors.indextts2 import _strip_stop_token
    from vllm_omni.utils.mm_outputs import snapshot_mm_payload

    # 1. snapshot_mm_payload packs per-request lists into shared storage views
    raw_mm = {
        "codes": {"mel": [torch.tensor([[11]]), torch.tensor([[22]])]},
        "hidden_states": {"latent": [torch.ones(1, 8), torch.full((1, 8), 2.0)]},
    }
    snap = snapshot_mm_payload(raw_mm)
    assert snap["codes"]["mel"][0].tolist() == [[11]]
    assert snap["codes"]["mel"][1].tolist() == [[22]]
    assert snap["hidden_states"]["latent"][0].untyped_storage().data_ptr() == (
        snap["hidden_states"]["latent"][1].untyped_storage().data_ptr()
    )

    # 2. _strip_stop_token single-request fast path trims before host transfer
    codes = torch.tensor([[10, 20, 8193, 99, 100]], dtype=torch.long)
    latent = torch.randn(1, 5, 8)
    clean_codes, clean_latent, code_lens = _strip_stop_token(codes, latent, stop_mel_token=8193)
    assert clean_codes.shape == (1, 2)
    assert clean_latent is not None and clean_latent.shape == (1, 2, 8)
    assert code_lens.tolist() == [2]

    # 3. _compute_emo_from_distribution selects speaker/emotion rows without .item() syncs
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker._ensure_matrix_loaded = lambda *_a, **_kw: None
    talker._spk_matrix = [torch.randn(4, 16) for _ in range(8)]
    talker._emo_matrix = [torch.randn(4, 8) for _ in range(8)]
    emo_dist = torch.full((1, 8), 0.125)
    style = torch.randn(1, 16)
    out_det = talker._compute_emo_from_distribution(emo_dist, style, torch.device("cpu"), use_random=False)
    out_rnd = talker._compute_emo_from_distribution(emo_dist, style, torch.device("cpu"), use_random=True)
    assert out_det.shape == (1, 8)
    assert out_rnd.shape == (1, 8)

    # 4. EnhancedCodec ragged batch decodes each sequence at its valid length
    # so temporal convolutions never bleed right-padding into shorter rows.
    decoder_25 = _make_decoder()
    decoder_25.semantic_codec_type = "enhanced"
    decoder_25.use_gpt_latent = False

    class _ConvEnhancedCodec:
        def decode(self, codes: torch.Tensor) -> torch.Tensor:
            # Temporal neighbor mixing: each position depends on adjacent tokens.
            x = codes.to(torch.float32).unsqueeze(1)  # [B, 1, T]
            mixed = torch.nn.functional.conv1d(x, torch.ones(1, 1, 3), padding=1)
            return mixed.transpose(1, 2).expand(-1, -1, 8).contiguous()

    codec_25 = _ConvEnhancedCodec()
    row0 = torch.tensor([[10, 20, 30]], dtype=torch.long)
    row1 = torch.tensor([[40, 50, 60, 70, 80]], dtype=torch.long)
    padded_batch = torch.tensor([[10, 20, 30, 0, 0], [40, 50, 60, 70, 80]], dtype=torch.long)
    sem_batch = decoder_25._build_semantic_embedding(
        codec=codec_25,
        mel_codes=padded_batch,
        latent=None,
        dtype=torch.float32,
        code_lens=[3, 5],
    )
    sem_row0 = decoder_25._build_semantic_embedding(
        codec=codec_25,
        mel_codes=row0,
        latent=None,
        dtype=torch.float32,
        code_lens=[3],
    )
    sem_row1 = decoder_25._build_semantic_embedding(
        codec=codec_25,
        mel_codes=row1,
        latent=None,
        dtype=torch.float32,
        code_lens=[5],
    )
    torch.testing.assert_close(sem_batch[0, :3], sem_row0[0])
    torch.testing.assert_close(sem_batch[1, :5], sem_row1[0])


@pytest.mark.parametrize("use_gpt_latent", [True, False])
def test_mixed_prefill_decode_and_replace_keys_prevent_duplicate_conditioning(use_gpt_latent: bool):
    """Verify real REPLACE-key resolution and single-emission S_ref/ref_mel/style contract."""
    from vllm_omni.data_entry_keys import flatten_payload

    plane, sent_packets, _ = _make_mrv2_data_plane_with_indextts()
    resolved_keys = plane._resolve_full_payload_replace_keys()
    assert resolved_keys == _FULL_PAYLOAD_REPLACE_KEYS

    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.use_gpt_latent = use_gpt_latent

    s_ref = torch.randn(1, 5, HIDDEN)
    ref_mel = torch.randn(1, 80, 5)
    style = torch.randn(1, 192)

    active_decode_info = {
        "codes": {"mel": torch.tensor([12])},
        "meta": {
            "mel_code_count": 1,
            "step_latent": torch.randn(1, GPT_DIM),
            "S_ref": torch.randn(1, 4, HIDDEN),
            "ref_mel": torch.randn(1, 80, 4),
            "style": torch.randn(1, 192),
            "duration_factor": 1.0,
        },
    }
    prefill_info = {
        "meta": {
            "mel_code_count": 0,
            "S_ref": s_ref,
            "ref_mel": ref_mel,
            "style": style,
            "duration_factor": 1.25,
        }
    }

    # Mixed batch: request 0 is decoding (mel_len=1), request 1 is prefilling (mel_len=0)
    mixed_out = talker.make_omni_output(
        torch.zeros(2, GPT_DIM),
        model_intermediate_buffer=[active_decode_info, prefill_info],
    )
    prefill_meta = mixed_out.multimodal_outputs["meta"][1]
    assert "S_ref" not in prefill_meta
    assert "ref_mel" not in prefill_meta
    assert "style" not in prefill_meta

    def _per_req_payload(mm: dict, req_idx: int) -> dict:
        out: dict = {"meta": mm["meta"][req_idx]}
        if use_gpt_latent:
            out["codes"] = {"mel": mm["codes"]["mel"][req_idx]}
            out["hidden_states"] = {"latent": mm["hidden_states"]["latent"][req_idx]}
        return flatten_payload(out)

    req = SimpleNamespace(
        req_id="req-c8",
        external_req_id="ext-c8",
        prompt_token_ids=[1, 2, 3],
        num_computed_tokens=3,
        resumable=False,
        aborted=False,
        additional_information=None,
        sampling_params=SimpleNamespace(stop_token_ids=[8193], seed=7),
    )
    plane.register_request(req)
    req_state = plane._native_requests["req-c8"]

    # Step 0: prefill step (from mixed batch) produces first sampled token 17
    plane.reserve_outputs(["req-c8"])
    assert (
        plane.complete_outputs(
            req_ids=["req-c8"],
            inter_stage_outputs=[_per_req_payload(mixed_out.multimodal_outputs, 1)],
            sampled_token_ids=[[17]],
        )
        == 0
    )

    # Also simulate an explicit duplicate conditioning emission before decode step 1
    # to verify _FULL_PAYLOAD_REPLACE_KEYS overwrites rather than concatenating.
    plane.accumulate_full_payload_output(
        "req-c8",
        {
            "meta.use_gpt_latent": use_gpt_latent,
            "meta.duration_factor": 1.25,
            "meta.S_ref": s_ref.clone(),
            "meta.ref_mel": ref_mel.clone(),
            "meta.style": style.clone(),
        },
        req_state,
    )

    # Step 1 (mel_code_count == 1, sampled token 23) and Step 2 (mel_code_count == 2, sampled stop token 8193)
    for step_idx, (mel_tok, next_sampled_tok) in enumerate(((17, 23), (23, 8193)), start=1):
        step_info = {
            "codes": {"mel": torch.tensor([mel_tok])},
            "meta": {
                "mel_code_count": step_idx,
                "step_latent": torch.randn(1, GPT_DIM),
                "S_ref": s_ref,
                "ref_mel": ref_mel,
                "style": style,
                "duration_factor": 1.25,
            },
        }
        step_out = talker.make_omni_output(torch.zeros(1, GPT_DIM), model_intermediate_buffer=[step_info])
        plane.reserve_outputs(["req-c8"])
        plane.complete_outputs(
            req_ids=["req-c8"],
            inter_stage_outputs=[_per_req_payload(step_out.multimodal_outputs, 0)],
            sampled_token_ids=[[next_sampled_tok]],
        )

    assert plane.request_terminal({"req-c8"}) == 1
    assert len(sent_packets) == 1
    _, stage1_info = sent_packets[0]
    assert stage1_info["S_ref"].shape == (1, 5, HIDDEN)
    assert stage1_info["ref_mel"].shape == (1, 80, 5)
    assert stage1_info["style"].shape == (1, 192)
    assert stage1_info["mel_codes"].shape == (1, 2)

    decoder = _make_decoder()
    decoder.use_gpt_latent = use_gpt_latent
    codec = _make_codec()
    sem = decoder._build_semantic_embedding(
        codec=codec,
        mel_codes=stage1_info["mel_codes"],
        latent=stage1_info.get("latent"),
        dtype=torch.float32,
        code_lens=[2],
    )
    cond = decoder._regulate_lengths(sem, [2], [4])
    prompt_cond = decoder._regulate_lengths(stage1_info["S_ref"], [5], [5])
    cat_condition = torch.cat([prompt_cond, cond], dim=1)
    assert cat_condition.shape == (1, 9, 24)


@pytest.mark.parametrize("use_gpt_latent", [True, False])
def test_immediate_stop_at_prefill_errors_cleanly_on_zero_mel_codes(use_gpt_latent: bool):
    """If step 0 samples STOP_MEL_TOKEN (mel_len=0), Stage-1 payload build fails cleanly."""
    from vllm_omni.model_executor.stage_input_processors.indextts2 import STOP_MEL_TOKEN

    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.use_gpt_latent = use_gpt_latent
    prefill_out = talker.make_omni_output(
        torch.zeros(1, GPT_DIM),
        model_intermediate_buffer=[
            {
                "meta": {
                    "mel_code_count": 0,
                    "S_ref": torch.randn(1, 5, HIDDEN),
                    "ref_mel": torch.randn(1, 80, 5),
                    "style": torch.randn(1, 192),
                    "duration_factor": 1.0,
                }
            }
        ],
    )
    req = SimpleNamespace(
        request_id="req-eos0",
        output_token_ids=[STOP_MEL_TOKEN],
    )
    payload = prefill_out.multimodal_outputs or {}
    if use_gpt_latent:
        with pytest.raises(ValueError, match="missing meta.use_gpt_latent"):
            talker2s2mel_full_payload(None, payload, req)
    else:
        flat_payload = {f"meta.{k}": v for k, v in payload["meta"][0].items()}
        with pytest.raises(ValueError, match="zero non-stop mel codes"):
            talker2s2mel_full_payload(None, flat_payload, req)


def test_ensure_dit_caches_avoids_runtime_reallocation_across_batch_growth():
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    decoder.s2mel_cfm_batch_size = 16
    decoder.inference_cfg_rate = 0.7

    setup_calls: list[tuple[int, int]] = []
    transformer = SimpleNamespace(freqs_cis=None, max_batch_size=-1, use_kv_cache=False)

    def setup_caches(max_batch_size: int, max_seq_length: int):
        setup_calls.append((max_batch_size, max_seq_length))
        transformer.freqs_cis = torch.ones(1)
        transformer.max_batch_size = max_batch_size

    estimator = SimpleNamespace(transformer=transformer, setup_caches=setup_caches)

    decoder._ensure_dit_caches(estimator, cfm_batch_size=1)
    decoder._ensure_dit_caches(estimator, cfm_batch_size=8)
    assert setup_calls == [(32, 16384)]

    # Even if an earlier step initialized a smaller max_batch_size with use_kv_cache=False,
    # batch growth must not trigger another 16384x16384 causal_mask allocation.
    transformer.max_batch_size = 2
    decoder._ensure_dit_caches(estimator, cfm_batch_size=8)
    assert setup_calls == [(32, 16384)]
    assert transformer.max_batch_size == 32


def test_indextts2_talker_mrv2_enables_full_cuda_graph_capture():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    model = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    nn.Module.__init__(model)
    assert model.have_multimodal_outputs is True
    assert model.mrv2_forward_returns_tensor is True

    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner._configure_cudagraph_output_contract()

    assert runner._model_has_structured_output is False
    assert runner._full_graph_aux_outputs is False
    assert runner._exclude_full_graph is False


def test_s2mel_forward_handles_mrv2_dummy_placeholders_and_rejects_partial_missing_payload():
    """MRv2 dummy/profile passes pass empty dict placeholders in model_intermediate_buffer.

    Verify Stage 1 returns a valid dummy OmniOutput for all-empty placeholder
    batches while still raising cleanly when a batch mixes a real request with a
    missing payload slot.
    """
    decoder = _make_decoder()

    single_dummy = decoder.forward(torch.tensor([0]), model_intermediate_buffer=[{}])
    assert single_dummy.text_hidden_states is None
    assert isinstance(single_dummy.multimodal_outputs["audio"], torch.Tensor)
    assert single_dummy.multimodal_outputs["audio"].ndim == 1
    assert int(single_dummy.multimodal_outputs["sr"].item()) == 22050

    multi_dummy = decoder.forward(torch.tensor([0, 0]), model_intermediate_buffer=[{}, {}])
    assert isinstance(multi_dummy.multimodal_outputs["audio"], list)
    assert len(multi_dummy.multimodal_outputs["audio"]) == 2

    valid_payload = _stage1_payload(4, seed=9)
    with pytest.raises(ValueError, match="input 1 is missing use_gpt_latent"):
        decoder.forward(torch.tensor([0, 0]), model_intermediate_buffer=[valid_payload, {}])


def test_ref_audio_and_emo_audio_cache_keys_reuse_speaker_cache_and_isolate_versions():
    """Inline ref_audio/emo_audio requests pass cache keys from the serving adapter.

    Verify Stage 0 reuses SpeakerEmbeddingCache on ref_audio_cache_key and
    emo_audio_cache_key without colliding between IndexTTS-2 and IndexTTS-2.5.
    """
    from vllm_omni.model_executor.models.indextts2.indextts2_talker import (
        resolve_indextts_conditioning_policy,
    )
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    shared_cache = SpeakerEmbeddingCache(max_bytes=64 * 1024 * 1024)

    def _build_minimal_talker(model_type: str) -> IndexTTS2TalkerForConditionalGeneration:
        talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
        nn.Module.__init__(talker)
        talker.model_path = "dummy-model"
        talker.config = SimpleNamespace(
            model_type=model_type,
            semantic_codec={"codebook_size": 8192},
        )
        talker.conditioning_policy = resolve_indextts_conditioning_policy(model_type)
        talker.use_gpt_latent = model_type == "indextts2"
        talker.model_dim = HIDDEN
        talker.start_mel_token = 8192
        talker._speaker_cache = shared_cache
        talker._w2v_stat = None
        talker.text_embedding = nn.Embedding(32, HIDDEN)
        talker.mel_embedding = nn.Embedding(8194, HIDDEN)
        talker.text_pos_embedding = lambda ids: torch.zeros(ids.shape[0], ids.shape[1], HIDDEN)
        talker.mel_pos_embedding = SimpleNamespace(
            get_fixed_embedding=lambda _pos, device: torch.zeros(1, 1, HIDDEN, device=device)
        )
        talker.emo_layer = nn.Linear(HIDDEN, HIDDEN)
        talker.emovec_layer = nn.Linear(1024, HIDDEN)
        if talker.conditioning_policy.uses_campplus_projection:
            talker.spk_emb_proj = nn.Linear(192, HIDDEN)
            talker.lang_embedding = nn.Embedding(8, HIDDEN)
        else:

            class _DummyCondEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = nn.Parameter(torch.zeros(1))

                def forward(self, emb, _lens):
                    return (
                        torch.zeros(1, 4, HIDDEN, dtype=emb.dtype, device=emb.device),
                        torch.ones(1, 1, 4, device=emb.device),
                    )

            talker.conditioning_encoder = _DummyCondEncoder()
            talker.cond_mask_pad = lambda mask: mask
            talker.perceiver_encoder = lambda cond, _mask: torch.zeros(1, 32, HIDDEN, dtype=cond.dtype)
            talker.speed_emb = nn.Embedding(2, HIDDEN)

        talker._ensure_w2v_stat_loaded = lambda _dev: None
        talker._compute_mel_22k = lambda _wav, dev: torch.ones(1, 80, 12, device=dev)
        talker._tokenize_text = lambda text, dev, **kw: (
            torch.tensor([[1, 2, 3]], dtype=torch.long, device=dev),
            torch.tensor([0], dtype=torch.long, device=dev)
            if talker.conditioning_policy.uses_language_embedding
            else None,
        )
        talker._merge_emovec = lambda spk, emo, _alpha: torch.zeros(1, HIDDEN, dtype=spk.dtype, device=spk.device)
        return talker

    load_calls: list[str] = []
    w2v_calls: list[str] = []
    quant_calls: list[str] = []

    def _fake_load_ref(_path, device, mode="speaker"):
        load_calls.append(mode)
        return torch.zeros(1600, device=device), torch.zeros(2205, device=device)

    def _fake_w2v(_wav, _model, _proc, device, _stat):
        w2v_calls.append("w2v")
        return torch.full((1, 5, 1024), 2.5, device=device)

    class _FakeCodec:
        def quantize(self, emb):
            quant_calls.append("quant")
            return None, torch.full((1, 5, 1024), 1.25, device=emb.device)

    with (
        patch("vllm_omni.model_executor.models.indextts2.indextts2_talker.load_reference_audio", _fake_load_ref),
        patch(
            "vllm_omni.model_executor.models.indextts2.indextts2_talker.load_wav2vec2",
            lambda _p, _d: (object(), object()),
        ),
        patch("vllm_omni.model_executor.models.indextts2.indextts2_talker.wav2vec_extract", _fake_w2v),
        patch(
            "vllm_omni.model_executor.models.indextts2.indextts2_talker.load_semantic_codec",
            lambda _p, _c, _d: _FakeCodec(),
        ),
        patch(
            "vllm_omni.model_executor.models.indextts2.indextts2_talker.load_campplus",
            lambda _p, _d: (lambda _fb: torch.ones(1, 192)),
        ),
        patch(
            "vllm_omni.model_executor.models.indextts2.indextts2_talker.compute_fbank",
            lambda _wav, dev: torch.ones(1, 10, 80, device=dev),
        ),
    ):
        t2 = _build_minimal_talker("indextts2")
        info_v2 = {
            "text": ["hello"],
            "voice": [["dummy.wav"]],
            "ref_audio_cache_key": ["sha256-same-ref"],
        }
        # IndexTTS-2: 34 prefix + 3 text + 1 start_mel = 38 tokens
        ids_v2 = torch.ones(38, dtype=torch.long)
        _, _, upd1 = t2._preprocess_prefill(ids_v2, None, info_v2, 38)
        _, _, upd2 = t2._preprocess_prefill(ids_v2, None, info_v2, 38)
        assert len(load_calls) == 1
        assert len(w2v_calls) == 1
        assert len(quant_calls) == 1
        torch.testing.assert_close(upd1["meta"]["S_ref"], upd2["meta"]["S_ref"])
        assert float(upd2["meta"]["S_ref"][0, 0, 0]) == pytest.approx(1.25)

        # IndexTTS-2.5 with the exact same ref_audio_cache_key must NOT hit 2.0's quantized S_ref
        t25 = _build_minimal_talker("indextts2_5")
        info_v25 = {
            "text": ["hello"],
            "voice": [["dummy.wav"]],
            "ref_audio_cache_key": ["sha256-same-ref"],
            "emo_audio": ["emo.wav"],
            "emo_audio_cache_key": ["sha256-same-emo"],
        }
        # IndexTTS-2.5: 3 prefix + 3 text + 1 start_mel = 7 tokens
        ids_v25 = torch.ones(7, dtype=torch.long)
        _, _, upd25_1 = t25._preprocess_prefill(ids_v25, None, info_v25, 7)
        _, _, upd25_2 = t25._preprocess_prefill(ids_v25, None, info_v25, 7)
        # 1 speaker load + 1 emo load on 1st call; 0 new loads on 2nd call
        assert load_calls == ["speaker", "speaker", "emotion"]
        assert len(w2v_calls) == 3
        assert float(upd25_1["meta"]["S_ref"][0, 0, 0]) == pytest.approx(2.5)
        torch.testing.assert_close(upd25_1["meta"]["S_ref"], upd25_2["meta"]["S_ref"])


@pytest.mark.parametrize("use_gpt_latent", [True, False])
def test_mrv2_batched_decode_preprocess_and_postprocess_match_serial(use_gpt_latent: bool):
    """Batched MRv2 decode preprocess and postprocess match serial hooks bit-for-bit."""
    import numpy as np

    from vllm_omni.model_executor.models.output_templates import OwnedBatchTensor
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    torch.manual_seed(17)
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.use_gpt_latent = use_gpt_latent
    talker.has_preprocess = True
    talker.has_postprocess = use_gpt_latent
    talker.model_dim = HIDDEN
    talker.mel_embedding = nn.Embedding(64, HIDDEN)
    talker.mel_pos_embedding = SimpleNamespace(emb=nn.Embedding(64, HIDDEN))
    talker.gpu_resident_buffer_keys = {
        ("codes", "mel"),
        ("meta", "mel_start_offset"),
        ("meta", "mel_code_count"),
        ("hidden_states", "latent"),
        ("meta", "latent_acc"),
        ("meta", "step_latent"),
    }

    batch_size = 3
    input_ids = torch.tensor([7, 19, 41], dtype=torch.long)
    prev_latents = [torch.randn(1, HIDDEN) for _ in range(batch_size)]
    req_infos = [
        {
            "req_id": f"req-{i}",
            "meta": {
                "mel_start_offset": 10 + i,
                "mel_code_count": i + 2,
                **({"latent_acc": prev_latents[i]} if use_gpt_latent else {}),
            },
        }
        for i in range(batch_size)
    ]

    # Compare batched MRv2 decode hook against serial _preprocess_decode
    serial_embeds = []
    serial_updates = []
    for i in range(batch_size):
        _, emb_i, upd_i = talker.preprocess(input_ids[i : i + 1], None, **req_infos[i])
        serial_embeds.append(emb_i)
        serial_updates.append(upd_i)

    batch_ids, batch_embeds, batch_updates = talker.preprocess_decode_batch_mrv2(
        input_ids=input_ids,
        input_embeds=torch.zeros(batch_size, HIDDEN),
        req_infos=req_infos,
    )
    assert batch_ids is input_ids
    torch.testing.assert_close(batch_embeds, torch.cat(serial_embeds, dim=0), atol=0.0, rtol=0.0)
    for i in range(batch_size):
        assert batch_updates[i]["meta"]["mel_code_count"] == serial_updates[i]["meta"]["mel_code_count"]
        if use_gpt_latent:
            torch.testing.assert_close(batch_updates[i]["codes"]["mel"], serial_updates[i]["codes"]["mel"])
            torch.testing.assert_close(
                batch_updates[i]["meta"]["step_latent"], serial_updates[i]["meta"]["step_latent"]
            )
        else:
            assert "codes" not in batch_updates[i]

    # Verify postprocess_batch_mrv2 matches serial postprocess
    hidden = torch.randn(7, HIDDEN)
    last_indices = torch.tensor([2, 4, 6], dtype=torch.long)
    if use_gpt_latent:
        key, owned = talker.postprocess_batch_mrv2(hidden_states=hidden, last_token_indices=last_indices)
        assert key == ("meta", "latent_acc")
        assert isinstance(owned, OwnedBatchTensor)
        assert owned.tensor.shape == (3, 1, HIDDEN)
        torch.testing.assert_close(owned.tensor[:, 0, :], hidden[last_indices], atol=0.0, rtol=0.0)

    # Verify OmniModelState.run_preprocess prefers num_computed_tokens_np over non-indexable StagedWriteTensor
    class _UnindexableStaged:
        pass

    state = object.__new__(OmniModelState)
    state.model = talker
    state.has_preprocess = True
    state.has_postprocess = use_gpt_latent
    state.preprocess_keeps_mm_inputs = False
    state._static_inputs_embeds = None
    state._has_mtp = False
    state._decode_preprocess_is_identity = False
    state.intermediate_buffer = OmniIntermediateBuffer(max_num_reqs=4)
    for i in range(batch_size):
        state.intermediate_buffer.buffers[i] = {
            "req_id": f"req-{i}",
            "meta": dict(req_infos[i]["meta"]),
        }

    input_batch = SimpleNamespace(
        num_reqs=batch_size,
        num_tokens=batch_size,
        input_ids=input_ids,
        idx_mapping_np=np.arange(batch_size, dtype=np.int32),
        query_start_loc_np=np.arange(batch_size + 1, dtype=np.int32),
        num_scheduled_tokens=np.ones(batch_size, dtype=np.int32),
    )
    req_states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([10, 11, 12, 0], dtype=np.int32)),
        num_computed_tokens=_UnindexableStaged(),
        num_computed_tokens_np=np.array([12, 14, 16, 0], dtype=np.int32),
    )
    model_inputs: dict[str, Any] = {"input_ids": input_ids}
    state.run_preprocess(input_batch, model_inputs, req_states=req_states)
    torch.testing.assert_close(model_inputs["inputs_embeds"], batch_embeds, atol=0.0, rtol=0.0)
    for i in range(batch_size):
        assert state.intermediate_buffer.buffers[i]["meta"]["mel_code_count"] == i + 3

    # Also verify OmniModelState.run_postprocess end-to-end (updates meta.latent_acc when use_gpt_latent=True,
    # skips completely when use_gpt_latent=False).
    decode_hidden = torch.randn(batch_size, HIDDEN)
    state.run_postprocess(decode_hidden, input_batch)
    for i in range(batch_size):
        if use_gpt_latent:
            stored = state.intermediate_buffer.buffers[i]["meta"]["latent_acc"]
            assert stored.shape == (1, HIDDEN)
            torch.testing.assert_close(stored, decode_hidden[i : i + 1], atol=0.0, rtol=0.0)
        else:
            assert "latent_acc" not in state.intermediate_buffer.buffers[i]["meta"]


def test_cfm_solve_euler_hoists_cond_projection_once_with_identical_output():
    """CFM.solve_euler projects static cond once before the Euler loop with bit-identical output."""
    from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import BASECFM

    class _CountingLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.calls = 0

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.linear(inp)

    class _TinyDiT(nn.Module):
        _supports_precomputed_cond = True
        style_as_token = False
        time_as_token = False
        is_causal = False

        def __init__(self):
            super().__init__()
            self.cond_projection = _CountingLinear(16, 16)
            self.merge = nn.Linear(8 + 8 + 16, 8)

        def forward(self, x, prompt_x, x_lens, t, style, cond, pre_mask=None, unpad_data=None, precomputed_cond=None):
            del x_lens, style, pre_mask, unpad_data
            c = self.cond_projection(cond) if precomputed_cond is None else precomputed_cond
            cat = torch.cat([x.transpose(1, 2), prompt_x.transpose(1, 2), c], dim=-1)
            return (self.merge(cat) + t[:, None, None]).transpose(1, 2)

    class _TinyCFM(BASECFM):
        def __init__(self, est):
            super().__init__(SimpleNamespace(DiT=SimpleNamespace(in_channels=8)))
            self.estimator = est
            object.__setattr__(self, "_eager_estimator", est)

    torch.manual_seed(0)
    est = _TinyDiT().eval()
    cfm = _TinyCFM(est)

    mu = torch.randn(2, 10, 16)
    x_lens = torch.tensor([10, 10], dtype=torch.long)
    prompt = torch.randn(2, 8, 4)
    style = torch.randn(2, 12)
    noise = torch.randn(2, 8, 10)

    out_hoisted = cfm.inference(
        mu, x_lens, prompt, style, None, n_timesteps=6, inference_cfg_rate=0.7, initial_noise=noise.clone()
    )
    assert est.cond_projection.calls == 1

    est.cond_projection.calls = 0
    est._supports_precomputed_cond = False
    out_baseline = cfm.inference(
        mu, x_lens, prompt, style, None, n_timesteps=6, inference_cfg_rate=0.7, initial_noise=noise.clone()
    )
    assert est.cond_projection.calls == 6
    torch.testing.assert_close(out_hoisted, out_baseline, atol=0.0, rtol=0.0)


def test_indextts2_deploy_config_includes_640_vocoder_graph_bucket():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "indextts2.yaml"
    cfg = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in cfg["stages"] if stage["stage_id"] == 1)
    overrides = stage1["hf_overrides"]
    assert 640 in overrides["s2mel_vocoder_capture_sizes"]
    assert overrides["s2mel_vocoder_compile_shapes"] == [256, 384, 512]

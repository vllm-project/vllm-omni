# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for IndexTTS 2 & 2.5 MRv2 data plane, Stage-1 concurrency, and scoped TF32."""

import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

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
from vllm_omni.worker_v2.delivery import OmniDeliveryManager
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
    plane = object.__new__(OmniRunnerDataPlane)
    sent_packets: list[tuple[str, dict]] = []
    cleaned_reqs: list[str] = []

    class _FakeConnector:
        def put(self, from_stage, to_stage, put_key, data):
            sent_packets.append((put_key, data))
            return True, 1, None

    plane.__dict__.update(
        _native_requests={},
        _native_output_lock=threading.RLock(),
        _native_send_lock=threading.Lock(),
        _native_output_error=None,
        _native_output_error_lock=threading.Lock(),
        _native_output_closed=False,
        _native_outputs_in_flight=defaultdict(int),
        _native_terminal_pending=set(),
        _deferred_shm_releases=set(),
        _async_chunk=False,
        _can_send=True,
        _pending_full_payload_send={},
        _request_ids_mapping={},
        _put_req_chunk=defaultdict(int),
        _ramp_chunk_count=defaultdict(int),
        _stage_id=0,
        _next_stage_id="1",
        _omni_connector=_FakeConnector(),
        _custom_process_func=talker2s2mel_full_payload,
        _custom_process_batch_func=None,
        _delivery_manager=OmniDeliveryManager(delivery_timeout_s=1.0, shutdown_timeout_s=1.0),
    )
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
    3. Both indextts2_mrv2.yaml and indextts2_5_mrv2.yaml overlays declare MRv2 on Stage 0 with
       supports_native_mrv2_data_plane=True and unchunked prefill.
    """
    plane, sent_packets, cleaned_reqs = _make_mrv2_data_plane_with_indextts()

    # 1. publish_output clears multimodal_outputs for non-async_chunk full-payload producer
    plane.enqueue_outputs = lambda **_kw: None
    shared_payload = [{"meta.use_gpt_latent": True}]
    runner_out = OmniModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
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
        req_id_to_index={"r0": 0},
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

    # 3. Verify both IndexTTS pipelines and MRv2 overlays
    for pipeline in (INDEXTTS2_PIPELINE, INDEXTTS25_PIPELINE):
        assert pipeline.stages[0].supports_native_mrv2_data_plane is True
        assert pipeline.stages[0].custom_process_next_stage_input_func.endswith("talker2s2mel_full_payload")

    deploy_dir = Path(__file__).parents[4] / "vllm_omni" / "deploy"
    for overlay_name in ("indextts2_mrv2.yaml", "indextts2_5_mrv2.yaml"):
        cfg = yaml.safe_load((deploy_dir / overlay_name).read_text(encoding="utf-8"))
        stage0 = next(s for s in cfg["stages"] if s["stage_id"] == 0)
        stage1 = next(s for s in cfg["stages"] if s["stage_id"] == 1)
        assert stage0["model_runner"] == "v2"
        assert stage0["enable_prefix_caching"] is False
        assert stage0["enable_chunked_prefill"] is False
        assert stage0["max_num_batched_tokens"] >= stage0["max_model_len"]
        assert stage1["hf_overrides"]["s2mel_allow_tf32"] is True


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
    """Verify:
    1. _resolve_full_payload_replace_keys() resolves _FULL_PAYLOAD_REPLACE_KEYS from
       stage_input_processors.indextts2 without fixture injection;
    2. When a prefill request (mel_code_count=0) shares a batch with an active decode
       request (mel_code_count=1) — or in IndexTTS-2.5 (use_gpt_latent=False) where
       prefill always emits meta — S_ref / ref_mel / style are emitted only once at
       mel_code_count == 1 and never concatenated into batch-2 tensors;
    3. Even if a duplicate conditioning chunk arrives in accumulate_full_payload_output,
       _FULL_PAYLOAD_REPLACE_KEYS keeps S_ref / ref_mel / style at batch size 1 so
       Stage 1 torch.cat([prompt_condition, cond], dim=1) succeeds.
    """
    from vllm_omni.data_entry_keys import flatten_payload

    plane, sent_packets, _ = _make_mrv2_data_plane_with_indextts()
    assert "_full_payload_replace_keys_cached" not in plane.__dict__
    resolved_keys = plane._resolve_full_payload_replace_keys()
    assert resolved_keys == _FULL_PAYLOAD_REPLACE_KEYS
    assert {"meta.S_ref", "meta.ref_mel", "meta.style", "meta.duration_factor", "meta.use_gpt_latent"} <= resolved_keys

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

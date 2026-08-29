# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav as batched_token2wav_module
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2Wav,
    _token2wav_sdpa_context,
    plan_token2wav_encode_slices,
    relpos_encode_token_budget,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []
        self.last_chunk_calls: list[bool] = []

    def forward_chunk(self, xs, last_chunk=False, cnn_cache=None, att_cache=None):
        batch, length, _ = xs.shape
        self.calls.append(batch)
        self.last_chunk_calls.append(last_chunk)
        old_length = 0 if att_cache is None else att_cache.shape[3]
        output = xs[:, : max(1, length - 1)]
        cnn = xs[:, :1, :].transpose(1, 2).contiguous()
        marker = xs[:, 0, 0].reshape(1, batch, 1, 1, 1)
        att = marker.expand(1, batch, 1, old_length + output.shape[1], 1).clone()
        return output, cnn, att


class _FakeBlock:
    def __init__(self):
        conv1 = SimpleNamespace(causal_padding=(1, 0))
        self.conv = SimpleNamespace(
            in_channels=1,
            out_channels=1,
            block=[None, conv1],
        )
        self.attn = SimpleNamespace(num_heads=1, head_dim=1)


class _FakeEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_FakeBlock()]
        self.cfg_batches: list[int] = []
        self.speaker_order: list[list[float]] = []
        self.attention_cache_dtypes: list[torch.dtype | None] = []
        self.register_buffer("att_cache_buffer", torch.ones(1), persistent=False)
        self.register_buffer("cnn_cache_buffer", torch.ones(1), persistent=False)

    def t_embedder(self, time):
        return time[:, None]

    def blocks_forward_chunk(
        self,
        inputs,
        time,
        mask,
        cnn_cache,
        att_cache,
        cnn_out,
        att_out,
    ):
        del time, mask, cnn_cache
        self.attention_cache_dtypes.append(att_cache.dtype if isinstance(att_cache, torch.Tensor) else None)
        self.cfg_batches.append(inputs.shape[0])
        self.speaker_order.append(inputs[:, 2, 0].tolist())
        marker = inputs[:, 1, 0]
        cnn_out.copy_(marker.reshape(1, -1, 1, 1).expand_as(cnn_out))
        att_out.copy_(marker.reshape(1, -1, 1, 1, 1).expand_as(att_out))
        return inputs[:, 1:2]


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.estimator = _FakeEstimator()
        self.inference_cfg_rate = 0.7
        self.register_buffer("rand_noise", torch.zeros(1, 1, 100), persistent=False)


class _FakeFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _FakeEncoder()
        self.encoder_proj = nn.Identity()
        self.decoder = _FakeDecoder()
        self.spk_embed_affine_layer = nn.Identity()

    def input_embedding(self, tokens):
        return tokens.to(torch.float32).unsqueeze(-1)


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[int] = []

    def forward(self, batch, device):
        mel = batch["speech_feat"].transpose(1, 2).to(device)
        return self.inference(mel, mel.new_zeros((mel.shape[0], 1, 0)))

    def inference(self, mel, source):
        del source
        self.calls.append(mel.shape[0])
        speech = mel[:, 0].repeat_interleave(3, dim=1)
        generated_source = speech[:, None]
        return speech, generated_source


class _FakeToken2Wav:
    def __init__(self):
        self.flow = _FakeFlow()
        self.hift = _FakeHiFT()
        self.float16 = False
        self.n_timesteps = 2
        self.mel_cache_len = 1
        self.source_cache_len = 2
        self.speech_window = torch.hamming_window(4, periodic=False)
        self.prompt_calls = 0

    def _prepare_prompt(self, prompt_wav):
        del prompt_wav
        self.prompt_calls += 1
        return (
            torch.tensor([[5, 6]], dtype=torch.long),
            torch.tensor([2], dtype=torch.int32),
            torch.ones(1, 1),
            torch.ones(1, 4, 1),
            torch.tensor([4], dtype=torch.int32),
        )

    def stream(self, *args, **kwargs):
        raise AssertionError("sequential stream fallback must never be called")

    def __call__(self, *args, **kwargs):
        raise AssertionError("sequential __call__ fallback must never be called")


def _config(minimum: int = 1, initial: int = 0):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/fake/model",
            stage_connector_config={
                "extra": {
                    "code2wav_min_batch_size": minimum,
                    "code2wav_initial_batch_size": initial,
                    "prompt_cache_id": "shared",
                    "prompt_wav": "/fake/prompt.wav",
                }
            },
        )
    )


def _model(initial: int = 0, minimum: int = 1):
    token2wav = _FakeToken2Wav()
    backend = BatchedToken2Wav(token2wav)
    _enable_fake_ragged_kernel(backend)
    model = MiniCPMO45Code2Wav(vllm_config=_config(minimum=minimum, initial=initial))
    model.backend = backend
    return model, token2wav


def _enable_fake_ragged_kernel(adapter: BatchedToken2Wav) -> None:
    """Keep fake-model tests focused on ragged grouping and row mapping."""
    estimator = adapter.flow.decoder.estimator
    estimator.in_proj = nn.Identity()

    def fake_ragged_kernel(
        estimator,
        estimator_input,
        time_embedding,
        attn_mask,
        cnn_cache,
        att_cache,
        cnn_cache_buffer,
        att_cache_buffer,
        valid_lengths,
    ):
        del valid_lengths
        return estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            attn_mask,
            cnn_cache,
            att_cache,
            cnn_cache_buffer,
            att_cache_buffer,
        )

    adapter._blocks_forward_chunk_ragged = fake_ragged_kernel  # type: ignore[method-assign]


def _info(
    request_id: str,
    chunk_seq: int,
    codes: list[int],
    *,
    last_chunk: bool = False,
    cache_epoch: int = 0,
):
    return {
        "codes": {"audio": torch.tensor(codes, dtype=torch.long)},
        "meta": {
            "request_id": request_id,
            "chunk_seq": chunk_seq,
            "cache_epoch": cache_epoch,
            "last_chunk": last_chunk,
            "prompt_cache_id": "shared",
        },
    }


def _forward(model, infos, placeholder_counts=None, request_ids=None):
    placeholder_counts = placeholder_counts or [1] * len(infos)
    input_ids = torch.zeros(sum(placeholder_counts), dtype=torch.long)
    return model(
        input_ids=input_ids,
        seq_token_counts=placeholder_counts,
        runtime_additional_information=infos,
        request_ids=request_ids,
    )


def test_adapter_runs_true_batch_cfg_and_splits_request_caches():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    audios, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.prompt_calls == 1
    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert all(order == [1.0, 1.0, 0.0, 0.0] for order in token2wav.flow.decoder.estimator.speaker_order)
    assert token2wav.hift.calls == [2]
    assert len(audios) == 2
    cache0 = states[0].flow_cache["estimator_cnn_cache"]
    cache1 = states[1].flow_cache["estimator_cnn_cache"]
    assert cache0.data_ptr() != cache1.data_ptr()
    assert cache0[0, 0, 0, 0, 0].item() == 10
    assert cache1[0, 0, 0, 0, 0].item() == 20


def test_relpos_encode_token_budget_leaves_room_for_upsample_and_cache():
    # The NPU crash was 6968 vs 985: 3333 tokens * 2 + cache overflowed max_pos=5000.
    assert relpos_encode_token_budget(max_pos=5000, stride=2, cache_offset=150, lookahead=3) <= 1024
    assert relpos_encode_token_budget(max_pos=5000, stride=2, cache_offset=150, lookahead=3, cap=3000) == 2346
    assert plan_token2wav_encode_slices(3333, max_frames=1024, min_nonfinal=4, last_chunk=True) == [
        (0, 1024),
        (1024, 2048),
        (2048, 3072),
        (3072, 3333),
    ]
    assert plan_token2wav_encode_slices(50, max_frames=40, min_nonfinal=20, last_chunk=False) == [
        (0, 30),
        (30, 50),
    ]


def test_decode_batch_splits_overlong_prefill_inside_relpos_budget():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)
    setup_encodes = len(token2wav.flow.encoder.calls)
    adapter._max_encode_token_frames = lambda _states: 4

    audios, _ = adapter.decode_batch(
        torch.arange(10, dtype=torch.long).reshape(1, -1),
        prompt,
        states,
        last_chunk=True,
    )

    assert token2wav.flow.encoder.calls[setup_encodes:] == [1, 1, 1]
    assert token2wav.flow.encoder.last_chunk_calls[setup_encodes:] == [False, False, True]
    assert len(audios) == 1
    assert audios[0].numel() > 0


def test_ragged_overlong_prefill_uses_relpos_safe_exact_slices():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    _enable_fake_ragged_kernel(adapter)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    setup_encodes = len(token2wav.flow.encoder.calls)
    adapter._max_encode_token_frames = lambda _states: 4

    tokens = [
        torch.arange(10, dtype=torch.long),
        torch.arange(4, dtype=torch.long),
    ]
    last_chunks = [True, False]
    audios, next_states = adapter.decode_ragged_batch(
        tokens,
        prompt,
        states,
        last_chunks=last_chunks,
    )

    assert token2wav.flow.encoder.calls[setup_encodes:] == [1, 1, 1, 1]
    assert token2wav.flow.encoder.last_chunk_calls[setup_encodes:] == [False, False, True, False]
    assert len(audios) == len(next_states) == 2
    assert all(audio.numel() > 0 for audio in audios)

    for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
        reference_token2wav = _FakeToken2Wav()
        reference_token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
        reference_adapter = BatchedToken2Wav(reference_token2wav)
        reference_prompt = reference_adapter.prepare_prompt("shared", "/fake/prompt.wav")
        reference_states = reference_adapter.setup_batch(reference_prompt, 1)
        reference_adapter._max_encode_token_frames = lambda _states: 4
        reference_audio, reference_state = reference_adapter.decode_batch(
            row_tokens.unsqueeze(0),
            reference_prompt,
            reference_states,
            last_chunk=last_chunk,
        )
        torch.testing.assert_close(audios[row], reference_audio[0])
        for cache_name, expected in reference_state[0].flow_cache.items():
            torch.testing.assert_close(
                next_states[row].flow_cache[cache_name],
                expected,
            )
        for cache_name, expected in reference_state[0].hift_cache.items():
            torch.testing.assert_close(
                next_states[row].hift_cache[cache_name],
                expected,
            )


def test_ragged_diffusion_batches_eight_rows_and_preserves_exact_results():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    _enable_fake_ragged_kernel(adapter)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 8)
    token2wav.flow.encoder.calls.clear()
    token2wav.flow.decoder.estimator.cfg_batches.clear()
    token2wav.hift.calls.clear()
    tokens = [
        *(torch.tensor([10 + row, 11 + row, 12 + row]) for row in range(6)),
        torch.tensor([20, 21]),
        torch.tensor([30, 31]),
    ]
    last_chunks = [False] * 6 + [True, True]

    audios, next_states = adapter.decode_ragged_batch(
        tokens,
        prompt,
        states,
        last_chunks=last_chunks,
    )

    assert token2wav.flow.encoder.calls == [6, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [16, 16]
    assert token2wav.hift.calls == [6, 2]
    assert len(audios) == len(next_states) == 8

    for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
        reference_adapter = BatchedToken2Wav(_FakeToken2Wav())
        reference_prompt = reference_adapter.prepare_prompt("shared", "/fake/prompt.wav")
        reference_states = reference_adapter.setup_batch(reference_prompt, 1)
        reference_audio, reference_state = reference_adapter.decode_batch(
            row_tokens.unsqueeze(0),
            reference_prompt,
            reference_states,
            last_chunk=last_chunk,
        )
        torch.testing.assert_close(audios[row], reference_audio[0])
        for cache_name, expected in reference_state[0].flow_cache.items():
            torch.testing.assert_close(
                next_states[row].flow_cache[cache_name],
                expected,
            )
        for cache_name, expected in reference_state[0].hift_cache.items():
            torch.testing.assert_close(
                next_states[row].hift_cache[cache_name],
                expected,
            )


def test_npu_patch_preserves_ragged_estimator_contract():
    script = r"""
import torch

from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import (
    _FakeToken2Wav,
    _enable_fake_ragged_kernel,
)
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
import vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav as npu_patch


class _FailGraphRunner:
    def run(self, *args, **kwargs):
        raise AssertionError("ragged decode must bypass the exact-shape NPU Graph")


npu_patch.apply_minicpmo_4_5_code2wav_patch()
adapter = BatchedToken2Wav(_FakeToken2Wav())
_enable_fake_ragged_kernel(adapter)
prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
states = adapter.setup_batch(prompt, 2)
npu_patch._backend_graph_runners[adapter] = _FailGraphRunner()
audios, next_states = adapter.decode_ragged_batch(
    [
        torch.tensor([10, 11, 12]),
        torch.tensor([20, 21]),
    ],
    prompt,
    states,
    last_chunks=[False, True],
)
assert len(audios) == len(next_states) == 2
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_decode_cfm_enters_platform_sdpa_context(monkeypatch):
    entered: list[str] = []

    @contextmanager
    def recording_context():
        entered.append("enter")
        yield
        entered.append("exit")

    monkeypatch.setattr(
        batched_token2wav_module,
        "_token2wav_sdpa_context",
        lambda _device: recording_context(),
    )
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    _enable_fake_ragged_kernel(adapter)
    adapter._decode_cfm(
        torch.ones((2, 1, 2)),
        torch.ones((2, 1)),
        torch.zeros((2, 1, 2)),
        cnn_cache=None,
        att_cache=None,
        valid_lengths=[2, 1],
    )

    assert entered == ["enter", "exit"]


def test_ragged_decode_bypasses_exact_shape_accelerators():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    _enable_fake_ragged_kernel(adapter)

    class ForbiddenAccelerator:
        def replay(self, *args):
            pytest.fail("ragged decode must not use the exact-shape CFM graph")

        def step(self, **kwargs):
            pytest.fail("ragged decode must not use the exact-shape TensorRT stepper")

    accelerator = ForbiddenAccelerator()
    adapter._cfm_graph_wrapper = accelerator
    adapter._trt_stepper = accelerator
    adapter._decode_cfm(
        torch.ones((2, 1, 2)),
        torch.ones((2, 1)),
        torch.zeros((2, 1, 2)),
        cnn_cache=None,
        att_cache=None,
        valid_lengths=[2, 1],
    )


def test_ragged_decode_fails_when_estimator_has_no_ragged_kernel():
    adapter = BatchedToken2Wav(_FakeToken2Wav())

    with pytest.raises(
        RuntimeError,
        match=r'"reason":"ragged_kernel_unavailable"',
    ):
        adapter._decode_cfm(
            torch.ones((2, 1, 2)),
            torch.ones((2, 1)),
            torch.zeros((2, 1, 2)),
            cnn_cache=None,
            att_cache=None,
            valid_lengths=[2, 1],
        )


def test_cfm_graph_receives_bfloat16_cache_in_compute_dtype():
    adapter = BatchedToken2Wav(_FakeToken2Wav(), bfloat16_attention_cache=True)
    seen_dtypes: list[torch.dtype] = []

    class RecordingGraph:
        def replay(self, estimator_input, time_embedding, cnn_cache, att_cache, cnn_out, att_out):
            del time_embedding, cnn_cache
            seen_dtypes.append(att_cache.dtype)
            return estimator_input[:, :1], cnn_out, att_out

    adapter._cfm_graph_wrapper = RecordingGraph()
    adapter._estimator_step(
        adapter.flow.decoder.estimator,
        x=torch.ones((2, 1, 2)),
        mu=torch.ones((2, 1, 2)),
        time=torch.zeros(2),
        speakers=torch.ones((2, 1)),
        cond=torch.zeros((2, 1, 2)),
        cnn_cache=torch.zeros((1, 2, 2, 1)),
        att_cache=torch.zeros((1, 2, 1, 1, 2), dtype=torch.bfloat16),
    )

    assert seen_dtypes == [torch.float32]


def test_npu_sdpa_context_uses_platform_patch(monkeypatch):
    entered: list[str] = []

    @contextmanager
    def recording_context():
        entered.append("enter")
        yield
        entered.append("exit")

    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms.npu.models.step_audio2_token2wav",
        SimpleNamespace(npu_token2wav_sdpa_context=recording_context),
    )
    with _token2wav_sdpa_context(SimpleNamespace(type="npu")):
        assert entered == ["enter"]

    assert entered == ["enter", "exit"]


def test_ragged_outputs_fail_closed_on_middle_hole():
    adapter = BatchedToken2Wav(_FakeToken2Wav())
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 3)
    audios: list[torch.Tensor | None] = [
        torch.zeros(1),
        None,
        torch.ones(1),
    ]
    next_states = [states[0], None, states[2]]

    with pytest.raises(
        RuntimeError,
        match=r'"reason":"incomplete_ragged_output","audio_rows":\[1\],"state_rows":\[1\]',
    ):
        adapter._require_complete_ragged_outputs(audios, next_states)


@pytest.mark.parametrize(
    "bfloat16_attention_cache",
    [False, True],
    ids=["fp32-cache", "bf16-cache"],
)
def test_ragged_dit_two_step_decode_matches_per_row_exact_decode(
    bfloat16_attention_cache: bool,
):
    from cosyvoice2.flow.decoder_dit import DiT

    torch.manual_seed(17)
    template = DiT(
        in_channels=4,
        out_channels=1,
        depth=2,
        num_heads=2,
        head_dim=2,
        hidden_size=4,
    ).eval()
    template_state = {name: value.detach().clone() for name, value in template.state_dict().items()}
    rand_noise = torch.randn(1, 1, 32)

    def make_adapter() -> BatchedToken2Wav:
        token2wav = _FakeToken2Wav()
        estimator = DiT(
            in_channels=4,
            out_channels=1,
            depth=2,
            num_heads=2,
            head_dim=2,
            hidden_size=4,
        ).eval()
        estimator.load_state_dict(template_state)
        token2wav.flow.decoder.estimator = estimator
        token2wav.flow.decoder.rand_noise = rand_noise.clone()
        return BatchedToken2Wav(
            token2wav,
            bfloat16_attention_cache=bfloat16_attention_cache,
        )

    ragged_adapter = make_adapter()
    ragged_prompt = ragged_adapter.prepare_prompt("shared", "/fake/prompt.wav")
    ragged_states = ragged_adapter.setup_batch(ragged_prompt, 2)

    exact_adapters = [make_adapter(), make_adapter()]
    exact_prompts = [adapter.prepare_prompt("shared", "/fake/prompt.wav") for adapter in exact_adapters]
    exact_states = [
        adapter.setup_batch(prompt, 1) for adapter, prompt in zip(exact_adapters, exact_prompts, strict=True)
    ]
    tolerance = (2e-2, 2e-3) if bfloat16_attention_cache else (1e-5, 1e-6)
    steps = [
        (
            [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22])],
            [False, True],
        ),
        (
            [torch.tensor([13, 14]), torch.tensor([22, 23, 24])],
            [True, True],
        ),
    ]
    first_step_history_width: int | None = None

    for step, (tokens, last_chunks) in enumerate(steps):
        if step == 1:
            assert first_step_history_width is not None
            assert all(
                int(state.flow_cache["estimator_att_cache"].shape[4]) == first_step_history_width
                for state in ragged_states
            )
        ragged_audios, ragged_states = ragged_adapter.decode_ragged_batch(
            tokens,
            ragged_prompt,
            ragged_states,
            last_chunks=last_chunks,
        )
        for row, (adapter, prompt) in enumerate(zip(exact_adapters, exact_prompts, strict=True)):
            exact_audios, exact_states[row] = adapter.decode_batch(
                tokens[row].unsqueeze(0),
                prompt,
                exact_states[row],
                last_chunk=last_chunks[row],
            )
            torch.testing.assert_close(
                ragged_audios[row],
                exact_audios[0],
                rtol=tolerance[0],
                atol=tolerance[1],
            )
            for cache_name, expected in exact_states[row][0].flow_cache.items():
                actual = ragged_states[row].flow_cache[cache_name]
                assert actual.dtype == expected.dtype
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=tolerance[0],
                    atol=tolerance[1],
                )
            for cache_name, expected in exact_states[row][0].hift_cache.items():
                torch.testing.assert_close(
                    ragged_states[row].hift_cache[cache_name],
                    expected,
                    rtol=tolerance[0],
                    atol=tolerance[1],
                )

        estimator_att = ragged_states[0].flow_cache["estimator_att_cache"]
        assert estimator_att.shape[4] > 0
        if step == 0:
            first_step_history_width = int(estimator_att.shape[4])
        else:
            assert first_step_history_width is not None
            assert int(estimator_att.shape[4]) > first_step_history_width


def test_fade_in_out_limits_overlap_to_available_previous_audio():
    speech = torch.arange(6, dtype=torch.float32).reshape(1, -1)
    previous = torch.full((1, 3), 2.0)
    window = torch.hamming_window(8, periodic=False)

    actual = BatchedToken2Wav._fade_in_out(speech, previous, window)

    expected = speech.clone()
    expected[..., :3] = speech[..., :3] * window[:3] + previous * window[-3:]
    torch.testing.assert_close(actual, expected)


def test_estimator_cache_stack_split_round_trip_preserves_cfg_rows():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)
    _, states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_cnn_cache"].shape[2] == 4
    assert stacked["estimator_att_cache"].shape[2] == 4
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        torch.testing.assert_close(
            round_tripped["estimator_cnn_cache"],
            original.flow_cache["estimator_cnn_cache"],
        )
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )


def test_bfloat16_estimator_attention_cache_materializes_only_current_timestep():
    token2wav = _FakeToken2Wav()
    adapter = BatchedToken2Wav(token2wav, bfloat16_attention_cache=True)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 2)

    assert all(state.flow_cache["estimator_att_cache"].dtype == torch.bfloat16 for state in states)
    assert (
        states[0].flow_cache["estimator_att_cache"].data_ptr() != states[1].flow_cache["estimator_att_cache"].data_ptr()
    )

    stacked = adapter._stack_flow_cache(states)
    assert stacked["estimator_att_cache"].dtype == torch.bfloat16
    restored = adapter._split_flow_cache(stacked, 2)
    for original, round_tripped in zip(states, restored, strict=True):
        assert round_tripped["estimator_att_cache"].dtype == torch.bfloat16
        torch.testing.assert_close(
            round_tripped["estimator_att_cache"],
            original.flow_cache["estimator_att_cache"],
        )

    _, next_states = adapter.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )
    assert all(state.flow_cache["estimator_att_cache"].dtype == torch.bfloat16 for state in next_states)
    assert token2wav.flow.decoder.estimator.attention_cache_dtypes[-2:] == [torch.float32, torch.float32]


def test_bfloat16_estimator_attention_cache_materializes_each_timestep():
    adapter = BatchedToken2Wav(_FakeToken2Wav(), bfloat16_attention_cache=True)

    _, estimator_cnn, estimator_att = adapter._decode_cfm(
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.ones((2, 1)),
        torch.zeros((2, 1, 2)),
        cnn_cache=None,
        att_cache=None,
    )

    assert estimator_cnn.dtype == torch.float32
    assert estimator_att.dtype == torch.bfloat16
    assert estimator_att.shape[:3] == (2, 1, 4)


def test_float32_estimator_attention_cache_materializes_each_timestep():
    adapter = BatchedToken2Wav(_FakeToken2Wav())

    _, estimator_cnn, estimator_att = adapter._decode_cfm(
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.ones((2, 1)),
        torch.zeros((2, 1, 2)),
        cnn_cache=None,
        att_cache=None,
    )

    assert estimator_cnn.dtype == torch.float32
    assert estimator_att.dtype == torch.float32
    assert estimator_att.shape[:3] == (2, 1, 4)


def test_model_preserves_output_slots_and_prefers_runtime_codes():
    model, token2wav = _model()
    output = _forward(
        model,
        [_info("a", 0, [10, 11]), _info("b", 0, [20, 21])],
        placeholder_counts=[3, 1],
    )

    audios = output.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert len(output.multimodal_outputs["sr"]) == 2
    assert all(sr.item() == 24000 for sr in output.multimodal_outputs["sr"])
    assert all(audio.dtype == torch.float32 for audio in audios)
    # Fake CFM uses two Euler steps whose deltas sum to one. Its conditional
    # row is mu and its unconditional row is zero, so CFG produces 1.7 * mu.
    torch.testing.assert_close(audios[0][0], torch.tensor(1.7 * 10))
    torch.testing.assert_close(audios[1][0], torch.tensor(1.7 * 20))
    assert token2wav.flow.encoder.calls[-1] == 2


def test_code2wav_projects_duplex_metadata_to_final_audio_output():
    model, token2wav = _model()
    segment = _info("duplex", 0, [10, 11])
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    segment["meta"].update(
        {
            "duplex_epoch": 3,
            "duplex_turn_id": 7,
            "llm_output_text_utf8": segment_text_utf8,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    segment_output = _forward(model, [segment])

    assert segment_output.multimodal_outputs["meta.turn_end"][0].item() is False
    # A Talker unit boundary only drains pending codec tokens. The official
    # streaming path keeps Token2wav open until the assistant turn ends.
    assert token2wav.flow.encoder.last_chunk_calls[-1] is False
    assert "duplex" in model._states

    final = _info("duplex", 1, [12, 13], last_chunk=True)
    final["meta"].update(segment["meta"])
    final["meta"]["chunk_seq"] = 1
    final["meta"]["last_chunk"] = True
    final["meta"]["turn_end"] = True
    output = _forward(model, [final])

    payload = output.multimodal_outputs
    assert "meta" not in payload
    assert payload["meta.duplex_epoch"][0].item() == 3
    assert payload["meta.duplex_turn_id"][0].item() == 7
    torch.testing.assert_close(
        payload["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert payload["meta.tts_is_last_chunk"][0].item() is True
    assert payload["meta.turn_end"][0].item() is True
    assert token2wav.flow.encoder.last_chunk_calls[-1] is True
    assert "duplex" not in model._states


def test_initial_empty_segment_marker_initializes_stream_without_audio():
    model, token2wav = _model()
    boundary = _info("duplex", 0, [])
    boundary["meta"].update(
        {
            "code_flat_numel": 0,
            "tts_is_last_chunk": True,
            "turn_end": False,
        }
    )

    output = _forward(model, [boundary])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert "duplex" in model._states
    assert token2wav.hift.calls == []

    resumed = _info(
        "duplex",
        1,
        [4218, 4218, 4218, 10, 11, 12, 13, 14],
    )
    output = _forward(model, [resumed])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert "duplex" in model._states


def test_initial_empty_segment_markers_respect_initial_batch_limit():
    model, token2wav = _model(initial=2)
    names = ["a", "b", "c", "d", "e"]
    boundaries = []
    for name in names:
        boundary = _info(name, 0, [])
        boundary["meta"].update(
            {
                "code_flat_numel": 0,
                "tts_is_last_chunk": True,
                "turn_end": False,
            }
        )
        boundaries.append(boundary)

    output = _forward(model, boundaries)

    assert token2wav.flow.encoder.calls == [2, 2, 1]
    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0] * len(names)
    assert set(model._states) == set(names)


def test_shared_runtime_prompt_recreates_missing_file_before_second_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    first = _info("voice-a", 0, [10, 11])
    first["codes"]["ref"] = reference
    first["meta"]["ref_audio_sr"] = 16000
    first["meta"].pop("prompt_cache_id")
    _forward(model, [first], request_ids=["internal-a"])

    prompt_key = model._request_prompt_keys["internal-a"]
    prompt_path = Path(model._runtime_prompts[prompt_key].path)
    prompt_path.unlink()

    second = _info("voice-b", 0, [12, 13])
    second["codes"]["ref"] = reference
    second["meta"]["ref_audio_sr"] = 16000
    second["meta"].pop("prompt_cache_id")
    _forward(model, [second], request_ids=["internal-b"])

    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-a", "internal-b"}

    model.on_requests_finished(["internal-a"])
    assert prompt_path.is_file()
    assert model._runtime_prompts[prompt_key].owners == {"internal-b"}

    model.on_requests_finished(["internal-b"])
    assert not prompt_path.exists()
    assert prompt_key not in model._runtime_prompts


def test_runtime_prompt_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def fail_after_partial_write(path, *_args, **_kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.sf.write",
        fail_after_partial_write,
    )

    with pytest.raises(OSError, match="simulated write failure"):
        model._materialize_runtime_prompt(reference, 16000)

    assert len(model._runtime_prompts) == 1
    entry = next(iter(model._runtime_prompts.values()))
    assert not Path(entry.path).exists()
    assert list(Path(entry.path).parent.iterdir()) == []


def test_runtime_prompt_files_are_isolated_between_model_instances(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    first_model, _ = _model()
    second_model, _ = _model()
    reference = torch.tensor([0.0, 0.25, -0.25, 0.0])

    def runtime_ref_info(request_id: str):
        info = _info(request_id, 0, [10, 11])
        info["codes"]["ref"] = reference
        info["meta"]["ref_audio_sr"] = 16000
        info["meta"].pop("prompt_cache_id")
        return info

    _forward(first_model, [runtime_ref_info("voice-a")], request_ids=["internal-a"])
    _forward(second_model, [runtime_ref_info("voice-b")], request_ids=["internal-b"])

    first_key = first_model._request_prompt_keys["internal-a"]
    second_key = second_model._request_prompt_keys["internal-b"]
    first_path = Path(first_model._runtime_prompts[first_key].path)
    second_path = Path(second_model._runtime_prompts[second_key].path)
    assert first_key == second_key
    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()

    first_model.on_requests_finished(["internal-a"])
    assert not first_path.exists()
    assert second_path.is_file()

    second_model.on_requests_finished(["internal-b"])
    assert not second_path.exists()


def test_mixed_final_exact_buckets_keep_order_and_release_only_final_states():
    model, _ = _model()
    _forward(
        model,
        [_info(name, 0, [index + 1, index + 2]) for index, name in enumerate(("a", "b", "c", "d"))],
    )
    output = _forward(
        model,
        [
            _info("a", 1, [11, 12]),
            _info("c", 1, [31, 32, 33], last_chunk=True),
            _info("b", 1, [21, 22]),
            _info("d", 1, [41, 42, 43], last_chunk=True),
        ],
    )

    audios = output.multimodal_outputs["model_outputs"]
    window = torch.hamming_window(4, periodic=False)
    overlap_scale = 1.7 * (window[0] + window[2])
    expected = torch.tensor([1, 3, 2, 4], dtype=torch.float32) * overlap_scale
    actual = torch.stack([audio[0] for audio in audios])
    torch.testing.assert_close(actual, expected)
    assert set(model._states) == {"a", "b"}


def test_empty_final_sentinel_emits_empty_and_releases_state_without_compute():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    hift_calls = list(token2wav.hift.calls)
    output = _forward(
        model,
        [
            _info("a", 1, [], last_chunk=True),
            _info("b", 1, [], last_chunk=True),
        ],
    )

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}
    assert token2wav.hift.calls == hift_calls


def test_empty_final_ignores_generation_scheduler_placeholder_token():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    infos = [_info("a", 1, [], last_chunk=True), _info("b", 1, [], last_chunk=True)]
    for info in infos:
        info.pop("codes")
        info["meta"]["code_flat_numel"] = 0

    output = _forward(model, infos, placeholder_counts=[1, 1])

    assert [audio.numel() for audio in output.multimodal_outputs["model_outputs"]] == [0, 0]
    assert model._states == {}


@pytest.mark.parametrize(
    "info",
    [
        # The runner injects the engine request id on every step (GPU
        # _preprocess, NPU _gather_runtime_additional_information)...
        {"request_id": "a", "meta": {"request_id": "a"}},
        # Runtime snapshot bookkeeping alone is not a Talker payload.
        {
            "request_id": "a",
            "meta": {
                "request_id": "a",
                "num_processed_tokens": 0,
                "resumable": True,
            },
        },
        # ...but a pre-warm step can also reach the model with nothing at all.
        {},
    ],
)
def test_prewarm_placeholder_step_emits_silence_without_touching_state(info):
    # async-chunk pre-warm submits Stage 2 with a reserved placeholder prompt.
    # If it gets scheduled before the first codec window lands, those reserved
    # tokens must neither be vocoded nor held to the codec payload contract.
    model, token2wav = _model()

    output = _forward(model, [info], request_ids=["a"])

    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    assert model._states == {}
    assert token2wav.hift.calls == []


def test_metadata_only_payload_still_decodes_codec_from_prompt_tokens():
    # The connector strips 1-D codec tensors out of additional_information and
    # leaves them in the prompt tokens, so a real chunk reaches the model as
    # producer metadata plus input ids. It must still be vocoded.
    model, _ = _model()
    info = {
        "request_id": "a",
        "meta": {
            "request_id": "a",
            "chunk_seq": 0,
            "code_flat_numel": 2,
            "prompt_cache_id": "shared",
        },
    }

    output = _forward(model, [info], placeholder_counts=[2])

    assert output.multimodal_outputs["model_outputs"][0].numel() > 0
    assert set(model._states) == {"a"}


def test_non_final_chunk_shorter_than_lookahead_window_is_rejected():
    token2wav = _FakeToken2Wav()
    token2wav.flow.encoder.pre_lookahead_layer = SimpleNamespace(pre_lookahead_len=3)
    adapter = BatchedToken2Wav(token2wav)
    prompt = adapter.prepare_prompt("shared", "/fake/prompt.wav")
    states = adapter.setup_batch(prompt, 1)

    with pytest.raises(RuntimeError, match="chunk_below_lookahead_window"):
        adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=False)

    # The final chunk is zero-padded by the encoder, so it stays decodable.
    audios, _ = adapter.decode_batch(torch.tensor([[10]]), prompt, states, last_chunk=True)
    assert len(audios) == 1


def test_forward_builds_backend_when_weight_loading_was_skipped(monkeypatch):
    # load_format=dummy never calls load_weights(), so Stage 2 would otherwise
    # reach its first request with no Token2wav assets at all.
    model = MiniCPMO45Code2Wav(vllm_config=_config())
    token2wav = _FakeToken2Wav()
    builds = 0

    def build_backend():
        nonlocal builds
        builds += 1
        model.backend = BatchedToken2Wav(token2wav)

    monkeypatch.setattr(model, "_build_backend", build_backend)

    output = _forward(model, [_info("a", 0, [10, 11])])
    _forward(model, [_info("a", 1, [12, 13])])

    assert builds == 1
    assert output.multimodal_outputs["model_outputs"][0].numel() > 0


@pytest.mark.parametrize(
    ("info", "reason"),
    [
        (_info("a", 0, [1, 2], cache_epoch=-1), "negative_stream_position"),
        (_info("a", 0, [1, 2]), "stale_or_reordered_chunk"),
        (_info("a", 2, [1, 2]), "stale_or_reordered_chunk"),
    ],
)
def test_stale_epoch_and_reordered_chunks_are_rejected(info, reason):
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])

    with pytest.raises(RuntimeError, match=reason):
        _forward(model, [info, _info("b", 1, [3, 4])])


def test_singleton_and_mixed_shape_buckets_use_same_batched_backend_without_fallback():
    model, token2wav = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    output = _forward(model, [_info("a", 1, [5, 6]), _info("b", 1, [7, 8, 9])])

    assert len(output.multimodal_outputs["model_outputs"]) == 2
    # Exact-shape buckets execute independently but both use the same vectorized
    # adapter; there is no Token2wav.stream/__call__ fallback.
    assert token2wav.hift.calls[-2:] == [1, 1]


def test_initial_batch_limit_allows_later_full_batch():
    model, token2wav = _model(initial=4)
    names = [f"request-{index}" for index in range(8)]

    for chunk_seq in (0, 1):
        infos = [_info(name, chunk_seq, [1, 2]) for name in names]
        _forward(model, infos)
        assert token2wav.hift.calls[-2:] == [4, 4]

    token2wav.hift.calls.clear()
    _forward(model, [_info(name, 2, [1, 2]) for name in names])
    assert token2wav.hift.calls == [8]


def test_initial_batch_partition_preserves_minimum():
    model, token2wav = _model(initial=4, minimum=3)
    names = [f"request-{index}" for index in range(6)]

    _forward(model, [_info(name, 0, [1, 2]) for name in names])

    assert token2wav.hift.calls == [3, 3]


def test_initial_batch_partition_rejects_impossible_remainder():
    model, _ = _model(initial=4, minimum=3)
    names = [f"request-{index}" for index in range(5)]

    with pytest.raises(RuntimeError, match="initial_batch_partition_below_minimum"):
        _forward(model, [_info(name, 0, [1, 2]) for name in names])


@pytest.mark.parametrize("chunk_seqs", [[0, 2, 2, 2], [0, 0, 0, 2]])
def test_initial_steady_partition_rejects_undersized_wave(chunk_seqs):
    model, _ = _model(initial=4, minimum=3)
    bucket = [SimpleNamespace(chunk_seq=chunk_seq) for chunk_seq in chunk_seqs]

    with pytest.raises(RuntimeError, match="decode_wave_below_minimum"):
        list(model._iter_decode_batches([bucket]))


def test_backend_failure_does_not_commit_any_request_state(monkeypatch):
    model, _ = _model()
    _forward(
        model,
        [
            _info("a", 0, [1, 2], cache_epoch=0),
            _info("b", 0, [2, 3], cache_epoch=0),
            _info("c", 0, [3, 4], cache_epoch=1),
            _info("d", 0, [4, 5], cache_epoch=1),
        ],
    )
    before = dict(model._states)
    original = model.backend.decode_batch
    call_count = 0

    def fail(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "decode_batch", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        _forward(
            model,
            [
                _info("a", 1, [5, 6]),
                _info("b", 1, [7, 8]),
                _info("c", 1, [9, 10], cache_epoch=1),
                _info("d", 1, [12, 13], cache_epoch=1),
            ],
        )
    assert call_count == 2
    assert model._states == before


def test_cleanup_and_profile_output_are_aligned():
    model, _ = _model()
    _forward(model, [_info("a", 0, [1, 2]), _info("b", 0, [3, 4])])
    model.on_requests_finished(["a"])
    assert set(model._states) == {"b"}

    profile = model(
        input_ids=torch.zeros(5, dtype=torch.long),
        seq_token_counts=[2, 3],
    )
    assert [audio.numel() for audio in profile.multimodal_outputs["model_outputs"]] == [0, 0]
    assert set(model._states) == {"b"}


def test_cleanup_uses_generation_runner_internal_request_ids():
    model, _ = _model()
    _forward(
        model,
        [_info("external-a", 0, [1, 2]), _info("external-b", 0, [3, 4])],
        request_ids=["internal-a", "internal-b"],
    )

    model.on_requests_finished(["internal-a"])

    assert set(model._states) == {"internal-b"}


def test_reference_voice_and_duplex_metadata_follow_request_lifecycle():
    model, _ = _model()
    first = _info("voice-a", 0, [1, 2])
    first["codes"]["ref"] = torch.linspace(-0.1, 0.1, 160)
    segment_text_utf8 = torch.tensor(list(b"hello"), dtype=torch.uint8)
    first["meta"].update(
        ref_audio_sr=16000,
        llm_output_text_utf8=segment_text_utf8,
        duplex_turn_id=7,
        duplex_epoch=3,
    )
    first["meta"].pop("prompt_cache_id")

    output = _forward(model, [first])
    prompt_key = model._request_prompt_keys["voice-a"]
    prompt = model._runtime_prompts[prompt_key]
    prompt_cache_id, prompt_wav = prompt.cache_id, prompt.path
    assert prompt_cache_id.startswith("runtime-ref-")
    assert Path(prompt_wav).is_file()
    torch.testing.assert_close(
        output.multimodal_outputs["meta.llm_output_text_utf8"][0],
        segment_text_utf8,
    )
    assert output.multimodal_outputs["meta.duplex_turn_id"][0].item() == 7
    assert output.multimodal_outputs["meta.duplex_epoch"][0].item() == 3

    final = _info("voice-a", 1, [3, 4], last_chunk=True)
    final["meta"].pop("prompt_cache_id")
    final["meta"]["tts_is_last_chunk"] = True
    output = _forward(model, [final])

    assert output.multimodal_outputs["meta.tts_is_last_chunk"][0].item() is True
    assert model._request_prompt_keys["voice-a"] == prompt_key
    model.on_requests_finished(["voice-a"])
    assert "voice-a" not in model._request_prompt_keys
    assert prompt_key not in model._runtime_prompts
    assert not Path(prompt_wav).exists()
    assert (prompt_cache_id, prompt_wav) not in model.backend._prompt_features

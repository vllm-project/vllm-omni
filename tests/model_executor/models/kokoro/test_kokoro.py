# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for Kokoro-82M: text chunking, checkpoint loading, the network
and the chunk-per-step generator lifecycle the AR runner drives.

The network runs with random weights at the real Kokoro-82M shapes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.kokoro.configuration_kokoro import (
    KOKORO_CONTINUE_TOKEN_ID,
    KOKORO_STOP_TOKEN_ID,
    KokoroConfig,
)
from vllm_omni.model_executor.models.kokoro.kokoro_modules import KokoroNetwork, fold_weight_norm
from vllm_omni.model_executor.models.kokoro.modeling_kokoro import (
    _COMPONENTS,
    KokoroForConditionalGeneration,
)
from vllm_omni.model_executor.models.kokoro.text_frontend import (
    MAX_PHONEMES,
    KokoroTextFrontend,
    _en_chunks,
    _sentence_chunks,
    phonemes_to_ids,
    resolve_lang_code,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Frames are expanded 2x by the prosody upsampler, then 10 * 6 * 5 by iSTFTNet.
SAMPLES_PER_FRAME = 600

_VOCAB = {p: i + 1 for i, p in enumerate("abcdefghijklmnopqrstuvwxyz .,!?")}


def _config() -> KokoroConfig:
    return KokoroConfig(
        vocab=_VOCAB,
        n_token=178,
        plbert={
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 2048,
            "max_position_embeddings": 512,
            "num_hidden_layers": 12,
            "dropout": 0.1,
        },
        istftnet={
            "upsample_kernel_sizes": [20, 12],
            "upsample_rates": [10, 6],
            "gen_istft_hop_size": 5,
            "gen_istft_n_fft": 20,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_initial_channel": 512,
        },
    )


@pytest.fixture(scope="module")
def network() -> KokoroNetwork:
    torch.manual_seed(0)
    net = KokoroNetwork(_config()).eval()
    net.reset_adain_affine()
    return net


# ---------------------------------------------------------------------------
# Text frontend
# ---------------------------------------------------------------------------


@dataclass
class _Tok:
    """Stand-in for misaki's MToken: only the fields chunking reads."""

    text: str
    phonemes: str
    whitespace: str = " "


def test_en_chunks_keep_short_text_whole() -> None:
    tokens = [_Tok("Hi", "hˈI", ""), _Tok(",", ",", " "), _Tok("you", "jˈu", ""), _Tok(".", ".", "")]
    assert list(_en_chunks(tokens)) == ["hˈI, jˈu."]


def test_en_chunks_split_at_sentence_break_within_budget() -> None:
    word = _Tok("word", "w" * 9)  # 9 phonemes + a space
    sentence = [word] * 20 + [_Tok(".", ".", " ")]  # ~201 phonemes per sentence
    chunks = list(_en_chunks(sentence * 4))
    assert len(chunks) == 2
    assert all(0 < len(c) <= MAX_PHONEMES for c in chunks)
    # The waterfall prefers the last full stop that keeps the chunk in budget.
    assert chunks[0].endswith(".")


def test_en_chunks_without_punctuation_still_respect_budget() -> None:
    chunks = list(_en_chunks([_Tok("word", "w" * 9)] * 120))
    assert len(chunks) == 3
    assert all(len(c) <= MAX_PHONEMES for c in chunks)


def test_sentence_chunks_group_sentences() -> None:
    text = "One. Two! Three? " * 40
    chunks = _sentence_chunks(text, chunk_size=100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(c.replace(" ", "") for c in chunks) == text.replace(" ", "")


def _frontend_with_g2p(lang_code: str, per_char: int) -> KokoroTextFrontend:
    """A frontend whose G2P emits ``per_char`` phonemes per non-space character."""

    def g2p(text: str) -> tuple[str, None]:
        return "".join(ch if ch.isspace() else ch * per_char for ch in text), None

    frontend = KokoroTextFrontend()
    frontend._g2p[lang_code] = g2p
    return frontend


def test_long_non_english_sentence_is_split_not_truncated() -> None:
    """One run-on sentence past the budget is split at clause marks; no speech is lost."""
    sentence = ", ".join(["el gato duerme en la casa grande"] * 30)  # 1018 chars, no . ! ?
    chunks = _frontend_with_g2p("e", 1).phoneme_chunks(sentence, "e")
    assert len(chunks) > 1
    assert all(0 < len(c) <= MAX_PHONEMES for c in chunks)
    assert "".join(chunks).replace(" ", "") == sentence.replace(" ", "")


def test_full_width_sentence_ends_split_cjk_text() -> None:
    text = "今天天气很好我们去公园散步吧。" * 30  # 450 chars, 1350 phonemes
    chunks = _frontend_with_g2p("z", 3).phoneme_chunks(text, "z")
    assert all(0 < len(c) <= MAX_PHONEMES for c in chunks)
    assert all(c.endswith("。" * 3) for c in chunks)
    assert "".join(chunks) == "".join(ch * 3 for ch in text)


def test_non_english_chunk_that_fits_is_phonemized_whole() -> None:
    text = "Hola, ¿qué tal? Muy bien."
    assert _frontend_with_g2p("e", 1).phoneme_chunks(text, "e") == [text]


@pytest.mark.parametrize(
    ("voice", "language", "expected"),
    [
        ("af_heart", None, "a"),
        ("bm_george", None, "b"),
        ("ef_dora", None, "e"),
        ("zf_xiaobei", None, "z"),
        ("af_heart", "en-GB", "b"),
        ("af_heart", "fr-fr", "f"),
        ("af_heart", "j", "j"),
    ],
)
def test_resolve_lang_code(voice: str, language: str | None, expected: str) -> None:
    assert resolve_lang_code(voice, language) == expected


@pytest.mark.parametrize(("voice", "language"), [("af_heart", "klingon"), ("xx_unknown", None)])
def test_resolve_lang_code_rejects_unknown(voice: str, language: str | None) -> None:
    with pytest.raises(ValueError):
        resolve_lang_code(voice, language)


def test_phonemes_to_ids_pads_and_drops_unknown() -> None:
    assert phonemes_to_ids("ab#c", {"a": 5, "b": 6, "c": 7}) == [0, 5, 6, 7, 0]


def test_misaki_english_g2p_fits_the_vocab() -> None:
    pytest.importorskip("misaki")
    pytest.importorskip("en_core_web_sm")
    chunks = KokoroTextFrontend().phoneme_chunks("Hello world.\nSecond paragraph here.", "a")
    assert len(chunks) == 2
    assert all(0 < len(c) <= MAX_PHONEMES for c in chunks)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def test_fold_weight_norm_matches_torch_weight_norm() -> None:
    torch.manual_seed(0)
    conv = nn.utils.weight_norm(nn.Conv1d(4, 6, 3))
    state = {k: v.detach().clone() for k, v in conv.state_dict().items()}
    assert {"weight_g", "weight_v", "bias"} <= {k.split(".")[-1] for k in state}
    folded = fold_weight_norm(state, "cpu")
    assert set(folded) == {"weight", "bias"}
    torch.testing.assert_close(folded["weight"], conv.weight.detach(), rtol=0, atol=0)


def _reference_layout_checkpoint(net: KokoroNetwork) -> dict[str, dict[str, torch.Tensor]]:
    """Re-express ``net``'s weights the way ``kokoro-v1_0.pth`` stores them.

    Per-component ``DataParallel`` state dicts (``module.`` prefix), conv
    weights split into weight-norm ``weight_g``/``weight_v``, and no AdaIN
    instance-norm affine params.
    """
    checkpoint = {}
    for component in _COMPONENTS:
        module = getattr(net, component)
        state: dict[str, torch.Tensor] = {}
        for name, tensor in module.state_dict().items():
            if KokoroNetwork.is_adain_affine_param(name):
                continue
            owner = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            if name.endswith(".weight") and isinstance(owner, (nn.Conv1d, nn.ConvTranspose1d)):
                prefix = name[: -len("weight")]
                g = torch.linalg.vector_norm(tensor, dim=tuple(range(1, tensor.dim())), keepdim=True)
                state[f"module.{prefix}weight_g"] = g
                state[f"module.{prefix}weight_v"] = tensor.clone()
            else:
                state[f"module.{name}"] = tensor.clone()
        checkpoint[component] = state
    return checkpoint


class _NoWarmupFrontend(KokoroTextFrontend):
    """Skips the misaki/spaCy warmup that ``load_weights()`` runs."""

    def warmup(self, lang_code: str = "a") -> None:
        pass


def _bare_model(config: KokoroConfig) -> KokoroForConditionalGeneration:
    """A model instance without vLLM engine construction."""
    model = KokoroForConditionalGeneration.__new__(KokoroForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = config
    model.model_path = "unused"
    model.revision = None
    model._device = torch.device("cpu")
    model.network = KokoroNetwork(config).eval()
    model.frontend = _NoWarmupFrontend()
    model._voices_dir = None
    model._voicepacks = {}
    model._stream_gens = {}
    model._finished_keys = set()
    model._ar_last_chunk_flags = []
    return model


def test_load_weights_from_pth_items(network: KokoroNetwork) -> None:
    checkpoint = _reference_layout_checkpoint(network)
    model = _bare_model(_config())
    # vLLM's .pt iterator yields the checkpoint's top-level items.
    loaded = model.load_weights(iter(checkpoint.items()))
    assert loaded == {f"network.{n}" for n, _ in model.network.named_parameters()}
    for (name, got), (_, want) in zip(model.network.state_dict().items(), network.state_dict().items()):
        torch.testing.assert_close(got, want, msg=name)


def test_load_weights_rejects_mismatched_checkpoint(network: KokoroNetwork) -> None:
    checkpoint = _reference_layout_checkpoint(network)
    model = _bare_model(_config())
    with pytest.raises(ValueError, match="missing components"):
        model.load_weights(iter({k: v for k, v in checkpoint.items() if k != "decoder"}.items()))
    checkpoint["bert_encoder"]["module.extra"] = torch.zeros(1)
    with pytest.raises(ValueError, match="bert_encoder"):
        model.load_weights(iter(checkpoint.items()))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def _inputs(length: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    input_ids = torch.tensor([[0, *torch.randint(1, 30, (length,)).tolist(), 0]])
    ref_s = torch.randn(1, 256) * 0.1
    return input_ids, ref_s


@torch.inference_mode()
def test_network_audio_length_follows_durations(network: KokoroNetwork) -> None:
    input_ids, ref_s = _inputs()
    audio, pred_dur = network(input_ids, ref_s, 1.0, torch.Generator().manual_seed(0))
    assert audio.dim() == 1
    assert pred_dur.shape == (input_ids.shape[1],)
    assert int(pred_dur.min()) >= 1
    assert audio.numel() == int(pred_dur.sum()) * SAMPLES_PER_FRAME
    assert torch.isfinite(audio).all()


@torch.inference_mode()
def test_network_is_deterministic_per_generator(network: KokoroNetwork) -> None:
    input_ids, ref_s = _inputs()
    a, _ = network(input_ids, ref_s, 1.0, torch.Generator().manual_seed(7))
    b, _ = network(input_ids, ref_s, 1.0, torch.Generator().manual_seed(7))
    c, _ = network(input_ids, ref_s, 1.0, torch.Generator().manual_seed(8))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


@torch.inference_mode()
def test_speed_shortens_durations(network: KokoroNetwork) -> None:
    input_ids, ref_s = _inputs()
    _, slow = network(input_ids, ref_s, 0.5, torch.Generator().manual_seed(0))
    _, fast = network(input_ids, ref_s, 2.0, torch.Generator().manual_seed(0))
    assert int(fast.sum()) < int(slow.sum())


# ---------------------------------------------------------------------------
# Generator lifecycle under the AR runner
# ---------------------------------------------------------------------------


class _CountingModel:
    """Wraps a bare model so each request yields ``n`` scripted chunks."""

    def __init__(self, chunks_per_request: dict[str, int]):
        self.model = _bare_model(_config())
        self.created: list[str] = []
        self.closed: list[str] = []

        def create(info):
            text = info["text"][0]
            self.created.append(text)
            n = chunks_per_request[text]
            try:
                for i in range(n):
                    yield torch.full((10,), float(i)), i == n - 1
            finally:
                self.closed.append(text)

        self.model._create_stream_gen = create

    def step(self, infos):
        out = self.model.forward(
            input_ids=torch.zeros(len(infos), dtype=torch.long), runtime_additional_information=infos
        )
        logits = self.model.compute_logits(out.text_hidden_states)
        return out.multimodal_outputs["model_outputs"], logits.argmax(dim=-1).tolist()


def _info(req_id: str, text: str) -> dict:
    # The engine list-wraps additional_information values, including its own ids.
    return {"text": [text], "global_request_id": [req_id]}


def test_chunks_stream_one_per_step_and_stop_on_last() -> None:
    m = _CountingModel({"two": 2, "one": 1})
    infos = [_info("r1", "two"), _info("r2", "one")]
    outputs, tokens = m.step(infos)
    assert [o.tolist() for o in outputs] == [[0.0] * 10, [0.0] * 10]
    assert tokens == [KOKORO_CONTINUE_TOKEN_ID, KOKORO_STOP_TOKEN_ID]
    outputs, tokens = m.step([_info("r1", "two")])
    assert outputs[0].tolist() == [1.0] * 10
    assert tokens == [KOKORO_STOP_TOKEN_ID]
    assert m.created == ["two", "one"]


def test_late_step_after_last_chunk_does_not_resynthesize() -> None:
    """Async scheduling runs one more step before the stop token is seen."""
    m = _CountingModel({"one": 1})
    m.step([_info("r1", "one")])
    outputs, tokens = m.step([_info("r1", "one")])
    assert outputs[0].numel() == 0
    assert tokens == [KOKORO_STOP_TOKEN_ID]
    assert m.created == ["one"]
    # Finishing the request releases its bookkeeping.
    m.model.on_requests_finished({"r1"})
    assert not m.model._finished_keys
    assert not m.model._stream_gens


def test_aborted_request_closes_its_generator() -> None:
    m = _CountingModel({"three": 3})
    m.step([_info("r1", "three")])
    assert "r1" in m.model._stream_gens
    m.model.on_requests_finished(["r1"])
    assert "r1" not in m.model._stream_gens
    assert m.closed == ["three"]


def test_rows_without_a_request_id_finish_without_synthesis() -> None:
    """Profiling rows, and rows whose buffer is already gone, carry no id."""
    m = _CountingModel({})
    outputs, tokens = m.step([{}, {}, {}])
    assert all(o.numel() == 0 for o in outputs)
    assert tokens == [KOKORO_STOP_TOKEN_ID] * 3
    assert m.created == []


def test_failed_request_is_flagged_and_stops() -> None:
    """A failure ends only its request, with meta.error set so serving answers 500."""
    m = _CountingModel({"ok": 2})
    create_ok = m.model._create_stream_gen

    def create(info):
        if info["text"][0] == "bad":
            yield None, True
        else:
            yield from create_ok(info)

    m.model._create_stream_gen = create
    out = m.model.forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        runtime_additional_information=[_info("r1", "ok"), _info("r2", "bad")],
    )
    assert [o.numel() for o in out.multimodal_outputs["model_outputs"]] == [10, 0]
    assert [e.tolist() for e in out.multimodal_outputs["meta"]["error"]] == [[0], [1]]
    tokens = m.model.compute_logits(out.text_hidden_states).argmax(dim=-1).tolist()
    assert tokens == [KOKORO_CONTINUE_TOKEN_ID, KOKORO_STOP_TOKEN_ID]
    # A step without failures carries no flag.
    out = m.model.forward(
        input_ids=torch.zeros(1, dtype=torch.long), runtime_additional_information=[_info("r1", "ok")]
    )
    assert "meta" not in out.multimodal_outputs

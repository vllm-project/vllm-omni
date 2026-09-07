# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("x_transformers")

from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (  # noqa: E402
    MingFlashOmniTalkerForConditionalGeneration,
    _normalize_last_hidden_for_step,
    _replace_hf_config,
    _sample_request_noise,
    _stop_decision_mask,
    _strip_model_prefix,
)
from vllm_omni.model_executor.models.ming_flash_omni.talker_request_state import (  # noqa: E402
    MingTalkerStateManager,
)
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import (  # noqa: E402
    build_ming_talker_prompt_token_ids_for_info,
    stamp_ming_talker_voice_meta,
)


class _FakeVllmConfig(SimpleNamespace):
    def __init__(self):
        super().__init__(model_config=SimpleNamespace(hf_config="root", hf_text_config="root-text"))
        self.calls = []

    def with_hf_config(self, hf_config):
        self.calls.append(hf_config)
        cloned = copy.copy(self)
        cloned.model_config = SimpleNamespace(hf_config=hf_config, hf_text_config="old")
        return cloned


def test_replace_hf_config_prefers_vllm_helper_and_sets_text_config():
    vllm_config = _FakeVllmConfig()

    replaced = _replace_hf_config(vllm_config, "talker-qwen2")

    assert replaced is not vllm_config
    assert vllm_config.calls == ["talker-qwen2"]
    assert replaced.model_config.hf_config == "talker-qwen2"
    assert replaced.model_config.hf_text_config == "talker-qwen2"


def test_strip_model_prefix_only_strips_backbone_weights():
    weights = [
        ("model.layers.0.weight", torch.empty(1)),
        ("aggregator.proj.weight", torch.empty(1)),
    ]

    stripped = [(name, tensor.shape) for name, tensor in _strip_model_prefix(weights)]

    assert stripped == [
        ("layers.0.weight", torch.Size([1])),
        ("aggregator.proj.weight", torch.Size([1])),
    ]


def test_stop_decision_mask_accepts_probabilities_and_logits():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    logits = torch.tensor([[3.0, -1.0], [-2.0, 0.5]])

    assert torch.equal(_stop_decision_mask(probs), torch.tensor([False, True]))
    assert torch.equal(_stop_decision_mask(logits), torch.tensor([False, True]))


def test_normalize_last_hidden_for_step_shapes():
    assert _normalize_last_hidden_for_step(torch.zeros(4)).shape == (1, 1, 4)
    assert _normalize_last_hidden_for_step(torch.zeros(3, 4)).shape == (1, 1, 4)
    assert _normalize_last_hidden_for_step(torch.zeros(2, 3, 4)).shape == (2, 1, 4)


def test_state_manager_creates_seeded_request_state_and_evicts():
    manager = MingTalkerStateManager()

    state = manager.create(
        "req-a",
        his_lat=torch.zeros(1, 2, 3),
        min_steps=1,
        max_steps=5,
        seed=123,
    )

    assert state.req_id == "req-a"
    assert state.seed == 123
    assert state.generator is not None
    assert "req-a" in manager

    manager.evict("req-a")
    assert "req-a" not in manager


def test_sample_request_noise_uses_request_local_generator():
    manager = MingTalkerStateManager()
    left = manager.create("req-a", seed=7)
    right = manager.create("req-b", seed=7)

    left_noise = _sample_request_noise(
        left,
        steps=2,
        patch_size=3,
        latent_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    right_noise = _sample_request_noise(
        right,
        steps=2,
        patch_size=3,
        latent_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(left_noise[0], right_noise[0])
    assert torch.equal(left_noise[1], right_noise[1])


class _TinyTokenizer:
    def __init__(self):
        self._ids = {}

    def encode(self, text):
        if text not in self._ids:
            self._ids[text] = len(self._ids) + 1
        return [self._ids[text]]


def test_build_prompt_slots_use_native_voice_metadata():
    tokenizer = _TinyTokenizer()
    info = {
        "ming_task": "instruct",
        "prompt": "system prompt",
        "instruction": "calm voice",
        "native_talker_prompt_text": "reference text",
        "native_talker_prompt_wav_len": 3,
        "native_talker_spk_emb_count": 2,
    }

    ids = build_ming_talker_prompt_token_ids_for_info(
        text="hello",
        additional_info=info,
        tokenizer=tokenizer,
    )

    assert ids is not None
    assert ids.count(tokenizer.encode("<audioPatch>")[0]) == 3
    assert ids.count(tokenizer.encode("<|vision_start|>")[0]) == 2
    assert tokenizer.encode("reference text")[0] in ids


def test_ming_prompt_wav_len_matches_audio_vae_frame_count():
    from vllm_omni.model_executor.models.ming_flash_omni.talker_module import ming_prompt_wav_len

    # 320 samples/latent * 4 latents/patch * 2 patches/token = 2560 samples/token.
    geometry = {"hop_size": 320, "vae_patch_size": 4, "patch_size": 2}
    assert ming_prompt_wav_len(2560, **geometry) == 1
    # Partial groups are zero-padded up to a whole token, matching _build_wav_embeddings.
    assert ming_prompt_wav_len(2561, **geometry) == 2
    assert ming_prompt_wav_len(5120, **geometry) == 2
    assert ming_prompt_wav_len(0, **geometry) == 0


def test_load_voice_manifest_entries_normalizes_single_and_multi_clip(tmp_path):
    from vllm_omni.model_executor.models.ming_flash_omni.voice_presets import _load_voice_manifest_entries

    (tmp_path / "voice_name.json").write_text(
        '{"A": {"prompt_wav_path": " a.wav ", "prompt_text": "ta"}, '
        '"B": {"prompt_wav_path": ["b1.wav", "/abs/b2.wav"]}, '
        '"C": {"prompt_text": "no wav"}}'
    )

    entries = _load_voice_manifest_entries(str(tmp_path / "voice_name.json"), str(tmp_path))

    # Paths are stripped and made absolute, exactly as VoicePresetRegistry.register does.
    assert entries == [
        ("A", [str(tmp_path / "a.wav")], "ta"),
        ("B", [str(tmp_path / "b1.wav"), "/abs/b2.wav"], ""),
    ]


def test_resolve_voice_preset_meta_sums_multi_clip_samples(tmp_path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    from vllm_omni.model_executor.models.ming_flash_omni import voice_presets

    # 2560 samples/token; two clips of 3000 + 1600 = 4600 -> ceil(4600/2560) = 2 tokens.
    for name, frames in (("c1.wav", 3000), ("c2.wav", 1600)):
        sf.write(str(tmp_path / name), np.zeros(frames, dtype="float32"), 16000)
    (tmp_path / "voice_name.json").write_text(
        '{"DB30": {"prompt_wav_path": ["c1.wav", "c2.wav"], "prompt_text": "ref"}}'
    )

    monkeypatch.setattr(
        voice_presets, "_locate_voice_manifest", lambda *_: (str(tmp_path / "voice_name.json"), str(tmp_path))
    )
    monkeypatch.setattr(voice_presets, "_resolve_prompt_wav_geometry", lambda *_: (16000, 320, 4, 2))
    monkeypatch.setattr(voice_presets, "_resolve_spkemb_path", lambda *_: "campplus.onnx")

    meta = voice_presets.resolve_voice_preset_meta.__wrapped__(str(tmp_path))

    assert meta == {"DB30": {"prompt_text": "ref", "prompt_wav_len": 2, "spk_emb_count": 2}}


def test_resolve_voice_preset_meta_probes_configs_in_the_resolved_local_dir(tmp_path, monkeypatch):
    """Config probing must reuse the directory the manifest lookup resolved.

    Probing a bare repo id instead costs one HF hub round-trip per candidate
    (~3 min on a mirrored hub) even though the snapshot is already on disk.
    """
    from vllm_omni.model_executor.models.ming_flash_omni import voice_presets

    (tmp_path / "voice_name.json").write_text("{}")
    seen: list[str] = []

    monkeypatch.setattr(
        voice_presets, "_locate_voice_manifest", lambda *_: (str(tmp_path / "voice_name.json"), str(tmp_path))
    )
    monkeypatch.setattr(
        voice_presets,
        "_resolve_prompt_wav_geometry",
        lambda talker_dir, model_path: (seen.append(talker_dir), (16000, 320, 4, 2))[1],
    )
    monkeypatch.setattr(voice_presets, "_resolve_spkemb_path", lambda *_: None)

    voice_presets.resolve_voice_preset_meta.__wrapped__("some-org/some-repo")

    assert seen == [str(tmp_path)]


def test_stamp_ming_talker_voice_meta_derives_preset_geometry(monkeypatch):
    from vllm_omni.model_executor.models.ming_flash_omni import voice_presets

    monkeypatch.setattr(
        voice_presets,
        "resolve_voice_preset_meta",
        lambda model_id, download_dir=None: {
            "DB30": {
                "prompt_text": "reference text",
                "prompt_wav_len": 5,
                "spk_emb_count": 1,
            }
        },
    )
    request_info = {"voice_name": "DB30"}
    stage_client = SimpleNamespace(model_config=SimpleNamespace(model="model-a"))

    stamp_ming_talker_voice_meta(request_info, stage_client=stage_client)

    assert request_info["native_talker_prompt_text"] == "reference text"
    assert request_info["native_talker_prompt_wav_len"] == 5
    assert request_info["native_talker_spk_emb_count"] == 1


def test_resolve_voice_injects_preset_only_when_slots_were_reserved():
    talker = MingFlashOmniTalkerForConditionalGeneration.__new__(MingFlashOmniTalkerForConditionalGeneration)
    preset = {
        "prompt_wav_lat": torch.ones(1, 2, 3),
        "prompt_wav_emb": torch.ones(1, 4, 5),
        "spk_emb": torch.ones(1, 6),
        "prompt_text": "reference",
    }
    talker.voice_presets = {"DB30": preset}

    no_slots = talker._resolve_voice({"voice_name": "DB30"})
    assert no_slots.spk_emb is None
    assert no_slots.prompt_wav_lat is None
    assert no_slots.already_projected is False

    with_slots = talker._resolve_voice(
        {
            "voice_name": "DB30",
            "native_talker_prompt_wav_len": 4,
        }
    )
    assert with_slots.spk_emb is preset["spk_emb"]
    assert with_slots.prompt_wav_lat is preset["prompt_wav_lat"]
    assert with_slots.prompt_wav_emb is preset["prompt_wav_emb"]
    assert with_slots.prompt_text == "reference"
    assert with_slots.already_projected is True


def test_audio_finalize_delays_stop_until_next_scheduler_step():
    talker = MingFlashOmniTalkerForConditionalGeneration.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.hidden_size = 4
    talker.state_manager = MingTalkerStateManager()
    talker._pending_requests = [("req-a", True, 1)]
    talker._pending_state_creations = set()
    talker._pending_prefill_done_updates = {}
    talker._results_queue = []
    talker._audio_queue = []

    state = talker.state_manager.create("req-a", his_lat=torch.zeros(1, 2, 3), seed=11)

    def fake_audio_step(step_state, last_hidden):
        assert step_state is state
        return (
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 4),
            torch.tensor([[0.1, 0.9]]),
        )

    talker._talker_audio_step = fake_audio_step
    # Force the per-request stop decision to fire on this step (instance
    # attribute shadows the staticmethod, so it is called unbound).
    talker._request_should_stop = lambda stop_hit, step, min_steps, max_steps: True
    talker._finalize_request = lambda state: SimpleNamespace(multimodal_outputs={"audio": torch.arange(2)})

    talker._run_pending_audio_steps(torch.zeros(1, 4))

    assert state.finished is True
    assert len(talker._audio_queue) == 1
    audio_req_id, audio_payload = talker._audio_queue[0]
    assert audio_req_id == "req-a"
    assert torch.equal(audio_payload["audio"], torch.arange(2))
    assert len(talker._results_queue) == 1
    _, first_logits = talker._results_queue[0]
    assert float(first_logits[0, 0]) == 0.0
    assert torch.isneginf(first_logits[0, 1])

    talker._results_queue.clear()
    talker._pending_requests = [("req-a", True, 1)]
    talker._run_pending_audio_steps(torch.zeros(1, 4))

    _, stop_logits = talker._results_queue[0]
    assert torch.isneginf(stop_logits[0, 0])
    assert float(stop_logits[0, 1]) == 0.0


def test_finished_mask_matches_original_zero_based_min_new_token_boundary():
    should_stop = MingFlashOmniTalkerForConditionalGeneration._request_should_stop

    # A model-signalled stop only counts once (step - 1) > min_steps.
    assert not should_stop(True, step=11, min_steps=10, max_steps=100)
    assert should_stop(True, step=12, min_steps=10, max_steps=100)
    # max_steps forces a stop even without a model stop signal.
    assert should_stop(False, step=100, min_steps=10, max_steps=100)

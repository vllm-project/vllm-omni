# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the Qwen3-Omni audio encoder graph path.

The audio tower is the production class built via ``object.__new__`` with tiny
real convolutions and a window-sensitive attention stand-in, so the chunk
layout, valid-token gather and attention windows are exercised end to end.
"""

from inspect import getattr_static
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata

import vllm_omni.model_executor.models.qwen3_omni.vision_encoder_cudagraph as graph_module
from tests.model_executor.models.qwen3_omni.test_vision_encoder_cudagraph import (
    _combined,
    _config,
    _images,
    _runner_factory,
    _thinker,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
)
from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# n_window=8: 16-frame chunks, 2 tokens per chunk; n_window_infer=64: 8-token windows.
N_WINDOW, N_WINDOW_INFER, MEL = 8, 64, 8


@pytest.fixture(autouse=True)
def _first_pipeline_rank(monkeypatch):
    monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    # Audio graphs are CUDA only; these CPU tests stand in for a CUDA platform unless a case changes it.
    monkeypatch.setattr(graph_module, "current_omni_platform", _platform("cuda"))
    # The tiny tower uses its own chunk size; production lengths assume 100-frame chunks.
    monkeypatch.setattr(
        graph_module,
        "_get_feat_extract_output_lengths",
        lambda lengths: _tiny_output_lengths(int(lengths)),
    )
    import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_module

    monkeypatch.setattr(
        thinker_module, "_get_feat_extract_output_lengths", lambda lengths: _tiny_output_lengths(int(lengths))
    )


def _cnn(length: int) -> int:
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def _tiny_output_lengths(length: int) -> int:
    chunk = 2 * N_WINDOW
    return (length // chunk) * _cnn(chunk) + (_cnn(length % chunk) if length % chunk else 0)


class _Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.output_size = out_features

    def forward(self, x):
        return super().forward(x), None


class _WindowLayer(nn.Module):
    """Adds each attention window's mean, so a wrong window boundary shows."""

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        bounds = cu_seqlens.tolist()
        assert int(max_seqlen) >= max(b - a for a, b in zip(bounds, bounds[1:]))
        out = hidden_states.clone()
        for a, b in zip(bounds, bounds[1:]):
            if b > a:
                out[a:b] += hidden_states[a:b].mean(0)
        return out


def _tower():
    torch.manual_seed(0)
    tower = object.__new__(Qwen3OmniMoeAudioEncoder)
    nn.Module.__init__(tower)
    tower.n_window, tower.n_window_infer, tower.num_mel_bins, tower.conv_chunksize = N_WINDOW, N_WINDOW_INFER, MEL, 500
    tower.conv2d1 = nn.Conv2d(1, 2, 3, 2, padding=1)
    tower.conv2d2 = nn.Conv2d(2, 2, 3, 2, padding=1)
    tower.conv2d3 = nn.Conv2d(2, 2, 3, 2, padding=1)
    tower.conv_out = _Linear(2 * _cnn(MEL), 4)
    tower.positional_embedding = SimpleNamespace(positional_embedding=torch.randn(64, 4))
    tower.layers = nn.ModuleList([_WindowLayer()])
    tower.ln_post = nn.Identity()
    tower.proj1 = _Linear(4, 4)
    tower.act = nn.GELU()
    tower.proj2 = _Linear(4, 6)
    tower.attn_backend = AttentionBackendEnum.FLASH_ATTN
    return tower


def _reference(tower, features, lengths):
    """The pre-refactor forward: device chunk lengths and boolean masks."""
    feature_lens = torch.tensor(lengths)
    chunk_size = tower.n_window * 2
    chunk_num = torch.ceil(feature_lens / chunk_size).long()
    chunk_lengths = torch.tensor([chunk_size] * int(chunk_num.sum()), dtype=torch.long)
    tail = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail] = feature_lens % chunk_size
    chunk_lengths[chunk_lengths == 0] = chunk_size
    padded = nn.utils.rnn.pad_sequence(features.T.split(chunk_lengths.tolist(), dim=0), batch_first=True)
    padded = padded.transpose(1, 2).unsqueeze(1)
    after = torch.tensor([_cnn(int(n)) for n in chunk_lengths])
    mask = torch.arange(int(after.max())).unsqueeze(0) < after.unsqueeze(1)
    embed = F.gelu(tower.conv2d3(F.gelu(tower.conv2d2(F.gelu(tower.conv2d1(padded))))))
    b, c, f, t = embed.size()
    embed, _ = tower.conv_out(embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
    embed = embed + tower.positional_embedding.positional_embedding[:t].unsqueeze(0)
    hidden = embed[mask]
    window = mask.shape[-1] * (tower.n_window_infer // chunk_size)
    cu = [0]
    for n in [_tiny_output_lengths(x) for x in lengths]:
        cu += [window] * (n // window) + ([n % window] if n % window else [])
    cu_seqlens = torch.tensor(cu).cumsum(-1, dtype=torch.int32)
    for layer in tower.layers:
        hidden = layer(hidden, cu_seqlens, (cu_seqlens[1:] - cu_seqlens[:-1]).max())
    hidden, _ = tower.proj1(tower.ln_post(hidden))
    hidden, _ = tower.proj2(tower.act(hidden))
    return hidden


def _audio(lengths, offset=0.0):
    generator = torch.Generator().manual_seed(len(lengths) + sum(lengths))
    return {
        "input_audio_features": torch.randn(MEL, sum(lengths), generator=generator) + offset,
        "audio_feature_lengths": torch.tensor(lengths),
    }


@pytest.mark.parametrize("lengths", [[16], [17], [5], [40], [16, 33, 7], [3, 9], [70, 16, 1]])
def test_host_metadata_forward_matches_the_device_mask_forward(lengths):
    tower = _tower()
    audio = _audio(lengths)
    with torch.no_grad():
        expected = _reference(tower, audio["input_audio_features"], lengths)
        actual = tower.forward_with_encoder_metadata(
            audio["input_audio_features"], tower.prepare_encoder_metadata(lengths)
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _audio_thinker(budgets=(2, 4, 8), max_items=2, **mm_changes):
    cfg = _config()
    cfg.compilation_config.encoder_cudagraph_token_budgets = list(budgets)
    cfg.compilation_config.encoder_cudagraph_max_vision_items_per_batch = max_items
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    for key, value in mm_changes.items():
        setattr(thinker.multimodal_config, key, value)
    thinker._enable_audio_encoder_cudagraph(cfg)
    return thinker


def _platform(name):
    return SimpleNamespace(is_cuda=lambda: name == "cuda", device_name=name)


@pytest.mark.parametrize(
    "change",
    [
        "no_audio_limit",
        "sdpa_backend",
        "fp8_attention",
        "later_pipeline_rank",
        "data_parallel",
        "budget_below_one_chunk",
        "embeds",
        "model_runner_v2",
        "opted_out",
        "opted_out_as_string",
        "rocm_aiter_backend",
        "rocm_platform",
        "npu_platform",
        "xpu_platform",
        None,
    ],
)
def test_audio_graph_gate(monkeypatch, change):
    cfg = _config()
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    mm = thinker.multimodal_config
    if change == "no_audio_limit":
        mm.get_limit_per_prompt = lambda modality: 0 if modality == "audio" else 4
    elif change == "sdpa_backend":
        thinker.audio_tower.attn_backend = AttentionBackendEnum.TORCH_SDPA
    elif change == "fp8_attention":
        mm.mm_encoder_attn_dtype = "fp8"
    elif change == "later_pipeline_rank":
        monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False))
    elif change == "data_parallel":
        mm.mm_encoder_tp_mode = "data"
    elif change == "budget_below_one_chunk":
        cfg.compilation_config.encoder_cudagraph_token_budgets = [1]
    elif change == "embeds":
        mm.enable_mm_embeds = True
    elif change == "model_runner_v2":
        cfg.use_v2_model_runner = True
    elif change == "opted_out":
        cfg.additional_config = {"audio_encoder_cudagraph": False}
    elif change == "opted_out_as_string":
        cfg.additional_config = {"audio_encoder_cudagraph": "false"}
    elif change == "rocm_aiter_backend":
        thinker.audio_tower.attn_backend = AttentionBackendEnum.ROCM_AITER_FA
    elif change in ("rocm_platform", "npu_platform", "xpu_platform"):
        # ROCm returns FLASH_ATTN on CDNA without AITER and on RDNA, so the backend alone does not exclude it.
        monkeypatch.setattr(graph_module, "current_omni_platform", _platform(change.removesuffix("_platform")))
    thinker._enable_audio_encoder_cudagraph(cfg)
    assert ("audio" in thinker.get_encoder_cudagraph_config().modalities) is (change is None)
    # The image path stays enabled either way.
    assert supports_encoder_cudagraph(thinker)


def test_unrecognised_audio_graph_setting_is_rejected():
    cfg = _config()
    cfg.additional_config = {"audio_encoder_cudagraph": "disabled"}
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    with pytest.raises(ValueError, match="audio_encoder_cudagraph"):
        thinker._enable_audio_encoder_cudagraph(cfg)


@pytest.mark.parametrize(
    "graphs_on, case, reason",
    [
        (True, "data_parallel", "the multimodal encoder runs data parallel"),
        (False, "data_parallel", None),
        (True, "rocm", "audio encoder graphs are validated on CUDA only, this platform is rocm"),
        (True, "npu", "audio encoder graphs are validated on CUDA only, this platform is npu"),
        (True, "xpu", "audio encoder graphs are validated on CUDA only, this platform is xpu"),
        (False, "npu", None),
    ],
)
def test_audio_graphs_left_off_are_logged_when_encoder_graphs_are_on(monkeypatch, graphs_on, case, reason):
    cfg = _config()
    cfg.compilation_config.cudagraph_mm_encoder = graphs_on
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    if case == "data_parallel":
        thinker.multimodal_config.mm_encoder_tp_mode = "data"
    else:
        monkeypatch.setattr(graph_module, "current_omni_platform", _platform(case))
    logged = []
    monkeypatch.setattr(graph_module.logger, "info", lambda msg, *args: logged.append(msg % args))
    thinker._enable_audio_encoder_cudagraph(cfg)
    assert "audio" not in thinker.get_encoder_cudagraph_config().modalities
    assert logged == ([f"Audio encoder CUDA graphs stay off: {reason}."] if reason else [])


def test_audio_replay_rejects_a_group_without_a_full_chunk():
    thinker = _audio_thinker()
    with pytest.raises(ValueError, match="full-length chunk"):
        thinker.prepare_encoder_cudagraph_replay_buffers(_audio([5, 9]), 2, 0, "audio")


def test_audio_graphs_stay_on_with_other_additional_config():
    cfg = _config()
    cfg.additional_config = {"audio_encoder_cudagraph": True, "unrelated": 1}
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    thinker._enable_audio_encoder_cudagraph(cfg)
    assert thinker.get_encoder_cudagraph_config().modalities == ["image", "audio"]


def test_config_keeps_the_image_only_path_and_adds_an_audio_path():
    image_only = _thinker().get_encoder_cudagraph_config()
    assert image_only.modalities == ["image"]
    assert list(image_only.paths) == ["default"] and not image_only.paths["default"].allow_zero_tokens

    config = _audio_thinker().get_encoder_cudagraph_config()
    assert config.modalities == ["image", "audio"]
    assert config.paths["default"].allow_zero_tokens and config.paths["audio"].allow_zero_tokens
    assert config.paths["audio"].min_token_budget == 2
    assert set(config.padding_logics) == {"cu_seqlens", "audio_cu_seqlens"}


def _cpu_manager(model):
    manager = SingleReplayEncoderCudaGraphManager(model.vllm_config, torch.device("cpu"), torch.float32, model)
    # Replace only CUDA launch/capture; keep production packing, selectors,
    # buffer copies, output slicing and ownership behavior.
    for path, budgets in manager.path_token_budgets.items():
        for budget in budgets:
            if budget == 0:
                continue
            values = model.prepare_encoder_cudagraph_capture_inputs(
                budget, manager.max_batch_size, manager.max_frames_per_batch, torch.device("cpu"), torch.float32, path
            ).values
            with torch.no_grad():
                output = torch.empty_like(model.encoder_cudagraph_forward(values, path=path))

            def replay(values=values, output=output, path=path):
                with torch.no_grad():
                    output.copy_(model.encoder_cudagraph_forward(values, path=path))

            manager.budget_graphs.setdefault(path, {})[budget] = BudgetGraphMetadata(
                token_budget=budget,
                max_batch_size=manager.max_batch_size,
                max_frames_per_batch=manager.max_frames_per_batch,
                graph=SimpleNamespace(replay=replay),
                input_buffers=values,
                output_buffer=output,
            )
    return manager


@pytest.mark.parametrize("lengths", [[16], [17, 5], [33], [40, 9]])
def test_audio_replay_matches_eager_and_keeps_earlier_outputs(lengths):
    model = _audio_thinker()
    manager = _cpu_manager(model)
    audio = _audio(lengths)
    with torch.no_grad():
        expected = model.encoder_eager_forward(audio).split([_tiny_output_lengths(n) for n in lengths])
        outputs = manager.execute(audio)
    assert manager.graph_hits == len(lengths)
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output, reference)
    saved = [output.clone() for output in outputs]
    with torch.no_grad():
        manager.execute(_audio([16, 16], offset=5.0))
    for output, reference in zip(outputs, saved):
        torch.testing.assert_close(output, reference)


@pytest.mark.parametrize(
    "inputs",
    [
        ("audio", [17, 5]),
        ("audio", [40, 9]),
        ("audio", [9, 33]),
        ("image", [[1, 2, 4], [1, 2, 2]]),
        ("audio", [5, 17]),
        ("image", [[1, 2, 4], [1, 2, 6]]),
    ],
)
def test_a_group_one_graph_holds_replays_once(inputs):
    # vLLM 0.32's packing splits [17, 5], [5, 17] and the 2+3-token images across
    # replays; the other groups also replay once on vLLM 0.31.
    kind, sizes = inputs
    model = _audio_thinker()
    manager = _cpu_manager(model)
    replays = []

    def counted(inner):
        def replay():
            replays.append(1)
            inner()

        return replay

    for graphs in manager.budget_graphs.values():
        for meta in graphs.values():
            meta.graph = SimpleNamespace(replay=counted(meta.graph.replay))
    group = _audio(sizes) if kind == "audio" else _images(sizes)
    selected = model.select_encoder_cudagraph_items
    model.select_encoder_cudagraph_items = lambda kwargs, indices: pytest.fail("whole group re-selected")
    with torch.no_grad():
        outputs = manager.execute(group)
        model.select_encoder_cudagraph_items = selected
        expected = model.encoder_eager_forward(group)
    assert outputs is not None and len(outputs) == len(sizes)
    assert len(replays) == 1
    torch.testing.assert_close(torch.cat(outputs), expected)


def test_images_still_replay_with_audio_enabled():
    model = _audio_thinker()
    manager = _cpu_manager(model)
    images = _images([[1, 2, 4], [1, 2, 2]])
    with torch.no_grad():
        outputs = manager.execute(images)
    for output, reference in zip(outputs, model.encoder_eager_forward(images).split([2, 1])):
        torch.testing.assert_close(output, reference)
    assert manager.graph_hits == 2


@pytest.mark.parametrize(
    "lengths",
    [
        [5, 9],  # every item shorter than one chunk: eager pads the convolution input differently
        [80],  # 5 chunks = 10 padded tokens above the largest budget
        [16, 16, 16],  # more items than one replay holds
    ],
)
def test_audio_group_that_a_graph_cannot_reproduce_goes_back_to_the_runner(lengths):
    model = _audio_thinker()
    manager = _cpu_manager(model)
    assert manager.execute(_audio(lengths)) is None
    assert manager.graph_hits == 0 and manager.graph_misses == len(lengths)


def test_ineligible_audio_group_stays_eager_with_a_data_parallel_encoder():
    manager = _cpu_manager(_audio_thinker())
    manager.use_dp = True
    assert manager.execute(_audio([5, 9])) is None
    assert manager.graph_misses == 2


def test_stage_wrapper_forwards_the_audio_hooks(monkeypatch):
    # The runner builds the manager on the stage wrapper, not on the thinker.
    model, cfg = _combined(monkeypatch, "thinker")
    # Upstream would give audio [2, 4, 8] from this list's maximum; the derived budgets are [2, 8].
    cfg.compilation_config.encoder_cudagraph_token_budgets = [3, 16]
    model.thinker.audio_tower = _tower()
    model.thinker._enable_audio_encoder_cudagraph(cfg)
    for name in (
        "encoder_cudagraph_group_eligible",
        "get_encoder_cudagraph_path_max_budget",
        "get_encoder_cudagraph_path_budgets",
    ):
        assert getattr_static(model, name).__self__ is model.thinker
    manager = _runner_factory(model, cfg)
    assert manager.supports_modality("audio")
    assert manager.path_token_budgets["audio"] == [0, 2, 8]
    assert manager.execute(_audio([5, 9])) is None


def test_audio_item_specs_size_graphs_by_whole_chunks():
    model = _audio_thinker()
    specs = model.get_encoder_cudagraph_item_specs(_audio([17, 5]))
    assert [spec.output_tokens for spec in specs] == [3, 1]
    assert [spec.get_path_output_tokens("audio") for spec in specs] == [4, 2]
    assert [spec.get_path_output_tokens("default") for spec in specs] == [0, 0]


def test_audio_selection_slices_frames_per_item():
    model = _audio_thinker()
    audio = _audio([17, 5, 16])
    selected = model.select_encoder_cudagraph_items(audio, [2, 0])
    assert selected["audio_feature_lengths"].tolist() == [16, 17]
    torch.testing.assert_close(
        selected["input_audio_features"],
        torch.cat([audio["input_audio_features"][:, 22:38], audio["input_audio_features"][:, :17]], dim=1),
    )
    empty = model.select_encoder_cudagraph_items(audio, [])
    assert empty["input_audio_features"].shape == (MEL, 0)


def test_audio_capture_keeps_the_window_bound_on_replay():
    model = _audio_thinker()
    values = model.prepare_encoder_cudagraph_capture_inputs(8, 2, 0, torch.device("cpu"), torch.float32, "audio")
    values = values.values
    assert values["audio_chunks"].shape == (4, MEL, 16)
    assert values["audio_max_seqlen"].item() == 8
    assert int(values["audio_valid_index"].max()) < 8
    replay = model.prepare_encoder_cudagraph_replay_buffers(_audio([17]), 2, 0, "audio").values
    assert "audio_max_seqlen" not in replay
    assert replay["audio_chunks"].shape == (2, MEL, 16)


def test_audio_budgets_stop_at_one_attention_window():
    # 2 tokens per chunk and 8-token windows: auto budgets 2..128 keep 2, 4 and 8 for audio.
    cfg = _config()
    cfg.compilation_config.encoder_cudagraph_token_budgets = []
    cfg.compilation_config.encoder_cudagraph_max_vision_items_per_batch = 0
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.model_config.max_model_len = 4096
    thinker = _thinker(cfg)
    thinker.audio_tower = _tower()
    thinker._enable_audio_encoder_cudagraph(cfg)
    manager = SingleReplayEncoderCudaGraphManager(cfg, torch.device("cpu"), torch.float32, thinker)
    assert manager.path_token_budgets["audio"] == [0, 2, 4, 8]
    assert manager.path_token_budgets["default"] == [0, 64, 128]
    # Explicit budgets raise the image path but not past the audio cap.
    explicit = _cpu_manager(_audio_thinker(budgets=(2, 4, 8, 16)))
    assert explicit.path_token_budgets["audio"] == [0, 2, 4, 8]
    assert explicit.path_token_budgets["default"] == [0, 2, 4, 8, 16]


@pytest.mark.parametrize(
    "budgets, audio",
    [
        # One large budget to keep graph memory small captures one audio graph.
        ((16,), [0, 8]),
        # Each budget rounds down to whole chunks; duplicates collapse.
        ((3, 5, 16), [0, 2, 4, 8]),
        ((3, 7, 9), [0, 2, 6, 8]),
    ],
)
def test_explicit_budgets_set_the_audio_budgets(budgets, audio):
    manager = _cpu_manager(_audio_thinker(budgets=budgets))
    assert manager.path_token_budgets["audio"] == audio
    assert manager.path_token_budgets["default"] == [0, *sorted(budgets)]
    # A clip that fills the largest audio budget (16 frames per 2-token chunk) replays.
    assert manager.execute(_audio([16 * (audio[-1] // 2)])) is not None


def test_audio_output_lengths_accept_python_ints():
    # Host metadata and item specs pass plain ints to the upstream helper.
    from vllm.model_executor.models.qwen3_omni_moe_thinker import _get_feat_extract_output_lengths

    for length in range(1, 3001):
        assert _get_feat_extract_output_lengths(length) == int(_get_feat_extract_output_lengths(torch.tensor(length)))

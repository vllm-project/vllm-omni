"""
Tests for Omni config utils. For stability, these tests should largely be
invariant to the specific attributes of vLLM config except in cases where we
explicitly patch values that differ from vLLM.
"""

import inspect
import json
import os
import shutil
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError
from transformers import PretrainedConfig, Qwen3OmniMoeConfig
from vllm.engine.arg_utils import EngineArgs

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_init_utils import build_engine_args_dict
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sync_config_is_omni():
    """Ensure create_model_config gives the right type."""
    cfg = OmniEngineArgs().create_model_config()
    assert isinstance(cfg, OmniModelConfig)


def test_session_mode_reaches_omni_model_config():
    assert OmniEngineArgs().create_model_config().session_mode == "turn"
    assert OmniEngineArgs(session_mode="duplex").create_model_config().session_mode == "duplex"


def test_default_stage_id_is_concrete_int():
    """Ensure `stage_id` stays safe for downstream arithmetic/indexing."""
    engine_args = OmniEngineArgs()

    assert engine_args.stage_id == 0
    assert isinstance(engine_args.stage_id, int)
    assert engine_args.log_stats is False

    cfg = engine_args.create_model_config()
    assert cfg.stage_id == 0


def test_full_payload_capability_reaches_omni_model_config(monkeypatch):
    """Keep the pipeline capability intact through the engine adapter."""
    captured: dict[str, object] = {}
    baseline_config = Mock()
    worker_module = types.ModuleType("test_connector_worker")

    class ConnectorRunner(OmniConnectorModelRunnerMixin):
        pass

    class ConnectorWorker:
        model_runner_cls = ConnectorRunner

    worker_module.ConnectorWorker = ConnectorWorker
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)

    monkeypatch.setattr(OmniEngineArgs, "_ensure_omni_models_registered", lambda _self: None)
    monkeypatch.setattr(EngineArgs, "create_model_config", lambda _self: baseline_config)

    def capture_omni_config(_cls, model_config, **omni_kwargs):
        assert model_config is baseline_config
        captured.update(omni_kwargs)
        return baseline_config

    monkeypatch.setattr(
        OmniModelConfig,
        "from_vllm_model_config",
        classmethod(capture_omni_config),
    )

    OmniEngineArgs(
        stage_id=1,
        requires_full_payload_input=True,
        worker_cls="test_connector_worker.ConnectorWorker",
    ).create_model_config()

    assert captured["requires_full_payload_input"] is True


def test_stage_without_connector_configuration_accepts_plain_runner(monkeypatch):
    worker_module = types.ModuleType("test_async_scheduler_worker")

    class WorkerWithoutConnector:
        model_runner_cls = object

    worker_module.WorkerWithoutConnector = WorkerWithoutConnector
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)

    args = OmniEngineArgs(
        stage_id=1,
        async_chunk=True,
        worker_cls="test_async_scheduler_worker.WorkerWithoutConnector",
    )

    assert args.requires_full_payload_input is False


def test_full_payload_capability_requires_selected_worker_connector(monkeypatch):
    worker_module = types.ModuleType("test_worker_without_connector")

    class WorkerWithoutConnector:
        model_runner_cls = object

    worker_module.WorkerWithoutConnector = WorkerWithoutConnector
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)

    with pytest.raises(ValueError, match="does not provide an Omni connector model runner"):
        OmniEngineArgs(
            stage_id=1,
            requires_full_payload_input=True,
            worker_cls="test_worker_without_connector.WorkerWithoutConnector",
        )


def test_full_payload_capability_validates_platform_selected_worker(monkeypatch):
    worker_module = types.ModuleType("test_platform_worker_without_connector")

    class WorkerWithoutConnector:
        model_runner_cls = object

    worker_module.WorkerWithoutConnector = WorkerWithoutConnector
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setattr(
        current_omni_platform,
        "get_omni_ar_worker_cls",
        lambda: "test_platform_worker_without_connector.WorkerWithoutConnector",
    )

    with pytest.raises(ValueError, match="does not provide an Omni connector model runner"):
        OmniEngineArgs(
            stage_id=1,
            requires_full_payload_input=True,
            worker_type="ar",
        )


def test_multimodal_kwarg_overrides(monkeypatch):
    """Ensure that overrides in the multimodal config are preserved."""
    sig = inspect.signature(OmniEngineArgs)
    default_mm_cache = sig.parameters["mm_processor_cache_gb"].default
    override_val = default_mm_cache + 1

    fake_model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(mm_processor_cache_gb=override_val),
    )

    def _fake_parent_create_model_config(self):
        assert self.mm_processor_cache_gb == override_val
        return fake_model_config

    monkeypatch.setattr(EngineArgs, "create_model_config", _fake_parent_create_model_config)
    monkeypatch.setattr(OmniModelConfig, "from_vllm_model_config", lambda model_config, **_: model_config)

    cfg = OmniEngineArgs(
        model="Qwen/Qwen2-VL-2B-Instruct",
        mm_processor_cache_gb=override_val,
    ).create_model_config()

    assert cfg.multimodal_config is not None
    assert cfg.multimodal_config.mm_processor_cache_gb == override_val


def test_from_vllm_config_validates_invalid_omni_kwargs():
    """Ensure omni-specific field validation catches invalid keys."""
    model_config = EngineArgs().create_model_config()
    with pytest.raises(ValueError, match="Unexpected omni kwarg"):
        OmniModelConfig.from_vllm_model_config(model_config, foo="bar")


def test_from_vllm_config_validates_bad_omni_kwarg_types():
    """Ensure omni-specific field validation catches type errors."""
    model_config = EngineArgs().create_model_config()
    with pytest.raises(ValidationError):
        OmniModelConfig.from_vllm_model_config(model_config, stage_id="not_an_int")


def test_default_all_values_are_initialized():
    """Ensure omni-specific field initializes all fields"""
    model_config = EngineArgs().create_model_config()
    cfg = OmniModelConfig.from_vllm_model_config(model_config)

    # Test a primitive
    assert cfg.model_stage == "thinker"
    # Test a field initialized with a default factory
    assert cfg.stage_connector_config == {
        "name": "SharedMemoryConnector",
        "extra": {},
    }

    # Ensure that hf_config is initialized on model_config in the vLLM by ModelConfig's
    # __post_init__, and that the hf_config is copied over to the OmniModelConfig;
    # we explicitly set this since the field sets init=False
    assert isinstance(model_config.hf_config, PretrainedConfig)
    assert cfg.hf_config is model_config.hf_config

    # Ensure that we can convert it to a string; this will convert
    # all attributes, so should raise if we have attributes that are
    # not initialized correctly, e.g., due to default factories
    str(cfg)


def test_qwen3_tts_codec_frame_rate_patching():
    """Ensure the patch for qwen3 tts is applied correctly when creating the omni config."""
    # Create a vLLM ModelConfig
    vllm_config = EngineArgs().create_model_config()

    # Create a mock talking config with a dummy value for position_id_per_seconds
    mock_talker_config = SimpleNamespace()
    mock_talker_config.position_id_per_seconds = 12.3
    vllm_config.hf_config.talker_config = mock_talker_config

    # Ensure creating the config for a Qwen3TTSTalkerForConditionalGenerationARVLLM
    # model calls the patch func to apply position_id_per_seconds from the talker
    # config to the config's codec_frame_rate_hz
    omni_config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        model_arch="Qwen3TTSTalkerForConditionalGenerationARVLLM",
    )

    # Verify codec_frame_rate_hz was patched
    assert omni_config.codec_frame_rate_hz == 12.3


def test_qwen3_tts_startup_task_type_is_validated():
    vllm_config = EngineArgs().create_model_config()

    config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        model_arch="Qwen3TTSTalkerForConditionalGenerationARVLLM",
        task_type="Base",
    )
    assert config.task_type == "Base"

    with pytest.raises(ValueError, match="Qwen3-TTS --task-type must be one of"):
        OmniModelConfig.from_vllm_model_config(
            vllm_config,
            model_arch="Qwen3TTSTalkerForConditionalGenerationARVLLM",
            task_type="fl2va",
        )


def test_qwen3_tts_code2wav_injects_max_position_embeddings(monkeypatch):
    """Ensure Code2Wav mirrors stage max_model_len into nested HF overrides.

    Qwen3-TTS Code2Wav is a pure decoder stage whose runtime max_model_len can
    legitimately exceed the base checkpoint's default text max length. Recent
    vLLM validates these values during ModelConfig creation, so we inject
    ``talker_config.max_position_embeddings`` before delegating to vLLM.
    """
    captured: dict[str, object] = {}
    baseline_config = Mock()

    def fake_create_model_config(self):
        captured["hf_overrides"] = self.hf_overrides
        return baseline_config

    monkeypatch.setattr(EngineArgs, "create_model_config", fake_create_model_config)
    monkeypatch.setattr(
        OmniModelConfig,
        "from_vllm_model_config",
        classmethod(lambda cls, model_config, **omni_kwargs: model_config),
    )

    OmniEngineArgs(
        model_arch="Qwen3TTSCode2Wav",
        max_model_len=65536,
    ).create_model_config()

    assert captured["hf_overrides"] == {
        "architectures": ["Qwen3TTSCode2Wav"],
        "talker_config": {
            "max_position_embeddings": 65536,
        },
    }


def test_patch_missing_local_hf_config(tmp_path):
    """Native model bundles may only contain a checkpoints/ directory."""
    (tmp_path / "checkpoints").mkdir()
    args = object.__new__(OmniEngineArgs)
    args.model = str(tmp_path)
    args.model_arch = "IndexTTS25TalkerForConditionalGeneration"
    args.hf_config_path = None

    try:
        args._patch_empty_hf_config("indextts2_5")

        assert args.hf_config_path is not None
        config_path = args.hf_config_path + "/config.json"
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        assert config == {
            "model_type": "indextts2_5",
            "architectures": ["IndexTTS25TalkerForConditionalGeneration"],
        }
    finally:
        if hasattr(args, "_temp_config_dir"):
            shutil.rmtree(args._temp_config_dir, ignore_errors=True)


@pytest.mark.parametrize(
    "config_entry",
    ["malformed", "directory", "broken_symlink", "permission_error"],
)
def test_non_missing_local_hf_config_error_reaches_parent_loader(tmp_path, monkeypatch, config_entry):
    """Non-missing config errors must reach vLLM's normal loader."""
    config_path = tmp_path / "config.json"
    if config_entry == "malformed":
        config_path.write_text("{not valid json", encoding="utf-8")
        loader_error = json.JSONDecodeError("invalid config", "{not valid json", 1)
    elif config_entry == "directory":
        config_path.mkdir()
        loader_error = IsADirectoryError(str(config_path))
    elif config_entry == "broken_symlink":
        config_path.symlink_to(tmp_path / "missing-target.json")
        loader_error = FileNotFoundError(str(config_path))
    else:
        config_path.write_text("{}", encoding="utf-8")
        loader_error = PermissionError(str(config_path))

    parent_inputs = {}

    def controlled_get_config_dict(cls, model, **kwargs):
        if model == str(tmp_path):
            raise loader_error
        with open(os.path.join(model, "config.json"), encoding="utf-8") as config_file:
            return json.load(config_file), {}

    def fake_parent_create_model_config(self):
        parent_inputs["called"] = True
        parent_inputs["model"] = self.model
        parent_inputs["hf_config_path"] = self.hf_config_path
        config_root = self.hf_config_path or self.model
        parent_inputs["config_root"] = config_root
        return PretrainedConfig.get_config_dict(config_root)[0]

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", classmethod(controlled_get_config_dict))
    monkeypatch.setattr(OmniEngineArgs, "_ensure_omni_models_registered", lambda self: True)
    monkeypatch.setattr(EngineArgs, "create_model_config", fake_parent_create_model_config)
    monkeypatch.setattr(
        OmniModelConfig,
        "from_vllm_model_config",
        classmethod(lambda cls, model_config, **omni_kwargs: model_config),
    )

    args = OmniEngineArgs(
        model=str(tmp_path),
        model_arch="IndexTTS25TalkerForConditionalGeneration",
    )

    with pytest.raises(type(loader_error)) as exc_info:
        args.create_model_config()

    assert str(exc_info.value) == str(loader_error)
    assert parent_inputs["called"] is True
    assert parent_inputs["model"] == str(tmp_path)
    assert parent_inputs["config_root"] == str(tmp_path)
    assert parent_inputs["hf_config_path"] is None
    assert args.model == str(tmp_path)
    assert args.hf_config_path is None
    assert not hasattr(args, "_temp_config_dir")


def test_patch_valid_hf_config_without_model_type_preserves_keys(tmp_path):
    """A valid partial config should be copied before required keys are added."""
    original_config = {
        "hidden_size": 1024,
        "custom_key": {"nested": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(original_config), encoding="utf-8")
    args = object.__new__(OmniEngineArgs)
    args.model = str(tmp_path)
    args.model_arch = "IndexTTS25TalkerForConditionalGeneration"
    args.hf_config_path = None

    try:
        args._patch_empty_hf_config("indextts2_5")

        assert args.hf_config_path is not None
        with open(os.path.join(args.hf_config_path, "config.json"), encoding="utf-8") as config_file:
            patched_config = json.load(config_file)
        assert patched_config["hidden_size"] == original_config["hidden_size"]
        assert patched_config["custom_key"] == original_config["custom_key"]
        assert patched_config["model_type"] == "indextts2_5"
        assert patched_config["architectures"] == ["IndexTTS25TalkerForConditionalGeneration"]
    finally:
        if hasattr(args, "_temp_config_dir"):
            shutil.rmtree(args._temp_config_dir, ignore_errors=True)


def test_remote_hf_config_error_reaches_parent_loader(monkeypatch):
    """Remote resolution failures must stay on vLLM's normal error path."""
    loader_error = OSError("remote config resolution failed")
    parent_inputs = {}

    def controlled_get_config_dict(cls, model, **kwargs):
        raise loader_error

    def fake_parent_create_model_config(self):
        parent_inputs["called"] = True
        parent_inputs["model"] = self.model
        parent_inputs["hf_config_path"] = self.hf_config_path
        return PretrainedConfig.get_config_dict(self.model)[0]

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", classmethod(controlled_get_config_dict))
    monkeypatch.setattr(OmniEngineArgs, "_ensure_omni_models_registered", lambda self: True)
    monkeypatch.setattr(EngineArgs, "create_model_config", fake_parent_create_model_config)

    args = OmniEngineArgs(
        model="remote/model",
        model_arch="IndexTTS25TalkerForConditionalGeneration",
    )

    with pytest.raises(OSError) as exc_info:
        args.create_model_config()

    assert str(exc_info.value) == str(loader_error)
    assert parent_inputs["called"] is True
    assert parent_inputs["model"] == "remote/model"
    assert parent_inputs["hf_config_path"] is None
    assert args.model == "remote/model"
    assert args.hf_config_path is None
    assert not hasattr(args, "_temp_config_dir")


def test_stage_specific_text_config_override():
    """Stage swap must refresh hf_text_config, dependent attrs, and model_arch_config."""
    vllm_config = EngineArgs().create_model_config()
    vllm_config.disable_sliding_window = True
    thinker_mac = vllm_config.model_arch_config

    talker_num_heads = max(2, thinker_mac.total_num_attention_heads // 2)
    talker_num_kv_heads = max(1, talker_num_heads // 8)
    talker_head_dim = 128
    stage_text_config = SimpleNamespace(
        sliding_window=4096,
        attention_chunk_size=2048,
        max_position_embeddings=4096,
        num_attention_heads=talker_num_heads,
        num_key_value_heads=talker_num_kv_heads,
        head_dim=talker_head_dim,
        hidden_size=talker_num_heads * talker_head_dim,
        vocab_size=thinker_mac.vocab_size,
        num_hidden_layers=4,
    )

    vllm_config.hf_text_config = SimpleNamespace()
    vllm_config.hf_config.thinker_config = SimpleNamespace(get_text_config=lambda: stage_text_config)

    omni_config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        hf_config_name="thinker_config",
    )

    assert omni_config.hf_text_config is stage_text_config
    assert omni_config.attention_chunk_size == 2048
    assert omni_config.max_model_len == 4096
    assert omni_config.hf_text_config.sliding_window is None

    stage_mac = omni_config.model_arch_config
    assert stage_mac is not thinker_mac
    assert stage_mac.total_num_attention_heads == talker_num_heads
    assert stage_mac.total_num_kv_heads == talker_num_kv_heads
    assert stage_mac.head_size == talker_head_dim

    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=1,
    )
    assert omni_config.get_num_attention_heads(parallel_config) == talker_num_heads
    assert omni_config.get_num_kv_heads(parallel_config) == talker_num_kv_heads
    assert omni_config.get_head_size() == talker_head_dim


def test_non_override_ar_stage_inputs_embeds_size_matches_hidden_size():
    """A stage without an embedding-size override keeps the default input width."""
    vllm_config = EngineArgs().create_model_config()
    vllm_config.hf_config = Qwen3OmniMoeConfig(enable_audio_output=True)

    omni_config = OmniModelConfig.from_vllm_model_config(
        vllm_config,
        model_stage="talker",
        hf_config_name="talker_config",
    )

    assert (
        getattr(
            omni_config.hf_config.talker_config,
            "embedding_size",
            None,
        )
        is None
    )
    assert omni_config.get_inputs_embeds_size() == omni_config.hf_text_config.hidden_size


# For https://github.com/vllm-project/vllm-omni/issues/3293
def test_tensor_parallel_size_none_is_handled():
    """Ensure the tensor parallel size of None isn't forwarded."""
    engine_args = OmegaConf.create({"stage_id": 0, "engine_args": {"tensor_parallel_size": None}})
    args = build_engine_args_dict(
        engine_args,
        model="snu-aidas/Dynin-Omni",
    )
    assert isinstance(args, dict)
    assert "tensor_parallel_size" not in args

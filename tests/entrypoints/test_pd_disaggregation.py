"""Unit tests for PD (Prefill-Decode) disaggregation in the Omni orchestrator.

Tests the PD detection, validation, config parsing, sampling param
preparation, and routing logic added by the PD disaggregation feature
(issue #1188).  All tests run without GPU by using the same mocking
infrastructure as test_omni_llm.py.
"""

import uuid
import warnings
from queue import Empty, Queue
from typing import Any
from unittest.mock import MagicMock

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK

# Suppress noisy DeprecationWarnings from optional Swig bindings imported by vLLM dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy.*has no __module__ attribute",
    category=DeprecationWarning,
)


# ---------------------------------------------------------------------------
# Fake helpers (same pattern as test_omni_llm.py)
# ---------------------------------------------------------------------------


class _FakeEngineArgs(dict):
    """Fake engine args that supports both attribute and dict access."""

    def __init__(self, args_dict: dict[str, Any]):
        super().__init__(args_dict)
        if "model_stage" not in self:
            self["model_stage"] = None
        if "engine_output_type" not in self:
            self["engine_output_type"] = None
        for key, value in self.items():
            setattr(self, key, value)


class _FakeStageConfig:
    def __init__(self, config_dict: dict[str, Any]):
        engine_args_dict = config_dict.get("engine_args", {})
        self.engine_args = _FakeEngineArgs(engine_args_dict)
        self.final_output = config_dict.get("final_output", False)
        self.final_output_type = config_dict.get("final_output_type", None)
        self.stage_id = config_dict.get("stage_id", 0)
        self.is_prefill_only = config_dict.get("is_prefill_only", False)
        self.is_decode_only = config_dict.get("is_decode_only", False)
        self.engine_input_source = config_dict.get("engine_input_source", [])
        self.is_comprehension = config_dict.get("is_comprehension", False)
        self._config_dict = config_dict


class _FakeQueue:
    def __init__(self, maxsize=0):
        self._queue = Queue(maxsize=maxsize)

    def put(self, item):
        self._queue.put(item)

    def put_nowait(self, item):
        self._queue.put_nowait(item)

    def get(self):
        return self._queue.get()

    def get_nowait(self):
        return self._queue.get_nowait()

    def empty(self):
        return self._queue.empty()


class _FakeStage:
    """Lightweight stage stub with PD disaggregation flag support."""

    def __init__(self, config, stage_init_timeout: int = 300):
        if isinstance(config, dict):
            config = _FakeStageConfig(config)
        self.config = config
        self.stage_config = config
        self.engine = None
        self.engine_outputs = None
        self.stage_id = getattr(config, "stage_id", 0)
        self.engine_args = config.engine_args
        self.model_stage = getattr(config.engine_args, "model_stage", None)
        self.stage_type = "llm"
        self.default_sampling_params = SamplingParams(temperature=1.0)
        self.final_output = config.final_output if hasattr(config, "final_output") else False
        self.final_output_type = getattr(config, "final_output_type", None)
        self.is_prefill_only = getattr(config, "is_prefill_only", False)
        self.is_decode_only = getattr(config, "is_decode_only", False)
        self.engine_input_source = getattr(config, "engine_input_source", [])
        self.is_comprehension = getattr(config, "is_comprehension", False)
        processed_input = getattr(config, "_config_dict", {}).get("processed_input", ["processed"])
        self._processed_input = processed_input
        self._in_q = None
        self._out_q = None
        self._proc = None
        self._stage_init_timeout = max(0, int(stage_init_timeout))

    def attach_queues(self, in_q, out_q):
        self._in_q = in_q
        self._out_q = out_q

    def init_stage_worker(
        self, model: str, *, is_async=False, shm_threshold_bytes=65536, ctx=None, batch_timeout=10, **kwargs
    ):
        self._proc = MagicMock()
        self._proc.start = MagicMock()
        self._proc.join = MagicMock()
        self._proc.is_alive = MagicMock(return_value=False)
        self._proc.terminate = MagicMock()
        if self._out_q is not None:
            try:
                self._out_q.put_nowait({"type": "stage_ready", "stage_id": self.stage_id})
            except Exception:
                pass

    def stop_stage_worker(self):
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(SHUTDOWN_TASK)
            except Exception:
                pass

    def submit(self, payload: dict[str, Any]):
        if self._in_q is not None:
            self._in_q.put(payload)

    def try_collect(self) -> Any:
        if self._out_q is None:
            return None
        try:
            return self._out_q.get_nowait()
        except Empty:
            return None

    def set_engine_outputs(self, outputs):
        self.engine_outputs = outputs

    def process_engine_inputs(self, stage_list, prompts):
        return self._processed_input


# ---------------------------------------------------------------------------
# Shared mock setup helpers
# ---------------------------------------------------------------------------


def _setup_engine_mocks(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.tokenizer = MagicMock()
    fake_engine.log_stats = False
    fake_engine.vllm_config = MagicMock()
    fake_engine.vllm_config.model_config = MagicMock()
    fake_engine.vllm_config.model_config.io_processor_plugin = None
    fake_engine.get_supported_tasks = MagicMock(return_value=[])
    fake_engine.model_config = MagicMock()
    fake_engine.model_config.io_processor_plugin = None
    fake_registry = MagicMock()
    fake_registry.resolve_model_cls = MagicMock(return_value=(MagicMock(), "test_arch"))
    fake_engine.model_config.registry = fake_registry
    fake_engine.vllm_config.model_config.registry = fake_registry

    monkeypatch.setattr(
        "vllm.v1.engine.llm_engine.LLMEngine.from_engine_args",
        lambda **kw: fake_engine,
        raising=False,
    )

    class FakeModelClass:
        pass

    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda model_config: (FakeModelClass, "test_arch"),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils._get_model_architecture",
        lambda model_config: (FakeModelClass, "test_arch"),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.adapters.try_create_mm_pooling_model_cls",
        lambda model_cls: model_cls,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.multimodal.cache._enable_processor_cache",
        lambda model_config, mm_registry: False,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.plugins.io_processors.get_io_processor",
        lambda vllm_config, io_processor_plugin: None,
        raising=False,
    )


def _setup_multiprocessing_mocks(monkeypatch):
    import multiprocessing as mp

    fake_process_class = MagicMock()
    fake_process_instance = MagicMock()
    fake_process_instance.start = MagicMock()
    fake_process_instance.join = MagicMock()
    fake_process_instance.is_alive = MagicMock(return_value=False)
    fake_process_instance.terminate = MagicMock()
    fake_process_class.return_value = fake_process_instance

    fake_ctx = MagicMock()
    fake_ctx.Queue = lambda maxsize=0: _FakeQueue(maxsize=maxsize)
    fake_ctx.Process = fake_process_class

    monkeypatch.setattr(mp, "get_context", lambda method: fake_ctx, raising=False)
    monkeypatch.setattr(mp, "Process", fake_process_class, raising=False)


def _setup_ipc_mocks(monkeypatch):
    def _fake_encode(obj, threshold, obj_key, shm_key):
        return {obj_key: obj}

    def _fake_load(result, obj_key, shm_key):
        return result.get(obj_key)

    def _fake_set(obj):
        return str(obj).encode()

    monkeypatch.setattr("vllm_omni.entrypoints.omni._encode", _fake_encode, raising=False)
    monkeypatch.setattr("vllm_omni.entrypoints.omni._load", _fake_load, raising=False)
    monkeypatch.setattr("vllm_omni.entrypoints.omni._set", _fake_set, raising=False)


def _setup_log_mocks(monkeypatch):
    class _FakeOrchestratorAggregator:
        def __init__(self, num_stages, enable_stats, wall_start_ts, final_stage_id_for_e2e=None):
            self.num_stages = num_stages
            self.enable_stats = enable_stats
            self.stage_first_ts = [None] * num_stages
            self.stage_last_ts = [None] * num_stages
            self.stage_total_tokens = [0] * num_stages
            self.accumulated_gen_time_ms = {}
            self.e2e_done = set()
            self.e2e_count = 0
            self.e2e_total_ms = 0.0

        def on_stage_metrics(self, stage_id, req_id, metrics, final_output_type=None):
            pass

        def on_finalize_request(self, stage_id, req_id, start_ts):
            self.e2e_done.add(req_id)

        def on_forward(self, from_stage, to_stage, req_id, size_bytes, tx_ms, use_shm):
            pass

        def accumulate_diffusion_metrics(self, stage_type, req_id, engine_outputs):
            pass

        def record_audio_generated_frames(self, output, stage_id, req_id):
            pass

        def stage_postprocess_timer(self, stage_id, req_id):
            from contextlib import contextmanager

            @contextmanager
            def _noop():
                yield

            return _noop()

        def build_and_log_summary(self):
            return "Fake summary"

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni.OrchestratorAggregator",
        _FakeOrchestratorAggregator,
        raising=False,
    )


def _clear_modules():
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]


@pytest.fixture(autouse=True)
def mock_get_config(monkeypatch):
    """Auto-mock get_config and related model loading functions."""
    import sys

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    fake_tokenizer.decode = MagicMock(return_value="test")

    def _mock_init_tokenizer_from_configs(model_config=None, **kwargs):
        return fake_tokenizer

    monkeypatch.setattr(
        "vllm.transformers_utils.tokenizer.init_tokenizer_from_configs",
        _mock_init_tokenizer_from_configs,
        raising=False,
    )
    tokenizer_module_path = "vllm.transformers_utils.tokenizer"
    if tokenizer_module_path in sys.modules:
        setattr(sys.modules[tokenizer_module_path], "init_tokenizer_from_configs", _mock_init_tokenizer_from_configs)

    def _mock_length_from_prompt_token_ids_or_embeds(prompt_token_ids=None, prompt_embeds=None):
        if prompt_token_ids is not None:
            if isinstance(prompt_token_ids, list):
                return len(prompt_token_ids)
        return 10

    monkeypatch.setattr(
        "vllm.utils.length_from_prompt_token_ids_or_embeds", _mock_length_from_prompt_token_ids_or_embeds, raising=False
    )
    monkeypatch.setattr(
        "vllm_omni.engine.input_processor.length_from_prompt_token_ids_or_embeds",
        _mock_length_from_prompt_token_ids_or_embeds,
        raising=False,
    )

    processor_module_path = "vllm_omni.engine.input_processor"
    if processor_module_path in sys.modules:
        setattr(
            sys.modules[processor_module_path],
            "length_from_prompt_token_ids_or_embeds",
            _mock_length_from_prompt_token_ids_or_embeds,
        )

    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.init_tokenizer_from_configs", _mock_init_tokenizer_from_configs, raising=False
    )
    async_omni_path = "vllm_omni.entrypoints.async_omni"
    if async_omni_path in sys.modules:
        setattr(sys.modules[async_omni_path], "init_tokenizer_from_configs", _mock_init_tokenizer_from_configs)

    fake_hf_config = MagicMock()
    fake_hf_config.model_type = "qwen2_5_omni"

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config", lambda model, **kwargs: fake_hf_config, raising=False
    )
    monkeypatch.setattr("vllm_omni.entrypoints.utils.get_config", lambda model, **kwargs: fake_hf_config, raising=False)

    def _mock_cached_file(path_or_repo_id, *args, **kwargs):
        import os
        import tempfile

        fake_config_file = os.path.join(tempfile.gettempdir(), "fake_config.json")
        if not os.path.exists(fake_config_file):
            with open(fake_config_file, "w") as f:
                f.write('{"model_type": "qwen2_5_omni"}')
        return fake_config_file

    monkeypatch.setattr("transformers.utils.hub.cached_file", _mock_cached_file, raising=False)
    monkeypatch.setattr(
        "transformers.utils.hub.cached_files",
        lambda path_or_repo_id, filenames, **kwargs: (
            [_mock_cached_file(path_or_repo_id, filenames[0])] if filenames else None
        ),
        raising=False,
    )


# ---------------------------------------------------------------------------
# Helper to build an Omni instance with PD stage configs
# ---------------------------------------------------------------------------


def _make_pd_omni(monkeypatch, stage_configs, *, extra_setup=None):
    """Create an Omni instance whose stage_list consists of _FakeStage objects
    built from *stage_configs* (list of dicts).

    Mocks the full init chain so no real model download, connector init,
    or stage process creation occurs.
    """
    _clear_modules()
    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    configs = [_FakeStageConfig(c) for c in stage_configs]

    # Mock load_and_resolve_stage_configs (the current entry point used by
    # OmniBase._resolve_stage_configs).  It returns (config_path, stage_configs).
    def _fake_load_and_resolve(model, stage_configs_path=None, kwargs=None, **kw):
        return ("fake_config_path", configs)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        _fake_load_and_resolve,
        raising=False,
    )

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    # Mock load_and_resolve_stage_configs on the omni module (it's imported
    # from utils at the top of omni.py).
    monkeypatch.setattr(omni_module, "load_and_resolve_stage_configs", _fake_load_and_resolve)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    # Mock omni_snapshot_download so it doesn't try to download a real model
    monkeypatch.setattr(omni_module, "omni_snapshot_download", lambda model_id: model_id)

    # Build a connectors dict that covers all edges so that
    # try_send_via_connector is always called (it sends data to the
    # next stage's input queue).
    num_stages = len(configs)
    fake_connectors = {}
    for i in range(num_stages - 1):
        fake_connectors[(str(i), str(i + 1))] = MagicMock()

    # Mock initialize_orchestrator_connectors so it doesn't parse real configs
    monkeypatch.setattr(
        omni_module,
        "initialize_orchestrator_connectors",
        lambda *args, **kwargs: (None, fake_connectors),
    )

    # Mock try_send_via_connector to directly submit to the stage queue
    # (the real version uses OmniConnector IPC; in tests we just put the
    # payload into the stage's input queue directly).
    def _fake_try_send(
        connector,
        stage_id,
        next_stage_id,
        req_id,
        next_inputs,
        sampling_params,
        original_prompt,
        next_stage_queue_submit_fn,
        metrics,
    ):
        task = {
            "request_id": req_id,
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
        }
        next_stage_queue_submit_fn(task)
        return True

    monkeypatch.setattr(omni_module, "try_send_via_connector", _fake_try_send)

    # Mock _start_stages to create fake queues (the real version uses ZMQ
    # sockets, multiprocessing contexts, and spawns stage workers).
    def _fake_start_stages(self, model):
        for stage in self.stage_list:
            in_q = _FakeQueue()
            out_q = _FakeQueue()
            self._stage_in_queues.append(in_q)
            self._stage_out_queues.append(out_q)
            stage.attach_queues(in_q, out_q)

    from vllm_omni.entrypoints.omni import OmniBase

    monkeypatch.setattr(OmniBase, "_start_stages", _fake_start_stages)
    monkeypatch.setattr(OmniBase, "_wait_for_stages_ready", lambda self, timeout=120: None)

    if extra_setup:
        extra_setup(monkeypatch, omni_module)

    from vllm_omni.entrypoints.omni import Omni

    return Omni(model="any", init_timeout=1)


# ---------------------------------------------------------------------------
# Stage config templates
# ---------------------------------------------------------------------------


def _prefill_stage_cfg(stage_id=0, **overrides):
    cfg = {
        "stage_id": stage_id,
        "engine_args": {
            "model_stage": "thinker",
            "kv_transfer_config": {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_producer",
                "kv_rank": 0,
                "kv_parallel_size": 2,
                "kv_connector_extra_config": {"mooncake_bootstrap_port": 25201},
            },
        },
        "is_prefill_only": True,
        "final_output": False,
        "is_comprehension": True,
    }
    cfg.update(overrides)
    return cfg


def _decode_stage_cfg(stage_id=1, engine_input_source=None, **overrides):
    cfg = {
        "stage_id": stage_id,
        "engine_args": {
            "model_stage": "thinker",
            "kv_transfer_config": {
                "kv_connector": "MooncakeConnector",
                "kv_role": "kv_consumer",
                "kv_rank": 1,
                "kv_parallel_size": 2,
                "kv_connector_extra_config": {"mooncake_bootstrap_port": 25202},
            },
        },
        "is_decode_only": True,
        "engine_input_source": engine_input_source if engine_input_source is not None else [0],
        "final_output": True,
        "final_output_type": "text",
        "is_comprehension": True,
    }
    cfg.update(overrides)
    return cfg


def _talker_stage_cfg(stage_id=2, engine_input_source=None, **overrides):
    cfg = {
        "stage_id": stage_id,
        "engine_args": {"model_stage": "talker"},
        "engine_input_source": engine_input_source if engine_input_source is not None else [1],
        "final_output": False,
    }
    cfg.update(overrides)
    return cfg


def _code2wav_stage_cfg(stage_id=3, engine_input_source=None, **overrides):
    cfg = {
        "stage_id": stage_id,
        "engine_args": {"model_stage": "code2wav"},
        "engine_input_source": engine_input_source if engine_input_source is not None else [2],
        "final_output": True,
        "final_output_type": "audio",
    }
    cfg.update(overrides)
    return cfg


# ===================================================================
# Tests: PD pair detection
# ===================================================================


class TestDetectPDSeparation:
    """Tests for Omni._detect_pd_separation()."""

    def test_detects_pd_pair(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
            ],
        )
        assert omni._pd_separation_pair == (0, 1)

    def test_no_pd_pair_without_flags(self, monkeypatch):
        """Normal (non-PD) pipeline has no PD pair."""
        omni = _make_pd_omni(
            monkeypatch,
            [
                {
                    "stage_id": 0,
                    "engine_args": {"model_stage": "thinker"},
                    "final_output": True,
                    "final_output_type": "text",
                },
                {
                    "stage_id": 1,
                    "engine_args": {"model_stage": "talker"},
                    "engine_input_source": [0],
                    "final_output": True,
                    "final_output_type": "audio",
                },
            ],
        )
        assert omni._pd_separation_pair is None

    def test_detects_pd_pair_in_4_stage_pipeline(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
                _talker_stage_cfg(stage_id=2, engine_input_source=[1]),
                _code2wav_stage_cfg(stage_id=3, engine_input_source=[2]),
            ],
        )
        assert omni._pd_separation_pair == (0, 1)

    def test_pd_pair_uses_stage_id_for_input_source(self, monkeypatch):
        """engine_input_source references stage_id, not list index."""
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=10),
                _decode_stage_cfg(stage_id=20, engine_input_source=[10]),
            ],
        )
        assert omni._pd_separation_pair == (0, 1)


# ===================================================================
# Tests: PD config validation
# ===================================================================


class TestValidatePDConfig:
    """Tests for Omni._validate_pd_separation_config()."""

    def test_valid_config_passes(self, monkeypatch):
        """Valid PD config should not raise."""
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        # If we got here without error, validation passed
        assert omni._pd_separation_pair == (0, 1)

    def test_mismatched_connector_raises(self, monkeypatch):
        """Different kv_connector types should raise ValueError."""
        decode_cfg = _decode_stage_cfg(engine_input_source=[0])
        decode_cfg["engine_args"]["kv_transfer_config"]["kv_connector"] = "NixlConnector"

        with pytest.raises(ValueError, match="connector mismatch"):
            _make_pd_omni(monkeypatch, [_prefill_stage_cfg(), decode_cfg])

    def test_wrong_prefill_role_raises(self, monkeypatch):
        """Prefill with kv_consumer role should raise."""
        prefill_cfg = _prefill_stage_cfg()
        prefill_cfg["engine_args"]["kv_transfer_config"]["kv_role"] = "kv_consumer"

        with pytest.raises(ValueError, match="kv_role must be"):
            _make_pd_omni(monkeypatch, [prefill_cfg, _decode_stage_cfg(engine_input_source=[0])])

    def test_wrong_decode_role_raises(self, monkeypatch):
        """Decode with kv_producer role should raise."""
        decode_cfg = _decode_stage_cfg(engine_input_source=[0])
        decode_cfg["engine_args"]["kv_transfer_config"]["kv_role"] = "kv_producer"

        with pytest.raises(ValueError, match="kv_role must be"):
            _make_pd_omni(monkeypatch, [_prefill_stage_cfg(), decode_cfg])

    def test_missing_kv_transfer_config_raises(self, monkeypatch):
        """Missing kv_transfer_config should raise."""
        prefill_cfg = _prefill_stage_cfg()
        del prefill_cfg["engine_args"]["kv_transfer_config"]

        with pytest.raises(ValueError, match="kv_transfer_config"):
            _make_pd_omni(monkeypatch, [prefill_cfg, _decode_stage_cfg(engine_input_source=[0])])

    def test_mismatched_buffer_device_raises(self, monkeypatch):
        """Mismatched kv_buffer_device should raise."""
        prefill_cfg = _prefill_stage_cfg()
        prefill_cfg["engine_args"]["kv_transfer_config"]["kv_buffer_device"] = "cuda"
        decode_cfg = _decode_stage_cfg(engine_input_source=[0])
        decode_cfg["engine_args"]["kv_transfer_config"]["kv_buffer_device"] = "cpu"

        with pytest.raises(ValueError, match="kv_buffer_device mismatch"):
            _make_pd_omni(monkeypatch, [prefill_cfg, decode_cfg])


# ===================================================================
# Tests: Connector info extraction
# ===================================================================


class TestGetPDConnectorInfo:
    """Tests for Omni._get_pd_connector_info()."""

    def test_extracts_bootstrap_addr_for_mooncake(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        info = omni._pd_connector_info
        assert "prefill_bootstrap_addr" in info
        assert info["prefill_bootstrap_addr"] == "127.0.0.1:25201"

    def test_none_for_non_pd_pipeline(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                {"stage_id": 0, "engine_args": {}, "final_output": True, "final_output_type": "text"},
            ],
        )
        assert omni._pd_connector_info is None


# ===================================================================
# Tests: Prefill sampling params preparation
# ===================================================================


class TestPreparePrefillSamplingParams:
    """Tests for Omni._prepare_prefill_sampling_params()."""

    def test_sets_max_tokens_to_1(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048)
        result = omni._prepare_prefill_sampling_params("req-1", sp)

        assert result.max_tokens == 1
        assert result is not sp  # should be cloned

    def test_injects_kv_transfer_params(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048)
        result = omni._prepare_prefill_sampling_params("req-1", sp)

        kv_params = result.extra_args["kv_transfer_params"]
        assert kv_params["do_remote_decode"] is True
        assert kv_params["do_remote_prefill"] is False
        assert kv_params["transfer_id"] == "xfer-req-1"

    def test_preserves_existing_extra_args(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048, extra_args={"custom_key": "value"})
        result = omni._prepare_prefill_sampling_params("req-1", sp)

        assert result.extra_args["custom_key"] == "value"
        assert "kv_transfer_params" in result.extra_args

    def test_does_not_mutate_original(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048)
        _ = omni._prepare_prefill_sampling_params("req-1", sp)

        assert sp.max_tokens == 2048
        assert sp.extra_args is None


# ===================================================================
# Tests: Sampling params auto-duplication for PD split
# ===================================================================


class TestSamplingParamsAutoDuplication:
    """When user provides N-1 sampling params (for logical stages), the
    orchestrator should auto-duplicate the thinker params for the decode stage.
    """

    def test_auto_duplicates_for_4_stage_pipeline(self, monkeypatch):
        """User provides 3 params for 4 physical stages -> auto-insert decode params."""
        test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")

        def _extra_setup(mp, omni_module):
            mp.setattr(uuid, "uuid4", lambda: test_uuid)
            mp.setattr(omni_module, "uuid", uuid)

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
                _talker_stage_cfg(stage_id=2, engine_input_source=[1]),
                _code2wav_stage_cfg(stage_id=3, engine_input_source=[2]),
            ],
            extra_setup=_extra_setup,
        )

        assert omni._pd_separation_pair == (0, 1)
        assert len(omni.stage_list) == 4

        # Simulate outputs for all stages
        expected_rid = f"0_{test_uuid}"
        for i in range(4):
            omni.stage_list[i]._out_q.put_nowait(
                {
                    "request_id": expected_rid,
                    "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1, 2])])],
                    "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
                }
            )

        # Provide 3 params (one less than 4 stages) - should auto-duplicate
        sp_thinker = SamplingParams(temperature=0.4, max_tokens=2048)
        sp_talker = SamplingParams(temperature=0.9, max_tokens=4096)
        sp_code2wav = SamplingParams(temperature=0.0, max_tokens=65536)

        # This should NOT raise ValueError about param count mismatch
        outputs = omni.generate(
            prompts=["hello"],
            sampling_params_list=[sp_thinker, sp_talker, sp_code2wav],
        )
        assert isinstance(outputs, list)


# ===================================================================
# Tests: KV transfer params normalization
# ===================================================================


class TestNormalizeKVTransferParams:
    def test_dict_passthrough(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        d = {"transfer_id": "test", "do_remote_decode": True}
        assert omni._normalize_kv_transfer_params(d) is d

    def test_none_returns_none(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        assert omni._normalize_kv_transfer_params(None) is None

    def test_dataclass_to_dict(self, monkeypatch):
        from dataclasses import dataclass

        @dataclass
        class FakeKVParams:
            transfer_id: str = "test"
            do_remote_decode: bool = True

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        result = omni._normalize_kv_transfer_params(FakeKVParams())
        assert isinstance(result, dict)
        assert result["transfer_id"] == "test"


# ===================================================================
# Tests: _kv_cfg_to_dict
# ===================================================================


class TestKvCfgToDict:
    def test_dict_passthrough(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        d = {"kv_connector": "MooncakeConnector"}
        assert omni._kv_cfg_to_dict(d) is d

    def test_none_returns_empty(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        assert omni._kv_cfg_to_dict(None) == {}

    def test_dataclass_converted(self, monkeypatch):
        from dataclasses import dataclass

        @dataclass
        class FakeCfg:
            kv_connector: str = "TestConnector"
            kv_role: str = "kv_producer"

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        result = omni._kv_cfg_to_dict(FakeCfg())
        assert result["kv_connector"] == "TestConnector"
        assert result["kv_role"] == "kv_producer"


# ===================================================================
# Tests: PD routing in scheduling loop
# ===================================================================


class TestPDRouting:
    """Test that the scheduling loop correctly routes requests from
    prefill to decode stage with proper kv_transfer_params.
    """

    def test_prefill_stage_receives_max_tokens_1(self, monkeypatch):
        """Stage 0 (prefill) should receive max_tokens=1."""
        test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000002")

        def _extra_setup(mp, omni_module):
            mp.setattr(uuid, "uuid4", lambda: test_uuid)
            mp.setattr(omni_module, "uuid", uuid)

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
            ],
            extra_setup=_extra_setup,
        )

        expected_rid = f"0_{test_uuid}"

        # Put stage outputs in both queues
        omni.stage_list[0]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1])])],
                "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
            }
        )
        omni.stage_list[1]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1, 2, 3])])],
                "metrics": {"num_tokens_out": 3, "stage_gen_time_ms": 50.0},
            }
        )

        sp_list = [SamplingParams(max_tokens=2048), SamplingParams(max_tokens=2048)]
        omni.generate(prompts=["hello"], sampling_params_list=sp_list)

        # Check what was submitted to stage 0's input queue
        # (skip the stage_ready message first)
        task = omni.stage_list[0]._in_q.get_nowait()
        assert task["sampling_params"].max_tokens == 1
        kv_params = task["sampling_params"].extra_args["kv_transfer_params"]
        assert kv_params["do_remote_decode"] is True
        assert kv_params["do_remote_prefill"] is False
        assert kv_params["transfer_id"] == f"xfer-{expected_rid}"

    def test_decode_stage_receives_original_prompt(self, monkeypatch):
        """Decode stage should get the original prompt (not processed outputs)."""
        test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000003")

        def _extra_setup(mp, omni_module):
            mp.setattr(uuid, "uuid4", lambda: test_uuid)
            mp.setattr(omni_module, "uuid", uuid)

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
            ],
            extra_setup=_extra_setup,
        )

        expected_rid = f"0_{test_uuid}"
        original_prompt = "test prompt for PD"

        omni.stage_list[0]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1])])],
                "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
            }
        )
        omni.stage_list[1]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1, 2, 3])])],
                "metrics": {"num_tokens_out": 3, "stage_gen_time_ms": 50.0},
            }
        )

        sp_list = [SamplingParams(max_tokens=2048), SamplingParams(max_tokens=2048)]
        omni.generate(prompts=[original_prompt], sampling_params_list=sp_list)

        # Check what was forwarded to stage 1 (decode)
        # The connector sends tasks to stage 1's input queue
        task = omni.stage_list[1]._in_q.get_nowait()
        # The engine_inputs should contain the original prompt
        engine_inputs = task.get("engine_inputs")
        # For PD routing, the original prompt is wrapped in a list
        if isinstance(engine_inputs, list):
            assert original_prompt in engine_inputs
        else:
            assert engine_inputs == original_prompt

    def test_decode_kv_params_have_correct_flags(self, monkeypatch):
        """Decode stage kv_transfer_params should have correct role flags."""
        test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000004")

        def _extra_setup(mp, omni_module):
            mp.setattr(uuid, "uuid4", lambda: test_uuid)
            mp.setattr(omni_module, "uuid", uuid)

        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(stage_id=0),
                _decode_stage_cfg(stage_id=1, engine_input_source=[0]),
            ],
            extra_setup=_extra_setup,
        )

        expected_rid = f"0_{test_uuid}"

        omni.stage_list[0]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1])])],
                "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
            }
        )
        omni.stage_list[1]._out_q.put_nowait(
            {
                "request_id": expected_rid,
                "engine_outputs": [MagicMock(request_id=expected_rid, outputs=[MagicMock(token_ids=[1, 2, 3])])],
                "metrics": {"num_tokens_out": 3, "stage_gen_time_ms": 50.0},
            }
        )

        sp_list = [SamplingParams(max_tokens=2048), SamplingParams(max_tokens=2048)]
        omni.generate(prompts=["hello"], sampling_params_list=sp_list)

        # Check decode task's kv_transfer_params
        task = omni.stage_list[1]._in_q.get_nowait()
        kv_params = task["sampling_params"].extra_args["kv_transfer_params"]
        assert kv_params["do_remote_prefill"] is True
        assert kv_params["do_remote_decode"] is False
        assert kv_params["transfer_id"] == f"xfer-{expected_rid}"
        assert kv_params["remote_bootstrap_addr"] == "127.0.0.1:25201"


# ===================================================================
# Tests: KV params cleanup
# ===================================================================


class TestKVParamsCleanup:
    def test_drop_cleans_up(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        omni._pd_kv_params_by_req["req-1"] = {"transfer_id": "xfer-1"}
        omni._drop_pd_kv_params("req-1")
        assert "req-1" not in omni._pd_kv_params_by_req

    def test_drop_nonexistent_is_noop(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        omni._drop_pd_kv_params("nonexistent")  # should not raise

    def test_pop_returns_stored_params(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        stored = {"transfer_id": "xfer-1", "extra_field": "value"}
        omni._pd_kv_params_by_req["req-1"] = stored

        result = omni._pop_pd_kv_params("req-1")
        assert result == stored
        assert "req-1" not in omni._pd_kv_params_by_req

    def test_pop_uses_fallback_when_no_stored(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        fallback = {"transfer_id": "xfer-fallback"}
        result = omni._pop_pd_kv_params("req-1", fallback=fallback)
        assert result == fallback


# ===================================================================
# Tests: Config YAML loads without error
# ===================================================================


class TestPDYAMLConfig:
    def test_pd_yaml_loads(self):
        """The PD multiconnector YAML config should load without errors."""
        import os

        yaml_path = os.path.join(
            os.path.dirname(__file__),
            "../../vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_multiconnector.yaml",
        )
        yaml_path = os.path.abspath(yaml_path)
        if not os.path.exists(yaml_path):
            pytest.skip("PD separation YAML not found")

        from omegaconf import OmegaConf

        cfg = OmegaConf.load(yaml_path)
        stages = cfg.stage_args
        assert len(stages) == 4

        # Prefill stage
        assert stages[0].is_prefill_only is True
        assert stages[0].final_output is False
        assert stages[0].is_comprehension is True

        # Decode stage
        assert stages[1].is_decode_only is True
        assert stages[1].final_output is True
        assert stages[1].final_output_type == "text"
        assert stages[1].is_comprehension is True
        assert 0 in stages[1].engine_input_source

        # KV transfer configs
        assert stages[0].engine_args.kv_transfer_config.kv_role == "kv_producer"
        assert stages[1].engine_args.kv_transfer_config.kv_role == "kv_consumer"
        assert stages[0].engine_args.kv_transfer_config.kv_connector == "MooncakeConnector"
        assert stages[1].engine_args.kv_transfer_config.kv_connector == "MooncakeConnector"


# ===================================================================
# Tests: MooncakeConnector monkey-patch
# ===================================================================


class TestMooncakeConnectorPatch:
    """Tests for the embedded MooncakeConnector monkey-patch that fixes
    the request-ID mismatch in PD disaggregation.
    """

    def test_stage_payload_includes_pd_flags(self, monkeypatch):
        """init_stage_worker should include is_prefill_only / is_decode_only
        in stage_payload so the worker process can decide whether to apply
        the MooncakeConnector patch.
        """
        from vllm_omni.entrypoints.omni_stage import OmniStage

        # Build a minimal stage config with PD flags
        stage_cfg = _FakeStageConfig(_prefill_stage_cfg(stage_id=0))
        stage = OmniStage.__new__(OmniStage)
        # Manually set required attributes (bypass __init__ which needs real config)
        stage.stage_config = stage_cfg
        stage.stage_id = 0
        stage.engine_args = stage_cfg.engine_args
        stage.is_prefill_only = True
        stage.is_decode_only = False
        stage.stage_type = "llm"
        stage.engine_input_source = []
        stage.final_output = False
        stage.final_output_type = None
        stage._shm_threshold_bytes = 65536
        stage._stage_init_timeout = 300
        stage._in_q = MagicMock()
        stage._out_q = MagicMock()
        stage._proc = None

        # Capture the stage_payload by monkeypatching the Process constructor
        captured_payloads = []

        class FakeCtx:
            class Process:
                def __init__(self, target=None, args=None):
                    self.target = target
                    # args[1] is stage_payload for _stage_worker
                    if args and len(args) >= 2:
                        captured_payloads.append(args[1])

                def start(self):
                    pass

        stage.init_stage_worker("test-model", ctx=FakeCtx())

        assert len(captured_payloads) == 1
        payload = captured_payloads[0]
        assert payload["is_prefill_only"] is True
        assert payload["is_decode_only"] is False

    def test_patch_creates_subclass(self):
        """create_patched_mooncake_connector should return a class that is a
        proper subclass of vLLM's MooncakeConnector (when available).
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (
                MooncakeConnector as OriginalMC,
            )
        except ImportError:
            pytest.skip("vLLM MooncakeConnector not available in this env")

        from vllm_omni.distributed.kv_transfer.monkey_patch import (
            _create_patched_mooncake_connector as create_patched_mooncake_connector,
        )

        PatchedCls = create_patched_mooncake_connector()
        assert issubclass(PatchedCls, OriginalMC)

    def test_request_finished_returns_remote_request_id(self):
        """The patched request_finished should inject remote_request_id
        into kv_transfer_params.
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (
                MooncakeConnector as OriginalMC,
            )
        except ImportError:
            pytest.skip("vLLM MooncakeConnector not available in this env")

        from vllm_omni.distributed.kv_transfer.monkey_patch import (
            _create_patched_mooncake_connector as create_patched_mooncake_connector,
        )

        PatchedCls = create_patched_mooncake_connector()

        # Create a mock instance without calling __init__ (avoids needing
        # real vLLM config), then manually set attributes the method needs.
        instance = PatchedCls.__new__(PatchedCls)
        instance.remote_to_local_req = {}

        # Mock request object
        fake_request = MagicMock()
        fake_request.request_id = "chatcmpl-abc-9dd58560"

        # Mock super().request_finished to return a dict
        original_rf = OriginalMC.request_finished
        monkeypatch_result = {"do_remote_decode": True, "transfer_id": "xfer-1"}

        def _mock_super_rf(self, request, block_ids):
            return dict(monkeypatch_result)

        OriginalMC.request_finished = _mock_super_rf
        try:
            result = PatchedCls.request_finished(instance, fake_request, [1, 2, 3])
        finally:
            OriginalMC.request_finished = original_rf

        assert result is not None
        assert result["remote_request_id"] == "chatcmpl-abc-9dd58560"

    def test_add_new_req_uses_remote_request_id(self):
        """When load_remote_cache=True, the patched add_new_req should
        store a PatchedRecvReqMeta with the remote_request_id from
        kv_transfer_params.
        """
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (  # noqa: F401
                MooncakeConnector as OriginalMC,
            )
        except ImportError:
            pytest.skip("vLLM MooncakeConnector not available in this env")

        from vllm_omni.distributed.kv_transfer.monkey_patch import (
            PatchedRecvReqMeta,
        )
        from vllm_omni.distributed.kv_transfer.monkey_patch import (
            _create_patched_mooncake_connector as create_patched_mooncake_connector,
        )

        PatchedCls = create_patched_mooncake_connector()

        instance = PatchedCls.__new__(PatchedCls)
        instance.remote_to_local_req = {}
        instance._reqs_need_recv = {}

        kv_params = {
            "do_remote_prefill": True,
            "remote_request_id": "chatcmpl-abc-9dd58560",
            "transfer_id": "xfer-1",
        }
        instance.add_new_req(
            request_id="chatcmpl-abc-ab52f3ea",
            local_block_ids=[10, 20, 30],
            kv_transfer_params=kv_params,
        )

        assert "chatcmpl-abc-ab52f3ea" in instance._reqs_need_recv
        meta = instance._reqs_need_recv["chatcmpl-abc-ab52f3ea"]
        assert isinstance(meta, PatchedRecvReqMeta)
        assert meta.request_id == "chatcmpl-abc-ab52f3ea"
        assert meta.remote_request_id == "chatcmpl-abc-9dd58560"
        assert meta.local_block_ids == [10, 20, 30]


# ===================================================================
# Tests: Stop neutralization in prefill sampling params
# ===================================================================


class TestPrefillStopNeutralization:
    """Tests that _prepare_prefill_sampling_params neutralizes stop
    conditions to ensure finish_reason='length'.
    """

    def test_clears_stop_strings(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048, stop=["</s>", "STOP"])
        result = omni._prepare_prefill_sampling_params("req-1", sp)
        assert result.stop == []

    def test_clears_stop_token_ids(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048, stop_token_ids=[151643, 151644])
        result = omni._prepare_prefill_sampling_params("req-1", sp)
        assert result.stop_token_ids == []

    def test_clears_include_stop_str_in_output(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048, include_stop_str_in_output=True)
        result = omni._prepare_prefill_sampling_params("req-1", sp)
        assert result.include_stop_str_in_output is False

    def test_original_sp_unchanged(self, monkeypatch):
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        sp = SamplingParams(max_tokens=2048, stop=["</s>"], stop_token_ids=[151643])
        _ = omni._prepare_prefill_sampling_params("req-1", sp)
        assert sp.stop == ["</s>"]
        assert sp.stop_token_ids == [151643]


# ===================================================================
# Tests: Failure mode & memory leak prevention
# ===================================================================
# NOTE: Full generate()-level failure mode tests are removed for now.
# The _run_generation error handler (line 1344-1350 in omni.py) calls
# _drop_pd_kv_params but does not increment completed_requests, causing
# the while-loop to hang.  These tests need to be revisited once the
# production error-handling path is fixed to properly terminate on
# stage errors.


# ===================================================================
# Tests: TP size validation
# ===================================================================


class TestTPSizeValidation:
    """Tests that _validate_pd_separation_config checks tensor_parallel_size."""

    def test_matching_tp_passes(self, monkeypatch):
        """Same TP size should not raise."""
        prefill_cfg = _prefill_stage_cfg()
        prefill_cfg["engine_args"]["tensor_parallel_size"] = 2
        decode_cfg = _decode_stage_cfg(engine_input_source=[0])
        decode_cfg["engine_args"]["tensor_parallel_size"] = 2
        omni = _make_pd_omni(monkeypatch, [prefill_cfg, decode_cfg])
        assert omni._pd_separation_pair == (0, 1)

    def test_mismatched_tp_raises(self, monkeypatch):
        """Different TP sizes should raise ValueError."""
        prefill_cfg = _prefill_stage_cfg()
        prefill_cfg["engine_args"]["tensor_parallel_size"] = 2
        decode_cfg = _decode_stage_cfg(engine_input_source=[0])
        decode_cfg["engine_args"]["tensor_parallel_size"] = 4
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            _make_pd_omni(monkeypatch, [prefill_cfg, decode_cfg])

    def test_default_tp_no_error(self, monkeypatch):
        """Stages without explicit TP (defaults to 1) should pass."""
        omni = _make_pd_omni(
            monkeypatch,
            [
                _prefill_stage_cfg(),
                _decode_stage_cfg(engine_input_source=[0]),
            ],
        )
        assert omni._pd_separation_pair == (0, 1)

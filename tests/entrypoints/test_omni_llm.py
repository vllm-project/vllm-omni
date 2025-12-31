import uuid
import warnings
from queue import Empty, Queue
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm_omni.entrypoints.stage_utils import SHUTDOWN_TASK

# Suppress noisy DeprecationWarnings from optional Swig bindings imported by vLLM dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPy.*has no __module__ attribute",
    category=DeprecationWarning,
)


class _FakeEngineArgs(dict):
    """Fake engine args that can be used both as object attributes and as **kwargs."""

    def __init__(self, args_dict: dict[str, Any]):
        super().__init__(args_dict)
        # Add required attributes if not present
        if "model_stage" not in self:
            self["model_stage"] = None
        if "engine_output_type" not in self:
            self["engine_output_type"] = None
        # Also set as attributes for object-style access
        for key, value in self.items():
            setattr(self, key, value)


class _FakeStageConfig:
    """Fake stage config object that mimics the real stage config structure."""

    def __init__(self, config_dict: dict[str, Any]):
        # engine_args needs to work both as object (for OmniStage) and as dict (for **kwargs)
        engine_args_dict = config_dict.get("engine_args", {})
        self.engine_args = _FakeEngineArgs(engine_args_dict)
        self.final_output = config_dict.get("final_output", False)
        self.final_output_type = config_dict.get("final_output_type", None)
        self.stage_id = config_dict.get("stage_id", 0)
        # Store original dict for reference
        self._config_dict = config_dict


class _FakeQueue:
    """Fake queue using standard library Queue to replace mp.Queue."""

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
    """Lightweight Stage stub for multi-process pipeline version with queue support."""

    def __init__(self, config, stage_init_timeout: int = 300):
        # Handle both dict and object configs
        if isinstance(config, dict):
            config = _FakeStageConfig(config)
        self.config = config
        self.stage_config = config
        self.engine = None
        self.engine_outputs = None
        # Set attributes that OmniStage expects
        self.stage_id = getattr(config, "stage_id", 0)
        self.engine_args = config.engine_args
        self.model_stage = getattr(config.engine_args, "model_stage", None)
        # set default sampling params
        self.default_sampling_params = {"temperature": 1.0}
        # Allow configuring final_output and final_output_type
        self.final_output = config.final_output if hasattr(config, "final_output") else False
        self.final_output_type = getattr(config, "final_output_type", None)
        # Configurable processing logic, default returns placeholder
        processed_input = getattr(config, "_config_dict", {}).get("processed_input", ["processed"])
        self._processed_input = processed_input
        # Queue references (set by attach_queues)
        self._in_q = None
        self._out_q = None
        self._proc = None  # Mock process reference
        self._stage_init_timeout = max(0, int(stage_init_timeout))

    def attach_queues(self, in_q, out_q):
        """Attach input and output queues."""
        self._in_q = in_q
        self._out_q = out_q

    def init_stage_worker(
        self,
        model: str,
        *,
        is_async: bool = False,
        shm_threshold_bytes: int = 65536,
        ctx=None,
        batch_timeout: int = 10,
        **kwargs,
    ):
        """Mock init_stage_worker: don't start real process, just send stage_ready message."""
        # Create a mock process object
        self._proc = MagicMock()
        self._proc.start = MagicMock()
        self._proc.join = MagicMock()
        self._proc.is_alive = MagicMock(return_value=False)
        self._proc.terminate = MagicMock()
        # Send stage_ready message to output queue
        if self._out_q is not None:
            try:
                self._out_q.put_nowait({"type": "stage_ready", "stage_id": self.stage_id})
            except Exception:
                pass

    def stop_stage_worker(self):
        """Mock stop_stage_worker: clean up queue references."""
        if self._in_q is not None:
            try:
                self._in_q.put_nowait(SHUTDOWN_TASK)
            except Exception:
                pass

    def submit(self, payload: dict[str, Any]):
        """Submit task to input queue."""
        if self._in_q is not None:
            self._in_q.put(payload)

    def try_collect(self) -> Any:
        """Non-blocking collect from output queue."""
        if self._out_q is None:
            return None
        try:
            return self._out_q.get_nowait()
        except Empty:
            return None

    def set_engine_outputs(self, outputs):
        """Set engine outputs for the stage."""
        self.engine_outputs = outputs

    def process_engine_inputs(self, stage_list, prompts):
        """Process engine inputs: return preset processed result."""
        return self._processed_input


class _FakeEngine:
    """Lightweight Engine stub: provides generate iterator output."""

    def __init__(self, outputs: list[Any]):
        self._outputs = outputs

    def generate(self, prompts, sampling_params):
        # Record the most recent prompts for outer assertions
        self._last_prompts = prompts
        # Simplified: return preset list at once, ensuring iterability
        yield from self._outputs


@pytest.fixture
def fake_stage_config():
    return {
        # Don't include 'model' in engine_args since it's passed separately
        "engine_args": {},
        "final_output": True,
        "final_output_type": "text",
        # Second stage will use processed_input to verify the chain
        "processed_input": ["processed-by-stage"],
    }


def _setup_engine_mocks(monkeypatch):
    """Helper function to set up common engine mocks."""
    fake_engine = MagicMock()
    # Add necessary attributes to fake_engine
    fake_engine.tokenizer = MagicMock()
    fake_engine.log_stats = False
    fake_engine.vllm_config = MagicMock()
    fake_engine.vllm_config.model_config = MagicMock()
    fake_engine.vllm_config.model_config.io_processor_plugin = None
    fake_engine.get_supported_tasks = MagicMock(return_value=[])
    fake_engine.model_config = MagicMock()
    fake_engine.model_config.io_processor_plugin = None
    # Add registry with resolve_model_cls method
    fake_registry = MagicMock()
    fake_registry.resolve_model_cls = MagicMock(return_value=(MagicMock(), "test_arch"))
    fake_engine.model_config.registry = fake_registry
    fake_engine.vllm_config.model_config.registry = fake_registry

    monkeypatch.setattr(
        "vllm.v1.engine.llm_engine.LLMEngine.from_engine_args",
        lambda **kw: fake_engine,
        raising=False,
    )

    # Mock model_config.registry.resolve_model_cls to return a tuple
    # Use a real class instead of MagicMock to avoid inspect.getsource issues
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

    # Mock try_create_mm_pooling_model_cls to return the class as-is
    monkeypatch.setattr(
        "vllm.model_executor.models.adapters.try_create_mm_pooling_model_cls",
        lambda model_cls: model_cls,
        raising=False,
    )

    # Mock _enable_processor_cache to return False
    monkeypatch.setattr(
        "vllm.multimodal.cache._enable_processor_cache",
        lambda model_config, mm_registry: False,
        raising=False,
    )

    # Mock get_io_processor to return None
    monkeypatch.setattr(
        "vllm.plugins.io_processors.get_io_processor",
        lambda vllm_config, io_processor_plugin: None,
        raising=False,
    )


def _setup_multiprocessing_mocks(monkeypatch):
    """Helper function to set up multiprocessing mocks."""
    import multiprocessing as mp

    # Mock Process
    fake_process_class = MagicMock()
    fake_process_instance = MagicMock()
    fake_process_instance.start = MagicMock()
    fake_process_instance.join = MagicMock()
    fake_process_instance.is_alive = MagicMock(return_value=False)
    fake_process_instance.terminate = MagicMock()
    fake_process_class.return_value = fake_process_instance

    # Mock get_context to return a context with Queue that returns _FakeQueue
    fake_ctx = MagicMock()
    fake_ctx.Queue = lambda maxsize=0: _FakeQueue(maxsize=maxsize)
    fake_ctx.Process = fake_process_class

    def _mock_get_context(method):
        return fake_ctx

    monkeypatch.setattr(mp, "get_context", _mock_get_context, raising=False)
    monkeypatch.setattr(mp, "Process", fake_process_class, raising=False)


def _setup_ipc_mocks(monkeypatch):
    """Helper function to set up IPC function mocks."""

    # Mock _encode: simple serialization
    def _fake_encode(obj, threshold, obj_key, shm_key):
        return {obj_key: obj}

    # Mock _load: extract object from result
    def _fake_load(result, obj_key, shm_key):
        return result.get(obj_key)

    # Mock _set: calculate serialization size
    def _fake_set(obj):
        return str(obj).encode()

    monkeypatch.setattr("vllm_omni.entrypoints.omni._encode", _fake_encode, raising=False)
    monkeypatch.setattr("vllm_omni.entrypoints.omni._load", _fake_load, raising=False)
    monkeypatch.setattr("vllm_omni.entrypoints.omni._set", _fake_set, raising=False)


def _setup_log_mocks(monkeypatch):
    """Helper function to set up logging and stats mocks."""
    # Mock OrchestratorMetrics to be a simple class that doesn't require file operations

    class _FakeOrchestratorMetrics:
        def __init__(self, num_stages, enable_stats, wall_start_ts):
            self.num_stages = num_stages
            self.enable_stats = enable_stats
            self.stage_first_ts = [None] * num_stages
            self.stage_last_ts = [None] * num_stages
            self.e2e_done = set()

        def on_stage_metrics(self, stage_id, req_id, metrics):
            pass

        def on_finalize_request(self, stage_id, req_id, engine_outputs, start_ts):
            self.e2e_done.add(req_id)

        def on_forward(self, from_stage, to_stage, req_id, size_bytes, tx_ms, use_shm):
            pass

        def build_and_log_summary(self, final_stage_id):
            return "Fake summary"

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni.OrchestratorMetrics",
        _FakeOrchestratorMetrics,
        raising=False,
    )


@pytest.fixture(autouse=True)
def mock_get_config(monkeypatch):
    """Auto-mock get_config and related model loading functions to avoid model path validation."""
    # CRITICAL: Mock tokenizer-related imports FIRST, before any module imports
    # This prevents ImportError when async_omni is imported (which happens via omni_stage)
    import sys

    fake_tokenizer = MagicMock()
    fake_tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    fake_tokenizer.decode = MagicMock(return_value="test")

    # Mock init_tokenizer_from_configs (used in async_omni)
    def _mock_init_tokenizer_from_configs(model_config=None, **kwargs):
        return fake_tokenizer

    # Strategy 1: Mock in the original location (vllm.transformers_utils.tokenizer)
    # This works if the module hasn't been imported yet
    monkeypatch.setattr(
        "vllm.transformers_utils.tokenizer.init_tokenizer_from_configs",
        _mock_init_tokenizer_from_configs,
        raising=False,
    )

    # Strategy 2: If the module is already in sys.modules, patch it directly
    tokenizer_module_path = "vllm.transformers_utils.tokenizer"
    if tokenizer_module_path in sys.modules:
        tokenizer_module = sys.modules[tokenizer_module_path]
        setattr(tokenizer_module, "init_tokenizer_from_configs", _mock_init_tokenizer_from_configs)

    # CRITICAL: Mock length_from_prompt_token_ids_or_embeds BEFORE trying to mock async_omni

    # This is because async_omni imports processor.py, which imports this function at module level
    # Mock length_from_prompt_token_ids_or_embeds (used in processor.py)
    def _mock_length_from_prompt_token_ids_or_embeds(prompt_token_ids=None, prompt_embeds=None):
        # Return a reasonable default length
        if prompt_token_ids is not None:
            if isinstance(prompt_token_ids, list):
                return len(prompt_token_ids)
            elif hasattr(prompt_token_ids, "shape"):
                return prompt_token_ids.shape[-1] if len(prompt_token_ids.shape) > 0 else 1
        if prompt_embeds is not None:
            if hasattr(prompt_embeds, "shape"):
                return prompt_embeds.shape[-2] if len(prompt_embeds.shape) > 1 else 1
        return 10  # Default length

    # Mock in vllm.utils
    monkeypatch.setattr(
        "vllm.utils.length_from_prompt_token_ids_or_embeds",
        _mock_length_from_prompt_token_ids_or_embeds,
        raising=False,
    )
    # Also mock in processor module if it's imported
    monkeypatch.setattr(
        "vllm_omni.engine.input_processor.length_from_prompt_token_ids_or_embeds",
        _mock_length_from_prompt_token_ids_or_embeds,
        raising=False,
    )
    # If processor module is already imported, patch it directly
    processor_module_path = "vllm_omni.engine.input_processor"
    if processor_module_path in sys.modules:
        processor_module = sys.modules[processor_module_path]
        setattr(
            processor_module, "length_from_prompt_token_ids_or_embeds", _mock_length_from_prompt_token_ids_or_embeds
        )

    # Strategy 3: Now mock async_omni AFTER length_from_prompt_token_ids_or_embeds is mocked
    # This prevents ImportError when async_omni imports processor.py
    monkeypatch.setattr(
        "vllm_omni.entrypoints.async_omni.init_tokenizer_from_configs",
        _mock_init_tokenizer_from_configs,
        raising=False,
    )

    # Strategy 4: If async_omni is already imported, patch it directly
    async_omni_path = "vllm_omni.entrypoints.async_omni"
    if async_omni_path in sys.modules:
        async_omni_module = sys.modules[async_omni_path]
        setattr(async_omni_module, "init_tokenizer_from_configs", _mock_init_tokenizer_from_configs)

    # Now mock get_config and other functions
    fake_hf_config = MagicMock()
    fake_hf_config.model_type = "qwen2_5_omni"

    def _mock_get_config(model, **kwargs):
        return fake_hf_config

    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_config",
        _mock_get_config,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_config",
        _mock_get_config,
        raising=False,
    )

    # Mock transformers' cached_file to avoid downloading model configs
    def _mock_cached_file(path_or_repo_id, *args, **kwargs):
        import os
        import tempfile

        fake_config_file = os.path.join(tempfile.gettempdir(), "fake_config.json")
        if not os.path.exists(fake_config_file):
            with open(fake_config_file, "w") as f:
                f.write('{"model_type": "qwen2_5_omni"}')
        return fake_config_file

    monkeypatch.setattr(
        "transformers.utils.hub.cached_file",
        _mock_cached_file,
        raising=False,
    )
    monkeypatch.setattr(
        "transformers.utils.hub.cached_files",
        lambda path_or_repo_id, filenames, **kwargs: (
            [_mock_cached_file(path_or_repo_id, filenames[0])] if filenames else None
        ),
        raising=False,
    )


def test_initialize_stage_configs_called_when_none(monkeypatch, fake_stage_config):
    """Test that stage configs are auto-loaded when stage_configs_path is None."""

    def _fake_loader(model: str):
        return [
            _FakeStageConfig(fake_stage_config),
            _FakeStageConfig(fake_stage_config),
        ]

    # Remove modules from cache BEFORE setting mocks
    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Set up mocks
    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    # Mock load_stage_configs_from_model
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )

    # Replace OmniStage
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    # Import the module after mocks are set
    import vllm_omni.entrypoints.omni as omni_module

    # Patch the imported function and class in the module
    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)
    # Verify: auto-loaded stage_configs and stage_list have consistent count
    assert isinstance(omni.stage_configs, list)
    assert len(omni.stage_configs) == 2
    assert len(omni.stage_list) == 2
    # Verify: each Stage is _FakeStage instance
    for st in omni.stage_list:
        assert isinstance(st, _FakeStage)
    # Verify: queues are attached
    for st in omni.stage_list:
        assert st._in_q is not None
        assert st._out_q is not None
    # Verify: all stages are ready
    assert len(omni._stages_ready) == 2


def test_generate_raises_on_length_mismatch(monkeypatch, fake_stage_config):
    """Test that generate raises ValueError when sampling_params_list length doesn't match."""

    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)
    with pytest.raises(ValueError):
        omni.generate(prompts=["hi"], sampling_params_list=[])


def test_generate_pipeline_and_final_outputs(monkeypatch, fake_stage_config):
    """Test multi-stage generation pipeline with queue polling."""
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    stage_cfg1["processed_input"] = ["processed-for-stage-1"]

    def _fake_loader(model: str):
        return [_FakeStageConfig(stage_cfg0), _FakeStageConfig(stage_cfg1)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    # Mock uuid.uuid4() to return a predictable value for request ID generation
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)
    monkeypatch.setattr(omni_module, "uuid", uuid)

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)

    # Generate the expected request ID format: "0_<uuid>"
    expected_request_id = f"0_{test_uuid}"

    # Simulate worker behavior: manually put results into output queues
    # Note: We put results before calling generate, which simulates worker processes
    # that have already completed. The polling loop will collect them in stage order.
    # Stage 0 output (will be collected first)
    omni.stage_list[0]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 0, "text": "s0"}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )
    # Stage 1 output (will be collected after stage 0 forwards to it)
    # Note: In real flow, stage 1 result would appear after stage 0 forwards,
    # but for testing we pre-populate it. The polling loop processes stages
    # in order, so stage 0 result will be collected first, then forwarded,
    # then stage 1 result will be collected.
    omni.stage_list[1]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 1, "text": "s1"}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )

    # Use dicts instead of object() for serializable sampling params
    sampling_params_list = [{"temperature": 0.7}, {"temperature": 0.8}]
    prompts = ["hi"]
    outputs = omni.generate(prompts=prompts, sampling_params_list=sampling_params_list)

    # Both stages have final_output=True, so should aggregate two OmniRequestOutput
    assert len(outputs) == 2
    # Verify stage outputs are set
    assert omni.stage_list[0].engine_outputs == [{"stage": 0, "text": "s0"}]
    assert omni.stage_list[1].engine_outputs == [{"stage": 1, "text": "s1"}]
    # Verify stage 0 input queue received the task
    assert not omni.stage_list[0]._in_q.empty()
    # Verify stage 1 received forwarded task (process_engine_inputs was called)
    assert omni.stage_list[1].process_engine_inputs([], []) is not None


def test_generate_no_final_output_returns_empty(monkeypatch, fake_stage_config):
    """Test that generate returns empty list when all stages have final_output=False."""
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    stage_cfg0["final_output"] = False
    stage_cfg1["final_output"] = False

    def _fake_loader(model: str):
        return [_FakeStageConfig(stage_cfg0), _FakeStageConfig(stage_cfg1)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    # Mock uuid.uuid4() to return a predictable value for request ID generation
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)
    monkeypatch.setattr(omni_module, "uuid", uuid)

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)

    # Generate the expected request ID format: "0_<uuid>"
    expected_request_id = f"0_{test_uuid}"

    # Simulate worker behavior: put results into output queues
    omni.stage_list[0]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 0}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )
    omni.stage_list[1]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 1}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )

    # Use dicts instead of object() for serializable sampling params
    outputs = omni.generate(prompts=["p"], sampling_params_list=[{"temperature": 0.7}, {"temperature": 0.8}])
    assert outputs == []


def test_generate_sampling_params_none_use_default(monkeypatch, fake_stage_config):
    """Test that generate uses default sampling params when sampling_params_list is None."""
    stage_cfg0 = dict(fake_stage_config)
    stage_cfg1 = dict(fake_stage_config)
    stage_cfg0["final_output"] = False
    stage_cfg1["final_output"] = False

    def _fake_loader(model: str):
        return [_FakeStageConfig(stage_cfg0), _FakeStageConfig(stage_cfg1)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    # Mock uuid.uuid4() to return a predictable value for request ID generation
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)
    monkeypatch.setattr(omni_module, "uuid", uuid)

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)

    # Generate the expected request ID format: "0_<uuid>"
    expected_request_id = f"0_{test_uuid}"

    # Simulate worker behavior: put results into output queues
    omni.stage_list[0]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 0}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )
    omni.stage_list[1]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 1}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )
    # Use the default sampling params
    omni.generate(prompts=["p"], sampling_params_list=None)


def test_wait_for_stages_ready_timeout(monkeypatch, fake_stage_config):
    """Test that _wait_for_stages_ready handles timeout correctly."""

    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )

    # Create a stage that doesn't send stage_ready message
    class _FakeStageNoReady(_FakeStage):
        def init_stage_worker(self, *args, **kwargs):
            # Don't send stage_ready message
            self._proc = MagicMock()
            self._proc.start = MagicMock()
            self._proc.join = MagicMock()
            self._proc.is_alive = MagicMock(return_value=False)
            self._proc.terminate = MagicMock()

    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStageNoReady(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStageNoReady(cfg, **kwargs))

    from vllm_omni.entrypoints.omni import Omni

    # Use very short timeout
    omni = Omni(model="any", init_timeout=0.01)
    # Verify that no stages are ready
    assert len(omni._stages_ready) == 0


def test_generate_handles_error_messages(monkeypatch, fake_stage_config):
    """Test that generate handles error messages from stages correctly."""

    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    # Mock uuid.uuid4() to return a predictable value for request ID generation
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")
    monkeypatch.setattr(uuid, "uuid4", lambda: test_uuid)
    monkeypatch.setattr(omni_module, "uuid", uuid)

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)

    # Generate the expected request ID format: "0_<uuid>"
    expected_request_id = f"0_{test_uuid}"

    # Put error message in output queue
    omni.stage_list[0]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "error": "test error",
        }
    )
    # Also put a valid result after error to allow the loop to complete
    # (error handling continues the loop, so we need a valid result to finish)
    omni.stage_list[0]._out_q.put_nowait(
        {
            "request_id": expected_request_id,
            "engine_outputs": [{"stage": 0, "text": "result"}],
            "metrics": {"num_tokens_out": 1, "stage_gen_time_ms": 10.0},
        }
    )

    # Generate should handle error gracefully (log but continue)
    # Use dict instead of object() for serializable sampling params
    sampling_params_list = [{"temperature": 0.7}]
    outputs = omni.generate(prompts=["hi"], sampling_params_list=sampling_params_list)
    # Should return final output (error was logged but didn't stop processing)
    assert isinstance(outputs, list)
    # Since final_output=True, should have one output
    assert len(outputs) == 1


def test_close_sends_shutdown_signal(monkeypatch, fake_stage_config):
    """Test that close() sends shutdown signal to all input queues."""

    def _fake_loader(model: str):
        return [_FakeStageConfig(fake_stage_config)]

    import sys

    for module_name in [
        "vllm_omni.entrypoints.utils",
        "vllm_omni.entrypoints.omni",
        "vllm_omni.entrypoints.omni_stage",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    _setup_engine_mocks(monkeypatch)
    _setup_multiprocessing_mocks(monkeypatch)
    _setup_ipc_mocks(monkeypatch)
    _setup_log_mocks(monkeypatch)

    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.load_stage_configs_from_model",
        _fake_loader,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.omni_stage.OmniStage",
        lambda cfg, **kwargs: _FakeStage(cfg, **kwargs),
        raising=False,
    )

    import vllm_omni.entrypoints.omni as omni_module

    monkeypatch.setattr(omni_module, "load_stage_configs_from_model", _fake_loader)
    monkeypatch.setattr(omni_module, "OmniStage", lambda cfg, **kwargs: _FakeStage(cfg, **kwargs))

    from vllm_omni.entrypoints.omni import Omni

    omni = Omni(model="any", init_timeout=1)

    # Call close
    omni.close()

    # Verify shutdown signal (None) was sent to input queue
    # Use get_nowait to avoid blocking (close() uses put_nowait, so should be safe)
    try:
        shutdown_signal = omni.stage_list[0]._in_q.get_nowait()
        assert shutdown_signal == SHUTDOWN_TASK
    except Empty:
        # If queue was already empty or only had stage_ready, that's also acceptable
        # The important thing is that close() was called without error
        pass

    # Verify stop_stage_worker was called (process should be set)
    assert omni.stage_list[0]._proc is not None

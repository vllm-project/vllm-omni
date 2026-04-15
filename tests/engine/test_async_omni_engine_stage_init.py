import importlib
import os
import threading
import types

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_engine_core_client_module_reload_keeps_forward_refs_deferred():
    """Regression test for forward references in make_async_mp_client."""
    import vllm_omni.engine.stage_engine_core_client as client_mod

    importlib.reload(client_mod)

    assert client_mod.StageEngineCoreClientBase.make_async_mp_client.__annotations__["return"] == (
        "StageEngineCoreClient | DPLBStageEngineCoreClient"
    )


def test_initialize_stages_restores_device_visibility_after_diffusion_init(monkeypatch):
    """Regression test for stage device env leakage across stage init.

    Diffusion init mutates process-level CUDA visibility. Ensure AsyncOmniEngine
    restores the previous value after diffusion stage setup.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 1
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._single_stage_id_filter = None
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace(stage_id=0, stage_type="diffusion")]

    env_var = current_omni_platform.device_control_env_var
    old_env = os.environ.get(env_var)
    os.environ[env_var] = "0,1"

    diffusion_client = types.SimpleNamespace(is_comprehension=False)

    metadata = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime_cfg={"devices": "1"},
        prompt_expand_func=None,
    )

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "extract_stage_metadata", lambda _cfg: metadata)
    monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(engine_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))

    def _fake_setup_stage_devices(_stage_id, _runtime_cfg):
        # Simulate diffusion setup mutating process-global visibility.
        current_omni_platform.set_device_control_env_var("1")

    monkeypatch.setattr(engine_mod, "setup_stage_devices", _fake_setup_stage_devices)
    monkeypatch.setattr(engine_mod, "inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(engine_mod, "initialize_diffusion_stage", lambda *_, **__: diffusion_client)
    monkeypatch.setattr(
        engine_mod,
        "finalize_initialized_stages",
        lambda stage_clients, _input_processor: (
            stage_clients,
            [types.SimpleNamespace()],
            [{"final_output_type": "image"}],
        ),
    )

    try:
        engine._initialize_stages(stage_init_timeout=1)
        assert os.environ.get(env_var) == "0,1"
    finally:
        if old_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_env


def test_initialize_stages_passes_stage_init_timeout_to_diffusion_handshake(monkeypatch):
    """Regression test for stage_init_timeout passing to complete_diffusion_handshake
    in the diffusion stage path.
    """
    import vllm_omni.diffusion.data as diffusion_data_mod
    import vllm_omni.diffusion.stage_diffusion_client as client_mod
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.config_path = "dummy-config"
    engine.num_stages = 2
    engine.async_chunk = False
    engine.diffusion_batch_size = 1
    engine.single_stage_mode = False
    engine._omni_master_server = None
    engine.stage_configs = [types.SimpleNamespace(stage_id=0, stage_type="diffusion", engine_args={})]

    metadata = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        final_output=True,
        final_output_type="image",
        default_sampling_params=None,
        custom_process_input_func=None,
        engine_input_source=None,
        cfg_kv_collect_func=None,
    )

    captured_timeout = None
    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(engine_mod, "load_omni_transfer_config_for_model", lambda *_: None)
    monkeypatch.setattr(engine_mod, "extract_stage_metadata", lambda _cfg: metadata)
    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(
        engine_mod,
        "finalize_initialized_stages",
        lambda stage_clients, _input_processor: (
            stage_clients,
            [types.SimpleNamespace()],
            [{"final_output_type": "image"}],
        ),
    )
    monkeypatch.setattr(
        diffusion_data_mod.OmniDiffusionConfig,
        "from_kwargs",
        classmethod(lambda cls, **kwargs: types.SimpleNamespace(parallel_config=types.SimpleNamespace(world_size=1))),
    )
    monkeypatch.setattr(
        client_mod,
        "spawn_diffusion_proc",
        lambda model, od_cfg: (object(), "ipc://handshake", "ipc://request", "ipc://response"),
    )

    def _capture_handshake_timeout(proc, handshake_address, handshake_timeout):
        nonlocal captured_timeout
        captured_timeout = handshake_timeout

    monkeypatch.setattr(client_mod, "complete_diffusion_handshake", _capture_handshake_timeout)
    monkeypatch.setattr(
        client_mod.zmq,
        "Context",
        lambda: types.SimpleNamespace(socket=lambda _: types.SimpleNamespace(connect=lambda _: None)),
    )

    try:
        engine._initialize_stages(stage_init_timeout=302)
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert captured_timeout == 302


def test_launch_llm_stage_passes_stage_init_timeout_to_complete_stage_handshake(monkeypatch):
    """Regression test for stage_init_timeout reaching complete_stage_handshake
    in the LLM stage path.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.single_stage_mode = False
    engine._omni_master_server = None

    metadata = types.SimpleNamespace(stage_id=0, runtime_cfg={"devices": "0"})
    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace()
    fake_proc = types.SimpleNamespace()

    captured_timeout = None

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(engine_mod, "build_vllm_config", lambda *_, **__: (fake_vllm_config, object))
    monkeypatch.setattr(engine_mod, "acquire_device_locks", lambda *_: [])
    monkeypatch.setattr(
        engine_mod,
        "spawn_stage_core",
        lambda **_: (fake_addresses, fake_proc, "ipc://handshake"),
    )

    def _capture_stage_timeout(_proc, _handshake_addr, _addresses, _vllm_cfg, handshake_timeout):
        nonlocal captured_timeout
        captured_timeout = handshake_timeout

    monkeypatch.setattr(engine_mod, "complete_stage_handshake", _capture_stage_timeout)

    try:
        engine._launch_llm_stage(
            stage_cfg=types.SimpleNamespace(engine_args={}),
            metadata=metadata,
            stage_connector_spec={},
            stage_init_timeout=302,
            llm_stage_launch_lock=threading.Lock(),
        )
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert captured_timeout == 302


def test_launch_llm_stage_releases_launch_lock_before_complete_stage_handshake(monkeypatch):
    """Regression test for parallel LLM stage startup during handshake wait."""
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.platforms import current_omni_platform

    engine = object.__new__(AsyncOmniEngine)
    engine.log_stats = False
    engine.model = "dummy-model"
    engine.single_stage_mode = False
    engine._omni_master_server = None

    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace()
    shared_launch_lock = threading.Lock()
    counter_lock = threading.Lock()
    first_handshake_started = threading.Event()
    second_stage_spawned = threading.Event()
    allow_first_handshake_to_finish = threading.Event()
    launch_errors: list[BaseException] = []
    spawn_count = 0

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(engine_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(engine_mod, "build_vllm_config", lambda *_, **__: (fake_vllm_config, object))
    monkeypatch.setattr(engine_mod, "acquire_device_locks", lambda *_: [])

    def _spawn_stage_core(**_):
        nonlocal spawn_count
        with counter_lock:
            spawn_count += 1
            call_idx = spawn_count
        if call_idx == 2:
            second_stage_spawned.set()
        return fake_addresses, types.SimpleNamespace(), f"ipc://handshake-{call_idx}"

    def _complete_stage_handshake(_proc, handshake_address, _addresses, _vllm_cfg, _timeout):
        if handshake_address == "ipc://handshake-1":
            first_handshake_started.set()
            assert second_stage_spawned.wait(timeout=1), (
                "second stage did not reach spawn_stage_core while first stage waited in handshake"
            )
            assert allow_first_handshake_to_finish.wait(timeout=1), (
                "second stage did not enter handshake while first stage was still waiting"
            )
        else:
            allow_first_handshake_to_finish.set()

    monkeypatch.setattr(engine_mod, "spawn_stage_core", _spawn_stage_core)
    monkeypatch.setattr(engine_mod, "complete_stage_handshake", _complete_stage_handshake)

    def _launch_stage(stage_id: int) -> None:
        metadata = types.SimpleNamespace(stage_id=stage_id, runtime_cfg={"devices": str(stage_id)})
        try:
            engine._launch_llm_stage(
                stage_cfg=types.SimpleNamespace(engine_args={}),
                metadata=metadata,
                stage_connector_spec={},
                stage_init_timeout=302,
                llm_stage_launch_lock=shared_launch_lock,
            )
        except BaseException as exc:  # pragma: no cover - surfaced through assertion below
            launch_errors.append(exc)

    try:
        first_thread = threading.Thread(target=_launch_stage, args=(0,))
        first_thread.start()
        assert first_handshake_started.wait(timeout=1), "first stage never entered handshake"

        second_thread = threading.Thread(target=_launch_stage, args=(1,))
        second_thread.start()

        first_thread.join(timeout=3)
        second_thread.join(timeout=3)
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert second_stage_spawned.is_set()
    assert not launch_errors


def test_attach_llm_stage_uses_omni_input_preprocessor(monkeypatch):
    """Regression test for GLM-Image t2i preprocessing path.

    Stage-0 InputProcessor must use OmniInputPreprocessor so text prompts with
    mm_processor_kwargs go through multimodal preprocessing.
    """
    import vllm_omni.engine.async_omni_engine as engine_mod

    class DummyStageEngineCoreClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def shutdown(self):
            return None

    class DummyOutputProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyInputProcessor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.renderer = object()
            self.input_preprocessor = object()

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer

    monkeypatch.setattr(
        engine_mod.StageEngineCoreClientBase,
        "make_async_mp_client",
        staticmethod(lambda **kwargs: DummyStageEngineCoreClient(**kwargs)),
    )
    monkeypatch.setattr(engine_mod, "MultimodalOutputProcessor", DummyOutputProcessor)
    monkeypatch.setattr(engine_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(engine_mod, "OmniInputPreprocessor", DummyOmniInputPreprocessor)

    started = types.SimpleNamespace(
        stage_id=0,
        metadata=types.SimpleNamespace(stage_id=0, engine_output_type="token_ids"),
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(skip_tokenizer_init=True)),
        executor_class=object,
        engine_manager=object(),
        coordinator=object(),
        proc=None,
        addresses=types.SimpleNamespace(
            inputs=["inproc://input"],
            outputs=["inproc://output"],
            frontend_stats_publish_address=None,
        ),
    )

    engine = object.__new__(AsyncOmniEngine)

    _stage_client, _out_proc, _vllm_cfg, input_processor = engine._attach_llm_stage(started)

    assert input_processor is not None
    assert isinstance(input_processor.input_preprocessor, DummyOmniInputPreprocessor)
    assert input_processor.input_preprocessor.renderer is input_processor.renderer

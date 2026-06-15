import contextlib
import importlib
import os
import time
import types

import pytest

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.stage_init_utils import (
    LogicalStageInitPlan,
    ReplicaInitPlan,
    build_stage0_input_processor,
    compute_replica_layout,
)
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_llm_metadata(
    stage_id: int,
    *,
    replica_id: int = 0,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        runtime_cfg={},
        prompt_expand_func=None,
        final_output=final_output,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"sp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[] if stage_id == 0 else [stage_id - 1],
        engine_output_type="token_ids",
        replica_id=replica_id,
        is_comprehension=is_comprehension,
    )


def _make_diffusion_metadata(stage_id: int, *, replica_id: int = 0, final_output_type: str = "image"):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="diffusion",
        runtime_cfg={"devices": str(replica_id)},
        prompt_expand_func=None,
        final_output=True,
        final_output_type=final_output_type,
        default_sampling_params=types.SimpleNamespace(name=f"dsp-{stage_id}-{replica_id}"),
        custom_process_input_func=None,
        engine_input_source=[],
        cfg_kv_collect_func=None,
        replica_id=replica_id,
    )


def _make_llm_plan(
    stage_idx: int,
    *,
    stage_id: int,
    vllm_config: object,
    num_replicas: int = 1,
    final_output: bool = False,
    final_output_type: str | None = None,
    is_comprehension: bool = False,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=stage_id,
            stage_type="llm",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_llm_metadata(
                    stage_id,
                    replica_id=replica_id,
                    final_output=final_output,
                    final_output_type=final_output_type,
                    is_comprehension=is_comprehension and replica_id == 0,
                ),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
                stage_vllm_config=vllm_config,
                executor_class=object,
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=replicas,
    )


def _make_diffusion_plan(
    stage_idx: int,
    *,
    stage_id: int,
    num_replicas: int = 1,
):
    replicas: list[ReplicaInitPlan] = []
    for replica_id in range(num_replicas):
        stage_cfg = types.SimpleNamespace(
            stage_id=stage_id,
            stage_type="diffusion",
            runtime=types.SimpleNamespace(devices=str(replica_id)),
            engine_args={},
        )
        replicas.append(
            ReplicaInitPlan(
                replica_id=replica_id,
                num_replicas=num_replicas,
                launch_mode="local",
                stage_cfg=stage_cfg,
                metadata=_make_diffusion_metadata(stage_id, replica_id=replica_id),
                stage_connector_spec={},
                omni_kv_connector=(None, None, None),
            )
        )
    return LogicalStageInitPlan(
        stage_idx=stage_idx,
        stage_id=stage_id,
        replicas=replicas,
    )


def test_stage_engine_core_client_module_reload_keeps_forward_refs_deferred():
    """Regression test for forward references in make_async_mp_client."""
    import vllm_omni.engine.stage_engine_core_client as client_mod

    importlib.reload(client_mod)

    assert client_mod.StageEngineCoreClientBase.make_async_mp_client.__annotations__["return"] == (
        "StageEngineCoreClient | DPLBStageEngineCoreClient"
    )


def test_compute_replica_layout_splits_diffusion_devices_by_world_size():
    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={"parallel_config": {"tensor_parallel_size": 2}},
        runtime={"devices": "0,1,2,3", "num_replicas": 2},
    )

    replicas_per_stage, replica_devices_map = compute_replica_layout([stage_cfg])

    assert replicas_per_stage == [2]
    assert replica_devices_map == {0: ["0,1", "2,3"]}


def test_collect_initialized_clients_for_cleanup_deduplicates_clients():
    shared = types.SimpleNamespace(name="shared")
    extra = types.SimpleNamespace(name="extra")

    cleanup_clients = StageRuntime._collect_initialized_clients_for_cleanup(
        stage_pools=[types.SimpleNamespace(clients=[shared, None])],
        initialized_clients_by_stage={0: [shared], 1: [extra]},
    )

    assert cleanup_clients == [shared, extra]


def test_initialize_local_diffusion_replica_restores_device_visibility_after_local_init(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.engine.stage_engine_startup import StageReplicaResources
    from vllm_omni.platforms import current_omni_platform

    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    plan = _make_diffusion_plan(0, stage_id=0).replicas[0]

    env_var = current_omni_platform.device_control_env_var
    old_env = os.environ.get(env_var)
    os.environ[env_var] = "0,1"
    runtime._init_visible_devices_baseline = "0,1"

    monkeypatch.setattr(runtime_mod, "inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(
        runtime_mod,
        "launch_diffusion_stage_replica",
        lambda **_: (types.SimpleNamespace(), StageReplicaResources()),
    )

    try:
        runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=1)
        assert os.environ.get(env_var) == "0,1"
    finally:
        if old_env is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = old_env


def test_initialize_local_diffusion_replica_passes_stage_init_timeout_and_inline_flag(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.engine.stage_engine_startup import StageReplicaResources

    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=4,
        async_chunk=False,
    )

    plan = _make_diffusion_plan(0, stage_id=0).replicas[0]

    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime_mod, "inject_kv_stage_info", lambda *_: None)

    def _capture_launch_diffusion_stage_replica(**kwargs):
        captured["stage_id"] = kwargs["metadata"].stage_id
        captured["stage_init_timeout"] = kwargs["stage_init_timeout"]
        captured["batch_size"] = kwargs["batch_size"]
        captured["use_inline"] = kwargs["use_inline"]
        captured["omni_master_server"] = kwargs["omni_master_server"]
        return types.SimpleNamespace(), StageReplicaResources()

    monkeypatch.setattr(runtime_mod, "launch_diffusion_stage_replica", _capture_launch_diffusion_stage_replica)

    runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=302)

    assert captured == {
        "stage_id": 0,
        "stage_init_timeout": 302,
        "batch_size": 4,
        "use_inline": True,
        "omni_master_server": None,
    }


def test_stage_runtime_initializes_stage_pools(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod

    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace(), types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2, is_comprehension=True),
        _make_llm_plan(1, stage_id=1, vllm_config=cfg1, final_output=True),
    ]

    stage0_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=True,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0"),
    )
    stage0_client_r1 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=False,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp0r1"),
    )
    stage1_client_r0 = types.SimpleNamespace(
        stage_type="llm",
        is_comprehension=False,
        final_output=True,
        final_output_type=None,
        default_sampling_params=types.SimpleNamespace(name="sp1"),
    )
    initialized_clients = {
        0: [stage0_client_r0, stage0_client_r1],
        1: [stage1_client_r0],
    }

    stage0_output_processor = object()
    stage1_output_processor = object()
    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)
    monkeypatch.setattr(runtime, "_initialize_stage_replicas", lambda *_: initialized_clients)
    monkeypatch.setattr(
        runtime_mod,
        "build_llm_stage_output_processor",
        lambda plan, _cfg, **_kw: stage0_output_processor if plan.stage_idx == 0 else stage1_output_processor,
    )

    runtime.initialize()

    assert len(runtime.stage_pools) == 2
    assert runtime.stage_pools[0].stage_client is stage0_client_r0
    assert runtime.stage_pools[1].stage_client is stage1_client_r0
    assert runtime.stage_pools[0].stage_vllm_config is cfg0
    assert runtime.stage_pools[1].stage_vllm_config is cfg1
    assert runtime.stage_pools[0].output_processor is stage0_output_processor
    assert runtime.stage_pools[1].output_processor is stage1_output_processor


def test_build_logical_stage_init_plans_applies_replica_device_splits(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod

    runtime = StageRuntime(
        stage_configs=[
            types.SimpleNamespace(
                stage_id=0, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="0")
            ),
            types.SimpleNamespace(
                stage_id=1, stage_type="llm", engine_args={}, runtime=types.SimpleNamespace(devices="1,2,3")
            ),
        ],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    metadata_by_stage = {
        0: _make_llm_metadata(0),
        1: _make_llm_metadata(1),
    }

    monkeypatch.setattr(
        runtime_mod,
        "extract_stage_metadata",
        lambda cfg: types.SimpleNamespace(**metadata_by_stage[cfg.stage_id].__dict__),
    )
    monkeypatch.setattr(runtime_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(runtime_mod, "resolve_omni_kv_config_for_stage", lambda *_: (None, None, None))
    monkeypatch.setattr(runtime_mod, "build_engine_args_dict", lambda *_, **__: {})
    monkeypatch.setattr(
        runtime_mod,
        "build_vllm_config",
        lambda stage_cfg, *_args, **_kwargs: (types.SimpleNamespace(tag=f"cfg-{stage_cfg.stage_id}"), object),
    )

    stage_plans = runtime._build_logical_stage_init_plans(
        omni_transfer_config=None,
        replicas_per_stage=[1, 3],
        replica_devices_map={1: ["1", "2", "3"]},
    )

    assert [plan.stage_id for plan in stage_plans] == [0, 1]
    assert [replica.stage_cfg.runtime.devices for replica in stage_plans[1].replicas] == ["1", "2", "3"]
    assert [replica.replica_id for replica in stage_plans[1].replicas] == [0, 1, 2]
    assert all(replica.num_replicas == 3 for replica in stage_plans[1].replicas)


def test_initialize_stage_replicas_collects_results_by_stage_and_replica_id(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=123,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    cfg1 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [
        _make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2),
        _make_llm_plan(1, stage_id=1, vllm_config=cfg1, num_replicas=2),
    ]

    clients = {
        (0, 0): types.SimpleNamespace(name="stage0-replica0"),
        (0, 1): types.SimpleNamespace(name="stage0-replica1"),
        (1, 0): types.SimpleNamespace(name="stage1-replica0"),
        (1, 1): types.SimpleNamespace(name="stage1-replica1"),
    }

    def _initialize_replica(plan, _stage_init_timeout):
        time.sleep(0.02 * (3 - plan.metadata.stage_id - plan.replica_id))
        return clients[(plan.metadata.stage_id, plan.replica_id)]

    monkeypatch.setattr(runtime, "_initialize_replica", _initialize_replica)

    initialized_clients = runtime._initialize_stage_replicas(stage_plans, stage_init_timeout=123)

    assert initialized_clients == {
        0: [clients[(0, 0)], clients[(0, 1)]],
        1: [clients[(1, 0)], clients[(1, 1)]],
    }


def test_remote_replicas_use_distinct_init_group_keys():
    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=123,
        diffusion_batch_size=1,
        async_chunk=False,
    )
    plan = _make_llm_plan(
        1,
        stage_id=1,
        vllm_config=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64)),
        num_replicas=2,
    )

    for replica in plan.replicas:
        replica.launch_mode = "remote"
        replica.metadata.runtime_cfg = None

    assert [runtime._replica_init_group_key(replica) for replica in plan.replicas] == [
        "remote:1:0",
        "remote:1:1",
    ]


def test_initialize_stages_cleans_up_successful_replicas_after_partial_multi_replica_failure(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)

    def _initialize_replica(plan, _stage_init_timeout):
        if plan.replica_id == 0:
            return initialized_client
        time.sleep(0.05)
        raise RuntimeError("replica launch failed")

    monkeypatch.setattr(runtime, "_initialize_replica", _initialize_replica)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(runtime, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        runtime.initialize()

    assert captured_cleanup == [[initialized_client]]


def test_initialize_stages_cleans_up_late_successful_replicas_after_early_multi_replica_failure(monkeypatch):
    runtime = StageRuntime(
        stage_configs=[types.SimpleNamespace()],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=1,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    cfg0 = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=64))
    stage_plans = [_make_llm_plan(0, stage_id=0, vllm_config=cfg0, num_replicas=2)]
    initialized_client = types.SimpleNamespace(shutdown=lambda: None)

    monkeypatch.setattr(runtime, "_prepare_stage_plans", lambda: stage_plans)

    def _initialize_stage_replicas(_stage_plans, _stage_init_timeout):
        exc = RuntimeError("replica launch failed")
        exc._initialized_clients_by_stage = {0: [None, initialized_client]}
        raise exc

    monkeypatch.setattr(runtime, "_initialize_stage_replicas", _initialize_stage_replicas)

    captured_cleanup: list[list[object]] = []

    def _capture_shutdown(clients):
        captured_cleanup.append(list(clients))

    monkeypatch.setattr(runtime, "_shutdown_initialized_clients", _capture_shutdown)

    with pytest.raises(RuntimeError, match="replica launch failed"):
        runtime.initialize()

    assert captured_cleanup == [[initialized_client]]


def test_initialize_local_llm_replica_passes_stage_init_timeout_to_complete_stage_handshake(monkeypatch):
    import vllm_omni.engine.stage_runtime as runtime_mod
    from vllm_omni.platforms import current_omni_platform

    runtime = StageRuntime(
        stage_configs=[],
        model="dummy-model",
        config_path="dummy-config",
        stage_init_timeout=302,
        diffusion_batch_size=1,
        async_chunk=False,
    )

    fake_vllm_config = types.SimpleNamespace()
    fake_addresses = types.SimpleNamespace(inputs=["in"], outputs=["out"], frontend_stats_publish_address=None)
    captured_timeout: int | None = None

    plan = ReplicaInitPlan(
        replica_id=0,
        num_replicas=1,
        launch_mode="local",
        stage_cfg=types.SimpleNamespace(engine_args={}, runtime=types.SimpleNamespace(devices="0")),
        metadata=types.SimpleNamespace(stage_id=0, runtime_cfg={"devices": "0"}),
        stage_connector_spec={},
        omni_kv_connector=(None, None, None),
        stage_vllm_config=fake_vllm_config,
        executor_class=object,
        engine_args_dict={},
    )

    device_env_var = current_omni_platform.device_control_env_var
    prev_device_env = os.environ.get(device_env_var)
    os.environ[device_env_var] = "0"

    def _capture_acquire_device_locks(*_args):
        nonlocal captured_timeout
        captured_timeout = _args[2]
        return []

    monkeypatch.setattr(runtime_mod, "acquire_device_locks", _capture_acquire_device_locks)

    from vllm_omni.engine.stage_engine_startup import StageReplicaResources

    @contextlib.contextmanager
    def _fake_launch_stage_replica(**_kwargs):
        yield StageReplicaResources(
            manager=types.SimpleNamespace(shutdown=lambda: None),
            addresses=fake_addresses,
        )

    monkeypatch.setattr(runtime_mod, "launch_stage_replica", _fake_launch_stage_replica)
    monkeypatch.setattr(
        runtime_mod.StageEngineCoreClientBase,
        "make_async_mp_client",
        staticmethod(lambda **_: types.SimpleNamespace(shutdown=lambda: None)),
    )

    try:
        runtime._initialize_local_llm_replica(plan, 302)
    finally:
        if prev_device_env is None:
            os.environ.pop(device_env_var, None)
        else:
            os.environ[device_env_var] = prev_device_env

    assert captured_timeout == 302


def test_build_engine_args_cli_tokenizer_overrides_inferred_base_tokenizer(tmp_path):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"model_subdir": "llm"},
        default_sampling_params={},
    )

    engine_args = build_engine_args_dict(
        stage_cfg,
        str(tmp_path),
        cli_tokenizer="/external/tokenizer",
    )

    assert engine_args["model"] == os.path.join(str(tmp_path), "llm")
    assert engine_args["tokenizer"] == "/external/tokenizer"


def test_build_engine_args_keeps_stage_owned_tokenizer_subdir(tmp_path):
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args={"model_subdir": "llm", "tokenizer_subdir": "tokenizer"},
        default_sampling_params={},
    )

    engine_args = build_engine_args_dict(
        stage_cfg,
        str(tmp_path),
        cli_tokenizer="/external/tokenizer",
    )

    assert engine_args["model"] == os.path.join(str(tmp_path), "llm")
    assert engine_args["tokenizer"] == os.path.join(str(tmp_path), "tokenizer")


def test_build_stage0_input_processor_uses_omni_input_preprocessor(monkeypatch):
    import vllm_omni.engine.stage_init_utils as init_mod

    class DummyInputProcessor:
        def __init__(self, vllm_config):
            self.vllm_config = vllm_config
            self.renderer = object()
            self.input_preprocessor = None

    class DummyOmniInputPreprocessor:
        def __init__(self, vllm_config, renderer=None):
            self.vllm_config = vllm_config
            self.renderer = renderer

    monkeypatch.setattr(init_mod, "InputProcessor", DummyInputProcessor)
    monkeypatch.setattr(init_mod, "OmniInputPreprocessor", DummyOmniInputPreprocessor)

    input_processor = build_stage0_input_processor(
        types.SimpleNamespace(model_config=types.SimpleNamespace(try_get_generation_config=lambda: {}))
    )

    assert isinstance(input_processor.input_preprocessor, DummyOmniInputPreprocessor)
    assert input_processor.input_preprocessor.renderer is input_processor.renderer


def test_inject_kv_stage_info_infers_sender_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {
                "need_send_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {"need_recv_cache": True},
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage0, 0, [stage0, stage1])

    assert stage0.engine_args["omni_kv_config"]["stage_id"] == 0
    assert stage0.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}


def test_inject_kv_stage_info_infers_receiver_tp_topology():
    from vllm_omni.engine.stage_init_utils import inject_kv_stage_info

    stage0 = types.SimpleNamespace(
        stage_id=0,
        engine_args={
            "tensor_parallel_size": 4,
            "omni_kv_config": {"need_send_cache": True},
        },
        engine_input_source=[],
    )
    stage1 = types.SimpleNamespace(
        stage_id=1,
        engine_args={
            "parallel_config": {
                "tensor_parallel_size": 2,
                "cfg_parallel_size": 1,
            },
            "omni_kv_config": {
                "need_recv_cache": True,
                "omni_from_stage": "0",
                "omni_to_stage": "1",
            },
        },
        engine_input_source=[0],
    )

    inject_kv_stage_info(stage1, 1, [stage0, stage1])

    assert stage1.engine_args["omni_kv_config"]["stage_id"] == 1
    assert stage1.engine_args["omni_kv_config"]["engine_input_source"] == [0]
    assert stage1.engine_args["omni_kv_config"]["rank_mapping"] == {"from_tp": 4, "to_tp": 2}


def test_resolve_stage_configs_injects_global_diffusion_attention_when_missing(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    stage_cfg = types.SimpleNamespace(
        stage_type="diffusion",
        engine_args=types.SimpleNamespace(
            diffusion_attention_config=None,
            lora_path=None,
            lora_scale=None,
            enable_sleep_mode=None,
            quantization_config=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg]),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "FLASH_ATTN"},
    )

    diffusion_attention_config = stage_configs[0].engine_args.diffusion_attention_config
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"


def test_resolve_stage_configs_preserves_stage_diffusion_attention(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    existing_attention = AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA"))
    stage_cfg = types.SimpleNamespace(
        stage_type="diffusion",
        engine_args=types.SimpleNamespace(
            diffusion_attention_config=existing_attention,
            lora_path=None,
            lora_scale=None,
            enable_sleep_mode=None,
            quantization_config=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg]),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "FLASH_ATTN"},
    )

    assert stage_configs[0].engine_args.diffusion_attention_config is existing_attention


def test_resolve_stage_configs_does_not_inject_diffusion_attention_into_llm_stage(monkeypatch):
    import vllm_omni.engine.async_omni_engine as engine_mod

    engine = object.__new__(AsyncOmniEngine)
    stage_cfg = types.SimpleNamespace(
        stage_type="llm",
        engine_args=types.SimpleNamespace(
            attention_config={"backend": "FLASH_ATTN"},
            enable_sleep_mode=None,
        ),
    )

    monkeypatch.setattr(
        engine_mod,
        "load_and_resolve_stage_configs",
        lambda *args, **kwargs: ("dummy-config", [stage_cfg]),
    )

    _config_path, stage_configs = engine._resolve_stage_configs(
        model="dummy-model",
        kwargs={"diffusion_attention_backend": "TORCH_SDPA"},
    )

    assert stage_configs[0].engine_args.attention_config == {"backend": "FLASH_ATTN"}
    assert not hasattr(stage_configs[0].engine_args, "diffusion_attention_config")


def test_extract_stage_metadata_rocm_does_not_inject_diffusion_attention(monkeypatch):
    """ROCm default attention logic only applies to LLM stages, not diffusion."""
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    monkeypatch.setattr("vllm_omni.engine.stage_init_utils.current_omni_platform.is_rocm", lambda: True)

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={},
        runtime={},
        engine_input_source=[],
        final_output=False,
        final_output_type=None,
    )

    metadata = extract_stage_metadata(stage_cfg)

    assert metadata.stage_type == "diffusion"
    assert "diffusion_attention_config" not in stage_cfg.engine_args


def test_build_engine_args_dict_normalizes_diffusion_attention_config():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={
            "diffusion_attention_config": {
                "default": {"backend": "FLASH_ATTN"},
                "per_role": {"cross": {"backend": "TORCH_SDPA"}},
            }
        },
        runtime={},
    )

    engine_args_dict = build_engine_args_dict(stage_cfg, model="dummy-model")

    diffusion_attention_config = engine_args_dict["diffusion_attention_config"]
    assert isinstance(diffusion_attention_config, AttentionConfig)
    assert diffusion_attention_config.default is not None
    assert diffusion_attention_config.default.backend == "FLASH_ATTN"
    assert diffusion_attention_config.per_role["cross"].backend == "TORCH_SDPA"


def test_build_engine_args_dict_uses_diffusion_attention_config_key():
    from vllm_omni.engine.stage_init_utils import build_engine_args_dict

    stage_cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        engine_args={
            "diffusion_attention_config": {
                "default": {"backend": "FLASH_ATTN"},
            }
        },
        runtime={},
    )

    engine_args_dict = build_engine_args_dict(stage_cfg, model="dummy-model")

    assert "attention_config" not in engine_args_dict
    assert engine_args_dict["diffusion_attention_config"].default.backend == "FLASH_ATTN"

"""Tests for prompt_preprocess_func stage config field."""

import types

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# A trivial preprocessor used by tests (avoids depending on any model code).
# ---------------------------------------------------------------------------


def _identity_preprocess(prompt):
    """No-op preprocessor for testing."""
    return prompt


# ---------------------------------------------------------------------------
# extract_stage_metadata loading tests
# ---------------------------------------------------------------------------


def _make_llm_stage_config(prompt_preprocess_func=None):
    """Build a minimal LLM stage config namespace for extract_stage_metadata."""
    engine_args = {
        "model_stage": "ar",
        "engine_output_type": "token_ids",
    }
    cfg = types.SimpleNamespace(
        stage_id=0,
        stage_type="llm",
        engine_args=engine_args,
        runtime={},
        engine_input_source=[],
        final_output=False,
        final_output_type=None,
        default_sampling_params={},
        is_comprehension=True,
    )
    if prompt_preprocess_func is not None:
        cfg.prompt_preprocess_func = prompt_preprocess_func
    # Ensure other optional func fields exist so getattr doesn't fall through
    cfg.prompt_expand_func = None
    cfg.cfg_kv_collect_func = None
    cfg.custom_process_input_func = None
    return cfg


def test_prompt_preprocess_func_loaded_from_config():
    """Verify prompt_preprocess_func is resolved from a dotted path."""
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    # Point at a known built-in function to verify importlib resolution.
    stage_config = _make_llm_stage_config(
        prompt_preprocess_func="copy.copy",
    )
    metadata = extract_stage_metadata(stage_config)
    assert metadata.prompt_preprocess_func is not None
    assert callable(metadata.prompt_preprocess_func)


def test_prompt_preprocess_func_none_when_not_configured():
    """Backward compat: missing field results in None."""
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    stage_config = _make_llm_stage_config(prompt_preprocess_func=None)
    metadata = extract_stage_metadata(stage_config)
    assert metadata.prompt_preprocess_func is None


def test_prompt_preprocess_func_none_when_attr_missing():
    """Backward compat: attribute not present at all results in None."""
    from vllm_omni.engine.stage_init_utils import extract_stage_metadata

    stage_config = _make_llm_stage_config()
    # Remove the attribute entirely
    if hasattr(stage_config, "prompt_preprocess_func"):
        delattr(stage_config, "prompt_preprocess_func")
    metadata = extract_stage_metadata(stage_config)
    assert metadata.prompt_preprocess_func is None


# ---------------------------------------------------------------------------
# _initialize_stages collects prompt_preprocess_func
# ---------------------------------------------------------------------------


def test_initialize_stages_collects_prompt_preprocess_func(monkeypatch):
    """Verify _initialize_stages stores prompt_preprocess_func on self."""
    import vllm_omni.engine.async_omni_engine as engine_mod
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
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
    engine.stage_configs = [
        types.SimpleNamespace(stage_id=0, stage_type="diffusion"),
    ]

    env_var = current_omni_platform.device_control_env_var
    old_env = __import__("os").environ.get(env_var)
    __import__("os").environ[env_var] = "0"

    _sentinel = lambda p: p  # noqa: E731

    metadata = types.SimpleNamespace(
        stage_id=0,
        stage_type="diffusion",
        runtime_cfg={"devices": "0"},
        prompt_expand_func=None,
        prompt_preprocess_func=_sentinel,
    )

    monkeypatch.setattr(engine_mod, "prepare_engine_environment", lambda: None)
    monkeypatch.setattr(
        engine_mod,
        "load_omni_transfer_config_for_model",
        lambda *_: None,
    )
    monkeypatch.setattr(
        engine_mod,
        "extract_stage_metadata",
        lambda _cfg: metadata,
    )
    monkeypatch.setattr(engine_mod, "get_stage_connector_spec", lambda **_: {})
    monkeypatch.setattr(
        engine_mod,
        "resolve_omni_kv_config_for_stage",
        lambda *_: (None, None, None),
    )
    monkeypatch.setattr(engine_mod, "setup_stage_devices", lambda *_: None)
    monkeypatch.setattr(engine_mod, "inject_kv_stage_info", lambda *_: None)
    monkeypatch.setattr(
        engine_mod,
        "initialize_diffusion_stage",
        lambda *_, **__: types.SimpleNamespace(is_comprehension=False),
    )
    monkeypatch.setattr(
        engine_mod,
        "finalize_initialized_stages",
        lambda stage_clients, _ip: (
            stage_clients,
            [types.SimpleNamespace()],
            [{"final_output_type": "image", "stage_type": "diffusion"}],
        ),
    )

    try:
        engine._initialize_stages(stage_init_timeout=1)
        assert engine.prompt_preprocess_func is _sentinel
    finally:
        if old_env is None:
            __import__("os").environ.pop(env_var, None)
        else:
            __import__("os").environ[env_var] = old_env

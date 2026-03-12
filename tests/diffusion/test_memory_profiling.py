from vllm_omni.diffusion.memory_profiling import (
    format_cuda_memory_snapshot,
    get_memory_log_env_var,
    is_memory_profiling_enabled,
)


def test_memory_log_env_var_name_is_stable():
    assert get_memory_log_env_var() == "VLLM_OMNI_DIFFUSION_LOG_MEMORY"


def test_memory_profiling_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", raising=False)
    assert is_memory_profiling_enabled() is False


def test_memory_profiling_truthy_values(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_DIFFUSION_LOG_MEMORY", "true")
    assert is_memory_profiling_enabled() is True


def test_format_cuda_memory_snapshot_none():
    assert format_cuda_memory_snapshot(None) == "cuda=unavailable"

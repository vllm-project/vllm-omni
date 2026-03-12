import logging

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler


class DummyRequest:
    def __init__(self):
        self.request_id = "req-1"
        self.num_computed_tokens = 3
        self.prompt_token_ids = [1, 2, 3, 4, 5]


def test_log_allocation_failure_includes_memory_snapshot(monkeypatch, caplog):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._memory_profiling_enabled = True

    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_generation_scheduler.capture_cuda_memory_snapshot",
        lambda: {
            "device": 0,
            "allocated_bytes": 1024,
            "reserved_bytes": 2048,
            "max_allocated_bytes": 4096,
            "max_reserved_bytes": 8192,
        },
    )
    monkeypatch.setattr(
        "vllm_omni.core.sched.omni_generation_scheduler.format_cuda_memory_snapshot",
        lambda snapshot: "cuda:0 allocated=0.00GiB reserved=0.00GiB max_allocated=0.00GiB max_reserved=0.00GiB",
    )

    with caplog.at_level(logging.WARNING):
        scheduler._log_allocation_failure(DummyRequest(), required_tokens=2, token_budget=1)

    assert "Diffusion scheduler allocation failed" in caplog.text
    assert "request_id=req-1" in caplog.text
    assert "token_budget=1" in caplog.text


def test_log_allocation_failure_noop_when_disabled(caplog):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler._memory_profiling_enabled = False

    with caplog.at_level(logging.WARNING):
        scheduler._log_allocation_failure(DummyRequest(), required_tokens=2, token_budget=1)

    assert caplog.text == ""


from types import SimpleNamespace

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler


def _request(cohort_id: int, cohort_size: int = 2):
    return SimpleNamespace(
        additional_information={
            "ming_tts_admission_cohort": cohort_id,
            "ming_tts_admission_cohort_size": cohort_size,
        }
    )


def _scheduler(*waiting, running=(), max_num_running_reqs: int = 8):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.waiting = list(waiting)
    scheduler.running = list(running)
    scheduler.max_num_running_reqs = max_num_running_reqs
    scheduler._ming_tts_cohort_hold_deadlines = {}
    scheduler._ming_tts_cohort_hold_s = 0.05
    return scheduler


def test_ming_tts_cohort_hold_waits_for_same_cohort_only():
    current = _request(1, cohort_size=2)
    other = _request(2, cohort_size=2)
    scheduler = _scheduler(current, other)

    assert scheduler._should_hold_ming_tts_batch(current, now=1.0) is True


def test_ming_tts_cohort_hold_releases_when_same_cohort_is_ready():
    current = _request(1, cohort_size=2)
    peer = _request(1, cohort_size=2)
    scheduler = _scheduler(current, peer)

    assert scheduler._should_hold_ming_tts_batch(current, now=1.0) is False


def test_ming_tts_cohort_hold_respects_available_slots():
    current = _request(1, cohort_size=8)
    scheduler = _scheduler(current, running=range(7), max_num_running_reqs=8)

    assert scheduler._should_hold_ming_tts_batch(current, now=1.0) is False

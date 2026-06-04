from types import SimpleNamespace

from vllm_omni.worker.ming_tts_stage_step_runner import MingTTSStageStepRunner


class MingFlashOmniTalkerForConditionalGeneration:
    pass


def test_ming_tts_step_runner_accepts_homogeneous_ming_batch():
    runner = SimpleNamespace(
        model=MingFlashOmniTalkerForConditionalGeneration(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model_stage="ming_flash_omni_tts", enforce_eager=True)
        ),
    )
    infos = [
        {"text": "a", "cfg": 2.0, "sigma": 0.25, "temperature": 0.0, "max_decode_steps": 20},
        {"text": "b", "cfg": 2.0, "sigma": 0.25, "temperature": 0.0, "max_decode_steps": 20},
    ]

    step_runner = MingTTSStageStepRunner(max_batch_size=8)

    assert step_runner.supports_batch(runner=runner, runtime_infos=infos, is_graph_capturing=False) is True


def test_ming_tts_step_runner_rejects_heterogeneous_params():
    runner = SimpleNamespace(
        model=MingFlashOmniTalkerForConditionalGeneration(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model_stage="ming_flash_omni_tts", enforce_eager=True)
        ),
    )
    infos = [
        {"text": "a", "max_decode_steps": 20},
        {"text": "b", "max_decode_steps": 21},
    ]

    step_runner = MingTTSStageStepRunner(max_batch_size=8)

    assert step_runner.supports_batch(runner=runner, runtime_infos=infos, is_graph_capturing=False) is False


def test_ming_tts_step_runner_rejects_heterogeneous_prompt_inputs():
    runner = SimpleNamespace(
        model=MingFlashOmniTalkerForConditionalGeneration(),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(model_stage="ming_tts", enforce_eager=True)),
    )
    infos = [
        {"text": "a", "instruction": "calm", "max_decode_steps": 20},
        {"text": "b", "instruction": "excited", "max_decode_steps": 20},
    ]

    step_runner = MingTTSStageStepRunner(max_batch_size=8)

    assert step_runner.supports_batch(runner=runner, runtime_infos=infos, is_graph_capturing=False) is False


def test_ming_tts_step_runner_rejects_graph_capture():
    runner = SimpleNamespace(
        model=MingFlashOmniTalkerForConditionalGeneration(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model_stage="ming_flash_omni_tts", enforce_eager=True)
        ),
    )
    infos = [{"text": "a"}, {"text": "b"}]

    step_runner = MingTTSStageStepRunner(max_batch_size=8)

    assert step_runner.supports_batch(runner=runner, runtime_infos=infos, is_graph_capturing=True) is False


def test_ming_tts_step_runner_run_and_commit_writes_per_request_audio():
    class AudioGenerator:
        def __init__(self):
            self.calls = []

        def generate_latents_batch(self, inputs_embeds, **kwargs):
            self.calls.append((tuple(inputs_embeds.shape), kwargs))
            return [[f"lat-{i}"] for i in range(inputs_embeds.shape[0])]

    class Model:
        def __init__(self):
            self.audio_generator = AudioGenerator()

        def decode_batch_latents_for_runner(self, latents_by_request, stream_decode=True):
            return [f"audio-{item[0]}" for item in latents_by_request], 44100

    runner = SimpleNamespace(
        model=Model(),
        model_intermediate_buffer={},
    )
    step_runner = MingTTSStageStepRunner(max_batch_size=8)
    runtime_infos = [{"text": "a", "max_decode_steps": 2}, {"text": "b", "max_decode_steps": 2}]
    inputs_embeds = __import__("torch").zeros(2, 3, 4)

    prepared = step_runner.prepare_step(
        request_ids=["r1", "r2"], runtime_infos=runtime_infos, inputs_embeds=inputs_embeds
    )
    step_runner.run_step(prepared=prepared, runner=runner)
    step_runner.commit_step(prepared=prepared, runner=runner)

    assert runner.model.audio_generator.calls == [
        ((2, 3, 4), {"max_steps": 2, "cfg": None, "sigma": 0.25, "temperature": 0.0, "use_static_cache": True})
    ]
    assert runner.model_intermediate_buffer["r1"]["audio"]["audio"] == "audio-lat-0"
    assert runner.model_intermediate_buffer["r2"]["audio"]["audio"] == "audio-lat-1"
    assert runner.model_intermediate_buffer["r1"]["audio"]["sr"] == 44100

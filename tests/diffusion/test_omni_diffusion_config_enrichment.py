from vllm_omni.diffusion.data import OmniDiffusionConfig


def test_enrich_config_maps_audiodit(monkeypatch):
    def fake_get_hf_file_to_dict(filename, model):
        assert model == "audio-model"
        if filename == "model_index.json":
            return None
        if filename == "config.json":
            return {"model_type": "audiodit"}
        raise AssertionError(f"unexpected filename: {filename}")

    monkeypatch.setattr(
        "vllm_omni.diffusion.data.get_hf_file_to_dict",
        fake_get_hf_file_to_dict,
    )

    od_config = OmniDiffusionConfig(model="audio-model")

    od_config.enrich_config()

    assert od_config.model_class_name == "LongCatAudioDiTPipeline"


def test_populate_audio_output_metadata(monkeypatch):
    class FakeAudioPipeline:
        support_audio_output = True
        sample_rate = 44100
        audio_channel_first = True

    monkeypatch.setattr(
        "vllm_omni.diffusion.registry.DiffusionModelRegistry._try_load_model_cls",
        staticmethod(lambda model_class_name: FakeAudioPipeline),
    )

    od_config = OmniDiffusionConfig(model="audio-model", model_class_name="FakeAudioPipeline")

    od_config.populate_audio_output_metadata()
    assert od_config.supports_audio_output is True
    assert od_config.audio_sample_rate == 44100
    assert od_config.audio_channel_first is True

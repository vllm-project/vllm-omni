from vllm_omni.diffusion.marey_serving_runtime import (
    allow_raw_latent_output,
    build_request_missing_vae_error,
    build_startup_missing_vae_error,
)


def test_allow_raw_latent_output_defaults_false(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_ALLOW_MAREY_RAW_LATENTS", raising=False)
    assert allow_raw_latent_output(default_output_type="np", model_config={}) is False


def test_allow_raw_latent_output_respects_default_output_type(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_OMNI_ALLOW_MAREY_RAW_LATENTS", raising=False)
    assert allow_raw_latent_output(default_output_type="latent", model_config={}) is True


def test_allow_raw_latent_output_model_config_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_ALLOW_MAREY_RAW_LATENTS", "0")
    assert (
        allow_raw_latent_output(
            default_output_type="np",
            model_config={"allow_raw_latent_output": True},
        )
        is True
    )


def test_startup_missing_vae_error_mentions_contract() -> None:
    message = build_startup_missing_vae_error("missing dask")
    assert "missing dask" in message
    assert "MOONVALLEY_AI_PATH" in message
    assert "PYTHONPATH" in message


def test_request_missing_vae_error_mentions_output_type() -> None:
    message = build_request_missing_vae_error("broken cv2", "np")
    assert "output_type='np'" in message
    assert "broken cv2" in message

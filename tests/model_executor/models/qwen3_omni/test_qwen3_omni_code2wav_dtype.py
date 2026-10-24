# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from vllm.config import CacheConfig, DeviceConfig, VllmConfig
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _codec_config():
    return Qwen3OmniMoeCode2WavConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        decoder_dim=32,
        upsample_rates=[2, 2],
        upsampling_ratios=[2],
        codebook_size=16,
        num_quantizers=2,
    )


def _codec(mocker, dtype, device="cuda", connector=None):
    vllm_config = mocker.Mock(
        spec=VllmConfig,
        model_config=mocker.Mock(spec=OmniModelConfig, hf_config=_codec_config(), stage_connector_config=connector),
        device_config=mocker.Mock(spec=DeviceConfig, device=torch.device(device)),
        quant_config=None,
    )
    # Simulate the loader's dtype context, but keep all tensors on CPU.
    with set_default_torch_dtype(dtype):
        return Qwen3OmniMoeCode2Wav(vllm_config=vllm_config).eval()


@pytest.fixture
def blackwell_cudnn(mocker):
    mocker.patch.object(current_omni_platform, "is_cuda", return_value=True)
    mocker.patch.object(torch.backends.cudnn, "version", return_value=92000)
    mocker.patch.object(torch.accelerator, "current_device_index", return_value=1)
    return mocker.patch.object(current_omni_platform, "get_device_capability", return_value=DeviceCapability(10, 3))


@pytest.mark.parametrize("capability", [DeviceCapability(10, 0), DeviceCapability(10, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_affected_codec_uses_float32(mocker, blackwell_cudnn, capability, dtype):
    blackwell_cudnn.return_value = capability
    with set_default_torch_dtype(dtype):
        model = _codec(mocker, dtype)
        # A nested Code2Wav copy must not change the enclosing Talker's dtype.
        assert torch.get_default_dtype() == dtype
        assert torch.nn.Linear(2, 2).weight.dtype == dtype

    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert model.code_offset.dtype == torch.long
    # Device 0 can be a different GPU: query the worker's selected device.
    blackwell_cudnn.assert_called_once_with(1)


@pytest.mark.parametrize(
    ("cuda", "device", "capability", "cudnn", "dtype"),
    [
        (False, "cpu", DeviceCapability(10, 3), 92000, torch.bfloat16),
        (True, "cpu", DeviceCapability(10, 3), 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(9, 0), 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(12, 0), 92000, torch.bfloat16),
        (True, "cuda", None, 92000, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), None, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 91900, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92001, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92100, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92400, torch.bfloat16),
        (True, "cuda", DeviceCapability(10, 3), 92000, torch.float32),
    ],
)
def test_other_environments_keep_dtype(mocker, blackwell_cudnn, cuda, device, capability, cudnn, dtype):
    mocker.patch.object(current_omni_platform, "is_cuda", return_value=cuda)
    mocker.patch.object(torch.backends.cudnn, "version", return_value=cudnn)
    blackwell_cudnn.return_value = capability
    model = _codec(mocker, dtype, device)

    assert {p.dtype for p in model.parameters()} == {dtype}


@pytest.mark.parametrize("typed_connector", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_opt_out_preserves_requested_dtype_and_caches(mocker, blackwell_cudnn, typed_connector, dtype):
    extra = {"codec_cudnn_fp32_fallback": False}
    connector = ConnectorSpec("SharedMemoryConnector", extra) if typed_connector else {"extra": extra}
    model = _codec(mocker, dtype, connector=connector)
    model.precompute_snake_caches()

    assert {p.dtype for p in model.parameters()} == {dtype}
    for module in model.modules():
        if isinstance(module, SnakeBeta):
            assert module._exp_alpha.dtype == module._inv_beta.dtype == dtype
    assert extra == {"codec_cudnn_fp32_fallback": False}
    blackwell_cudnn.assert_not_called()


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_invalid_fallback_option_fails_before_model_construction(mocker, value):
    with pytest.raises(ValueError, match="codec_cudnn_fp32_fallback must be a boolean"):
        _codec(mocker, torch.float32, device="cpu", connector={"extra": {"codec_cudnn_fp32_fallback": value}})


@pytest.mark.parametrize("profile", ["qwen3_omni_moe.yaml", "qwen3_omni_moe_mrv2.yaml"])
@pytest.mark.parametrize("fallback", [False, True])
def test_deploy_option_reaches_both_decoder_placements(mocker, blackwell_cudnn, tmp_path, profile, fallback):
    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.stage_config import resolve_deploy_yaml
    from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.engine.stage_init_utils import get_stage_connector_spec
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

    mocker.patch.dict("os.environ", {"VLLM_OMNI_TALKER_FIRST_AUDIO": "1"})
    overlay = tmp_path / "deploy.yaml"
    overlay.write_text(
        f"base_config: {get_deploy_config_path(profile)}\n"
        "connectors:\n  connector_of_shared_memory:\n    extra:\n"
        f"      codec_cudnn_fp32_fallback: {str(fallback).lower()}\n"
    )
    transfer = load_omni_transfer_config(config_dict=resolve_deploy_yaml(overlay))
    config = Qwen3OmniMoeConfig(
        architectures=["Qwen3OmniMoeForConditionalGeneration"], code2wav_config=_codec_config().to_dict()
    )
    checkpoint_dir = tmp_path / "model"
    config.save_pretrained(checkpoint_dir)
    decoders = []
    for stage_id in (1, 2):
        spec = get_stage_connector_spec(transfer, stage_id, async_chunk=True)
        model_config = OmniEngineArgs(
            model=str(checkpoint_dir),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            stage_id=stage_id,
            model_stage="talker" if stage_id == 1 else "code2wav",
            worker_cls=(
                "vllm_omni.worker.gpu_ar_worker.GPUARWorker"
                if stage_id == 1
                else "vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker"
            ),
            stage_connector_spec=spec,
            async_chunk=True,
            use_v2_model_runner=profile.endswith("mrv2.yaml"),
        ).create_model_config()
        vllm_config = VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(enable_prefix_caching=False),
        )
        # Detection is simulated; no tensor or CUDA context is allocated.
        vllm_config.device_config.device = torch.device("cuda")
        with set_default_torch_dtype(model_config.dtype):
            if stage_id == 1:
                first = Qwen3OmniMoeForConditionalGeneration._build_first_frame_decoder(
                    vllm_config, config.code2wav_config, ""
                )
                assert (first is not None) == profile.endswith("mrv2.yaml")
                if first is not None:
                    decoders.append(first.code2wav)
            else:
                decoders.append(
                    Qwen3OmniMoeCode2Wav(
                        vllm_config=vllm_config.with_hf_config(
                            config.code2wav_config, architectures=["Qwen3OmniMoeCode2Wav"]
                        )
                    )
                )
        assert model_config.dtype == torch.bfloat16
        assert model_config.stage_connector_config["extra"]["codec_cudnn_fp32_fallback"] is fallback
    expected = torch.float32 if fallback else torch.bfloat16
    assert all({p.dtype for p in decoder.parameters()} == {expected} for decoder in decoders)


def test_explicit_device_index_takes_precedence(mocker, blackwell_cudnn):
    model = _codec(mocker, torch.bfloat16, torch.device("cuda:2"))

    blackwell_cudnn.assert_called_once_with(2)
    assert model.code_embedding.weight.dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fallback_loads_checkpoint_and_matches_explicit_float32(mocker, blackwell_cudnn, dtype):
    torch.manual_seed(8935)
    reference = _codec(mocker, torch.float32)
    checkpoint = [("code2wav." + name, p.to(dtype)) for name, p in reference.named_parameters()]
    model = _codec(mocker, dtype)
    assert model.load_weights(checkpoint) == reference.load_weights(checkpoint)
    model.precompute_snake_caches()
    reference.precompute_snake_caches()

    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert all(b.dtype == torch.float32 for b in model.buffers() if b.is_floating_point())
    for frames in (4, 29, 50):
        codes = torch.randint(0, 16, (1, 2, frames))
        actual = model(codes)
        assert actual.dtype == torch.float32
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference(codes), rtol=0, atol=0)

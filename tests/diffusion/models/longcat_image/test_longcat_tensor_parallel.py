import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image import prepare_pos_ids

DTYPES = {"fp32": torch.float32}


@pytest.fixture(scope="module")
def device():
    init_distributed_environment()
    return torch.device(f"cuda:{torch.distributed.get_rank()}")


@pytest.fixture(scope="module", autouse=True)
def init_distributed(device):
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device=device))):
        yield
    # Teardown distributed environment
    destroy_distributed_environment()


@pytest.fixture(scope="module")
def inputs(device) -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    torch.distributed.barrier()  # FIXME: idk why but doesn't work without this

    batch_size = 1
    latent_size = (32, 32)
    in_channels = 64
    txt_seq_len = 512
    joint_attention_dim = 3584

    img_ids = prepare_pos_ids(
        modality_id=1, type="image", start=(512, 512), height=latent_size[0], width=latent_size[1]
    ).to(device)
    txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=txt_seq_len).to(device)

    hidden_states = torch.rand(
        batch_size, latent_size[0] * latent_size[1], in_channels, device=device, dtype=torch.float32
    )
    encoder_hidden_states = torch.rand(batch_size, txt_seq_len, joint_attention_dim, device=device, dtype=torch.float32)
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }


def init_model(od_config, device) -> LongCatImageTransformer2DModel:
    torch.manual_seed(42)
    return LongCatImageTransformer2DModel(od_config=od_config, num_layers=2, num_single_layers=2).eval().to(device)


def model_inference(model, inputs, dtype) -> torch.Tensor:
    model.to(dtype)
    with torch.inference_mode():
        sample = model(**{arg: data.to(dtype) for arg, data in inputs.items()}).sample
    return sample.cpu()


@pytest.fixture(scope="module")
def single_device_reference(inputs, device) -> dict[str, torch.Tensor]:
    dp_size = torch.distributed.get_world_size()
    initialize_model_parallel(data_parallel_size=dp_size)
    parallel_config = DiffusionParallelConfig(data_parallel_size=dp_size)

    out = {}
    for name, dtype in DTYPES.items():
        od_config = OmniDiffusionConfig(model="test_model", dtype=dtype, parallel_config=parallel_config)
        model = init_model(od_config, device)

        with set_forward_context(omni_diffusion_config=od_config):
            out[name] = model_inference(model, inputs, dtype)

    destroy_model_parallel()
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="LongCat Tensor Parallel unit test requires >= 2 CUDA devices.",
)
@pytest.mark.parametrize("dtype", DTYPES.keys())
def test_longcat_tensor_parallel(inputs, single_device_reference, device, dtype):
    tp_size = torch.distributed.get_world_size()
    initialize_model_parallel(tensor_parallel_size=tp_size)
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=tp_size)

    od_config = OmniDiffusionConfig(model="test_model", dtype=DTYPES[dtype], parallel_config=parallel_config)
    model = init_model(od_config, device)

    with set_forward_context(omni_diffusion_config=od_config):
        tp_output = model_inference(model, inputs, DTYPES[dtype])

    destroy_model_parallel()
    torch.testing.assert_close(tp_output, single_device_reference[dtype])

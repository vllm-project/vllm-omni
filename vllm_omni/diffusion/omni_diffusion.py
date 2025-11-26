import logging
import multiprocessing as mp

import PIL.Image
import torch
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.launch_engine import launch_engine
from vllm_omni.diffusion.req import OmniDiffusionRequest
from vllm_omni.diffusion.sample import DiffusionSamplingParams
from vllm_omni.diffusion.schedule import sync_scheduler

# TODO configure logging properly
logging.basicConfig(level=logging.INFO)

logger = init_logger(__name__)


def prepare_requests(prompt: str | list[str], sampling_params: DiffusionSamplingParams | None):
    pass


class OmniDiffusion:
    def __init__(self, od_config: OmniDiffusionConfig | None = None, **kwargs):
        """Create an OmniDiffusion instance.

        You can pass either an `OmniDiffusionConfig` via `od_config`, or
        pass kwargs such as `model="Qwen/Qwen-Image"`,
        which will be forwarded to `OmniDiffusionConfig.from_kwargs`.
        """
        if od_config is None:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)
        elif isinstance(od_config, dict):
            od_config = OmniDiffusionConfig.from_kwargs(**od_config)

        self.od_config = od_config

        config_dict = get_hf_file_to_dict(
            "model_index.json",
            od_config.model,
        )
        od_config.model_class_name = config_dict.get("_class_name", None)

        self.scheduler_process: list[mp.Process] | None = None
        # TODO:use another way to get post process func
        from vllm_omni.diffusion.models.qwen_image.qwen_image import (
            get_qwen_image_post_process_func,
        )

        self.post_process_func = get_qwen_image_post_process_func(od_config)

        self.scheduler_process = self._make_client()

    def _make_client(self) -> list[mp.Process]:
        sync_scheduler.initialize(self.od_config)

        # Get the broadcast handle from the initialized scheduler
        broadcast_handle = sync_scheduler.get_broadcast_handle()

        processes, result_handle = launch_engine(
            self.od_config,
            broadcast_handle=broadcast_handle,
            launch_http_server=False,
        )

        if result_handle is not None:
            sync_scheduler.initialize_result_queue(result_handle)
        else:
            logger.error("Failed to get result queue handle from workers")

        return processes

    def _send_to_scheduler_and_wait_for_response(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Sends a request to the scheduler and waits for a response.
        """
        return sync_scheduler.add_req(requests)

    def generate(
        self,
        prompt: str | list[str],
        sampling_params: DiffusionSamplingParams | None = None,
        **kwargs,
    ) -> list[PIL.Image.Image | None | torch.Tensor] | None:
        # Placeholder for diffusion generation logic
        prompts = []
        if isinstance(prompt, str):
            prompts.append(prompt)
        elif isinstance(prompt, list):
            prompts.extend(prompt)
        else:
            raise ValueError("Prompt must be a string or a list of strings")

        requests: list[OmniDiffusionRequest] = []
        for p in prompts:
            requests.append(
                OmniDiffusionRequest(
                    request_id=None,
                    prompt=p,
                    negative_prompt=(getattr(sampling_params, "negative_prompt", None) if sampling_params else None),
                    num_inference_steps=(
                        getattr(sampling_params, "num_inference_steps", 50) if sampling_params else 50
                    ),
                )
            )
        logger.info(f"Prepared {len(requests)} requests for generation.")
        try:
            output = self._send_to_scheduler_and_wait_for_response(requests)
            if output.error:
                raise Exception(f"{output.error}")

            logger.info("Generation completed successfully.")
            return self.post_process_func(output.output)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
        return f"Generated content based on prompt: {prompt} with sampling parameters: {sampling_params}"

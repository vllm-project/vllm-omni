import logging
import multiprocessing as mp

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig, OutputBatch
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
    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config

        self.local_scheduler_process: list[mp.Process] | None = None
        self.owns_scheduler_client: bool = False

    @classmethod
    def from_pretrained(
        cls,
        **kwargs,
    ) -> "OmniDiffusion":
        if (od_config := kwargs.get("od_config", None)) is not None:
            if isinstance(od_config, OmniDiffusionConfig):
                pass
            elif isinstance(od_config, dict):
                od_config = OmniDiffusionConfig.from_kwargs(**od_config)
        else:
            od_config = OmniDiffusionConfig.from_kwargs(**kwargs)

        return cls.from_engine_args(od_config)

    @classmethod
    def from_engine_args(cls, od_config: OmniDiffusionConfig) -> "OmniDiffusion":
        """
        Create a engine with the specified arguments.
        """
        instance = cls(
            od_config=od_config,
        )
        instance.local_scheduler_process = instance._make_client()
        instance.owns_scheduler_client = True
        return instance

    def _make_client(
        self,
    ) -> list[mp.Process]:
        sync_scheduler.initialize(self.od_config)

        # Get the broadcast handle from the initialized client
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

    def _send_to_scheduler_and_wait_for_response(self, requests: list[OmniDiffusionRequest]) -> OutputBatch:
        """
        Sends a request to the scheduler and waits for a response.
        """
        return sync_scheduler.add_req(requests)

    def generate(
        self,
        prompt: str | list[str],
        sampling_params: DiffusionSamplingParams | None = None,
        **kwargs,
    ):
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
            output_batch = self._send_to_scheduler_and_wait_for_response(requests)
            if output_batch.error:
                raise Exception(f"{output_batch.error}")

            logger.info("Generation completed successfully.")
            return output_batch.output
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
        return f"Generated content based on prompt: {prompt} with sampling parameters: {sampling_params}"

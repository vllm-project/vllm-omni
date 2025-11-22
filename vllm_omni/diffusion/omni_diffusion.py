from .req import OmniDiffusionRequest
from .sample import DiffusionSamplingParams
from .data import OutputBatch, EngineArgs, PortArgs
from .schedule import sync_scheduler_client
from .launch_engine import launch_engine
from vllm.logger import init_logger
import multiprocessing as mp
from typing import Any
import logging

# TODO configure logging properly
logging.basicConfig(level=logging.INFO)

logger = init_logger(__name__)


def prepare_requests(
    prompt: str | list[str], sampling_params: DiffusionSamplingParams | None
):
    pass


class OmniDiffusion:
    def __init__(self, engine_args: EngineArgs):
        self.engine_args = engine_args
        self.port_args = PortArgs.from_engine_args(engine_args)

        self.local_scheduler_process: list[mp.Process] | None = None
        self.owns_scheduler_client: bool = False

    @classmethod
    def from_pretrained(
        cls,
        **kwargs,
    ) -> "OmniDiffusion":
        """
        Create a DiffGenerator from a pretrained model.

        Args:
            **kwargs: Additional arguments to customize model loading, set any ServerArgs or PipelineConfig attributes here.

        Returns:
            The created DiffGenerator

        Priority level: Default pipeline config < User's pipeline config < User's kwargs
        """
        # If users also provide some kwargs, it will override the ServerArgs and PipelineConfig.

        if (engine_args := kwargs.get("engine_args", None)) is not None:
            if isinstance(engine_args, EngineArgs):
                pass
            elif isinstance(engine_args, dict):
                engine_args = EngineArgs.from_kwargs(**engine_args)
        else:
            engine_args = EngineArgs.from_kwargs(**kwargs)

        return cls.from_engine_args(engine_args)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "OmniDiffusion":
        """
        Create a engine with the specified arguments.

        Args:
            engine_args: The inference arguments

        Returns:
            The created OmniDiffusion
        """
        instance = cls(
            engine_args=engine_args,
        )
        instance.local_scheduler_process = instance._make_client()
        instance.owns_scheduler_client = True
        return instance

    def _make_client(
        self,
    ) -> list[mp.Process]:
        """Check if a local server is running; if not, start it and return the process handles."""
        # First, we need a client to test the server. Initialize it temporarily.
        sync_scheduler_client.initialize(self.engine_args)

        processes = launch_engine(self.engine_args, launch_http_server=False)

        return processes

    def _send_to_scheduler_and_wait_for_response(
        self, batch: list[OmniDiffusionRequest]
    ) -> OutputBatch:
        """
        Sends a request to the scheduler and waits for a response.
        """
        return sync_scheduler_client.forward(batch)

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
                    negative_prompt=(
                        getattr(sampling_params, "negative_prompt", None)
                        if sampling_params
                        else None
                    ),
                    num_inference_steps=(
                        getattr(sampling_params, "num_inference_steps", 50)
                        if sampling_params
                        else 50
                    ),
                    true_cfg_scale=(
                        getattr(sampling_params, "guidance_scale", 4.0)
                        if sampling_params
                        else 4.0
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
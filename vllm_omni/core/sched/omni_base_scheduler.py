from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler
from vllm.logger import init_logger


logger = init_logger(__name__)

class BaseOmniScheduler(VLLMScheduler):
    @classmethod
    def validate_stage_config(
        cls,
        global_policy: str | None,
        stage_id: int,
        is_comprehension: bool,
        is_dit: bool,
    ):
        logger.warning(f"Stage {stage_id}: "
                    "Class {cls.__name__} has not implemented validate_stage_config method."
                    "it is possible that the scheduler is not compatible with the stage.") 
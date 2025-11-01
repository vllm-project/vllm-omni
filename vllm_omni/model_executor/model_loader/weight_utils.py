import time
from typing import Optional, Union

import huggingface_hub
from huggingface_hub import snapshot_download
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import DisabledTqdm, get_lock

logger = init_logger(__name__)


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: list[str],
    revision: Optional[str] = None,
    ignore_patterns: Optional[Union[str, list[str]]] = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (list[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.
        ignore_patterns (Optional[Union[str, list[str]]]): The patterns to
            filter out the weight files. Files matched by any of the patterns
            will be ignored.

    Returns:
        str: The path to the downloaded model weights.
    """
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=local_only,
        )
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder

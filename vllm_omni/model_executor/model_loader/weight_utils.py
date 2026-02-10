import time
from pathlib import Path

import huggingface_hub
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import DisabledTqdm, get_lock

if envs.VLLM_USE_MODELSCOPE:
    from modelscope.hub.snapshot_download import snapshot_download
else:
    from huggingface_hub import snapshot_download

logger = init_logger(__name__)


def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    require_all: bool = False,
) -> str:
    """Download model weights from Hugging Face Hub. Users can specify the
    allow_patterns to download only the necessary weights.

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
        require_all (bool): If True, will iterate through and download files
            matching all patterns in allow_patterns. If False, will stop after
            the first pattern that matches any files.

    Returns:
        str: The path to the downloaded model weights.
    """
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    download_kwargs = {"tqdm_class": DisabledTqdm} if not envs.VLLM_USE_MODELSCOPE else {}

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        if require_all:
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_only,
                **download_kwargs,
            )
        else:
            for allow_pattern in allow_patterns:
                hf_folder = snapshot_download(
                    model_name_or_path,
                    allow_patterns=allow_pattern,
                    ignore_patterns=ignore_patterns,
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_only,
                    **download_kwargs,
                )
                # If we have downloaded weights for this allow_pattern,
                # we don't need to check the rest, unless require_all is set.
                if any(Path(hf_folder).glob(allow_pattern)):
                    break
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder

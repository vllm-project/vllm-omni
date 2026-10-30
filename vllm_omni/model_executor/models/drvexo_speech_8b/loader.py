# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM loader for stage views into the single DRVEXO checkpoint."""

import time
from pathlib import Path

from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from .checkpoint import iter_view, open_view


@register_model_loader("drvexo_unified")
class DRVEXOUnifiedModelLoader(DefaultModelLoader):
    def _get_weights_iterator(self, source):
        directory = Path(source.model_or_path)
        if source.subfolder:
            directory /= source.subfolder
        if self.counter_before_loading_weights == 0:
            self.counter_before_loading_weights = time.perf_counter()
        return ((source.prefix + name, value) for name, value in iter_view(directory / "weights.json"))

    def download_model(self, model_config):
        # Deliberately local-only: the complete bundle must be downloaded first.
        open_view(Path(model_config.model) / "weights.json")

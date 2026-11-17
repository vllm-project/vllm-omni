# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Focused regression for Bagel's device-placement guard.

The guard decides whether ``BagelPipeline.__init__` may run the eager
``self.to(self.device)``. Both streaming offload strategies must skip it:
``LAYER_WISE`` and ``DISTRIBUTED_LAYER_WISE``. ``DISTRIBUTED_LAYER_WISE`` is
selected as soon as ``diffusion_offload_config.layer_options`` is non-empty
(even for a plain ``resident_layers`` entry), which is exactly the shape that
used to fall into the eager branch and OOM on a 24 GB card.

These cases are mock-level on purpose: no pipeline build, no device work.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.bagel.pipeline_bagel import should_defer_device_placement

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _od_config(offload=None, use_hsdp=False):
    return SimpleNamespace(
        diffusion_offload_config=offload,
        parallel_config=SimpleNamespace(data_parallel_size=1, use_hsdp=use_hsdp),
    )


def test_no_offload_keeps_eager_placement():
    # Nothing streams weights: the eager .to(device) must still happen.
    assert should_defer_device_placement(_od_config()) is False


def test_legacy_layerwise_flag_defers():
    od = _od_config()
    od.enable_layerwise_offload = True
    assert should_defer_device_placement(od) is True


@pytest.mark.parametrize(
    "layer_options",
    [
        # The reproduction from #10464: a non-empty layer_options entry alone
        # resolves to DISTRIBUTED_LAYER_WISE, so the guard must defer.
        {"dit": {"resident_layers": 6}},
        {"dit": {"weight_transfer": "allgather"}},
    ],
)
def test_public_config_with_layer_options_defers(layer_options):
    od = _od_config(
        {
            "mode": "layer",
            "components": ["dit"],
            "layer_options": layer_options,
            "pin_memory": True,
        }
    )
    assert should_defer_device_placement(od) is True


def test_public_config_without_layer_options_defers():
    # mode=layer with empty layer_options resolves to plain LAYER_WISE, which
    # already skipped the eager branch before the fix; keep it deferred.
    od = _od_config({"mode": "layer", "components": ["dit"]})
    assert should_defer_device_placement(od) is True


def test_quantization_defers():
    # vLLM linears live on meta until the loader materializes them.
    assert should_defer_device_placement(_od_config(), quant_config=object()) is True


def test_hsdp_defers():
    assert should_defer_device_placement(_od_config(use_hsdp=True)) is True


def test_repro_config_old_predicate_used_eager_placement():
    """Characterizes the bug this PR fixes (issue #10464).

    The pre-fix predicate only accepted plain LAYER_WISE, so a YAML that sets
    layer_options (which selects DISTRIBUTED_LAYER_WISE) fell through to the
    eager self.to(self.device) and OOMed on a 24 GB card. Both halves are
    asserted here so the regression cannot silently come back.
    """
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload_strategy

    od = _od_config(
        {
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"resident_layers": 6}},
            "pin_memory": True,
        }
    )

    assert resolve_offload_strategy(od) is OffloadStrategy.DISTRIBUTED_LAYER_WISE

    old_defers = resolve_offload_strategy(od) is OffloadStrategy.LAYER_WISE or od.parallel_config.use_hsdp
    assert old_defers is False

    assert should_defer_device_placement(od) is True

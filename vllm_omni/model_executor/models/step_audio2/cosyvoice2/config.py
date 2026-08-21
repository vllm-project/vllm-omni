# SPDX-License-Identifier: Apache-2.0
"""Build the in-tree CosyVoice2 Flow model from its HyperPyYAML config."""

from pathlib import Path

import yaml

from .cfm import CausalConditionalCFM
from .dit import DiT
from .flow import CausalMaskedDiffWithXvec
from .upsample_encoder import UpsampleConformerEncoderV2


class _FlowConfigLoader(yaml.SafeLoader):
    """Load HyperPyYAML object declarations as plain dictionaries."""


def _construct_object_config(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> dict:
    del suffix
    return loader.construct_mapping(node, deep=True)


_FlowConfigLoader.add_multi_constructor("!new:", _construct_object_config)


def load_flow_model(config_path: str | Path) -> CausalMaskedDiffWithXvec:
    """Instantiate the vendored Flow modules without importing CosyVoice2."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        flow_config = yaml.load(config_file, Loader=_FlowConfigLoader)["flow"]

    decoder_config = dict(flow_config.get("decoder", {}))
    estimator_config = decoder_config.pop("estimator", {})

    estimator = DiT(**estimator_config)
    encoder = UpsampleConformerEncoderV2(**flow_config.get("encoder", {}))
    decoder = CausalConditionalCFM(estimator=estimator, **decoder_config)
    return CausalMaskedDiffWithXvec(
        input_size=flow_config.get("input_size", 512),
        output_size=flow_config.get("output_size", 80),
        spk_embed_dim=flow_config.get("spk_embed_dim", 192),
        output_type=flow_config.get("output_type", "mel"),
        vocab_size=flow_config.get("vocab_size", 6561),
        encoder=encoder,
        decoder=decoder,
        input_embedding=flow_config.get("input_embedding"),
    )

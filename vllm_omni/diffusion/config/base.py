from abc import ABC

from transformers.configuration_utils import PretrainedConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.data import TransformerConfig

logger = init_logger(__name__)


class BaseDiTConfig(ABC, PretrainedConfig):
    _class_name: str | None = None

    @classmethod
    def from_tf_config(cls, cfg: TransformerConfig):
        """This name is confusing - TransformersConfig is our generic
        wrapper around the Diffusers DiT config. Here, we are converting
        the wrapped class to a dict so that we can wrap it as a subclass.

        TODO (Alex) further consolidate this in the future.
        """
        model_dict = cfg.to_dict()
        if "_class_name" in model_dict:
            parsed_class_name = model_dict.pop("_class_name")
            cls._validate_class_name(parsed_class_name)

        return cls.from_dict(model_dict)

    @classmethod
    def _validate_class_name(cls, parsed_class_name):
        """Warn if th class name looks incorrect."""
        if cls._class_name is None:
            logger.warn("Model config does not have a set defined _class_name; couldn't validate!")
        if cls._class_name != parsed_class_name:
            logger.warn(
                "Model config expected _class_name %s, but got %s",
                cls._class_name,
                parsed_class_name,
            )

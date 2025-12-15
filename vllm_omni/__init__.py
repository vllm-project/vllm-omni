"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

# vllm_omni/__init__.py
import vllm
from transformers import AutoConfig, Qwen2Config
from vllm.model_executor.models import ModelRegistry

from vllm_omni.model_executor.models.bagel.bagel import BagelForConditionalGeneration
from vllm_omni.model_executor.models.bagel.configuration_bagel import BagelConfig
from vllm_omni.model_executor.models.bagel.qwen2_bagel import Qwen2ForCausalLM

from .config import OmniModelConfig
from .entrypoints.async_omni import AsyncOmni

# Main entry points
from .entrypoints.omni import Omni

from .version import __version__, __version_tuple__  # isort:skip


# 注册给 Transformers
AutoConfig.register("bagel", BagelConfig)
# =========================================================
# 拦截 vLLM Config，修复 Worker 里的字典问题
# =========================================================

_original_with_hf_config = vllm.config.VllmConfig.with_hf_config


def _patched_with_hf_config(self, hf_config, *args, **kwargs):
    """
    拦截函数：如果 hf_config 是字典（发生在 Worker 进程中），
    强行将其转换回 BagelConfig 对象。
    """
    # 检查是不是字典 (这就是你报错的原因)
    if isinstance(hf_config, dict):
        try:
            # 尝试用我们定义的类重新实例化它
            # 使用 **hf_config 将字典解包为参数
            hf_config = BagelConfig(**hf_config)
        except Exception:
            # 如果实例化失败，使用通用 Qwen2Config 兜底
            hf_config = Qwen2Config(**hf_config)

    # 再次检查 text_config (防止之前的 get_text_config 报错)
    if not hasattr(hf_config, "get_text_config"):
        # 动态给对象绑定一个方法
        hf_config.get_text_config = lambda: hf_config

    # 调用 vLLM 原来的逻辑，这时候 hf_config 已经是对象了，不会报错
    return _original_with_hf_config(self, hf_config, *args, **kwargs)


# 覆盖 vLLM 类的方法
vllm.config.VllmConfig.with_hf_config = _patched_with_hf_config

# =========================================================
#  注册 vLLM 模型实现
# =========================================================


ModelRegistry.register_model("BagelForConditionalGeneration", BagelForConditionalGeneration)
ModelRegistry.register_model("Qwen2ForCausalLM", Qwen2ForCausalLM)

print(" vLLM-Omni 初始化完成")

try:
    from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]

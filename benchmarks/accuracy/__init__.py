"""Accuracy Benchmarks for T2I and I2I Generation.

This module provides evaluation metrics for Text-to-Image (T2I) and
Image-to-Image (I2I) generation quality.

T2I Metrics:
    - VQAScore: Prompt faithfulness evaluation
    - GenEval/TIFA: Compositional and fine-grained correctness

I2I Metrics:
    - VLM-Judge: Edit success and instruction following
    - LPIPS: Background preservation

Example:
    >>> from benchmarks.accuracy import T2IEvaluator, I2IEvaluator
    >>> t2i_eval = T2IEvaluator()
    >>> scores = t2i_eval.evaluate(prompts, generated_images)
"""

__version__ = "0.1.0"

# Optional imports - metrics may not be available if dependencies are not installed
try:
    from .i2i import I2IEvaluator, LPIPSMetric
    from .t2i import GenEval, T2IEvaluator, VQAScore

    __all__ = [
        "T2IEvaluator",
        "VQAScore",
        "GenEval",
        "I2IEvaluator",
        "LPIPSMetric",
    ]
except ImportError:
    # Dependencies not installed
    __all__ = []

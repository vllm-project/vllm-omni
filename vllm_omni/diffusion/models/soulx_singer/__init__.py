"""SoulX-Singer SVS / SVC flow-matching singing models for vLLM-Omni."""

__all__ = [
    "FlowMatchingAudioPipeline",
    "PipelineSoulXSingerSVS",
    "PipelineSoulXSingerSVC",
    "get_soulxsinger_post_process_func",
    "get_soulxsinger_pre_process_func",
    "get_soulxsinger_svc_pre_process_func",
]


def __getattr__(name: str):
    if name in {"FlowMatchingAudioPipeline", "get_soulxsinger_post_process_func"}:
        from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
            FlowMatchingAudioPipeline,
            get_soulxsinger_post_process_func,
        )

        return {
            "FlowMatchingAudioPipeline": FlowMatchingAudioPipeline,
            "get_soulxsinger_post_process_func": get_soulxsinger_post_process_func,
        }[name]
    if name in {"PipelineSoulXSingerSVC", "get_soulxsinger_svc_pre_process_func"}:
        from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_svc import (
            PipelineSoulXSingerSVC,
            get_soulxsinger_svc_pre_process_func,
        )

        return {
            "PipelineSoulXSingerSVC": PipelineSoulXSingerSVC,
            "get_soulxsinger_svc_pre_process_func": get_soulxsinger_svc_pre_process_func,
        }[name]
    if name in {"PipelineSoulXSingerSVS", "get_soulxsinger_pre_process_func"}:
        from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_svs import (
            PipelineSoulXSingerSVS,
            get_soulxsinger_pre_process_func,
        )

        return {
            "PipelineSoulXSingerSVS": PipelineSoulXSingerSVS,
            "get_soulxsinger_pre_process_func": get_soulxsinger_pre_process_func,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

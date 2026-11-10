# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace

from vllm_omni.diffusion.capability_resolvers import (
    COSMOS3_CAPABILITIES,
    MINIMAX_H3_CAPABILITIES,
    MINIMAX_H3_CAPABILITIES_RESOLVER,
)
from vllm_omni.request_capabilities import (
    InputConstraint,
    MaskConstraint,
    ParameterConstraint,
    RequestConstraints,
    RequestSupport,
    SourceResize,
)
from vllm_omni.serving_capabilities import (
    AUDIO_RESPONSE,
    IMAGE_RESPONSE,
    IMAGE_TO_IMAGE,
    IMAGE_TO_VIDEO,
    OPERATIONS,
    TEXT_RESPONSE,
    TEXT_TO_AUDIO,
    TEXT_TO_IMAGE,
    TEXT_TO_SPEECH,
    TEXT_TO_VIDEO,
    VIDEO_TO_VIDEO,
    ServingCapabilities,
    declare_capabilities,
)


@dataclass(frozen=True)
class DiffusionModelMetadata:
    # Keep serving-facing capability metadata in a lightweight shared module so
    # config/model plumbing can read it without importing concrete pipelines.
    supports_multimodal_inputs: bool = False
    max_multimodal_image_inputs: int | None = None
    supports_mixed_reference_inputs: bool = False
    # Multipart controls are exposed as ``control_reference`` plus
    # ``control_type`` by the video API.  A pipeline that opts in receives the
    # persisted upload through ``extra_args[control_type]["control_path"]``.
    supported_control_upload_types: tuple[str, ...] = ()
    attention_mask_free: bool = False
    final_output_type: str | None = None
    # Whether ``/v1/videos`` accepts source media plus per-token video/audio
    # noise masks for latent initialization. Unknown pipelines must remain
    # opted out so uploaded files never reach a model that cannot consume them.
    supports_latent_mask_editing: bool = False
    # Whether ``/v1/videos`` accepts the ``timeline_guides`` manifest plus
    # ``guide_files`` uploads, forwarded as ``multi_modal_data["timeline_guides"]``.
    supports_timeline_guides: bool = False
    # Whether the pipeline keeps finished clips' latents under
    # ``extra_args["latent_cache_key"]`` and accepts ``extra_args["continue_from"]``
    # to continue one; the video API tags requests with their job id.
    supports_latent_continuation: bool = False
    # Whether ``/v1/images/segmentations`` may serve this model: it takes one
    # image plus text/box prompts and returns a ``segmentation`` payload.
    supports_segmentation: bool = False
    # Whether ``/v1/videos/segmentations`` may serve this model: it takes one
    # video path plus text prompts and returns a ``video_segmentation`` payload.
    supports_video_segmentation: bool = False
    # Whether ``/v1/videos/depth`` may serve this model: it takes one video path
    # and returns a ``video_depth`` payload of per-frame depth maps.
    supports_video_depth: bool = False
    serving_capabilities: ServingCapabilities | None = None
    serving_capabilities_resolver: str | None = None


# ``extra_args`` keys that only name a request (the video API's job-id tag for
# ``supports_latent_continuation``) and never change its forward schedule, so
# requests differing only in them may share a DLO data-parallel wave.
REQUEST_IDENTITY_EXTRA_ARGS = frozenset({"latent_cache_key"})


# FLUX.2 Klein supports up to four reference images.
FLUX2_KLEIN_MAX_INPUT_IMAGES = 4
QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES = 4
# Qwen-Image 2.1 image-conditioned generation caps condition images at 4.
QWEN_IMAGE_21_MAX_INPUT_IMAGES = 4
# Upstream HunyuanImage-3.0 "Multi-Image Fusion" caps reference images at 3.
HUNYUAN_IMAGE3_MAX_INPUT_IMAGES = 3
JOY_IMAGE_EDIT_MAX_INPUT_IMAGES = 1
# Boogu-Image editing (TI2I) supports a single reference image for now.
BOOGU_IMAGE_MAX_INPUT_IMAGES = 1
FLUX3_MAX_INPUT_IMAGES = 10


def _image_edit_profile(max_images: int | None = None, *, native_mask: bool = False):
    """Native image pipeline input ownership; geometry is supplied at startup.

    Klein is the native image pipeline consuming mask_image/reference_image.
    Other built-in image pipelines consume the ordinary image role only.
    External DiffusersAdapterPipeline remains unknown below.
    """
    extra_support: RequestSupport = "supported" if native_mask else "unsupported"
    return replace(
        IMAGE_TO_IMAGE,
        request=RequestConstraints(
            inputs={
                "text": InputConstraint(support="supported", min_count=1),
                "image": InputConstraint(support="supported", min_count=1, max_count=max_images),
                "mask_image": InputConstraint(support=extra_support, min_count=0, max_count=1 if native_mask else 0),
                "reference_image": InputConstraint(
                    support=extra_support, min_count=0, max_count=1 if native_mask else 0
                ),
            },
            mask=MaskConstraint(
                support="supported",
                max_source_count=1,
                white="edit",
                black="preserve",
                binarize_threshold=0.5,
                requires_same_size=True,
                source_resize=SourceResize(max_pixels=1024 * 1024),
            )
            if native_mask
            else MaskConstraint(support="unsupported"),
        ),
    )


_DIFFUSION_MODEL_METADATA: dict[str, DiffusionModelMetadata] = {
    "HyNomasPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/hy_nomas/pipeline_hy_nomas.py",),
        ),
    ),
    "HyNomasModularPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/hy_nomas/pipeline_hy_nomas.py",),
        ),
    ),
    "SeedVR2Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            VIDEO_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/seedvr2/pipeline_seedvr2.py",),
        ),
    ),
    "Sam3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        supports_segmentation=True,
        supports_video_segmentation=True,
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/sam3/pipeline_sam3.py",),
        ),
    ),
    "VideoDepthAnythingPipeline": DiffusionModelMetadata(
        supports_video_depth=True,
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/video_depth_anything/pipeline_video_depth_anything.py",),
        ),
    ),
    "Flux2KleinPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=FLUX2_KLEIN_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(FLUX2_KLEIN_MAX_INPUT_IMAGES, native_mask=True),
            sources=("vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py",),
        ),
    ),
    "QwenImageEditPlusPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            replace(IMAGE_RESPONSE, required_inputs=("text", "image")),
            _image_edit_profile(QWEN_IMAGE_EDIT_PLUS_MAX_INPUT_IMAGES),
            sources=("vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit_plus.py",),
        ),
    ),
    # Qwen-Image 2.1 handles both text-to-image (no image) and
    # image-conditioned requests through the same pipeline class.
    "QwenImage21Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=QWEN_IMAGE_21_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(QWEN_IMAGE_21_MAX_INPUT_IMAGES),
            sources=("vllm_omni/diffusion/models/qwen_image_21/pipeline_qwen_image_21.py",),
        ),
    ),
    "HunyuanImage3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=HUNYUAN_IMAGE3_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(HUNYUAN_IMAGE3_MAX_INPUT_IMAGES),
            sources=("vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py",),
        ),
    ),
    # Shared by the Base (text-to-image) and Edit (TI2I) checkpoints, which use
    # the same ``BooguImagePipeline`` class. Text-to-image requests simply carry
    # no reference image.
    "BooguImagePipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=BOOGU_IMAGE_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(BOOGU_IMAGE_MAX_INPUT_IMAGES),
            sources=("vllm_omni/diffusion/models/boogu_image/pipeline_boogu_image.py",),
        ),
    ),
    "JoyImageEditPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=JOY_IMAGE_EDIT_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            replace(IMAGE_RESPONSE, required_inputs=("text", "image")),
            _image_edit_profile(JOY_IMAGE_EDIT_MAX_INPUT_IMAGES),
            sources=("vllm_omni/diffusion/models/joy_image/pipeline_joy_image_edit.py",),
        ),
    ),
    "MiniMaxH3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
        supported_control_upload_types=("canny", "depth", "hed", "mlsd", "pose", "inpaint"),
        supports_latent_mask_editing=True,
        supports_timeline_guides=True,
        supports_latent_continuation=True,
        final_output_type="video",
        # H3 represents alignment padding as a second packed sequence.  The
        # packed TRTLLM backend consumes cu_seqlens and isolates that padding.
        attention_mask_free=True,
        serving_capabilities=MINIMAX_H3_CAPABILITIES,
        serving_capabilities_resolver=MINIMAX_H3_CAPABILITIES_RESOLVER,
    ),
    # The modular alias is served by MiniMaxH3Pipeline and has the same
    # Ref2VA request contract and packed-sequence layout. Every field must stay
    # in sync with it: the repository root ``model_index.json`` declares this
    # alias, so serving the checkpoint by repo id resolves here rather than to
    # ``MiniMaxH3Pipeline``.
    "MiniMaxH3ModularPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=9,
        supports_mixed_reference_inputs=True,
        supported_control_upload_types=("canny", "depth", "hed", "mlsd", "pose", "inpaint"),
        supports_latent_mask_editing=True,
        supports_timeline_guides=True,
        supports_latent_continuation=True,
        final_output_type="video",
        attention_mask_free=True,
        serving_capabilities=MINIMAX_H3_CAPABILITIES,
        serving_capabilities_resolver=MINIMAX_H3_CAPABILITIES_RESOLVER,
    ),
    # TaoMate-H3 is a text-prompted streaming model on the MiniMax-H3 packed
    # layout; alignment padding is a second packed document, as for H3.
    "TaoMateH3Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        attention_mask_free=True,
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/taomate_h3/pipeline.py",),
        ),
    ),
    "Magi2Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            *(
                replace(
                    profile,
                    request=RequestConstraints(
                        inputs={
                            name: InputConstraint(support="supported", min_count=1, max_count=1)
                            for name in profile.required_inputs
                        },
                        parameters={
                            "num_frames": ParameterConstraint(support="supported", type="integer", enum=(125,)),
                            "fps": ParameterConstraint(support="supported", type="number", enum=(12.5,)),
                            "seconds": ParameterConstraint(support="supported", type="number", enum=(10.0,)),
                            "resolution": ParameterConstraint(
                                support="supported", type="string", enum=("272p", "540p")
                            ),
                        },
                    ),
                )
                for profile in (TEXT_TO_VIDEO, IMAGE_TO_VIDEO)
            ),
            sources=("vllm_omni/diffusion/models/magi2/pipeline_magi2.py", "vllm_omni/model_extras/magi2.py"),
        ),
    ),
    # Joint text/image-to-video-and-audio, same shape as MiniMaxH3Pipeline
    # above (an MP4 with both tracks) — declared "video" for the same reason:
    # the final container is a video file, so /v1/videos* is the right API
    # surface even though the model also produces audio.
    "Kandinsky6TI2VAPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio"), constraints=(("audio_can_be_disabled", True),)),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio"), constraints=(("audio_can_be_disabled", True),)),
            sources=("vllm_omni/diffusion/models/kandinsky6/pipeline_kandinsky6.py",),
        ),
    ),
    "WanPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            replace(IMAGE_TO_VIDEO, support="unknown", reason_code="checkpoint_requires_expand_timesteps"),
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",),
        ),
    ),
    "Flux3Pipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=FLUX3_MAX_INPUT_IMAGES,
        final_output_type="image",
    ),
    "WanImageToVideoPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(
                IMAGE_TO_VIDEO,
                request=RequestConstraints(
                    inputs={
                        "text": InputConstraint(support="supported", min_count=1),
                        "image": InputConstraint(support="supported", min_count=1, max_count=1),
                        "last_frame": InputConstraint(support="supported", min_count=0, max_count=1),
                    }
                ),
            ),
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py",),
        ),
    ),
    "WanVACEPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            IMAGE_TO_VIDEO,
            VIDEO_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py",),
        ),
    ),
    "WanS2VPipeline": DiffusionModelMetadata(
        attention_mask_free=True,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(IMAGE_TO_VIDEO, required_inputs=("text", "image", "audio"), outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_s2v.py",),
        ),
    ),
    "WanT2VDMD2Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",),
        ),
    ),
    "WanI2VDMD2Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py",),
        ),
    ),
    "LTX2Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py",),
        ),
    ),
    "LTX2DistilledPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2_two_stage.py",),
        ),
    ),
    "LTX2T2VDMD2Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py",),
        ),
    ),
    "LTX2I2VDMD2Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py",),
        ),
    ),
    "HeliosPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            IMAGE_TO_VIDEO,
            VIDEO_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/helios/pipeline_helios.py",),
        ),
    ),
    "HeliosPyramidPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            IMAGE_TO_VIDEO,
            VIDEO_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/helios/pipeline_helios.py",),
        ),
    ),
    "HunyuanVideo15Pipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5.py",),
        ),
    ),
    "HunyuanVideo15ImageToVideoPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5_i2v.py",),
        ),
    ),
    "LingBotVideoPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            TEXT_TO_VIDEO,
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/lingbot_video/pipeline_lingbot_video.py",),
        ),
    ),
    "LongCatVideoAvatarPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, required_inputs=("text", "audio")),
            replace(IMAGE_TO_VIDEO, required_inputs=("text", "image", "audio"), outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/longcat_video/pipeline_longcat_video_avatar.py",),
        ),
    ),
    "MagiHumanPipeline": DiffusionModelMetadata(final_output_type="video"),
    "DreamIDOmniPipeline": DiffusionModelMetadata(final_output_type="video"),
    "Cosmos3OmniDiffusersPipeline": DiffusionModelMetadata(
        supported_control_upload_types=("edge", "blur", "depth", "seg", "wsm"),
        final_output_type="video",
        serving_capabilities=COSMOS3_CAPABILITIES,
    ),
    "Cosmos3OmniPipeline": DiffusionModelMetadata(
        supported_control_upload_types=("edge", "blur", "depth", "seg", "wsm"),
        final_output_type="video",
        serving_capabilities=COSMOS3_CAPABILITIES,
    ),
    "SanaVideoPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/sana_video/pipeline_sana_video.py",),
        ),
    ),
    "SanaImageToVideoPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/sana_video/pipeline_sana_video_i2v.py",),
        ),
    ),
    "SanaWmPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=1,
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(IMAGE_TO_VIDEO, required_inputs=("image", "camera_or_action")),
            sources=("vllm_omni/diffusion/models/sana_wm/pipeline_sana_wm.py",),
        ),
    ),
    "ABotWorldCausalPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/abot_world/pipeline_abot_world.py",),
        ),
    ),
    "Alpamayo2SuperPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/alpamayo2_super/pipeline_alpamayo2_super.py",),
        ),
    ),
    "AnimaPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/anima/pipeline_anima.py",),
        ),
    ),
    "AuKPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/auk/pipeline_auk.py",),
        ),
    ),
    "BagelPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_RESPONSE,
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/bagel/pipeline_bagel.py",),
        ),
    ),
    "BooguImageTurboPipeline": DiffusionModelMetadata(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=BOOGU_IMAGE_MAX_INPUT_IMAGES,
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/boogu_image/pipeline_boogu_image.py",),
        ),
    ),
    "CausVidWanPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/causvid_wan/pipeline_causvid_wan.py",),
        ),
    ),
    "DiffusersAdapterPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unknown_operations=OPERATIONS,
            unknown_reason="external_pipeline_declaration_missing",
            sources=("vllm_omni/diffusion/models/diffusers_adapter/pipeline_diffusers_adapter.py",),
        ),
    ),
    "DreamZeroPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/dreamzero/pipeline_dreamzero.py",),
        ),
    ),
    "ErnieImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/ernie_image/pipeline_ernie_image.py",),
        ),
    ),
    "Flux2Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/flux2/pipeline_flux2.py",),
        ),
    ),
    "FluxDMD2Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/flux/pipeline_flux.py",),
        ),
    ),
    "FluxKontextPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/flux/pipeline_flux_kontext.py",),
        ),
    ),
    "FluxPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/flux/pipeline_flux.py",),
        ),
    ),
    "GlmImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/glm_image/pipeline_glm_image.py",),
        ),
    ),
    "Gr00tN1d7Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/gr00t/pipeline_gr00t.py",),
        ),
    ),
    "HiDreamO1ImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/hidream_o1_image/pipeline_hidream_o1_image.py",),
        ),
    ),
    "InternVLAA1Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/internvla_a1/pipeline_internvla_a1.py",),
        ),
    ),
    "Krea2Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/krea2/pipeline_krea2.py",),
        ),
    ),
    "LTX2DistilledOneStagePipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2.py",),
        ),
    ),
    "LTX2DistilledTwoStagePipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2_two_stage.py",),
        ),
    ),
    "LTX2TextToAudioPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            AUDIO_RESPONSE,
            TEXT_TO_AUDIO,
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2_audio.py",),
        ),
    ),
    "LTX2TwoStagePipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            replace(TEXT_TO_VIDEO, outputs=("video", "audio")),
            replace(IMAGE_TO_VIDEO, outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/ltx2/pipeline_ltx2_two_stage.py",),
        ),
    ),
    "LancePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_RESPONSE,
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            unknown_operations=("video_generation",),
            unknown_reason="checkpoint_specific_video_contract",
            sources=("vllm_omni/diffusion/models/lance/pipeline_lance.py",),
        ),
    ),
    "LingBotWorldCausalDMDPipeline": DiffusionModelMetadata(
        final_output_type="video",
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/lingbot_world/pipeline.py",),
        ),
    ),
    "LongCatImageEditPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            replace(IMAGE_RESPONSE, required_inputs=("text", "image")),
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image_edit.py",),
        ),
    ),
    "LongCatImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image.py",),
        ),
    ),
    "MOSSImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/mossimage/pipeline_mossimage.py",),
        ),
    ),
    "MammothModa2DiTPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py",),
        ),
    ),
    "MingImageDiffusionPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/ming_image/pipeline.py",),
        ),
    ),
    "MingImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/ming_flash_omni/pipeline_ming_imagegen.py",),
        ),
    ),
    "MiniMaxH3DecoderPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="internal_stage_requires_parent",
            sources=("vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_decoder.py",),
        ),
    ),
    "NextStep11Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/nextstep_1_1/pipeline_nextstep_1_1.py",),
        ),
    ),
    "OmniGen2Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/omnigen2/pipeline_omnigen2.py",),
        ),
    ),
    "OmniVoicePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_SPEECH,
            sources=("vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py",),
        ),
    ),
    "OvisImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/ovis_image/pipeline_ovis_image.py",),
        ),
    ),
    "PhysisH3Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/physis/h3/pipeline_physis_h3.py",),
        ),
    ),
    "PhysisLTXPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/physis/ltx/pipeline_physis_ltx.py",),
        ),
    ),
    "PhysisWanPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/physis/wan/pipeline_physis_wan.py",),
        ),
    ),
    "Pi05Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/pi05/pipeline_pi05.py",),
        ),
    ),
    "Pi0Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            unsupported_reason="outside_standard_operations",
            sources=("vllm_omni/diffusion/models/pi0/pipeline_pi0.py",),
        ),
    ),
    "QwenImageDMD2Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py",),
        ),
    ),
    "QwenImageEditPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            replace(IMAGE_RESPONSE, required_inputs=("text", "image")),
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py",),
        ),
    ),
    "QwenImageLayeredPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            replace(IMAGE_RESPONSE, required_inputs=("text", "image")),
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_layered.py",),
        ),
    ),
    "QwenImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py",),
        ),
    ),
    "SenseNovaU1Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_RESPONSE,
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(),
            sources=("vllm_omni/diffusion/models/sensenova_u1/pipeline_sensenova_u1.py",),
        ),
    ),
    "StableAudio3Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            AUDIO_RESPONSE,
            TEXT_TO_AUDIO,
            sources=("vllm_omni/diffusion/models/stable_audio_3/pipeline_stable_audio_3.py",),
        ),
    ),
    "StableAudioPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            AUDIO_RESPONSE,
            TEXT_TO_AUDIO,
            sources=("vllm_omni/diffusion/models/stable_audio/pipeline_stable_audio.py",),
        ),
    ),
    "StableDiffusion3Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/sd3/pipeline_sd3.py",),
        ),
    ),
    "StableDiffusionXLPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            sources=("vllm_omni/diffusion/models/sdxl/pipeline_sdxl.py",),
        ),
    ),
    "ZImagePipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_RESPONSE,
            TEXT_TO_IMAGE,
            _image_edit_profile(1),
            sources=("vllm_omni/diffusion/models/z_image/pipeline_z_image.py",),
        ),
    ),
    "HunyuanVideo15I2VPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/hunyuan_video/pipeline_hunyuan_video_1_5_i2v.py",),
        ),
    ),
    "Wan22I2VPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            IMAGE_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_i2v.py",),
        ),
    ),
    "Wan22Pipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            replace(IMAGE_TO_VIDEO, support="unknown", reason_code="checkpoint_requires_expand_timesteps"),
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py",),
        ),
    ),
    "Wan22S2VPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            replace(IMAGE_TO_VIDEO, required_inputs=("text", "image", "audio"), outputs=("video", "audio")),
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_s2v.py",),
        ),
    ),
    "Wan22VACEPipeline": DiffusionModelMetadata(
        serving_capabilities=declare_capabilities(
            TEXT_TO_VIDEO,
            IMAGE_TO_VIDEO,
            VIDEO_TO_VIDEO,
            sources=("vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2_vace.py",),
        ),
    ),
}

_DIFFUSION_MODEL_METADATA_ALIASES = {
    "WanDMDPipeline": "WanPipeline",
    "LTX2TwoStagePipeline": "LTX2Pipeline",
    "LTX2DistilledOneStagePipeline": "LTX2DistilledPipeline",
    "LTX2DistilledTwoStagePipeline": "LTX2DistilledPipeline",
    "LingBotWorldCausalDMDPipeline": "LingBotVideoPipeline",
    "BooguImageTurboPipeline": "BooguImagePipeline",
}


def get_diffusion_model_metadata(model_class_name: str | None) -> DiffusionModelMetadata:
    # Unknown models fall back to "no special multimodal capabilities" so new
    # pipelines do not accidentally inherit limits meant for other models.
    if model_class_name is None:
        return DiffusionModelMetadata()
    metadata = _DIFFUSION_MODEL_METADATA.get(model_class_name)
    if metadata is not None:
        return metadata
    canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(model_class_name)
    if canonical_name is not None:
        return _DIFFUSION_MODEL_METADATA[canonical_name]
    # Some checkpoints report the HF architecture name diff from internal pipeline class name
    # (e.g. HunyuanImage3ForCausalMM, WanVACEPipeline, OmniVoice ...).
    from vllm_omni.diffusion.registry import _DIFFUSION_MODELS

    entry = _DIFFUSION_MODELS.get(model_class_name)
    if entry is not None:
        # Unpack instead of indexing so a future change to the registry tuple
        # shape fails loudly instead of silently reading the wrong element.
        _, _, pipeline_cls_name = entry
        # Note: the registry ``cls_name`` and the metadata keys are two separate
        # key spaces. Aliases whose pipeline class has no metadata entry (e.g.
        # external plugin classes) still fall back to the defaults;
        # that is not a regression, it just means no capability override.
        canonical_name = _DIFFUSION_MODEL_METADATA_ALIASES.get(pipeline_cls_name, pipeline_cls_name)
        return _DIFFUSION_MODEL_METADATA.get(canonical_name, DiffusionModelMetadata())
    return DiffusionModelMetadata()

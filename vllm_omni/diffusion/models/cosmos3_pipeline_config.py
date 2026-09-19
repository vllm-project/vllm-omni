# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 topologies: co-located (one stage) and disaggregated (one per tower).

Cosmos3 is an Omni-modal foundation model built on a Mixture-of-Transformers
(MoT) architecture with two complementary transformer towers:

  * UND / ``reasoner``  -- the autoregressive transformer for discrete token
    generation (``Cosmos3VFMTransformer.language_model``).
  * GEN / ``generator`` -- the diffusion transformer for continuous multimodal
    generation (``Cosmos3VFMTransformer.gen_layers``).

``Cosmos3OmniDiffusersPipeline`` co-locates both towers in one stage. This
topology splits them into two independently-scheduled stage workers:

    stage 0 (reasoner) --per-layer text K/V--> stage 1 (generator) --> image

WHY THIS IS SEPARABLE
---------------------
``Cosmos3VFMTransformer.forward`` already contains the seam: the UND tower runs
*once per request* and its per-layer K/V is memoized in ``self.cached_kv``, so
every subsequent denoising step skips it. The UND tower is therefore a run-once
prologue whose entire contribution to the rest of generation is that K/V.
Feeding a GEN-only worker the K/V a separate UND-only worker computed reproduces
the co-located result, which is what makes the split numerically faithful rather
than an approximation. See ``cosmos3/pipeline_cosmos3_disagg.py`` for the
interception point and why it is the ``self.language_model(...)`` call rather
than the ``cached_kv`` attribute.

Structurally this is the same handoff as GLM-Image's AR->DiT
``prior_token_ids`` bridge; the payload is per-layer K/V tensors instead of
token ids.

PAYLOAD SIZE
------------
K/V is grouped-query (``num_key_value_heads=8``, ``head_dim=128``) and bf16, so
one token costs ``8 * 128 * 2 bytes = 2 KiB`` per tensor, ``4 KiB`` per layer for
K and V together, and ~``256 KiB`` per token per branch across 64 layers. It is
trimmed to the real text length, but that length is what drives the total: a
256-token formatted T2I prompt hands off ~64 MiB per branch, doubled to ~128 MiB
when guidance is active and both branches are encoded. ``max_sequence_length``
(4096 by default) is the hard ceiling and puts the worst case in the GiB range,
so the reasoner logs the size and warns past
``COSMOS3_UND_PAYLOAD_WARN_MIB``. That cost is paid once per request, against a
GEN tower that then runs ``num_inference_steps`` times.

Those figures are TP-independent. UND K/V is born sharded across the reasoner's
tensor-parallel ranks, but the reasoner all-gathers the KV-head dimension before
the payload leaves the tower and each generator rank slices back out the head
range its own cross-attention owns, so what crosses the edge is always the full
8-head set -- the same bytes a co-located TP-1 tower would produce. The two
stages are therefore free to run different ``tensor_parallel_size`` values.

WHAT DISAGGREGATION BUYS
------------------------
The towers are near-symmetric in size -- 31.2 B parameters each (58.1 GiB in
bf16 apiece) -- but wildly asymmetric in duty cycle: UND runs once, GEN runs
once per denoising step. Co-located, the idle UND tower's 58 GiB stays resident
on the same GPUs for the whole denoise loop. Split, each tower gets its own
GPUs and its own parallelism, the GEN stage keeps denoising while the UND stage
admits the next request, and the two scale independently.

Keep this topology outside the ``cosmos3`` package: importing a package
submodule executes ``cosmos3/__init__``, which imports the runtime pipeline and
``diffusion.data``. The two payload-key constants live here for the same
reason -- the stage input processor needs them in the orchestrator process,
which must not pay for the diffusion pipeline import to read two strings.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

#: Payload keys on the reasoner -> generator edge. Shared by the tower
#: pipelines and the stage input processor so the two cannot drift apart.
COSMOS3_UND_KV_KEY = "cosmos3_und_kv"
COSMOS3_UND_META_KEY = "cosmos3_und_meta"

#: Identifier of the conditioning contract those two keys carry, owned by the code
#: that produces it (``Cosmos3TextConditioning`` in
#: ``cosmos3/pipeline_cosmos3_disagg.py``). It is model-specific on purpose: what
#: crosses this edge is Cosmos3 UND per-layer text K/V with a Cosmos3-specific
#: layout, not a generic conditioning artifact. The generator refuses a payload
#: that does not declare exactly this string, so a future layout change is a named
#: mismatch on the first request rather than a wrong image.
COSMOS3_UND_SCHEMA = "cosmos3.text_conditioning/v1"

#: Warn once per request when the reasoner -> generator K/V payload exceeds this
#: size. Not a hard limit: an oversized payload is still correct, just expensive
#: to serialize and ship (see PAYLOAD SIZE above). Sized so a normal T2I prompt
#: with guidance stays quiet and a runaway ``max_sequence_length`` does not.
COSMOS3_UND_PAYLOAD_WARN_MIB = 512.0

COSMOS3_ARCH = "Cosmos3OmniDiffusersPipeline"
COSMOS3_REASONER_ARCH = "Cosmos3ReasonerPipeline"
COSMOS3_GENERATOR_ARCH = "Cosmos3GeneratorPipeline"

_COSMOS3_INPUT_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.cosmos3"


# The co-located topology: both towers in one Cosmos3OmniDiffusersPipeline stage
# doing text encode + denoise + VAE decode together, parallelized *within* the
# stage (CFG x Ulysses, with HSDP sharding the weights).
#
# WHY THIS IS REGISTERED AT ALL
# -----------------------------
# Nothing about co-located Cosmos3 needs a multi-stage topology, and it ran for a
# long time without one -- an unregistered model_type resolves to no
# PipelineConfig and the engine builds a lone default stage from CLI kwargs. But
# a deploy YAML is only read *through* this registry: with no entry,
# ``create_stage_configs`` returns None and every field in a ``stages:`` YAML is
# silently discarded. So without a registered topology, no deploy YAML can
# configure co-located Cosmos3 at all -- no per-stage ``devices``,
# ``max_num_seqs``, ``parallel_config`` or ``guardrails`` gate.
#
# WHY IT IS KEYED ``cosmos3_omni_colocated`` AND NOT ``cosmos3_omni``
# -------------------------------------------------------------------
# Three different topologies share one set of HF metadata. T2I *and* T2V/I2V/V2V
# checkpoints, and the policy checkpoints, all report ``model_type=cosmos3_omni``
# with ``model_index.json`` ``_class_name=Cosmos3OmniDiffusersPipeline``, so the
# checkpoint cannot tell them apart. Keying this entry on the bare
# ``cosmos3_omni`` (or declaring ``hf_architectures`` /
# ``diffusers_class_name``) would make it the auto-detected answer for all of
# them, which breaks two things:
#
#   * ``final_output_type``. This stage pins ``"image"``, while the single-stage
#     fallback resolves it dynamically through ``get_diffusion_output_type`` and
#     gets ``"video"`` for ``Cosmos3OmniDiffusersPipeline`` (see
#     ``_DIFFUSION_MODEL_METADATA``). The registry path passes ``final_output_type``
#     through verbatim, so every T2V/I2V/V2V deployment would silently get its
#     output relabelled as an image.
#   * Policy checkpoints, which ``model_executor/models/cosmos3/pipeline.py``
#     keeps on that same fallback for exactly this reason.
#
# So this topology follows the convention its two siblings already use --
# ``cosmos3_policy`` and ``cosmos3_omni_disagg`` below: a distinct registry key,
# no auto-detect hooks, and the only route in is an explicit
# ``pipeline: cosmos3_omni_colocated`` in a deploy YAML, which
# ``_get_deploy_override_pipe_config`` honours ahead of any inference.
# ``deploy/cosmos3_super_t2i.yaml`` carries that key and is the recommended
# single-GPU layout: ``--deploy-config cosmos3_super_t2i.yaml``. A bare
# ``serve`` with no deploy config is untouched by this entry and keeps building
# one default stage from CLI kwargs, exactly as before.
COSMOS3_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_colocated",
    default_deploy_config_name="cosmos3_super_t2i.yaml",
    model_arch="Cosmos3ForConditionalGeneration",
    # Deliberately empty, and no ``diffusers_class_name`` -- see WHY IT IS KEYED
    # ... above. Naming either would hand this config to every Cosmos3
    # checkpoint, including the video and policy ones it is wrong for.
    hf_architectures=(),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="diffusion",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            owns_tokenizer=True,
            # Cosmos3 also does image- and video-conditioned generation, so mm
            # profiling stays on even though T2I never uses it.
            requires_multimodal_data=True,
            final_output=True,
            final_output_type="image",
            model_arch=COSMOS3_ARCH,
        ),
    ),
)


# Unlike the co-located entry above, naming a default deploy YAML here cannot
# change anyone's defaults: this topology is unreachable without a deploy config
# that selects it by name (see ``hf_architectures`` below), so by the time this
# config is resolved the caller has already supplied one and
# ``_get_deploy_config`` returns that instead. It is declared for the one path
# that does reach it -- selecting the topology programmatically, e.g.
# ``get_pipeline_config(pipeline="cosmos3_omni_disagg")`` with no deploy path --
# where the two-stage device map is not something a caller should have to
# reconstruct by hand.
COSMOS3_DISAGG_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_disagg",
    default_deploy_config_name="cosmos3_super_t2i_disagg.yaml",
    model_arch="Cosmos3ForConditionalGeneration",
    # Deliberately empty, and no ``diffusers_class_name``: this is a second
    # topology over the *same* checkpoint as the ``cosmos3_omni_colocated``
    # deployment, so it must never be auto-detected. Both auto-detect paths in
    # ``StageConfigFactory`` scan every registered pipeline -- the arch fallback
    # matches ``hf_architectures`` against ``hf_config.architectures``, and the
    # diffusers fallback matches ``diffusers_class_name`` against
    # ``model_index.json:_class_name``. Declaring either would hand this
    # 2-stage config to a co-located deploy merely because it is registered.
    # Same convention as ``hunyuan_image3_ar`` / ``hunyuan_image3_dit``: the
    # only route in is an explicit ``pipeline: cosmos3_omni_disagg`` in the
    # deploy YAML, which ``_get_deploy_override_pipe_config`` honours ahead of
    # any inference.
    hf_architectures=(),
    stages=(
        # UND tower only. Emits per-layer text K/V, never allocates latents and
        # never touches the VAE.
        StagePipelineConfig(
            stage_id=0,
            model_stage="reasoner",
            # DIFFUSION, not LLM_AR: the UND tower is a prologue inside the MoT
            # diffusion pipeline, not a separate sampling LLM engine.
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            owns_tokenizer=True,
            # T2I: no image/video input reaches this stage, so mm profiling is
            # skipped (``merge_pipeline_deploy`` sets skip_mm_profiling).
            requires_multimodal_data=False,
            final_output=False,
            model_arch=COSMOS3_REASONER_ARCH,
            # No ``engine_output_type``: it is an AR-engine knob (``OutputModality``
            # has no value for "per-layer K/V tensors"), and a DIFFUSION stage's
            # output shape is decided by its postprocessor -- here
            # ``get_cosmos3_reasoner_post_process_func``, which parks the K/V on
            # ``multimodal_output``.
        ),
        # GEN tower + VAE decode: run once per denoising step, so this is where
        # essentially all the FLOPs are. The UND tower is replaced by a replay
        # stub fed from stage 0, so no UND weights load here.
        StagePipelineConfig(
            stage_id=1,
            model_stage="generator",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            requires_multimodal_data=False,
            final_output=True,
            final_output_type="image",
            model_arch=COSMOS3_GENERATOR_ARCH,
            custom_process_input_func=f"{_COSMOS3_INPUT_PROCESSOR}.reasoner2generator",
            # The K/V handoff travels in the stage payload, not through the AR
            # KV-transfer machinery; mirrors GLM-Image's DiT stage.
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)

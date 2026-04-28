#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

IMPORT_SENTINEL = "from server_ingress_plugin.image_ingress_scheduler_single import ("
GLOBAL_SENTINEL = "_IMAGE_INGRESS_DISPATCHER = None"
FUNC_SENTINEL = "async def _get_image_ingress_dispatcher(raw_request: Request):"
ENABLE_SENTINEL = (
    'if os.environ.get("OMNI_INGRESS_BATCH_DRR_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}:'
)


IMPORT_BLOCK = """from server_ingress_plugin.image_ingress_scheduler_single import (
    ImageIngressDispatcherService,
    ImageIngressResult,
    RequestType,
    parse_batch_caps,
)
"""


GLOBAL_BLOCK = """
# BEGIN OMNI_INGRESS_SINGLEFILE_PATCH globals
_IMAGE_INGRESS_DISPATCHER = None
_IMAGE_INGRESS_LOCK = asyncio.Lock()
# END OMNI_INGRESS_SINGLEFILE_PATCH globals
"""


HELPER_BLOCK = """
# BEGIN OMNI_INGRESS_SINGLEFILE_PATCH helpers
def _default_ingress_batch_caps() -> dict[str, int]:
    return {
        "512x512_20": 4,
        "768x768_20": 4,
        "1024x1024_25": 1,
        "1536x1536_35": 1,
    }


def _resolve_ingress_batch_caps() -> dict[str, int]:
    raw = os.environ.get("OMNI_INGRESS_BATCH_CAPS", "")
    parsed = parse_batch_caps(raw)
    if parsed:
        return parsed
    return _default_ingress_batch_caps()


async def _get_image_ingress_dispatcher(raw_request: Request):
    global _IMAGE_INGRESS_DISPATCHER
    if _IMAGE_INGRESS_DISPATCHER is not None:
        return _IMAGE_INGRESS_DISPATCHER

    async with _IMAGE_INGRESS_LOCK:
        if _IMAGE_INGRESS_DISPATCHER is not None:
            return _IMAGE_INGRESS_DISPATCHER

        async def _batch_execute(req_type: RequestType, payloads: list) -> list[ImageIngressResult]:
            engine_client, _model_name, stage_configs = _get_engine_and_model(raw_request)

            async def _run_same_n_group(group_payloads: list, num_outputs_per_prompt: int) -> list[ImageIngressResult]:
                prompts = []
                for p in group_payloads:
                    item = {"prompt": p.prompt}
                    if p.negative_prompt is not None:
                        item["negative_prompt"] = p.negative_prompt
                    prompts.append(item)

                gen_params = OmniDiffusionSamplingParams(
                    num_outputs_per_prompt=max(1, int(num_outputs_per_prompt)),
                    width=req_type.width,
                    height=req_type.height,
                    num_inference_steps=req_type.steps,
                )
                # Use first request's optional controls for the batch.
                if group_payloads and group_payloads[0].seed is not None:
                    gen_params.seed = group_payloads[0].seed

                result = await _generate_with_async_omni(
                    engine_client=engine_client,
                    gen_params=gen_params,
                    stage_configs=stage_configs,
                    prompt=prompts,
                    request_id=f"img_ingress_batch-{random_uuid()}",
                )
                images = _extract_images_from_result(result) if result is not None else []
                created = int(time.time())
                outputs: list[ImageIngressResult] = []
                for idx, _p in enumerate(group_payloads):
                    start = idx * num_outputs_per_prompt
                    end = start + num_outputs_per_prompt
                    if end <= len(images):
                        outputs.append(
                            ImageIngressResult(
                                created=created,
                                images_b64=[encode_image_base64(img) for img in images[start:end]],
                            )
                        )
                    else:
                        outputs.append(
                            ImageIngressResult(
                                created=created,
                                images_b64=[encode_image_base64(img) for img in images[start:len(images)]],
                                error=(
                                    f"missing images for index={idx}, "
                                    f"expected={num_outputs_per_prompt}, "
                                    f"available={max(0, len(images) - start)}"
                                ),
                            )
                        )
                return outputs

            grouped: dict[int, list[tuple[int, object]]] = {}
            for idx, p in enumerate(payloads):
                n = max(1, int(getattr(p, "n", 1) or 1))
                grouped.setdefault(n, []).append((idx, p))

            merged: list[ImageIngressResult | None] = [None] * len(payloads)
            for n, nodes in grouped.items():
                group_payloads = [p for _, p in nodes]
                group_outputs = await _run_same_n_group(group_payloads, n)
                for (orig_idx, _), out in zip(nodes, group_outputs, strict=True):
                    merged[orig_idx] = out

            outputs: list[ImageIngressResult] = []
            for idx, item in enumerate(merged):
                if item is None:
                    outputs.append(
                        ImageIngressResult(
                            created=int(time.time()),
                            images_b64=[],
                            error=f"missing merged output at index={idx}",
                        )
                    )
                else:
                    outputs.append(item)
            return outputs

        svc = ImageIngressDispatcherService(
            batch_execute_fn=_batch_execute,
            batch_caps=_resolve_ingress_batch_caps(),
        )
        await svc.start()
        _IMAGE_INGRESS_DISPATCHER = svc
        return svc


# END OMNI_INGRESS_SINGLEFILE_PATCH helpers
"""


ENABLE_BLOCK = """
    if os.environ.get("OMNI_INGRESS_BATCH_DRR_ENABLE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        _engine_client, model_name, _stage_configs = _get_engine_and_model(raw_request)
        if request.model is not None and request.model != model_name:
            logger.warning(
                f"Model mismatch: request specifies '{request.model}' but "
                f"server is running '{model_name}'. Using server model."
            )

        if not isinstance(request.prompt, str):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Ingress scheduler expects single prompt per request.",
            )

        if request.size:
            width, height = parse_size(request.size)
        else:
            width, height = 1024, 1024
        steps = request.num_inference_steps if request.num_inference_steps is not None else 20
        # Keep the same max-size protection as the standard generation flow when helper exists.
        size_checker = globals().get("_check_max_generated_image_size")
        if callable(size_checker):
            size_checker(width, height)

        dispatcher = await _get_image_ingress_dispatcher(raw_request)
        ingress_result = await dispatcher.submit(
            model=model_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            steps=steps,
            seed=request.seed,
            n=request.n,
            request_id=f"img_ingress-{random_uuid()}",
        )
        if ingress_result.error:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value, detail=ingress_result.error)

        image_data = [ImageData(b64_json=x, revised_prompt=None) for x in ingress_result.images_b64]
        return ImageGenerationResponse(created=ingress_result.created, data=image_data)
"""


def patch_api_server(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if IMPORT_SENTINEL not in text:
        lines = text.splitlines(keepends=True)
        for idx, line in enumerate(lines):
            if line.startswith("from vllm_omni.inputs.data import "):
                lines.insert(idx + 1, IMPORT_BLOCK + "\n")
                text = "".join(lines)
                break
        else:
            raise RuntimeError("Import anchor not found")

    if GLOBAL_SENTINEL not in text:
        anchor = "profiler_router = APIRouter()\n"
        if anchor not in text:
            raise RuntimeError("Global anchor not found")
        text = text.replace(anchor, anchor + GLOBAL_BLOCK + "\n", 1)

    if FUNC_SENTINEL not in text:
        anchor = "\n\n# Image generation API endpoints\n"
        if anchor not in text:
            raise RuntimeError("Helper anchor not found")
        text = text.replace(anchor, "\n\n" + HELPER_BLOCK + "\n# Image generation API endpoints\n", 1)

    if ENABLE_SENTINEL not in text:
        func_sig = (
            "async def generate_images("
            "request: ImageGenerationRequest, raw_request: Request"
            ") -> ImageGenerationResponse:\n"
        )
        start = text.find(func_sig)
        if start < 0:
            raise RuntimeError("generate_images signature not found")
        try_pos = text.find("\n    try:\n", start)
        if try_pos < 0:
            raise RuntimeError("generate_images try block anchor not found")
        text = text[: try_pos + 1] + ENABLE_BLOCK + text[try_pos + 1 :]

    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="/vllm-workspace/vllm-omni/vllm_omni/entrypoints/openai/api_server.py",
    )
    args = parser.parse_args()
    target = Path(args.target)
    if not target.exists():
        raise SystemExit(f"target file not found: {target}")
    patch_api_server(target)
    print(f"[OK] patched {target}")


if __name__ == "__main__":
    main()

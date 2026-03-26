from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_image(path: Path, image: Image.Image) -> None:
    ensure_dir(path.parent)
    image.save(path)


def find_first_image(folder: Path, stem: str | None = None) -> Path | None:
    patterns = [f"{stem}.*"] if stem else ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    for pattern in patterns:
        for candidate in sorted(folder.glob(pattern)):
            if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                return candidate
    return None


def extract_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    delimiter = "||V^=^V||"
    if raw_text.count(delimiter) >= 2:
        start = raw_text.find(delimiter) + len(delimiter)
        end = raw_text.rfind(delimiter)
        raw_text = raw_text[start:end].strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in: {raw_text[:200]}")
    return json.loads(raw_text[start : end + 1])


def build_openai_url(base_url: str, api_path: str) -> str:
    base = base_url.rstrip("/")
    normalized_path = api_path if api_path.startswith("/") else f"/{api_path}"
    if base.endswith(normalized_path):
        return base
    if base.endswith("/v1"):
        return f"{base}{normalized_path}"
    return f"{base}/v1{normalized_path}"


def pil_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    return f"data:image/{image_format.lower()};base64,{pil_to_base64(image, image_format=image_format)}"


def decode_base64_image(encoded: str) -> Image.Image:
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    image.load()
    return image.convert("RGB")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class VllmOmniImageClient:
    """Thin OpenAI-compatible image client for vLLM-Omni serving."""

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_text_to_image(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int = 20,
        guidance_scale: float | None = None,
        seed: int | None = None,
        output_compression: int | None = None,
    ) -> Image.Image:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
            "num_inference_steps": num_inference_steps,
        }
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale
        if seed is not None:
            payload["seed"] = seed
        if output_compression is not None:
            payload["output_compression"] = output_compression

        response = requests.post(
            build_openai_url(self.base_url, "/images/generations"),
            json=payload,
            headers=self._headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return decode_base64_image(response.json()["data"][0]["b64_json"])

    def generate_image_edit(
        self,
        *,
        model: str,
        prompt: str,
        images: Image.Image | list[Image.Image],
        width: int,
        height: int,
        num_inference_steps: int = 20,
        guidance_scale: float | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        output_compression: int | None = None,
    ) -> Image.Image:
        if not isinstance(images, list):
            images = [images]
        data: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
            "num_inference_steps": str(num_inference_steps),
        }
        if guidance_scale is not None:
            data["guidance_scale"] = str(guidance_scale)
        if seed is not None:
            data["seed"] = str(seed)
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if output_compression is not None:
            data["output_compression"] = str(output_compression)

        files = [
            (
                "image[]" if len(images) > 1 else "image",
                (f"image_{index}.png", pil_to_png_bytes(image), "image/png"),
            )
            for index, image in enumerate(images)
        ]

        edit_paths = ["/images/edits", "/images/edit"]
        last_response: requests.Response | None = None
        for api_path in edit_paths:
            response = requests.post(
                build_openai_url(self.base_url, api_path),
                data=data,
                files=files,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            last_response = response
            if response.status_code != 404:
                response.raise_for_status()
                return decode_base64_image(response.json()["data"][0]["b64_json"])

        assert last_response is not None
        last_response.raise_for_status()
        raise ValueError("No image payload returned from image edit endpoint")

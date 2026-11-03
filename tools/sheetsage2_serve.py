# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve SheetSage2 audio-to-score transcription over HTTP.

``POST /v1/audio/transcriptions`` follows the OpenAI multipart upload shape and
returns the recording's ABC score as ``text``, ready for YuE2's
``extra_params.abc``. ``melody_only`` and ``max_seconds`` are extra form fields
with the same meaning as the CLI flags. The model loads once at startup and
requests run one at a time on its device.

Run it in the separate environment documented in
recipes/m-a-p/SheetSage2-H200.md. Like the CLI, it uses the upstream
transcriber, not an Omni engine.
"""

# No `from __future__ import annotations`: FastAPI reads the endpoint's
# annotations at definition time, and they name locally imported types.
import argparse
import asyncio
import math
import tempfile
from pathlib import Path

try:
    from tools.sheetsage2_transcribe import add_model_arguments, checked_score, load_model
except ModuleNotFoundError:  # Run as a script: tools/ itself is on sys.path.
    from sheetsage2_transcribe import add_model_arguments, checked_score, load_model

UPLOAD_CHUNK = 1 << 20


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--served-model-name", help="Model name clients send; defaults to --model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-upload-mb", type=int, default=200, help="Largest accepted recording")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.trust_remote_code:
        raise ValueError(
            "SheetSage2 uses custom model code; pass --trust-remote-code after reviewing the model repository"
        )
    if args.base_model_path is not None and not args.base_model_path.is_dir():
        raise ValueError(f"MERT-v2 directory does not exist: {args.base_model_path}")
    if args.max_upload_mb <= 0:
        raise ValueError("--max-upload-mb must be positive")


def create_app(model, served_model_name: str, dtype: str, max_upload_bytes: int):
    from fastapi import FastAPI, Form, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse

    app = FastAPI(title="SheetSage2")
    # One model on one device: transcriptions queue instead of sharing it.
    lock = asyncio.Lock()

    def error(status: int, message: str) -> JSONResponse:
        kind = "invalid_request_error" if status < 500 else "server_error"
        return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("")

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": served_model_name, "object": "model", "owned_by": "m-a-p"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile,
        model_name: str | None = Form(None, alias="model"),
        response_format: str = Form("json"),
        melody_only: bool = Form(False),
        max_seconds: float | None = Form(None),
    ):
        if model_name not in (None, "", served_model_name):
            return error(404, f"The model {model_name!r} does not exist; this server serves {served_model_name!r}")
        if response_format not in ("json", "text"):
            return error(400, f"response_format must be json or text; got {response_format!r}")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            return error(400, "max_seconds must be finite and positive")
        with tempfile.TemporaryDirectory(prefix="sheetsage2-") as work:
            # Keep the extension: FFmpeg probes the container, but some demuxers need the hint.
            audio = Path(work) / f"input{Path(file.filename or '').suffix.lower()[:8]}"
            size = 0
            with audio.open("wb") as out:
                while chunk := await file.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        return error(413, f"Recording exceeds {max_upload_bytes // (1 << 20)} MiB")
                    out.write(chunk)
            if size == 0:
                return error(400, "file is empty")
            output = Path(work) / "score"
            output.mkdir()
            try:
                async with lock:
                    result = await asyncio.to_thread(
                        model.transcribe,
                        str(audio),
                        output_dir=str(output),
                        melody_only=melody_only,
                        dtype=dtype,
                        max_seconds=max_seconds,
                    )
                abc = checked_score(result)
            except ValueError as exc:  # Upstream rejects recordings it cannot decode with ValueError.
                return error(400, f"SheetSage2 cannot transcribe this recording: {exc}")
            except Exception as exc:  # noqa: BLE001 - surface other upstream failures to the client
                return error(500, f"SheetSage2 transcription failed: {exc}")
        if response_format == "text":
            return PlainTextResponse(abc)
        return {"text": abc}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        import uvicorn

        model, device, _ = load_model(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"SheetSage2: {exc}\nSee recipes/m-a-p/SheetSage2-H200.md for the isolated environment setup.\n")
    name = args.served_model_name or args.model
    app = create_app(model, name, args.dtype, args.max_upload_mb << 20)
    print(f"Serving {name} on {device} at http://{args.host}:{args.port}/v1/audio/transcriptions")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

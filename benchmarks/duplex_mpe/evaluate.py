# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""CLI and evaluation pipeline for Duplex-MPE on vLLM-Omni."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
import soundfile
import torch
import websockets
from pydantic import JsonValue

from benchmarks.duplex_mpe.dataset import (
    DUPLEX_MPE_DATASET_ID,
    DUPLEX_MPE_DATASET_REVISION,
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    AddressingSelection,
    AudioInventoryEntry,
    DuplexMpeDatasetIdentity,
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    download_duplex_mpe_dataset,
    inspect_duplex_mpe_dataset,
    load_duplex_mpe_scenarios,
    read_json_file,
    resolve_dataset_audio_path,
    resolve_retained_scenario_audio_path,
    validate_duplex_mpe_dataset,
    validate_safe_identifier,
    write_json_atomic,
)
from benchmarks.duplex_mpe.metrics import (
    REFERENCE_SILERO_VAD_VERSION,
    format_markdown_score_table,
    score_scenario_reports,
    validate_scenario_reports,
)
from benchmarks.duplex_mpe.protocol import (
    compute_protocol_fingerprint,
    load_reference_seeds,
    resolve_scenario_seed,
)
from benchmarks.duplex_mpe.realtime_session import (
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    SEED_POLICY_METADATA_ONLY,
    InputAudioEncoding,
    ProtocolSessionError,
    RealtimeEndpointSpec,
    RealtimeProtocolSelection,
    RealtimeSessionOptions,
    encode_audio_file_as_data_url,
    negotiate_realtime_endpoint,
    resolve_effective_instructions,
    run_realtime_scenario,
    summarize_transport_exception,
    validate_realtime_base_url,
)
from benchmarks.duplex_mpe.semantic import (
    DEFAULT_ASR_MODEL_ID,
    JudgeEndpointConfig,
    build_semantic_tasks,
    judge_tasks_via_http,
    load_judge_endpoint_config,
    load_validated_existing_verdicts,
    make_deterministic_empty_asr_verdict,
    public_judge_metadata,
    resolve_chat_completions_url,
    resolve_transcriptions_url,
    sanitize_endpoint_origin,
    semantic_task_key,
    transcribe_tasks_via_http,
    validate_http_endpoint_url,
    validate_transcribed_against_tasks,
)
from benchmarks.duplex_mpe.vad import silero_package_version

logger = logging.getLogger(__name__)

EvaluationPhase = Literal["all", "generate", "asr", "judge", "score"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_quiet_http_loggers() -> None:
    for logger_name in ("httpx", "httpcore", "websockets"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def query_package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_short_command(*args: str, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_first_prefixed_line(file_path: str, prefix: str) -> str | None:
    target = Path(file_path)
    if not target.is_file():
        return None
    try:
        for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(prefix):
                _, _, rhs = line.partition(":")
                return rhs.strip() or line.strip()
    except OSError:
        return None
    return None


def collect_benchmark_provenance(
    *,
    model_id: str,
    model_revision: str | None,
    dataset_id: str,
    dataset_revision: str | None,
    launch_command: str | None,
    server_config: dict[str, JsonValue],
    evaluation_input_sha256: str | None = None,
) -> dict[str, JsonValue]:
    """Collect reproducible host, GPU, git, and package metadata for vLLM-Omni."""
    inventory: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.name
        if isinstance(name, str) and name:
            inventory[name.lower()] = dist.version
    ordered_inventory: dict[str, JsonValue] = {name: inventory[name] for name in sorted(inventory)}
    package_payload = json.dumps(ordered_inventory, sort_keys=True, separators=(",", ":"))
    repo_status = run_short_command("git", "status", "--porcelain", cwd=REPO_ROOT)
    try:
        cudnn_raw = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except (AttributeError, OSError, RuntimeError):
        cudnn_raw = None
    cudnn_version: int | None = int(cudnn_raw) if isinstance(cudnn_raw, int) else None

    return {
        "schema_version": 1,
        "repository": {
            "commit": run_short_command("git", "rev-parse", "HEAD", cwd=REPO_ROOT),
            "branch": run_short_command("git", "branch", "--show-current", cwd=REPO_ROOT),
            "dirty": None if repo_status is None else bool(repo_status),
        },
        "host": {
            "platform": platform.platform(),
            "cpu_model": read_first_prefixed_line("/proc/cpuinfo", "model name"),
            "logical_cpus": os.cpu_count(),
            "memory_total_kib": read_first_prefixed_line("/proc/meminfo", "MemTotal"),
        },
        "gpu": {
            "nvidia_smi_csv": run_short_command(
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version,pstate,"
                "power.limit,clocks.sm,clocks.mem,compute_cap",
                "--format=csv,noheader,nounits",
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_build": torch.version.cuda,
            "cudnn_version": cudnn_version,
        },
        "packages": {
            name: query_package_version(name)
            for name in (
                "torch",
                "vllm",
                "vllm-omni",
                "transformers",
                "silero-vad",
                "soundfile",
                "scipy",
            )
        },
        "dependency_inventory": ordered_inventory,
        "dependency_freeze_sha256": hashlib.sha256(package_payload.encode("utf-8")).hexdigest(),
        "artifacts": {
            "requested_model_id": model_id,
            "declared_model_revision": model_revision,
            "dataset_id": dataset_id,
            "dataset_revision": dataset_revision,
            "evaluation_input_sha256": evaluation_input_sha256,
        },
        "launch_command": launch_command,
        "declared_server_config": server_config,
    }


@dataclass(frozen=True, kw_only=True)
class DuplexMpeEvalConfig:
    dataset_root: Path
    output_dir: Path
    phase: EvaluationPhase
    addressing: AddressingSelection = "all"
    mini: bool = False
    max_scenarios: int | None = None
    model: str | None = None
    model_revision: str | None = None
    launch_command: str | None = None
    base_url: str | None = None
    realtime_api_key_env: str | None = None
    protocol: RealtimeProtocolSelection = "auto"
    input_audio_format: InputAudioEncoding | None = None
    input_sample_rate_hz: int | None = None
    output_sample_rate_hz: int | None = None
    voice: str | None = None
    instructions: str | None = None
    ref_audio_path: Path | None = None
    temperature: float | None = None
    max_concurrency: int = 1
    warmup_sessions: int = 1
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    asr_base_url: str | None = None
    asr_model: str | None = None
    asr_api_key_env: str | None = None
    asr_max_concurrency: int = 8
    judge_config_path: Path | None = None
    strict_reference_vad_version: bool = False
    verify_retained_audio: bool = True
    trust_env: bool = False

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.warmup_sessions < 0:
            raise ValueError("warmup_sessions must be >= 0")
        if not math.isfinite(self.session_timeout_seconds) or self.session_timeout_seconds <= 0.0:
            raise ValueError("session_timeout_seconds must be finite and positive")
        if self.asr_max_concurrency < 1:
            raise ValueError("asr_max_concurrency must be >= 1")
        if self.asr_model is not None and not self.asr_model.strip():
            raise ValueError("asr_model must be non-empty when provided")
        if self.realtime_api_key_env is not None and (
            not self.realtime_api_key_env.strip() or self.realtime_api_key_env != self.realtime_api_key_env.strip()
        ):
            raise ValueError("realtime_api_key_env must be null or a trimmed variable name")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0.0):
            raise ValueError("temperature must be finite and >= 0.0 when provided")
        resolved_dataset = self.dataset_root.expanduser().resolve()
        resolved_output = self.output_dir.expanduser().resolve()
        if resolved_output == resolved_dataset or resolved_output.is_relative_to(resolved_dataset):
            raise ValueError("output_dir must be outside dataset_root")
        if self.phase in ("all", "generate"):
            if not self.base_url:
                raise ValueError(f"--base-url is required for phase={self.phase}")
            validate_realtime_base_url(self.base_url)
        if self.phase == "asr" and not self.asr_base_url:
            raise ValueError("--asr-base-url is required for phase=asr")
        elif self.asr_base_url:
            validate_http_endpoint_url(self.asr_base_url)
        if self.phase == "judge" and self.judge_config_path is None:
            raise ValueError("--judge-config is required for phase=judge")
        if (
            self.phase == "all"
            and self.judge_config_path is not None
            and not self.asr_base_url
            and not (resolved_output / "transcribed.json").is_file()
        ):
            raise ValueError(
                "--asr-base-url is required when --judge-config is set for phase=all "
                "without an existing transcribed.json"
            )
        if self.ref_audio_path is not None and self.phase in ("all", "generate"):
            encode_audio_file_as_data_url(self.ref_audio_path)
        if self.launch_command is not None:
            sanitize_launch_command(self.launch_command)


SENSITIVE_ENV_SEGMENTS: frozenset[str] = frozenset(
    {
        "KEY",
        "KEYS",
        "APIKEY",
        "TOKEN",
        "SECRET",
        "SECRETS",
        "PASS",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "CREDENTIAL",
        "CREDENTIALS",
        "AUTH",
        "AUTHORIZATION",
        "BEARER",
        "COOKIE",
        "COOKIES",
        "SESSIONID",
        "HEADER",
        "HEADERS",
        "JWT",
        "NETRC",
        "USERINFO",
        "CERT",
        "OAUTH",
        "OAUTH2",
        "SASL",
    }
)
SENSITIVE_FLAG_SEGMENTS: frozenset[str] = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "token",
        "secret",
        "secrets",
        "pass",
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "sessionid",
        "header",
        "headers",
        "jwt",
        "user",
        "username",
        "login",
        "netrc",
        "cert",
        "oauth",
        "oauth2",
        "sasl",
    }
)
SAFE_KEY_FLAGS: frozenset[str] = frozenset({"sort-key", "sort-keys", "cache-key", "partition-key", "extra-keys"})
SAFE_ENV_NAMES: frozenset[str] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "TORCH_CUDA_ARCH_LIST",
        "PYTHONUNBUFFERED",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_LOGGING_LEVEL",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "NEMOTRON_VOICECHAT_LLM_PATH",
    }
)
SHELL_WRAPPER_NAMES: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh"})
SAFE_SERVER_EXECUTABLES: frozenset[str] = frozenset(
    {
        "env",
        "uv",
        "python",
        "python3",
        "vllm",
        "vllm-omni",
        "sgl-omni",
        "sglang_omni",
        "sglang",
        "torchrun",
    }
)
SAFE_SUBCOMMANDS: frozenset[str] = frozenset({"run", "serve", "launch"})
SAFE_BOOLEAN_FLAGS: frozenset[str] = frozenset(
    {
        "--omni",
        "--no-sync",
        "--duplex-engine-backend-as-process",
        "--enforce-eager",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--enable-realtime",
        "--no-enable-realtime",
        "--text-only",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--enable-metrics",
        "--enable-torch-compile",
        "--skip-server-warmup",
        "--disable-log-requests",
        "--no-enable-log-requests",
    }
)
SAFE_VALUE_FLAGS: frozenset[str] = frozenset(
    {
        "--model-path",
        "--model",
        "--model-name",
        "--served-model-name",
        "--deploy-config",
        "--stage-configs-path",
        "--api-server-count",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--kv-cache-memory-bytes",
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
        "--mm-processor-cache-gb",
        "--generation-config",
        "--variant",
        "--port",
        "--host",
        "--tp",
        "--tp-size",
        "--dp",
        "--dp-size",
        "--pp-size",
        "--ep-size",
        "--dtype",
        "--quantization",
        "--mem-fraction-static",
        "--max-running-requests",
        "--max-total-tokens",
        "--max-seq-len",
        "--context-length",
        "--attention-backend",
        "--sampling-backend",
        "--reasoning-parser",
        "--device",
        "--seed",
        "--log-level",
        "--chat-template",
        "--config",
        "--config-path",
        "--profile",
        "--max-sessions",
        "--max_sessions",
        "--voice",
        "--timeout-s",
        "--timeout_s",
        "--revision",
        "--download-dir",
    }
)
SAFE_DOTTED_CONFIG_PREFIXES: frozenset[str] = frozenset(
    {
        "sampling",
        "vision",
        "speech",
        "window",
        "preprocessing",
        "perception",
        "thinker",
        "talker",
        "talker_ar",
        "code2wav",
        "mimi_encode",
        "lm",
        "asr",
        "tts",
        "vad",
        "turn_detection",
        "session_nodes",
        "services",
        "stages",
        "engine",
        "duplex_session",
    }
)
SAFE_URL_OPTION_SUFFIXES: frozenset[str] = frozenset(
    {".base_url", ".alignment_url", ".heard_asr_url", ".history_asr_url"}
)
SAFE_EMPTY_VALUE_OPTION_SUFFIXES: frozenset[str] = frozenset({".alignment_url", ".heard_asr_url", ".history_asr_url"})
SAFE_DOTTED_CONFIG_FLAG_RE = re.compile(r"^--[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+$")
SAFE_PLAIN_LONG_FLAG_RE = re.compile(r"^--[A-Za-z_][A-Za-z0-9_-]*$")
BLOCKED_LONG_CODE_FLAGS: frozenset[str] = frozenset({"c", "e", "eval", "command", "exec", "code", "script"})
SAFE_EXECUTABLE_TOKEN_RE = re.compile(r"^(?:/[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+)$")
SAFE_SCALAR_VALUE_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
SAFE_HOST_VALUE_RE = re.compile(r"^\[?[A-Za-z0-9_.:-]+\]?$")
SAFE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,:/-]+$")
SAFE_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
EMBEDDED_URL_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"/\\<>]+@")
SAFE_DOTTED_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
KNOWN_TOKEN_PREFIX_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|xox[bp]-[A-Za-z0-9-]{8,}|"
    r"AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{10,}\.)"
)
CANONICAL_SENSITIVE_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "openai_api_key",
        "anthropic_api_key",
        "hf_token",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "secret",
        "client_secret",
        "password",
        "passwd",
        "pass",
        "passphrase",
        "credential",
        "credentials",
        "authorization",
        "header",
        "headers",
        "cookie",
        "cookies",
        "user",
        "username",
        "proxy_user",
        "cert",
    }
)


def split_identifier_segments(identifier: str) -> list[str]:
    decamelized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    return [part for part in re.split(r"[^A-Za-z0-9]+", decamelized) if part]


def is_sensitive_env_name(variable_name: str) -> bool:
    segments = [part.upper() for part in split_identifier_segments(variable_name)]
    return any(part in SENSITIVE_ENV_SEGMENTS for part in segments)


def is_sensitive_cli_flag(flag_token: str) -> bool:
    if flag_token == "-H":
        return True
    if not flag_token.startswith("--"):
        return False
    segments = [part.lower() for part in split_identifier_segments(flag_token.lstrip("-"))]
    hyphenated = "-".join(segments)
    if hyphenated in SAFE_KEY_FLAGS:
        return False
    return any(part in SENSITIVE_FLAG_SEGMENTS for part in segments)


def has_url_userinfo(token_value: str) -> bool:
    if EMBEDDED_URL_USERINFO_RE.search(token_value) is not None:
        return True
    if "://" not in token_value:
        return False
    try:
        parsed = urlsplit(token_value)
        return parsed.username is not None or parsed.password is not None
    except ValueError:
        return True


def contains_embedded_credential_syntax(argument_value: str) -> bool:
    if has_url_userinfo(argument_value):
        return True
    if re.search(r"\bBearer\s+\S+", argument_value, flags=re.IGNORECASE):
        return True
    if KNOWN_TOKEN_PREFIX_RE.search(argument_value) is not None:
        return True
    for key_match in re.finditer(r'([A-Za-z][A-Za-z0-9_.-]*)["\']*\s*(?:[:=]|->)', argument_value):
        key_identifier = key_match.group(1)
        if is_sensitive_cli_flag(f"--{key_identifier}") or is_sensitive_env_name(key_identifier):
            return True
    for xml_match in re.finditer(r"<([A-Za-z][A-Za-z0-9_.-]*)>", argument_value):
        tag_identifier = xml_match.group(1)
        if tag_identifier.upper() == "REDACTED":
            continue
        if is_sensitive_cli_flag(f"--{tag_identifier}") or is_sensitive_env_name(tag_identifier):
            return True
    if not any(ch.isspace() for ch in argument_value) and not argument_value.startswith("-"):
        return False
    try:
        inner_tokens = shlex.split(argument_value, posix=True)
    except ValueError:
        return True
    for subtoken in inner_tokens:
        is_short_opt = subtoken.startswith("-") and not subtoken.startswith("--") and len(subtoken) >= 2
        if is_short_opt and not subtoken[1:].isdigit():
            return True
        if subtoken.startswith("-") and "=" in subtoken:
            subflag, subval = subtoken.split("=", 1)
            if is_sensitive_cli_flag(subflag) or has_url_userinfo(subval):
                return True
        elif is_sensitive_cli_flag(subtoken) or has_url_userinfo(subtoken):
            return True
    return False


def is_allowed_server_executable(executable_name: str) -> bool:
    return (
        executable_name in SAFE_SERVER_EXECUTABLES
        or executable_name in SHELL_WRAPPER_NAMES
        or bool(re.fullmatch(r"python3\.\d+", executable_name))
    )


def is_safe_executable_token(raw_token: str) -> bool:
    if (
        ".." in raw_token
        or SAFE_EXECUTABLE_TOKEN_RE.fullmatch(raw_token) is None
        or has_url_userinfo(raw_token)
        or contains_embedded_credential_syntax(raw_token)
    ):
        return False
    return is_allowed_server_executable(Path(raw_token).name.lower())


def is_safe_positional_model_token(raw_token: str) -> bool:
    """Allow `vllm serve <org/model-or-local-path>` positional checkpoint identifiers."""
    if (
        ".." in raw_token
        or "://" in raw_token
        or has_url_userinfo(raw_token)
        or contains_embedded_credential_syntax(raw_token)
    ):
        return False
    return bool(SAFE_SCALAR_VALUE_RE.fullmatch(raw_token))


def is_safe_dotted_config_flag(flag_token: str) -> bool:
    if SAFE_DOTTED_CONFIG_FLAG_RE.fullmatch(flag_token) is None:
        return False
    if KNOWN_TOKEN_PREFIX_RE.search(flag_token) is not None:
        return False
    if is_sensitive_cli_flag(flag_token):
        return False
    dotted_parts = flag_token.lstrip("-").split(".")
    if len(dotted_parts) > 5:
        return False
    if not all(SAFE_DOTTED_PATH_COMPONENT_RE.fullmatch(part) is not None for part in dotted_parts):
        return False
    top_namespace = dotted_parts[0].lower()
    segments = [part.lower() for part in split_identifier_segments(flag_token.lstrip("-"))]
    return top_namespace in SAFE_DOTTED_CONFIG_PREFIXES and not any(
        part in BLOCKED_LONG_CODE_FLAGS for part in segments
    )


def is_safe_value_flag(flag_token: str) -> bool:
    return flag_token.lower() in SAFE_VALUE_FLAGS or is_safe_dotted_config_flag(flag_token)


def is_safe_http_endpoint_value(value_token: str) -> bool:
    if has_url_userinfo(value_token) or contains_embedded_credential_syntax(value_token):
        return False
    try:
        parsed = urlsplit(value_token)
        return (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and bool(re.fullmatch(r"^[A-Za-z0-9_./-]*$", parsed.path))
        )
    except ValueError:
        return False


def is_well_formed_sensitive_flag_name(flag_token: str) -> bool:
    if has_url_userinfo(flag_token) or KNOWN_TOKEN_PREFIX_RE.search(flag_token) is not None:
        return False
    if SAFE_PLAIN_LONG_FLAG_RE.fullmatch(flag_token) is not None:
        normalized_plain = flag_token.lstrip("-").lower().replace("-", "_")
        return normalized_plain in CANONICAL_SENSITIVE_LEAF_NAMES
    if SAFE_DOTTED_CONFIG_FLAG_RE.fullmatch(flag_token) is not None:
        dotted_parts = flag_token.lstrip("-").split(".")
        if len(dotted_parts) > 5:
            return False
        leaf_normalized = dotted_parts[-1].lower().replace("-", "_")
        non_leaf_parts = dotted_parts[:-1]
        return (
            leaf_normalized in CANONICAL_SENSITIVE_LEAF_NAMES
            and all(SAFE_DOTTED_PATH_COMPONENT_RE.fullmatch(part) is not None for part in non_leaf_parts)
            and not any(is_sensitive_cli_flag(f"--{part}") for part in non_leaf_parts)
        )
    return False


def is_safe_flag_scalar_value(flag_token: str, value_token: str) -> bool:
    lowered_flag = flag_token.lower()
    if value_token == "" and any(lowered_flag.endswith(suffix) for suffix in SAFE_EMPTY_VALUE_OPTION_SUFFIXES):
        return True
    if ".." in value_token or has_url_userinfo(value_token) or contains_embedded_credential_syntax(value_token):
        return False
    if lowered_flag == "--host":
        return "://" not in value_token and bool(SAFE_HOST_VALUE_RE.fullmatch(value_token))
    if lowered_flag == "--device":
        return bool(re.fullmatch(r"^(?:cpu|cuda(?::\d+)?|xpu(?::\d+)?|npu(?::\d+)?)$", value_token))
    if any(lowered_flag.endswith(suffix) for suffix in SAFE_URL_OPTION_SUFFIXES):
        return is_safe_http_endpoint_value(value_token)
    return bool(SAFE_SCALAR_VALUE_RE.fullmatch(value_token))


def sanitize_launch_command(launch_command: str | None, *, nesting_depth: int = 0) -> str | None:
    if launch_command is None:
        return None
    if nesting_depth > 3:
        raise ValueError("Nested shell wrappers exceed maximum depth in --launch-command")
    if any(char in launch_command for char in ("\n", "\r", "\x00")):
        raise ValueError("Control characters are not permitted in --launch-command")
    if "`" in launch_command or "$(" in launch_command:
        raise ValueError("Command substitution is not permitted in --launch-command")
    try:
        lexer = shlex.shlex(launch_command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        punct_tokens = list(lexer)
        raw_tokens = shlex.split(launch_command, posix=True)
    except ValueError:
        raise ValueError("Invalid shell quoting in --launch-command") from None
    if any(punct_tok in {";", "|", "||", "&&", "&", "(", ")"} for punct_tok in punct_tokens):
        raise ValueError("Shell compound operators are not permitted in --launch-command")

    sanitized_tokens: list[str] = []
    executable_seen = False
    serve_positional_allowed = False
    serve_positional_consumed = False
    current_executable = ""
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        if not executable_seen:
            if (
                not token.startswith("-")
                and "=" in token
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.split("=", 1)[0])
            ):
                env_name, env_val = token.split("=", 1)
                if has_url_userinfo(env_val) or contains_embedded_credential_syntax(env_val):
                    if is_sensitive_env_name(env_name):
                        sanitized_tokens.append(f"{env_name}=<REDACTED>")
                    else:
                        raise ValueError(
                            "Embedded credentials in environment assignment are not permitted in --launch-command"
                        )
                elif is_sensitive_env_name(env_name):
                    sanitized_tokens.append(f"{env_name}=<REDACTED>")
                elif env_name in SAFE_ENV_NAMES and ".." not in env_val and bool(SAFE_ENV_VALUE_RE.fullmatch(env_val)):
                    sanitized_tokens.append(token)
                else:
                    raise ValueError(
                        "Unsupported environment assignment in --launch-command; "
                        "only allowlisted runtime variables or redacted credential variables are permitted"
                    )
                index += 1
            elif token.startswith("-"):
                raise ValueError("Unexpected option before executable in --launch-command")
            else:
                if not is_safe_executable_token(token):
                    raise ValueError("Unsupported or unsafe executable token in --launch-command")
                candidate_name = Path(token).name.lower()
                if candidate_name == "env":
                    sanitized_tokens.append(token)
                    index += 1
                elif candidate_name in SHELL_WRAPPER_NAMES:
                    if index + 3 != len(raw_tokens):
                        raise ValueError(
                            f"Shell wrapper {candidate_name} in --launch-command must have exact form "
                            f"'{candidate_name} -c|-lc <quoted-command>'"
                        )
                    wrapper_flag = raw_tokens[index + 1]
                    script_body = raw_tokens[index + 2]
                    if wrapper_flag not in ("-c", "-lc") or script_body.startswith("-"):
                        raise ValueError(f"Unsupported {candidate_name} wrapper flags in --launch-command")
                    nested_sanitized = sanitize_launch_command(script_body, nesting_depth=nesting_depth + 1)
                    assert nested_sanitized is not None
                    sanitized_tokens.extend([token, wrapper_flag, nested_sanitized])
                    index += 3
                    executable_seen = True
                    current_executable = candidate_name
                else:
                    executable_seen = True
                    current_executable = candidate_name
                    sanitized_tokens.append(token)
                    index += 1
        elif token.startswith("-") and not token.startswith("--"):
            if token == "-H":
                if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith("-"):
                    raise ValueError("Option -H requires a header value in --launch-command")
                sanitized_tokens.extend(["-H", "<REDACTED>"])
                index += 2
            elif token == "-m" and current_executable.startswith("python"):
                if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith("-"):
                    raise ValueError("Option -m requires a Python module name in --launch-command")
                module_token = raw_tokens[index + 1]
                if not SAFE_MODULE_NAME_RE.fullmatch(module_token) or contains_embedded_credential_syntax(module_token):
                    raise ValueError("Invalid Python module name in --launch-command")
                sanitized_tokens.extend(["-m", module_token])
                index += 2
            else:
                raise ValueError(
                    "Unsupported or ambiguous short option in --launch-command; use explicit long --flag options"
                )
        elif token.startswith("--") and "=" in token:
            flag_part, value_part = token.split("=", 1)
            normalized_flag = flag_part.lstrip("-").lower()
            if normalized_flag in BLOCKED_LONG_CODE_FLAGS:
                raise ValueError("Inline code option is not permitted in --launch-command")
            if "." in normalized_flag and not is_safe_value_flag(flag_part):
                if (
                    is_sensitive_cli_flag(flag_part)
                    and is_well_formed_sensitive_flag_name(flag_part)
                    and normalized_flag.split(".", 1)[0] in SAFE_DOTTED_CONFIG_PREFIXES
                ):
                    sanitized_tokens.append(f"{flag_part}=<REDACTED>")
                    index += 1
                    continue
                raise ValueError("Unsupported dotted option in --launch-command")
            if is_sensitive_cli_flag(flag_part):
                if not is_well_formed_sensitive_flag_name(flag_part):
                    raise ValueError("Malformed sensitive option name in --launch-command")
                sanitized_tokens.append(f"{flag_part}=<REDACTED>")
            elif is_safe_value_flag(flag_part) and is_safe_flag_scalar_value(flag_part, value_part):
                sanitized_tokens.append(token)
            else:
                raise ValueError("Unsupported option or non-scalar value in --launch-command")
            index += 1
        elif token.startswith("--"):
            normalized_flag = token.lstrip("-").lower()
            if normalized_flag in BLOCKED_LONG_CODE_FLAGS:
                raise ValueError("Inline code option is not permitted in --launch-command")
            if "." in normalized_flag and not is_safe_value_flag(token):
                if (
                    is_sensitive_cli_flag(token)
                    and is_well_formed_sensitive_flag_name(token)
                    and normalized_flag.split(".", 1)[0] in SAFE_DOTTED_CONFIG_PREFIXES
                ):
                    if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith("-"):
                        raise ValueError("Sensitive option requires a non-flag value in --launch-command")
                    sanitized_tokens.extend([token, "<REDACTED>"])
                    index += 2
                    continue
                raise ValueError("Unsupported dotted option in --launch-command")
            if is_sensitive_cli_flag(token):
                if not is_well_formed_sensitive_flag_name(token):
                    raise ValueError("Malformed sensitive option name in --launch-command")
                if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith("-"):
                    raise ValueError("Sensitive option requires a non-flag value in --launch-command")
                sanitized_tokens.extend([token, "<REDACTED>"])
                index += 2
            elif normalized_flag in {flag.lstrip("-") for flag in SAFE_BOOLEAN_FLAGS}:
                sanitized_tokens.append(token)
                index += 1
            elif is_safe_value_flag(token):
                if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith("-"):
                    raise ValueError("Allowlisted option requires a scalar value in --launch-command")
                val_tok = raw_tokens[index + 1]
                if not is_safe_flag_scalar_value(token, val_tok):
                    raise ValueError("Unsafe or non-scalar option value in --launch-command")
                sanitized_tokens.extend([token, val_tok])
                index += 2
            else:
                raise ValueError("Unsupported CLI option in --launch-command")
        elif has_url_userinfo(token) or contains_embedded_credential_syntax(token):
            raise ValueError("Embedded credentials are not permitted in --launch-command")
        elif token in SAFE_SUBCOMMANDS:
            sanitized_tokens.append(token)
            if token == "serve":
                serve_positional_allowed = True
            index += 1
        elif is_safe_executable_token(token):
            candidate_name = Path(token).name.lower()
            if candidate_name in SHELL_WRAPPER_NAMES:
                if index + 3 != len(raw_tokens):
                    raise ValueError(
                        f"Shell wrapper {candidate_name} in --launch-command must have exact form "
                        f"'{candidate_name} -c|-lc <quoted-command>'"
                    )
                wrapper_flag = raw_tokens[index + 1]
                script_body = raw_tokens[index + 2]
                if wrapper_flag not in ("-c", "-lc") or script_body.startswith("-"):
                    raise ValueError(f"Unsupported {candidate_name} wrapper flags in --launch-command")
                nested_sanitized = sanitize_launch_command(script_body, nesting_depth=nesting_depth + 1)
                assert nested_sanitized is not None
                sanitized_tokens.extend([token, wrapper_flag, nested_sanitized])
                index += 3
                current_executable = candidate_name
            else:
                if candidate_name.startswith("python"):
                    current_executable = candidate_name
                sanitized_tokens.append(token)
                index += 1
        elif serve_positional_allowed and not serve_positional_consumed and is_safe_positional_model_token(token):
            serve_positional_consumed = True
            sanitized_tokens.append(token)
            index += 1
        else:
            raise ValueError("Unsupported positional argument in --launch-command")
    if not executable_seen:
        raise ValueError("Missing command executable in --launch-command")
    return shlex.join(sanitized_tokens)


def compute_selected_scenarios_sha256(scenarios: list[DuplexMpeScenarioRecord]) -> str:
    canonical_rows = [
        {
            "scenario_id": scenario.scenario_id,
            "pair_id": scenario.pair_id,
            "addressing": scenario.addressing,
            "trigger_index": scenario.trigger_index,
            "supporting_indices": scenario.supporting_indices,
            "gold_answer": scenario.gold_answer,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "event_type": turn.event_type,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "audio_path": turn.audio_path,
                }
                for turn in scenario.turns
            ],
        }
        for scenario in scenarios
    ]
    return hashlib.sha256(json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compute_selected_turn_audio_sha256(
    dataset_root: Path,
    scenarios: list[DuplexMpeScenarioRecord],
) -> str:
    unique_relative_paths = sorted({turn.audio_path for scenario in scenarios for turn in scenario.turns})
    asset_digests = {
        relative_path: compute_file_sha256(resolve_dataset_audio_path(dataset_root, relative_path))
        for relative_path in unique_relative_paths
    }
    return hashlib.sha256(json.dumps(asset_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_scenario_bindings(
    dataset_root: Path,
    scenarios: list[DuplexMpeScenarioRecord],
) -> dict[str, JsonValue]:
    return {
        scenario.scenario_id: {
            "scenario_sha256": compute_selected_scenarios_sha256([scenario]),
            "turn_audio_sha256": compute_selected_turn_audio_sha256(dataset_root, [scenario]),
        }
        for scenario in scenarios
    }


def verify_saved_dataset_bindings(
    *,
    dataset_root: Path,
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
    reports_by_id: dict[str, dict[str, JsonValue]],
    saved_run_manifest: dict[str, JsonValue],
) -> None:
    saved_dataset_sha256 = saved_run_manifest.get("dataset_sha256")
    if isinstance(saved_dataset_sha256, str) and saved_dataset_sha256:
        dataset_json_path = resolve_dataset_audio_path(dataset_root, "dataset.json")
        current_dataset_sha256 = compute_file_sha256(dataset_json_path) if dataset_json_path.is_file() else None
        if current_dataset_sha256 != saved_dataset_sha256:
            raise ValueError("Current dataset.json SHA-256 differs from recorded run.json dataset_sha256")

    saved_preamble_sha256 = saved_run_manifest.get("preamble_sha256")
    if isinstance(saved_preamble_sha256, str) and saved_preamble_sha256:
        preamble_path = resolve_dataset_audio_path(dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH)
        current_preamble_sha256 = compute_file_sha256(preamble_path) if preamble_path.is_file() else None
        if current_preamble_sha256 != saved_preamble_sha256:
            raise ValueError("Current duty preamble WAV SHA-256 differs from recorded run.json preamble_sha256")

    saved_manifest_digests = saved_run_manifest.get("manifest_sha256")
    if isinstance(saved_manifest_digests, dict):
        selected_conditions = {
            scenarios_by_id[scenario_id].addressing for scenario_id in reports_by_id if scenario_id in scenarios_by_id
        }
        for condition_name in sorted(selected_conditions):
            expected_manifest_digest = saved_manifest_digests.get(condition_name)
            if isinstance(expected_manifest_digest, str) and expected_manifest_digest:
                manifest_path = resolve_dataset_audio_path(dataset_root, f"manifests/{condition_name}.json")
                actual_manifest_digest = compute_file_sha256(manifest_path) if manifest_path.is_file() else None
                if actual_manifest_digest != expected_manifest_digest:
                    raise ValueError(f"Current manifests/{condition_name}.json SHA-256 differs from recorded run.json")

    saved_scenario_bindings = saved_run_manifest.get("scenario_bindings")
    if isinstance(saved_scenario_bindings, dict):
        for scenario_id in sorted(reports_by_id):
            scenario_record = scenarios_by_id[scenario_id]
            recorded_binding = saved_scenario_bindings.get(scenario_id)
            if not isinstance(recorded_binding, dict):
                raise ValueError(f"Scenario {scenario_id} is missing from recorded run.json scenario_bindings")
            expected_scenario_sha = recorded_binding.get("scenario_sha256")
            expected_audio_sha = recorded_binding.get("turn_audio_sha256")
            if expected_scenario_sha != compute_selected_scenarios_sha256([scenario_record]):
                raise ValueError(f"Scenario manifest fields for {scenario_id} differ from recorded run.json")
            if expected_audio_sha != compute_selected_turn_audio_sha256(dataset_root, [scenario_record]):
                raise ValueError(f"Referenced human-turn audio for {scenario_id} differs from recorded run.json")


def verify_selected_human_audio_inventory(
    dataset_root: Path,
    scenarios: list[DuplexMpeScenarioRecord],
) -> bool:
    """Verify SHA-256 digests of all selected human-turn WAVs against `audio_inventory.json`."""
    inventory_path = resolve_dataset_audio_path(dataset_root, "audio_inventory.json")
    if not inventory_path.is_file():
        return False
    raw_inventory = read_json_file(inventory_path)
    if not isinstance(raw_inventory, list):
        return False
    inventory_by_path = {
        entry.path: entry for entry in (AudioInventoryEntry.model_validate(item) for item in raw_inventory)
    }
    referenced_paths = sorted(
        {DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH}
        | {turn.audio_path for scenario in scenarios for turn in scenario.turns}
    )
    for rel_path in referenced_paths:
        inv_entry = inventory_by_path.get(rel_path)
        if inv_entry is None:
            return False
        audio_path = resolve_dataset_audio_path(dataset_root, rel_path)
        if not audio_path.is_file():
            return False
        if audio_path.stat().st_size != inv_entry.bytes or compute_file_sha256(audio_path) != inv_entry.sha256:
            raise ValueError(f"Human-turn audio SHA-256 mismatch against audio_inventory.json: {rel_path}")
    return True


def verify_retained_assistant_audio(
    reports_by_id: dict[str, dict[str, JsonValue]],
    audio_directory: Path,
) -> int:
    verified_count = 0
    for scenario_id, report in reports_by_id.items():
        validate_safe_identifier(scenario_id, "scenario_id")
        expected_sha256 = report.get("assistant_audio_sha256")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not expected_sha256):
            raise ValueError(f"Invalid assistant_audio_sha256 in {scenario_id}")
        if expected_sha256 is None:
            continue
        audio_path = resolve_retained_scenario_audio_path(audio_directory, scenario_id, must_exist=True)
        if compute_file_sha256(audio_path) != expected_sha256:
            raise ValueError(f"Retained assistant audio modified on disk for {scenario_id}")
        audio_header = soundfile.info(str(audio_path))
        expected_sr = report.get("assistant_wav_sr")
        if type(expected_sr) is int and expected_sr > 0 and audio_header.samplerate != expected_sr:
            raise ValueError(
                f"Retained assistant WAV sample rate {audio_header.samplerate} != {expected_sr} for {scenario_id}"
            )
        verified_count += 1
    return verified_count


@dataclass(frozen=True, kw_only=True)
class GeneratePhaseResult:
    reports_by_id: dict[str, dict[str, JsonValue]]
    failed_scenarios: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.reports_by_id)

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self.reports_by_id

    def __getitem__(self, scenario_id: str) -> dict[str, JsonValue]:
        return self.reports_by_id[scenario_id]


def load_validated_failure_ledger(ledger_path: Path) -> dict[str, str]:
    if not ledger_path.is_file():
        return {}
    raw_ledger = read_json_file(ledger_path)
    if not isinstance(raw_ledger, dict):
        raise ValueError(f"{ledger_path} must contain a JSON object")
    validated_ledger: dict[str, str] = {}
    for entry_key, entry_value in raw_ledger.items():
        if not isinstance(entry_key, str) or not isinstance(entry_value, str):
            raise ValueError(f"{ledger_path} entries must map string keys to string reasons")
        validated_ledger[entry_key] = entry_value
    return validated_ledger


def parse_json_object_rows(raw_rows: list[JsonValue], source_path: Path) -> list[dict[str, JsonValue]]:
    typed_rows: list[dict[str, JsonValue]] = []
    for row_index, entry in enumerate(raw_rows):
        if not isinstance(entry, dict):
            raise ValueError(f"{source_path} element at index {row_index} must be a JSON object")
        typed_rows.append(entry)
    return typed_rows


def is_compatible_saved_run_manifest(
    saved_manifest: dict[str, JsonValue],
    current_manifest: dict[str, JsonValue],
) -> bool:
    comparable_saved = {key: value for key, value in saved_manifest.items() if key != "concurrency_history"}
    comparable_current = {key: value for key, value in current_manifest.items() if key != "concurrency_history"}
    if set(comparable_saved) != set(comparable_current):
        return False
    for field_name, current_value in comparable_current.items():
        saved_value = comparable_saved.get(field_name)
        if field_name == "session_timeout_seconds":
            if (
                not isinstance(saved_value, int | float)
                or type(saved_value) is bool
                or not isinstance(current_value, int | float)
                or type(current_value) is bool
                or float(current_value) < float(saved_value)
            ):
                return False
        elif field_name in ("max_concurrency", "warmup_sessions"):
            expected_min = 1 if field_name == "max_concurrency" else 0
            if (
                type(saved_value) is not int
                or saved_value < expected_min
                or type(current_value) is not int
                or current_value < expected_min
            ):
                return False
        elif saved_value != current_value:
            return False
    return True


def load_saved_reports(report_path: Path) -> dict[str, dict[str, JsonValue]]:
    if not report_path.is_file():
        return {}
    raw_reports = read_json_file(report_path)
    if not isinstance(raw_reports, dict):
        raise ValueError(f"{report_path} must contain a JSON object")
    typed_reports: dict[str, dict[str, JsonValue]] = {}
    for scenario_id, entry in raw_reports.items():
        validate_safe_identifier(scenario_id, "scenario_id")
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed scenario report for {scenario_id} in {report_path}")
        typed_reports[scenario_id] = entry
    return typed_reports


def resolve_path_sync(path_value: Path) -> Path:
    return path_value.expanduser().resolve()


def remove_file_if_present_sync(target_path: Path) -> None:
    if target_path.is_file():
        target_path.unlink()


def persist_scenario_checkpoint_sync(
    report_path: Path,
    reports_snapshot: dict[str, dict[str, JsonValue]],
    failures_path: Path,
    failures_snapshot: dict[str, str],
) -> None:
    write_json_atomic(report_path, reports_snapshot)
    if failures_snapshot:
        write_json_atomic(failures_path, failures_snapshot)
    elif failures_path.is_file():
        failures_path.unlink()


def persist_judge_failures_sync(
    judge_failures_path: Path,
    merged_judge_failures: dict[str, str],
) -> None:
    if merged_judge_failures:
        write_json_atomic(judge_failures_path, merged_judge_failures)
    elif judge_failures_path.is_file():
        judge_failures_path.unlink()


def load_ref_audio_artifacts_sync(ref_audio_path: Path | None) -> tuple[Path | None, str | None, str | None]:
    if ref_audio_path is None:
        return None, None, None
    resolved_ref = ref_audio_path.expanduser().resolve()
    data_url = encode_audio_file_as_data_url(ref_audio_path)
    sha256_digest = compute_file_sha256(resolved_ref)
    return resolved_ref, data_url, sha256_digest


@dataclass(frozen=True, kw_only=True)
class PreparedGenerateState:
    output_dir: Path
    run_manifest_path: Path
    report_path: Path
    failures_path: Path
    audio_directory: Path
    seeds: dict[str, int]
    saved_reports: dict[str, dict[str, JsonValue]]
    pending_scenarios: list[DuplexMpeScenarioRecord]
    existing_failure_ledger: dict[str, str]


def prepare_generate_manifest_and_reports_sync(
    config: DuplexMpeEvalConfig,
    scenarios: list[DuplexMpeScenarioRecord],
    *,
    dataset_root: Path,
    endpoint_spec: RealtimeEndpointSpec,
    effective_model: str,
    effective_max_concurrency: int,
    current_vad_version: str | None,
    ref_audio_sha256: str | None,
) -> PreparedGenerateState:
    seeds = load_reference_seeds()
    for scenario in scenarios:
        resolve_scenario_seed(scenario.scenario_id, seeds)

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run.json"
    report_path = output_dir / "report.json"
    failures_path = output_dir / "failures.json"
    audio_directory = output_dir / "audio"

    if report_path.is_file() and not run_manifest_path.is_file():
        raise ValueError(f"Existing {report_path} has no {run_manifest_path} manifest; choose a new --output-dir")

    manifest_digests: dict[str, JsonValue] = {}
    for condition_name in ("explicit", "implicit"):
        condition_path = resolve_dataset_audio_path(dataset_root, f"manifests/{condition_name}.json")
        if condition_path.is_file():
            manifest_digests[condition_name] = compute_file_sha256(condition_path)
    dataset_json_path = resolve_dataset_audio_path(dataset_root, "dataset.json")
    dataset_sha256 = compute_file_sha256(dataset_json_path) if dataset_json_path.is_file() else None
    preamble_path = resolve_dataset_audio_path(dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH)
    preamble_sha256 = compute_file_sha256(preamble_path) if preamble_path.is_file() else None
    endpoint_websocket_sha256 = hashlib.sha256(endpoint_spec.websocket_url.encode("utf-8")).hexdigest()
    effective_instructions = resolve_effective_instructions(
        model=endpoint_spec.reported_model or config.model,
        requires_auto_response_on_open=endpoint_spec.requires_auto_response_on_open,
        explicit_instructions=config.instructions,
    )
    instructions_sha256 = (
        hashlib.sha256(effective_instructions.encode("utf-8")).hexdigest()
        if effective_instructions is not None
        else None
    )

    assert config.base_url is not None
    run_manifest: dict[str, JsonValue] = {
        "model": effective_model,
        "reported_model": endpoint_spec.reported_model,
        "model_revision": config.model_revision,
        "launch_command": sanitize_launch_command(config.launch_command),
        "base_url": sanitize_endpoint_origin(config.base_url),
        "endpoint_websocket_sha256": endpoint_websocket_sha256,
        "wire_mode": endpoint_spec.wire_mode,
        "protocol": endpoint_spec.protocol,
        "input_audio_encoding": endpoint_spec.input_audio_encoding,
        "input_sample_rate_hz": endpoint_spec.input_sample_rate_hz,
        "output_sample_rate_hz": endpoint_spec.output_sample_rate_hz,
        "turn_detection": endpoint_spec.turn_detection,
        "voice": config.voice,
        "instructions_sha256": instructions_sha256,
        "ref_audio_sha256": ref_audio_sha256,
        "temperature": config.temperature,
        "max_concurrency": effective_max_concurrency,
        "warmup_sessions": config.warmup_sessions,
        "session_timeout_seconds": config.session_timeout_seconds,
        "addressing": config.addressing,
        "mini": config.mini,
        "max_scenarios": config.max_scenarios,
        "evaluation_clock": "realtime_wall_clock",
        "seed_policy": SEED_POLICY_METADATA_ONLY,
        "silero_vad_version": current_vad_version,
        "code_fingerprint": compute_protocol_fingerprint(),
        "dataset_sha256": dataset_sha256,
        "preamble_sha256": preamble_sha256,
        "manifest_sha256": manifest_digests,
        "selected_scenarios_sha256": compute_selected_scenarios_sha256(scenarios),
        "selected_turn_audio_sha256": compute_selected_turn_audio_sha256(dataset_root, scenarios),
        "scenario_bindings": build_scenario_bindings(dataset_root, scenarios),
        "seed_map_sha256": hashlib.sha256(json.dumps(seeds, sort_keys=True).encode("utf-8")).hexdigest(),
    }

    existing_manifest: dict[str, JsonValue] | None = None
    if run_manifest_path.is_file():
        loaded_manifest = read_json_file(run_manifest_path)
        if not isinstance(loaded_manifest, dict) or not is_compatible_saved_run_manifest(
            loaded_manifest,
            run_manifest,
        ):
            raise ValueError(f"Run configuration in {run_manifest_path} changed; choose a new --output-dir")
        existing_manifest = loaded_manifest

    saved_reports = load_saved_reports(report_path)
    pending_scenarios: list[DuplexMpeScenarioRecord] = []
    for scenario in scenarios:
        if scenario.scenario_id in saved_reports:
            recorded_sha256 = saved_reports[scenario.scenario_id].get("assistant_audio_sha256")
            try:
                wav_path = resolve_retained_scenario_audio_path(
                    audio_directory,
                    scenario.scenario_id,
                    must_exist=True,
                )
            except FileNotFoundError as exc:
                raise ValueError(f"Saved audio missing or modified for {scenario.scenario_id}") from exc
            if compute_file_sha256(wav_path) != recorded_sha256:
                raise ValueError(f"Saved audio missing or modified for {scenario.scenario_id}")
            continue
        pending_scenarios.append(scenario)

    if not pending_scenarios:
        if existing_manifest is None:
            run_manifest["concurrency_history"] = [effective_max_concurrency]
            write_json_atomic(run_manifest_path, run_manifest)
        if failures_path.is_file():
            failures_path.unlink()
        return PreparedGenerateState(
            output_dir=output_dir,
            run_manifest_path=run_manifest_path,
            report_path=report_path,
            failures_path=failures_path,
            audio_directory=audio_directory,
            seeds=seeds,
            saved_reports=saved_reports,
            pending_scenarios=[],
            existing_failure_ledger={},
        )

    prior_history: list[int] = []
    if existing_manifest is not None:
        raw_history = existing_manifest.get("concurrency_history")
        if isinstance(raw_history, list):
            for item in raw_history:
                if type(item) is int and item >= 1 and item not in prior_history:
                    prior_history.append(item)
        saved_concurrency = existing_manifest.get("max_concurrency")
        if type(saved_concurrency) is int and saved_concurrency >= 1 and saved_concurrency not in prior_history:
            prior_history.append(saved_concurrency)
    if effective_max_concurrency not in prior_history:
        prior_history.append(effective_max_concurrency)
    run_manifest["concurrency_history"] = [int(level) for level in prior_history]
    if existing_manifest != run_manifest:
        write_json_atomic(run_manifest_path, run_manifest)

    existing_failure_ledger = load_validated_failure_ledger(failures_path)
    return PreparedGenerateState(
        output_dir=output_dir,
        run_manifest_path=run_manifest_path,
        report_path=report_path,
        failures_path=failures_path,
        audio_directory=audio_directory,
        seeds=seeds,
        saved_reports=saved_reports,
        pending_scenarios=pending_scenarios,
        existing_failure_ledger=existing_failure_ledger,
    )


async def run_generate_phase(
    config: DuplexMpeEvalConfig,
    scenarios: list[DuplexMpeScenarioRecord],
) -> GeneratePhaseResult:
    assert config.base_url is not None
    current_vad_version = await asyncio.to_thread(silero_package_version)
    if config.strict_reference_vad_version and current_vad_version != REFERENCE_SILERO_VAD_VERSION:
        raise ValueError(
            f"--strict-reference-vad-version requires silero-vad=={REFERENCE_SILERO_VAD_VERSION}, "
            f"found {current_vad_version}"
        )

    dataset_root = await asyncio.to_thread(resolve_path_sync, config.dataset_root)
    _, ref_audio_data_url, ref_audio_sha256 = await asyncio.to_thread(
        load_ref_audio_artifacts_sync,
        config.ref_audio_path,
    )
    session_options = RealtimeSessionOptions(
        instructions=config.instructions,
        voice=config.voice,
        ref_audio_data_url=ref_audio_data_url,
        temperature=config.temperature,
    )
    endpoint_spec = await negotiate_realtime_endpoint(
        config.base_url,
        protocol_selection=config.protocol,
        requested_model=config.model,
        input_audio_encoding=config.input_audio_format,
        input_sample_rate_hz=config.input_sample_rate_hz,
        output_sample_rate_hz=config.output_sample_rate_hz,
        session_options=session_options,
        api_key_env=config.realtime_api_key_env,
        trust_env=config.trust_env,
    )
    effective_max_concurrency = (
        min(config.max_concurrency, endpoint_spec.max_sessions)
        if endpoint_spec.max_sessions is not None
        else config.max_concurrency
    )
    if effective_max_concurrency < config.max_concurrency:
        logger.warning(
            "Clamping --max-concurrency to server max_sessions",
            extra={
                "requested_concurrency": config.max_concurrency,
                "effective_concurrency": effective_max_concurrency,
            },
        )
    effective_model = config.model if config.model is not None else (endpoint_spec.reported_model or "unknown")
    prep_state = await asyncio.to_thread(
        prepare_generate_manifest_and_reports_sync,
        config,
        scenarios,
        dataset_root=dataset_root,
        endpoint_spec=endpoint_spec,
        effective_model=effective_model,
        effective_max_concurrency=effective_max_concurrency,
        current_vad_version=current_vad_version,
        ref_audio_sha256=ref_audio_sha256,
    )
    output_dir = prep_state.output_dir
    report_path = prep_state.report_path
    failures_path = prep_state.failures_path
    audio_directory = prep_state.audio_directory
    seeds = prep_state.seeds
    saved_reports = prep_state.saved_reports
    pending_scenarios = prep_state.pending_scenarios

    if not pending_scenarios:
        logger.info(
            "All selected scenarios already recorded",
            extra={"scenarios": len(scenarios), "report_path": str(report_path)},
        )
        return GeneratePhaseResult(reports_by_id=saved_reports, failed_scenarios={})

    if config.warmup_sessions > 0:
        warmup_directory = output_dir / "warmup"
        warmup_scenario = scenarios[0]
        logger.info(
            "Running unscored warmup sessions",
            extra={"warmup_sessions": config.warmup_sessions, "scenario_id": warmup_scenario.scenario_id},
        )
        warmup_semaphore = asyncio.Semaphore(effective_max_concurrency)

        async def run_warmup_one(warmup_index: int) -> None:
            async with warmup_semaphore:
                try:
                    await run_realtime_scenario(
                        warmup_scenario,
                        dataset_root=dataset_root,
                        endpoint_spec=endpoint_spec,
                        output_audio_path=warmup_directory / f"warmup_{warmup_index}.wav",
                        session_timeout_seconds=config.session_timeout_seconds,
                        seed_mapping=seeds,
                    )
                except (
                    OSError,
                    ProtocolSessionError,
                    RuntimeError,
                    TimeoutError,
                    asyncio.TimeoutError,
                    ValueError,
                    httpx.HTTPError,
                    websockets.WebSocketException,
                ) as exc:
                    warmup_reason = (
                        str(exc)
                        if isinstance(
                            exc,
                            ProtocolSessionError | RuntimeError | TimeoutError | asyncio.TimeoutError | ValueError,
                        )
                        and str(exc)
                        else summarize_transport_exception(exc)
                    )
                    logger.warning(
                        "Warmup session failed; continuing to measured scenarios",
                        extra={"warmup_index": warmup_index, "reason": warmup_reason},
                    )

        await asyncio.gather(*(run_warmup_one(index) for index in range(config.warmup_sessions)))

    semaphore = asyncio.Semaphore(effective_max_concurrency)
    persist_semaphore = asyncio.Semaphore(1)
    existing_failure_ledger = prep_state.existing_failure_ledger
    failed_scenarios: dict[str, str] = {
        scenario_key: reason
        for scenario_key, reason in existing_failure_ledger.items()
        if scenario_key not in saved_reports
    }

    async def record_scenario(scenario: DuplexMpeScenarioRecord) -> None:
        async with semaphore:
            wav_path = await asyncio.to_thread(
                resolve_retained_scenario_audio_path,
                audio_directory,
                scenario.scenario_id,
                must_exist=False,
            )
            try:
                scenario_report = await run_realtime_scenario(
                    scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=endpoint_spec,
                    output_audio_path=wav_path,
                    session_timeout_seconds=config.session_timeout_seconds,
                    seed_mapping=seeds,
                )
            except (
                OSError,
                ProtocolSessionError,
                RuntimeError,
                TimeoutError,
                asyncio.TimeoutError,
                ValueError,
                httpx.HTTPError,
                websockets.WebSocketException,
            ) as exc:
                failure_reason = (
                    str(exc)
                    if isinstance(
                        exc,
                        ProtocolSessionError | RuntimeError | TimeoutError | asyncio.TimeoutError | ValueError,
                    )
                    and str(exc)
                    else summarize_transport_exception(exc)
                )
                async with persist_semaphore:
                    failed_scenarios[scenario.scenario_id] = failure_reason
                    failures_snapshot = dict(failed_scenarios)
                    await asyncio.to_thread(write_json_atomic, failures_path, failures_snapshot)
                logger.error(
                    "Scenario failed; remaining pending for resume",
                    extra={"scenario_id": scenario.scenario_id, "reason": failure_reason},
                )
                return

            async with persist_semaphore:
                scenario_report["recorded_concurrency"] = effective_max_concurrency
                saved_reports[scenario.scenario_id] = scenario_report
                failed_scenarios.pop(scenario.scenario_id, None)
                reports_snapshot = dict(saved_reports)
                failures_snapshot = dict(failed_scenarios)
                await asyncio.to_thread(
                    persist_scenario_checkpoint_sync,
                    report_path,
                    reports_snapshot,
                    failures_path,
                    failures_snapshot,
                )

            target_block = scenario_report.get("T")
            outcome_label = target_block.get("outcome") if isinstance(target_block, dict) else None
            logger.info(
                "Recorded Duplex-MPE scenario",
                extra={
                    "scenario_id": scenario.scenario_id,
                    "target_outcome": outcome_label,
                    "completed": len(saved_reports),
                    "selected": len(scenarios),
                },
            )

    await asyncio.gather(*(record_scenario(scenario) for scenario in pending_scenarios))
    if not failed_scenarios:
        await asyncio.to_thread(remove_file_if_present_sync, failures_path)
    return GeneratePhaseResult(reports_by_id=saved_reports, failed_scenarios=failed_scenarios)


@dataclass(frozen=True, kw_only=True)
class InitialEvaluationBootstrap:
    dataset_root: Path
    output_dir: Path
    dataset_identity: DuplexMpeDatasetIdentity
    scenarios: list[DuplexMpeScenarioRecord]
    human_audio_inventory_verified: bool
    judge_config: JudgeEndpointConfig | None


def bootstrap_evaluation_sync(config: DuplexMpeEvalConfig) -> InitialEvaluationBootstrap:
    dataset_root = config.dataset_root.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_identity = inspect_duplex_mpe_dataset(dataset_root)
    scenarios = load_duplex_mpe_scenarios(
        dataset_root,
        addressing=config.addressing,
        mini=config.mini,
        max_scenarios=config.max_scenarios,
    )
    human_audio_inventory_verified = verify_selected_human_audio_inventory(dataset_root, scenarios)
    judge_config = (
        load_judge_endpoint_config(config.judge_config_path) if config.judge_config_path is not None else None
    )
    return InitialEvaluationBootstrap(
        dataset_root=dataset_root,
        output_dir=output_dir,
        dataset_identity=dataset_identity,
        scenarios=scenarios,
        human_audio_inventory_verified=human_audio_inventory_verified,
        judge_config=judge_config,
    )


def load_offline_reports_and_unselected_keys_sync(
    *,
    dataset_root: Path,
    report_path: Path,
    failures_path: Path,
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
) -> tuple[dict[str, dict[str, JsonValue]], dict[str, str], frozenset[str]]:
    raw_failures = load_validated_failure_ledger(failures_path)
    all_saved = load_saved_reports(report_path)
    reports_by_id = {scenario_id: all_saved[scenario_id] for scenario_id in scenarios_by_id if scenario_id in all_saved}
    failed_scenarios = {
        scenario_key: reason
        for scenario_key, reason in raw_failures.items()
        if scenario_key in scenarios_by_id and scenario_key not in reports_by_id
    }
    if not reports_by_id:
        raise FileNotFoundError(f"No matching scenario reports found in {report_path}")
    allowed_unselected_task_keys: frozenset[str] = frozenset()
    unselected_ids = set(all_saved) - set(reports_by_id)
    if unselected_ids:
        all_dataset_scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all")
        all_scenarios_by_id = {row.scenario_id: row for row in all_dataset_scenarios}
        unknown_saved_ids = sorted(unselected_ids - set(all_scenarios_by_id))
        if unknown_saved_ids:
            raise ValueError(f"Unknown scenario_id in {report_path}: {unknown_saved_ids[0]}")
        unselected_reports = {scenario_id: all_saved[scenario_id] for scenario_id in unselected_ids}
        unselected_tasks = build_semantic_tasks(unselected_reports, all_scenarios_by_id)
        allowed_unselected_task_keys = frozenset(semantic_task_key(task) for task in unselected_tasks)
    return reports_by_id, failed_scenarios, allowed_unselected_task_keys


def load_saved_run_manifest_sync(run_manifest_path: Path) -> dict[str, JsonValue]:
    if not run_manifest_path.is_file():
        return {}
    raw_run_manifest = read_json_file(run_manifest_path)
    if not isinstance(raw_run_manifest, dict):
        raise ValueError(f"{run_manifest_path} must contain a JSON object")
    return raw_run_manifest


def maybe_write_semantic_tasks_sync(
    *,
    tasks_path: Path,
    semantic_tasks: list[dict[str, JsonValue]],
    should_always_write: bool,
) -> None:
    if should_always_write or (not tasks_path.is_file()):
        write_json_atomic(tasks_path, semantic_tasks)


def load_offline_transcribed_cache_sync(
    *,
    transcribed_path: Path,
    semantic_tasks: list[dict[str, JsonValue]],
    expected_asr_model: str | None,
    expected_asr_endpoint_sha256: str | None,
    allowed_unselected_task_keys: frozenset[str],
    allow_partial: bool,
) -> tuple[bool, list[dict[str, JsonValue]]]:
    if not transcribed_path.is_file():
        return False, []
    raw_transcribed = read_json_file(transcribed_path)
    if not isinstance(raw_transcribed, list):
        raise ValueError(f"{transcribed_path} must contain a JSON array")
    all_transcribed_rows = parse_json_object_rows(raw_transcribed, transcribed_path)
    selected_transcribed_map = validate_transcribed_against_tasks(
        semantic_tasks,
        all_transcribed_rows,
        expected_asr_model=expected_asr_model,
        expected_asr_endpoint_sha256=expected_asr_endpoint_sha256,
        allowed_unselected_task_keys=allowed_unselected_task_keys,
        allow_partial=allow_partial,
    )
    return True, list(selected_transcribed_map.values())


def load_offline_verdicts_and_failures_sync(
    *,
    labels_path: Path,
    judge_failures_path: Path,
    transcribed_path: Path,
    has_transcribed_cache: bool,
    semantic_tasks: list[dict[str, JsonValue]],
    transcribed_tasks: list[dict[str, JsonValue]],
    judge_config: JudgeEndpointConfig | None,
    allowed_unselected_task_keys: frozenset[str],
    expected_selected_task_keys: frozenset[str],
    allow_partial: bool,
) -> tuple[list[dict[str, JsonValue]], tuple[str, ...]]:
    if labels_path.is_file():
        raw_labels = read_json_file(labels_path)
        if not isinstance(raw_labels, list):
            raise ValueError(f"{labels_path} must contain a JSON array")
        raw_verdict_rows = parse_json_object_rows(raw_labels, labels_path)
        if raw_verdict_rows:
            if not has_transcribed_cache:
                raise ValueError(f"Cannot verify {labels_path} without {transcribed_path}")
            indexed_transcribed = validate_transcribed_against_tasks(
                semantic_tasks,
                transcribed_tasks,
                allow_partial=allow_partial,
            )
            expected_judge_endpoint_sha256 = (
                hashlib.sha256(resolve_chat_completions_url(judge_config.base_url).encode("utf-8")).hexdigest()
                if judge_config is not None
                else None
            )
            validated_verdicts = load_validated_existing_verdicts(
                indexed_transcribed,
                raw_verdict_rows,
                judge_config=judge_config,
                expected_endpoint_sha256=expected_judge_endpoint_sha256,
                allowed_unselected_task_keys=allowed_unselected_task_keys,
            )
            verdicts = list(validated_verdicts.values())
        else:
            verdicts = []
        persisted_judge_failures = load_validated_failure_ledger(judge_failures_path)
        resolved_verdict_keys = {semantic_task_key(row) for row in verdicts}
        failed_keys = tuple(
            sorted(
                task_key
                for task_key in persisted_judge_failures
                if task_key in expected_selected_task_keys and task_key not in resolved_verdict_keys
            )
        )
        return verdicts, failed_keys

    persisted_judge_failures = load_validated_failure_ledger(judge_failures_path)
    failed_keys = tuple(
        sorted(task_key for task_key in persisted_judge_failures if task_key in expected_selected_task_keys)
    )
    return [], failed_keys


def persist_summary_outputs_sync(
    *,
    output_dir: Path,
    effective_model: str,
    heading_suffix: str,
    markdown_table: str,
    summary_payload: dict[str, JsonValue],
) -> Path:
    (output_dir / "RESULTS.md").write_text(
        f"# Duplex-MPE Evaluation ({effective_model}){heading_suffix}\n\n{markdown_table}",
        encoding="utf-8",
    )
    summary_path = output_dir / "duplex_mpe_results.json"
    write_json_atomic(summary_path, summary_payload)
    return summary_path


async def run_duplex_mpe_evaluation(config: DuplexMpeEvalConfig) -> dict[str, JsonValue]:
    """Execute the selected Duplex-MPE phase(s) and write summary artifacts."""
    boot = await asyncio.to_thread(bootstrap_evaluation_sync, config)
    dataset_root = boot.dataset_root
    output_dir = boot.output_dir
    dataset_identity = boot.dataset_identity
    scenarios = boot.scenarios
    human_audio_inventory_verified = boot.human_audio_inventory_verified
    judge_config = boot.judge_config
    if not scenarios:
        raise ValueError("No Duplex-MPE scenarios selected")
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    run_manifest_path = output_dir / "run.json"
    report_path = output_dir / "report.json"
    failures_path = output_dir / "failures.json"
    judge_failures_path = output_dir / "judge_failures.json"
    tasks_path = output_dir / "tasks.json"
    transcribed_path = output_dir / "transcribed.json"
    labels_path = output_dir / "labels.json"
    audio_directory = output_dir / "audio"

    failed_scenarios: dict[str, str] = {}
    failed_semantic_task_keys: tuple[str, ...] = ()
    allowed_unselected_task_keys: frozenset[str] = frozenset()
    if config.phase in ("all", "generate"):
        generate_result = await run_generate_phase(config, scenarios)
        reports_by_id = generate_result.reports_by_id
        failed_scenarios = dict(generate_result.failed_scenarios)
    else:
        reports_by_id, failed_scenarios, allowed_unselected_task_keys = await asyncio.to_thread(
            load_offline_reports_and_unselected_keys_sync,
            dataset_root=dataset_root,
            report_path=report_path,
            failures_path=failures_path,
            scenarios_by_id=scenarios_by_id,
        )

    saved_run_manifest = await asyncio.to_thread(load_saved_run_manifest_sync, run_manifest_path)

    saved_model = saved_run_manifest.get("model")
    if config.model is not None and isinstance(saved_model, str) and saved_model and config.model != saved_model:
        raise ValueError(f"Requested --model={config.model!r} conflicts with saved run.json model={saved_model!r}")
    saved_revision = saved_run_manifest.get("model_revision")
    if config.model_revision is not None and saved_revision is not None and config.model_revision != saved_revision:
        raise ValueError(f"Requested --model-revision={config.model_revision!r} conflicts with saved run.json")
    effective_model = (
        config.model
        if config.model is not None
        else (saved_model if isinstance(saved_model, str) and saved_model else "unknown")
    )
    effective_model_revision = (
        config.model_revision
        if config.model_revision is not None
        else (saved_revision if isinstance(saved_revision, str) else None)
    )
    sanitized_cli_launch = sanitize_launch_command(config.launch_command)
    raw_saved_launch = saved_run_manifest.get("launch_command")
    if raw_saved_launch is not None and not isinstance(raw_saved_launch, str):
        raise ValueError(f"Invalid launch_command in {run_manifest_path}; expected string or null")
    sanitized_saved_launch = sanitize_launch_command(raw_saved_launch) if isinstance(raw_saved_launch, str) else None
    if (
        sanitized_cli_launch is not None
        and sanitized_saved_launch is not None
        and sanitized_cli_launch != sanitized_saved_launch
    ):
        raise ValueError("Requested --launch-command conflicts with saved run.json")
    effective_launch_command = sanitized_cli_launch if sanitized_cli_launch is not None else sanitized_saved_launch

    missing_selected_scenarios = sorted(set(scenarios_by_id) - set(reports_by_id))

    if config.phase == "generate":
        is_generate_complete = (not missing_selected_scenarios) and (not failed_scenarios)
        missing_json: list[JsonValue] = list(missing_selected_scenarios)
        failed_json: dict[str, JsonValue] = dict(failed_scenarios)
        return {
            "status": "complete" if is_generate_complete else "incomplete",
            "phase": "generate",
            "scenarios_selected": len(scenarios_by_id),
            "scenarios_recorded": len(reports_by_id),
            "missing_scenarios": missing_json,
            "failed_scenarios": failed_json,
            "report_path": str(report_path),
        }
    if not reports_by_id:
        raise RuntimeError(
            f"All {len(scenarios_by_id)} selected scenarios failed during generation; inspect {failures_path}"
        )
    if config.phase not in ("all", "generate") and saved_run_manifest:
        await asyncio.to_thread(
            verify_saved_dataset_bindings,
            dataset_root=dataset_root,
            scenarios_by_id=scenarios_by_id,
            reports_by_id=reports_by_id,
            saved_run_manifest=saved_run_manifest,
        )

    semantic_tasks = build_semantic_tasks(reports_by_id, scenarios_by_id)
    await asyncio.to_thread(
        maybe_write_semantic_tasks_sync,
        tasks_path=tasks_path,
        semantic_tasks=semantic_tasks,
        should_always_write=(not allowed_unselected_task_keys and config.phase != "score"),
    )

    resolved_asr_model = config.asr_model if config.asr_model is not None else DEFAULT_ASR_MODEL_ID
    expected_offline_asr_endpoint_sha256 = (
        hashlib.sha256(resolve_transcriptions_url(config.asr_base_url).encode("utf-8")).hexdigest()
        if config.asr_base_url is not None
        else None
    )
    expected_offline_asr_model = (
        config.asr_model
        if config.asr_model is not None
        else (DEFAULT_ASR_MODEL_ID if config.asr_base_url is not None else None)
    )
    has_transcribed_cache = False
    should_run_asr = config.phase == "asr" or (config.phase == "all" and config.asr_base_url is not None)
    if should_run_asr:
        assert config.asr_base_url is not None
        transcribed_tasks = await transcribe_tasks_via_http(
            semantic_tasks,
            audio_directory=audio_directory,
            output_path=transcribed_path,
            asr_base_url=config.asr_base_url,
            asr_model_id=resolved_asr_model,
            api_key_env=config.asr_api_key_env,
            max_concurrency=config.asr_max_concurrency,
            trust_env=config.trust_env,
            allowed_unselected_task_keys=allowed_unselected_task_keys,
        )
        has_transcribed_cache = True
        if config.phase == "asr":
            is_asr_complete = (not missing_selected_scenarios) and (not failed_scenarios)
            asr_missing_json: list[JsonValue] = list(missing_selected_scenarios)
            asr_failed_json: dict[str, JsonValue] = dict(failed_scenarios)
            return {
                "status": "complete" if is_asr_complete else "incomplete",
                "phase": "asr",
                "tasks_transcribed": len(transcribed_tasks),
                "missing_scenarios": asr_missing_json,
                "failed_scenarios": asr_failed_json,
                "transcribed_path": str(transcribed_path),
            }
    else:
        has_transcribed_cache, transcribed_tasks = await asyncio.to_thread(
            load_offline_transcribed_cache_sync,
            transcribed_path=transcribed_path,
            semantic_tasks=semantic_tasks,
            expected_asr_model=expected_offline_asr_model,
            expected_asr_endpoint_sha256=expected_offline_asr_endpoint_sha256,
            allowed_unselected_task_keys=allowed_unselected_task_keys,
            allow_partial=(config.phase == "score"),
        )

    should_run_judge = config.phase == "judge" or (config.phase == "all" and judge_config is not None)
    expected_selected_task_keys: frozenset[str] = frozenset(semantic_task_key(task) for task in semantic_tasks)
    if should_run_judge:
        if not has_transcribed_cache and semantic_tasks:
            raise RuntimeError(f"Cannot run semantic judge without ASR transcripts in {transcribed_path}")
        assert judge_config is not None
        existing_judge_failures = await asyncio.to_thread(load_validated_failure_ledger, judge_failures_path)
        judge_result = await judge_tasks_via_http(
            transcribed_tasks,
            judge_config=judge_config,
            output_path=labels_path,
            trust_env=config.trust_env,
            allowed_unselected_task_keys=allowed_unselected_task_keys,
        )
        verdicts = list(judge_result.verdicts)
        resolved_verdict_keys = {semantic_task_key(row) for row in verdicts}
        preserved_unselected_judge_failures = {
            task_key: reason
            for task_key, reason in existing_judge_failures.items()
            if task_key not in expected_selected_task_keys and task_key not in resolved_verdict_keys
        }
        merged_judge_failures = {**preserved_unselected_judge_failures, **judge_result.failure_reasons}
        await asyncio.to_thread(persist_judge_failures_sync, judge_failures_path, merged_judge_failures)
        failed_semantic_task_keys = judge_result.failed_task_keys
        if config.phase == "judge":
            unresolved_selected_task_keys = sorted(
                expected_selected_task_keys - {semantic_task_key(row) for row in verdicts}
            )
            is_judge_complete = (
                (not missing_selected_scenarios)
                and (not failed_scenarios)
                and (not failed_semantic_task_keys)
                and (not unresolved_selected_task_keys)
            )
            judge_missing_json: list[JsonValue] = list(missing_selected_scenarios)
            judge_failed_json: dict[str, JsonValue] = dict(failed_scenarios)
            failed_keys_json: list[JsonValue] = list(failed_semantic_task_keys)
            return {
                "status": "complete" if is_judge_complete else "incomplete",
                "phase": "judge",
                "verdicts_written": len(verdicts),
                "missing_scenarios": judge_missing_json,
                "failed_scenarios": judge_failed_json,
                "failed_semantic_tasks": len(failed_semantic_task_keys),
                "failed_task_keys": failed_keys_json,
                "labels_path": str(labels_path),
            }
    else:
        verdicts, failed_semantic_task_keys = await asyncio.to_thread(
            load_offline_verdicts_and_failures_sync,
            labels_path=labels_path,
            judge_failures_path=judge_failures_path,
            transcribed_path=transcribed_path,
            has_transcribed_cache=has_transcribed_cache,
            semantic_tasks=semantic_tasks,
            transcribed_tasks=transcribed_tasks,
            judge_config=judge_config,
            allowed_unselected_task_keys=allowed_unselected_task_keys,
            expected_selected_task_keys=expected_selected_task_keys,
            allow_partial=(config.phase == "score"),
        )

    if transcribed_tasks:
        validated_transcribed = validate_transcribed_against_tasks(
            semantic_tasks,
            transcribed_tasks,
            allow_partial=(config.phase == "score"),
        )
        verdict_keys_present: set[tuple[str, int, str]] = set()
        for row in verdicts:
            row_sid = row.get("scenario_id")
            row_tidx = row.get("turn_index")
            row_kind = row.get("kind")
            if isinstance(row_sid, str) and type(row_tidx) is int and isinstance(row_kind, str):
                verdict_keys_present.add((row_sid, row_tidx, row_kind))
        for task in validated_transcribed.values():
            raw_turn_idx = task.get("turn_index")
            assert type(raw_turn_idx) is int
            task_tuple = (str(task["scenario_id"]), raw_turn_idx, str(task["kind"]))
            if not str(task.get("hyp", "")).strip() and task_tuple not in verdict_keys_present:
                verdicts.append(make_deterministic_empty_asr_verdict(task))
                verdict_keys_present.add(task_tuple)

    final_resolved_selected_keys = {
        semantic_task_key(row) for row in verdicts if semantic_task_key(row) in expected_selected_task_keys
    }
    failed_semantic_task_keys = tuple(
        key
        for key in failed_semantic_task_keys
        if key in expected_selected_task_keys and key not in final_resolved_selected_keys
    )

    verified_retained_audio_count = 0
    if config.verify_retained_audio:
        verified_retained_audio_count = await asyncio.to_thread(
            verify_retained_assistant_audio,
            reports_by_id,
            audio_directory,
        )
    all_retained_audio_verified = (
        bool(config.verify_retained_audio)
        and bool(reports_by_id)
        and verified_retained_audio_count == len(reports_by_id)
    )

    validate_scenario_reports(
        reports_by_id,
        scenarios_by_id,
        strict_reference_vad_version=config.strict_reference_vad_version,
    )
    first_report = next(iter(reports_by_id.values()))
    first_aed = first_report.get("aed_config")
    recorded_vad_version = first_aed.get("silero_vad_version") if isinstance(first_aed, dict) else None
    observed_report_protocols = {
        str(report.get("realtime_protocol"))
        for report in reports_by_id.values()
        if isinstance(report.get("realtime_protocol"), str) and report.get("realtime_protocol")
    }
    valid_protocols = {"native", "turn_based", "openai_turn", "legacy"}
    unknown_protocols = observed_report_protocols - valid_protocols
    if unknown_protocols:
        raise ValueError(f"Unsupported realtime_protocol values in {report_path}: {sorted(unknown_protocols)}")
    if len(observed_report_protocols) > 1:
        raise ValueError(f"Mixed realtime_protocol values in {report_path}: {sorted(observed_report_protocols)}")
    report_protocol = next(iter(observed_report_protocols)) if len(observed_report_protocols) == 1 else None

    raw_manifest_protocol = saved_run_manifest.get("protocol")
    manifest_protocol: str | None = None
    if raw_manifest_protocol is not None:
        if not isinstance(raw_manifest_protocol, str) or raw_manifest_protocol not in valid_protocols:
            raise ValueError(f"Unsupported protocol={raw_manifest_protocol!r} in {run_manifest_path}")
        manifest_protocol = raw_manifest_protocol

    if manifest_protocol is not None and report_protocol is not None and manifest_protocol != report_protocol:
        raise ValueError(
            f"Protocol mismatch between {run_manifest_path} ({manifest_protocol!r}) "
            f"and {report_path} ({report_protocol!r})"
        )
    if config.protocol != "auto" and manifest_protocol is not None and config.protocol != manifest_protocol:
        raise ValueError(
            f"Requested --protocol={config.protocol!r} conflicts with {run_manifest_path} ({manifest_protocol!r})"
        )
    if config.protocol != "auto" and report_protocol is not None and config.protocol != report_protocol:
        raise ValueError(f"Requested --protocol={config.protocol!r} conflicts with {report_path} ({report_protocol!r})")
    if manifest_protocol is not None:
        effective_protocol = manifest_protocol
    elif report_protocol is not None:
        effective_protocol = report_protocol
    elif config.protocol != "auto":
        effective_protocol = config.protocol
    else:
        raise ValueError(
            f"Reports in {report_path} lack explicit realtime_protocol and {run_manifest_path} is absent; "
            "pass --protocol explicitly when scoring imported legacy reports"
        )

    observed_clocks = {
        str(report.get("evaluation_clock"))
        for report in reports_by_id.values()
        if isinstance(report.get("evaluation_clock"), str) and report.get("evaluation_clock")
    }
    effective_evaluation_clock: str | None = (
        next(iter(observed_clocks))
        if len(observed_clocks) == 1
        else (
            str(saved_run_manifest["evaluation_clock"])
            if isinstance(saved_run_manifest.get("evaluation_clock"), str)
            else None
        )
    )
    effective_turn_detection = saved_run_manifest.get("turn_detection")
    is_probe_only = effective_protocol != "native"
    evaluation_mode = "unscored_turn_based_probe" if is_probe_only else "scored_full_duplex"
    scored_metrics = score_scenario_reports(
        reports_by_id,
        scenarios_by_id,
        verdicts,
        probe_only=is_probe_only,
    )
    missing_scenarios = sorted(set(scenarios_by_id) - set(reports_by_id))
    pooled_row = scored_metrics.get("overall_pooled", {})
    pending_semantic_count = 0
    for metric_name in ("conditional_answer_accuracy", "silence_preservation"):
        metric_cell = pooled_row.get(metric_name)
        if isinstance(metric_cell, dict):
            raw_pending = metric_cell.get("pending")
            if type(raw_pending) is int and raw_pending > 0:
                pending_semantic_count += raw_pending
    has_pending_semantics = pending_semantic_count > 0
    resolved_selected_verdict_count = len(final_resolved_selected_keys)

    is_run_complete = (
        (not missing_scenarios)
        and (not failed_scenarios)
        and (not has_pending_semantics)
        and (not failed_semantic_task_keys)
    )
    evaluation_input_sha256 = compute_selected_scenarios_sha256(scenarios)
    raw_saved_base_url = saved_run_manifest.get("base_url")
    if raw_saved_base_url is not None and not isinstance(raw_saved_base_url, str):
        raise ValueError(f"Invalid base_url in {run_manifest_path}; expected string or null")
    sanitized_saved_base_url = (
        sanitize_endpoint_origin(raw_saved_base_url)
        if isinstance(raw_saved_base_url, str) and raw_saved_base_url
        else None
    )
    if isinstance(sanitized_saved_base_url, str) and not urlsplit(sanitized_saved_base_url).hostname:
        raise ValueError(f"Invalid base_url in {run_manifest_path}")
    effective_base_url = sanitize_endpoint_origin(config.base_url) if config.base_url else sanitized_saved_base_url
    effective_concurrency = (
        saved_run_manifest.get("max_concurrency")
        if type(saved_run_manifest.get("max_concurrency")) is int
        else config.max_concurrency
    )
    raw_concurrency_history = saved_run_manifest.get("concurrency_history")
    effective_concurrency_history: list[JsonValue] | None = (
        [int(item) for item in raw_concurrency_history if type(item) is int and item >= 1]
        if isinstance(raw_concurrency_history, list)
        else None
    )
    effective_warmup_sessions = (
        saved_run_manifest.get("warmup_sessions")
        if type(saved_run_manifest.get("warmup_sessions")) is int
        else config.warmup_sessions
    )
    raw_saved_timeout = saved_run_manifest.get("session_timeout_seconds")
    effective_session_timeout_seconds = (
        float(raw_saved_timeout)
        if isinstance(raw_saved_timeout, int | float) and type(raw_saved_timeout) is not bool
        else config.session_timeout_seconds
    )
    observed_asr_models: set[str] = {
        str(row["asr_model"]) for row in transcribed_tasks if isinstance(row.get("asr_model"), str) and row["asr_model"]
    }
    if len(observed_asr_models) > 1:
        raise ValueError(f"Mixed asr_model values in {transcribed_path}: {sorted(observed_asr_models)}")
    if len(observed_asr_models) == 1:
        effective_asr_model: str | None = next(iter(observed_asr_models))
    elif config.asr_base_url is not None:
        effective_asr_model = resolved_asr_model
    else:
        effective_asr_model = None

    observed_pacing_lags_ms: list[float] = []
    for report in reports_by_id.values():
        raw_lag = report.get("max_pacing_lag_ms")
        if isinstance(raw_lag, int | float) and type(raw_lag) is not bool:
            observed_pacing_lags_ms.append(float(raw_lag))
    max_observed_pacing_lag_ms = round(max(observed_pacing_lags_ms), 3) if observed_pacing_lags_ms else None

    inferred_judge_metadata: JsonValue = None
    if judge_config is not None:
        inferred_judge_metadata = public_judge_metadata(judge_config)
    else:
        non_empty_verdicts = [row for row in verdicts if row.get("error_kind") != "asr_empty"]
        if non_empty_verdicts:
            sample_verdict = non_empty_verdicts[0]
            answer_decoding = next(
                (
                    row.get("judge_decoding")
                    for row in non_empty_verdicts
                    if row.get("kind") == "answer" and "judge_decoding" in row
                ),
                None,
            )
            silence_decoding = next(
                (
                    row.get("judge_decoding")
                    for row in non_empty_verdicts
                    if row.get("kind") == "silence" and "judge_decoding" in row
                ),
                None,
            )
            inferred_judge_metadata = {
                "model": sample_verdict.get("judge_model"),
                "model_revision": sample_verdict.get("judge_model_revision"),
                "enable_thinking": sample_verdict.get("judge_enable_thinking"),
                "answer_decoding": answer_decoding,
                "silence_decoding": silence_decoding,
            }

    server_declaration: dict[str, JsonValue] = {
        "base_url": effective_base_url,
        "protocol": effective_protocol,
        "evaluation_clock": effective_evaluation_clock,
        "evaluation_mode": evaluation_mode,
        "turn_detection": effective_turn_detection,
        "max_concurrency": effective_concurrency,
        "concurrency_history": effective_concurrency_history,
        "warmup_sessions": effective_warmup_sessions,
        "session_timeout_seconds": effective_session_timeout_seconds,
        "max_pacing_lag_ms": max_observed_pacing_lag_ms,
        "seed_policy": SEED_POLICY_METADATA_ONLY,
        "silero_vad_version": recorded_vad_version,
        "reference_vad_version_matched": recorded_vad_version == REFERENCE_SILERO_VAD_VERSION,
        "strict_reference_vad_version": config.strict_reference_vad_version,
        "retained_audio_verified": all_retained_audio_verified,
        "retained_audio_verified_scenarios": verified_retained_audio_count,
        "asr_base_url": sanitize_endpoint_origin(config.asr_base_url) if config.asr_base_url else None,
        "asr_model": effective_asr_model,
        "judge": inferred_judge_metadata,
        "trust_env": config.trust_env,
    }
    provenance = await asyncio.to_thread(
        collect_benchmark_provenance,
        model_id=effective_model,
        model_revision=effective_model_revision,
        dataset_id=DUPLEX_MPE_DATASET_ID,
        dataset_revision=(
            DUPLEX_MPE_DATASET_REVISION
            if (dataset_identity.metadata_matches_expected_revision and human_audio_inventory_verified)
            else None
        ),
        launch_command=effective_launch_command,
        server_config=server_declaration,
        evaluation_input_sha256=evaluation_input_sha256,
    )

    unattempted_scenarios = sorted(set(scenarios_by_id) - set(reports_by_id) - set(failed_scenarios))
    unattempted_semantic_keys = (
        expected_selected_task_keys - final_resolved_selected_keys - set(failed_semantic_task_keys)
    )
    markdown_table = format_markdown_score_table(
        scored_metrics,
        interaction_protocol=effective_protocol,
        evaluation_clock=effective_evaluation_clock,
        turn_detection=effective_turn_detection,
        selected_scenarios=len(scenarios_by_id),
        completed_scenarios=len(reports_by_id),
        missing_scenarios=len(unattempted_scenarios),
        failed_scenarios=len(failed_scenarios),
        required_semantic_tasks=len(expected_selected_task_keys),
        resolved_semantic_tasks=resolved_selected_verdict_count,
        pending_semantic_tasks=len(unattempted_semantic_keys),
        failed_semantic_tasks=len(failed_semantic_task_keys),
        is_run_complete=is_run_complete,
    )
    heading_suffix = "" if is_run_complete else " — PARTIAL / INCOMPLETE"
    missing_json_list: list[JsonValue] = list(missing_scenarios)
    failed_json_map: dict[str, JsonValue] = dict(failed_scenarios)
    results_json_map: dict[str, JsonValue] = {split_key: split_val for split_key, split_val in scored_metrics.items()}
    summary_payload: dict[str, JsonValue] = {
        "status": "complete" if is_run_complete else "incomplete",
        "evaluation_mode": evaluation_mode,
        "evaluation_clock": effective_evaluation_clock,
        "interaction": effective_protocol,
        "turn_detection": effective_turn_detection,
        "model": effective_model,
        "dataset_identity": asdict(dataset_identity),
        "selected_scenarios": len(scenarios_by_id),
        "completed_scenarios": len(reports_by_id),
        "missing_scenarios": missing_json_list,
        "failed_scenarios": failed_json_map,
        "semantic_tasks": len(semantic_tasks),
        "semantic_verdicts": len(verdicts),
        "failed_semantic_tasks": len(failed_semantic_task_keys),
        "results": results_json_map,
        "provenance": provenance,
    }
    summary_path = await asyncio.to_thread(
        persist_summary_outputs_sync,
        output_dir=output_dir,
        effective_model=effective_model,
        heading_suffix=heading_suffix,
        markdown_table=markdown_table,
        summary_payload=summary_payload,
    )
    print(markdown_table, end="")
    print(f"Saved summary to {summary_path}")
    return summary_payload


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/cache/duplex_mpe"),
        help="Path to materialized Duplex-MPE dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/duplex_mpe"),
        help="Output directory outside dataset-root for audio, reports, and scores.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "asr", "judge", "score"),
        default="all",
        help="Pipeline phase to run (default: all).",
    )
    parser.add_argument(
        "--addressing",
        choices=("all", "explicit", "implicit"),
        default="all",
        help="Addressing condition subset to evaluate.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Select deterministic 20-pair (40-scenario) smoke-test subset.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on the number of scenarios evaluated.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Served model identifier (inherited from run.json during offline phases).",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model checkpoint commit SHA recorded in provenance.",
    )
    parser.add_argument(
        "--launch-command",
        default=None,
        help="Optional server launch command recorded in provenance.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="vLLM-Omni server root or /v1/realtime(?duplex=1) URL (e.g., http://127.0.0.1:8091).",
    )
    parser.add_argument(
        "--realtime-api-key-env",
        default=None,
        help="Optional environment variable holding the /v1/realtime bearer token.",
    )
    parser.add_argument(
        "--protocol",
        choices=("auto", "native", "turn_based", "openai_turn"),
        default="auto",
        help="Session protocol (auto negotiates /v1/realtime?duplex=1 capabilities or falls back to /v1/realtime).",
    )
    parser.add_argument(
        "--input-audio-format",
        choices=("pcm16", "pcm_f32le"),
        default=None,
        help="Override input PCM wire encoding (default: pcm_f32le on duplex, pcm16 on openai_turn).",
    )
    parser.add_argument(
        "--input-sample-rate-hz",
        type=int,
        default=None,
        help="Override input sample rate in Hz.",
    )
    parser.add_argument(
        "--output-sample-rate-hz",
        type=int,
        default=None,
        help="Override output sample rate in Hz.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Optional speaker/voice name forwarded in session.update.",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional system instructions / persona forwarded in session.update.",
    )
    parser.add_argument(
        "--ref-audio",
        dest="ref_audio_path",
        type=Path,
        default=None,
        help="Optional reference voice WAV/FLAC path encoded as a data URL for native duplex models.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature forwarded in duplex session.update (unsupported on openai_turn).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent /v1/realtime scenario sessions.",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup_sessions",
        type=int,
        default=1,
        help="Unscored warmup sessions before measured recording (0 disables).",
    )
    parser.add_argument(
        "--session-timeout-s",
        dest="session_timeout_seconds",
        type=float,
        default=DEFAULT_SESSION_TIMEOUT_SECONDS,
        help="Per-scenario WebSocket session timeout in seconds.",
    )
    parser.add_argument(
        "--asr-base-url",
        default=None,
        help="OpenAI-compatible ASR server URL serving Qwen/Qwen3-ASR-1.7B.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=f"Model name sent to /v1/audio/transcriptions (default: {DEFAULT_ASR_MODEL_ID}).",
    )
    parser.add_argument(
        "--asr-api-key-env",
        default=None,
        help="Optional environment variable holding the ASR bearer token.",
    )
    parser.add_argument(
        "--asr-max-concurrency",
        type=int,
        default=8,
        help="Concurrent /v1/audio/transcriptions requests.",
    )
    parser.add_argument(
        "--judge-config",
        dest="judge_config_path",
        type=Path,
        default=None,
        help="Path to JSON judge endpoint configuration.",
    )
    parser.add_argument(
        "--strict-reference-vad-version",
        action="store_true",
        help="Require exact reference silero-vad==6.2.1 when validating scenario reports.",
    )
    parser.add_argument(
        "--verify-retained-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify retained assistant WAV SHA-256 digests before scoring (default: true).",
    )
    parser.add_argument(
        "--trust-env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Honor HTTP_PROXY/HTTPS_PROXY/NO_PROXY environment variables.",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duplex-MPE multi-party full-duplex interaction benchmark for vLLM-Omni."
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Download and verify the pinned ChengqianMa/Duplex-MPE dataset.",
    )
    prepare_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/cache/duplex_mpe"),
        help="Destination directory for Duplex-MPE dataset files.",
    )
    prepare_parser.add_argument(
        "--mini",
        action="store_true",
        help="Download only the deterministic 20-pair (40-scenario) smoke-test subset.",
    )
    prepare_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress log lines.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Verify a local Duplex-MPE dataset directory and WAV inventory.",
    )
    validate_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/cache/duplex_mpe"),
        help="Path to materialized Duplex-MPE dataset directory.",
    )
    validate_parser.add_argument(
        "--mini",
        action="store_true",
        help="Validate only the deterministic 20-pair (40-scenario) subset.",
    )
    validate_parser.add_argument(
        "--verify-audio-checksums",
        action="store_true",
        help="Verify SHA-256 digests and WAV headers of referenced audio clips.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run Duplex-MPE generation, ASR transcription, semantic judging, and scoring.",
    )
    add_run_arguments(run_parser)
    return parser


class StructuredExtraFormatter(logging.Formatter):
    _STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        base_line = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._STANDARD_RECORD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return base_line
        rendered = " ".join(f"{key}={extras[key]!r}" for key in sorted(extras))
        return f"{base_line} [{rendered}]"


def main(argv: list[str] | None = None) -> int:
    import sys

    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if not raw_args:
        effective_argv = ["--help"]
    elif raw_args[0] not in ("prepare", "validate", "run", "-h", "--help"):
        effective_argv = ["run", *raw_args]
    else:
        effective_argv = raw_args

    parser = build_argument_parser()
    parsed_args = parser.parse_args(effective_argv)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(StructuredExtraFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)
    configure_quiet_http_loggers()

    if parsed_args.subcommand == "prepare":
        download_duplex_mpe_dataset(
            destination_dir=parsed_args.dataset_root,
            mini=parsed_args.mini,
            quiet=parsed_args.quiet,
        )
        return 0

    if parsed_args.subcommand == "validate":
        identity = inspect_duplex_mpe_dataset(parsed_args.dataset_root)
        if not identity.metadata_matches_expected_revision:
            raise ValueError(
                f"Metadata SHA-256 digests in {parsed_args.dataset_root} do not match "
                f"pinned revision {DUPLEX_MPE_DATASET_REVISION}"
            )
        summary = validate_duplex_mpe_dataset(
            parsed_args.dataset_root,
            mini=parsed_args.mini,
            verify_audio_checksums=parsed_args.verify_audio_checksums,
        )
        print(json.dumps(summary, indent=2))
        return 0

    eval_config = DuplexMpeEvalConfig(
        dataset_root=parsed_args.dataset_root,
        output_dir=parsed_args.output_dir,
        phase=parsed_args.phase,
        addressing=parsed_args.addressing,
        mini=parsed_args.mini,
        max_scenarios=parsed_args.max_scenarios,
        model=parsed_args.model,
        model_revision=parsed_args.model_revision,
        launch_command=parsed_args.launch_command,
        base_url=parsed_args.base_url,
        realtime_api_key_env=parsed_args.realtime_api_key_env,
        protocol=parsed_args.protocol,
        input_audio_format=parsed_args.input_audio_format,
        input_sample_rate_hz=parsed_args.input_sample_rate_hz,
        output_sample_rate_hz=parsed_args.output_sample_rate_hz,
        voice=parsed_args.voice,
        instructions=parsed_args.instructions,
        ref_audio_path=parsed_args.ref_audio_path,
        temperature=parsed_args.temperature,
        max_concurrency=parsed_args.max_concurrency,
        warmup_sessions=parsed_args.warmup_sessions,
        session_timeout_seconds=parsed_args.session_timeout_seconds,
        asr_base_url=parsed_args.asr_base_url,
        asr_model=parsed_args.asr_model,
        asr_api_key_env=parsed_args.asr_api_key_env,
        asr_max_concurrency=parsed_args.asr_max_concurrency,
        judge_config_path=parsed_args.judge_config_path,
        strict_reference_vad_version=parsed_args.strict_reference_vad_version,
        verify_retained_audio=parsed_args.verify_retained_audio,
        trust_env=parsed_args.trust_env,
    )
    outcome = asyncio.run(run_duplex_mpe_evaluation(eval_config))
    return 0 if outcome.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

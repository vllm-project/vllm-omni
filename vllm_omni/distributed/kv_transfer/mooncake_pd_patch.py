from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import threading
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_patched = False
_bootstrap_patched = False

# How long (seconds) the decode (kv_consumer) side waits for a single KV
# pull to complete before it gives up on its own, independent of whatever
# the producer's VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT (default 480s) is set
# to. Upstream MooncakeConnector's decode-side receive path has NO local
# timeout of its own (see module docstring below `_patch_mooncake_worker`);
# without this watchdog a stuck pull can hang the whole PD pipeline for
# 480s+ (producer timeout) or indefinitely (if even that response is lost),
# silently backing up every other request behind it. Default is
# intentionally much shorter than the producer's 480s so TTS callers fail
# fast / recompute instead of waiting minutes.
_DEFAULT_KV_PULL_TIMEOUT_S = 60.0

# PD data-plane selector. Keep this separate from Mooncake's protocol name:
# operators choose a topology-oriented mode, while the connector receives the
# concrete backend protocol it understands.
_PD_TRANSPORT_TO_MOONCAKE_PROTOCOL = {
    "tcp": "tcp",
    "rdma": "rdma",
    # Mooncake names its same-host NVLink backend ``nvlink_intra``.
    "nvlink": "nvlink_intra",
}


def _configure_pd_transport(vllm_config: Any) -> str | None:
    """Validate and materialize the explicit PD transport selection.

    ``pd_transport`` is intentionally a PD-facing setting rather than a
    Mooncake implementation detail. It is read before the upstream worker
    initializes its TransferEngine, then translated to ``mooncake_protocol``.
    Configurations that omit it retain upstream behavior for compatibility;
    all shipped Qwen3-TTS PD configs set it explicitly.
    """
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is None:
        return None
    extra = getattr(kv_cfg, "kv_connector_extra_config", None)
    if not isinstance(extra, dict):
        return None

    configured = extra.get("pd_transport")
    if configured is None:
        return None
    transport = str(configured).strip().lower()
    if transport == "pcie":
        raise NotImplementedError(
            "PD transport 'pcie' was requested, but this build has no CUDA-IPC/"
            "PCIe data-plane backend. It must not silently fall back to TCP. "
            "Use pd_transport: tcp, rdma, or nvlink until a PCIe backend is added."
        )
    protocol = _PD_TRANSPORT_TO_MOONCAKE_PROTOCOL.get(transport)
    if protocol is None:
        valid = ", ".join((*_PD_TRANSPORT_TO_MOONCAKE_PROTOCOL, "pcie"))
        raise ValueError(f"Unknown pd_transport={configured!r}; expected one of: {valid}")

    existing = extra.get("mooncake_protocol")
    if existing is not None and str(existing).strip().lower() != protocol:
        raise ValueError(
            "PD transport configuration conflict: "
            f"pd_transport={transport!r} maps to mooncake_protocol={protocol!r}, "
            f"but mooncake_protocol={existing!r} was also supplied."
        )
    extra["mooncake_protocol"] = protocol
    logger.info(
        "[PD_TRANSPORT] selected=%s mooncake_protocol=%s data_plane=Mooncake; "
        "no automatic protocol fallback is permitted by this config.",
        transport,
        protocol,
    )
    return transport


def _pd_trace_enabled() -> bool:
    return os.getenv("VLLM_OMNI_PD_TRACE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _kv_pull_timeout_s() -> float:
    try:
        return float(os.environ.get("VLLM_OMNI_MOONCAKE_PULL_TIMEOUT_S", _DEFAULT_KV_PULL_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_KV_PULL_TIMEOUT_S


def _flatten_block_ids(local_block_ids: Any) -> set[int]:
    """Flatten a PullReqMeta.local_block_ids (list[list[int]]) into a flat set[int]."""
    flat: set[int] = set()
    if not local_block_ids:
        return flat
    for group in local_block_ids:
        if isinstance(group, (list, tuple, set)):
            for bid in group:
                try:
                    flat.add(int(bid))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                flat.add(int(group))
            except (TypeError, ValueError):
                continue
    return flat


def _import_first(paths: tuple[str, ...]):
    for path in paths:
        try:
            return importlib.import_module(path)
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _import_mooncake_module():
    return _import_first(
        (
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector",
        )
    )


def _import_mooncake_utils_module():
    return _import_first(
        (
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils",
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake_utils",
        )
    )


def _fill_producer_kv_params(kv_params: Any, request: Any, owner: Any) -> dict[str, Any]:
    req_id = getattr(request, "request_id", None)
    if kv_params is None or not isinstance(kv_params, dict):
        kv_params = {}
    if not req_id:
        return kv_params

    kv_params.setdefault("transfer_id", f"xfer-{req_id}")
    kv_params.setdefault("remote_request_id", req_id)
    # This hook publishes producer metadata, not an extraction acknowledgement.
    # `kv_ready` is reserved for OmniARAsyncScheduler after it attaches the
    # matching Qwen3-TTS non-KV runtime state to the KV acknowledgement.
    kv_params.pop("kv_ready", None)

    host = getattr(owner, "side_channel_host", None)
    port = getattr(owner, "side_channel_port", None)
    if host is not None:
        kv_params.setdefault("remote_host", host)
    if port is not None:
        kv_params.setdefault("remote_port", port)
    return kv_params


def _create_patched_mooncake_connector():
    mc_mod = _import_mooncake_module()
    if mc_mod is None:
        raise ImportError("Cannot import MooncakeConnector from upstream vLLM")

    original_cls = mc_mod.MooncakeConnector

    class PatchedMooncakeConnector(original_cls):
        @classmethod
        def get_required_kvcache_layout(cls, vllm_config: Any) -> str | None:
            return None

        def get_block_ids_with_load_errors(self) -> set:
            # Worker-side hook: surface KV-pull failures/timeouts detected by
            # our MooncakeConnectorWorker patch (see _patch_mooncake_worker)
            # through the standard invalid_block_ids contract so the
            # scheduler recomputes/aborts instead of hanging forever. The
            # upstream base class has no implementation for this connector,
            # so without this override the worker-side tracking added below
            # would never reach the scheduler.
            worker = getattr(self, "connector_worker", None)
            fn = getattr(worker, "get_block_ids_with_load_errors", None)
            if fn is not None:
                try:
                    return fn()
                except Exception:
                    logger.exception("[vllm-omni] get_block_ids_with_load_errors failed")
            try:
                return super().get_block_ids_with_load_errors()
            except AttributeError:
                return set()

        def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
            result = super().request_finished(request, block_ids)
            if isinstance(result, tuple) and len(result) == 2:
                delay_free, kv_params = result
            else:
                delay_free, kv_params = False, result

            owner = getattr(self, "connector_scheduler", None) or getattr(self, "connector_worker", None)
            if getattr(owner, "is_kv_producer", False):
                kv_params = _fill_producer_kv_params(kv_params, request, owner)
            return delay_free, kv_params

    PatchedMooncakeConnector.__qualname__ = original_cls.__qualname__
    return PatchedMooncakeConnector


def _check_port_available(host: str, port: int) -> tuple[bool, str | None]:
    bind_host = "0.0.0.0" if host in (None, "", "0.0.0.0") else host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((bind_host, port))
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()
    return True, None


def _patched_bootstrap_start(self) -> None:
    if self.server_thread is not None:
        return

    import uvicorn

    ok, err = _check_port_available(self.host, self.port)
    if not ok:
        raise RuntimeError(
            f"Mooncake bootstrap cannot bind {self.host}:{self.port}: {err}. "
            "Assign a unique VLLM_MOONCAKE_BOOTSTRAP_PORT for each PD stage."
        )

    kwargs = {}
    if "install_signal_handlers" in uvicorn.Config.__init__.__code__.co_varnames:
        kwargs["install_signal_handlers"] = False
    server = uvicorn.Server(uvicorn.Config(app=self.app, host=self.host, port=self.port, log_config=None, **kwargs))
    self.server = server

    def _run() -> None:
        try:
            server.run()
        except BaseException:
            return

    self.server_thread = threading.Thread(
        target=_run,
        name=f"mooncake_bootstrap_server[{self.host}:{self.port}]",
        daemon=True,
    )
    self.server_thread.start()

    deadline = time.monotonic() + 30.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"Mooncake bootstrap did not start within 30s ({self.host}:{self.port})")
        if not self.server_thread.is_alive():
            raise RuntimeError(f"Mooncake bootstrap thread died before startup ({self.host}:{self.port})")
        time.sleep(0.05)


def _patch_bootstrap_server() -> None:
    global _bootstrap_patched
    if _bootstrap_patched:
        return

    utils_mod = _import_mooncake_utils_module()
    if utils_mod is None:
        raise ImportError("Cannot import Mooncake bootstrap utilities")
    bootstrap_cls = getattr(utils_mod, "MooncakeBootstrapServer", None)
    if bootstrap_cls is None:
        raise ImportError("MooncakeBootstrapServer not found")

    bootstrap_cls.start = _patched_bootstrap_start  # type: ignore[assignment]

    original_register_worker = getattr(bootstrap_cls, "register_worker", None)
    payload_cls = getattr(utils_mod, "RegisterWorkerPayload", None)
    engine_entry_cls = getattr(utils_mod, "EngineEntry", None)
    if original_register_worker is not None and payload_cls is not None:

        def _register_routes(self):
            async def register_worker_endpoint(payload):
                addr = getattr(payload, "addr", None)
                if isinstance(addr, str) and "://" not in addr:
                    normalized = f"tcp://{addr}"
                    try:
                        payload.addr = normalized
                    except Exception:
                        if hasattr(payload, "model_copy"):
                            payload = payload.model_copy(update={"addr": normalized})
                        elif hasattr(payload, "copy"):
                            payload = payload.copy(update={"addr": normalized})
                return await original_register_worker(self, payload)

            register_worker_endpoint.__annotations__ = {"payload": payload_cls}
            self.app.post("/register")(register_worker_endpoint)
            if engine_entry_cls is not None:
                self.app.get("/query", response_model=dict[int, engine_entry_cls])(self.query)
            else:
                self.app.get("/query")(self.query)

        bootstrap_cls._register_routes = _register_routes  # type: ignore[assignment]

    _bootstrap_patched = True


def _read_remote_layout_from_shm(host: str, port: int) -> tuple[int, list[int]]:
    for shm_dir in ("/dev/shm", "/tmp"):
        path = f"{shm_dir}/vllm_omni_d_kv_layout_{host}_{port}.json"
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                payload = json.load(f)
            num_blocks = int(payload.get("num_blocks", 0) or 0)
            block_lens = [int(x) for x in (payload.get("block_lens", []) or [])]
            if num_blocks > 0 and block_lens:
                return num_blocks, block_lens
        except Exception:
            continue
    return 0, []


def _cache_remote_layout(self_worker: Any, meta: Any) -> None:
    host = getattr(meta, "remote_hostname", None)
    port = getattr(meta, "remote_port", None)
    if not host or not port:
        return

    key = f"{host}:{port}"
    nb_map = getattr(self_worker, "_vllm_omni_remote_num_blocks_map", None)
    bl_map = getattr(self_worker, "_vllm_omni_remote_block_lens_map", None)
    if nb_map is None:
        nb_map = {}
        self_worker._vllm_omni_remote_num_blocks_map = nb_map  # type: ignore[attr-defined]
    if bl_map is None:
        bl_map = {}
        self_worker._vllm_omni_remote_block_lens_map = bl_map  # type: ignore[attr-defined]
    if nb_map.get(key) and bl_map.get(key):
        return

    num_blocks = int(getattr(meta, "num_blocks", 0) or 0)
    block_lens = [int(x) for x in (getattr(meta, "block_lens", []) or [])]
    if num_blocks <= 0 or not block_lens:
        num_blocks, block_lens = _read_remote_layout_from_shm(str(host), int(port))
    if num_blocks > 0 and block_lens:
        nb_map[key] = num_blocks
        bl_map[key] = block_lens
        if _pd_trace_enabled():
            logger.info(
                "[PD_TRACE] mooncake_remote_layout_cached key=%s blocks=%d block_len0=%d layers=%d",
                key,
                num_blocks,
                block_lens[0],
                len(block_lens),
            )


def _export_consumer_layout(self_worker: Any) -> None:
    if getattr(self_worker, "is_kv_producer", False):
        return
    host = getattr(self_worker, "hostname", None)
    port = getattr(self_worker, "rpc_port", None)
    num_blocks = int(getattr(self_worker, "num_blocks", 0) or 0)
    block_lens = [int(x) for x in (getattr(self_worker, "block_len_per_layer", None) or [])]
    if not host or not port or num_blocks <= 0 or not block_lens:
        return

    shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
    path = f"{shm_dir}/vllm_omni_d_kv_layout_{host}_{port}.json"
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump({"session": f"{host}:{port}", "num_blocks": num_blocks, "block_lens": block_lens}, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _get_bootstrap_host_port(vllm_config: Any) -> tuple[str, int] | None:
    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is None:
        return None
    extra = getattr(kv_cfg, "kv_connector_extra_config", None) or {}
    if not isinstance(extra, dict):
        extra = {}
    host = getattr(kv_cfg, "kv_ip", None) or extra.get("kv_ip") or "127.0.0.1"
    port = os.environ.get("VLLM_MOONCAKE_BOOTSTRAP_PORT") or extra.get("mooncake_bootstrap_port") or 0
    try:
        port = int(port)
    except (TypeError, ValueError):
        return None
    if port <= 0:
        return None
    return str(host), port


def _get_scheduler_zmq_addr(owner: Any) -> str | None:
    host = getattr(owner, "side_channel_host", None) or getattr(owner, "hostname", None)
    port = getattr(owner, "side_channel_port", None)
    if not host or port is None:
        return None
    return f"tcp://{host}:{int(port)}"


def _ensure_producer_registered(self_worker: Any) -> None:
    if not getattr(self_worker, "is_kv_producer", False):
        return
    addr = _get_scheduler_zmq_addr(self_worker)
    if addr is None:
        return
    vllm_config = getattr(self_worker, "vllm_config", None)
    hp = _get_bootstrap_host_port(vllm_config)
    if hp is None:
        return
    host, port = hp

    # In some vLLM/Mooncake versions the producer bootstrap server is not
    # reliably started/registered before the decode side queries it. Make this
    # explicit and idempotent here.
    if not getattr(self_worker, "_vllm_omni_bootstrap_server", None):
        ok, _err = _check_port_available("0.0.0.0", port)
        if ok:
            utils_mod = _import_mooncake_utils_module()
            bootstrap_cls = getattr(utils_mod, "MooncakeBootstrapServer", None) if utils_mod is not None else None
            if bootstrap_cls is not None:
                server = bootstrap_cls("0.0.0.0", port)
                server.start()
                self_worker._vllm_omni_bootstrap_server = server  # type: ignore[attr-defined]

    try:
        import requests

        payload = {
            "engine_id": str(getattr(self_worker, "engine_id")),
            "dp_rank": int(getattr(self_worker, "dp_rank", 0)),
            "tp_rank": int(getattr(self_worker, "tp_rank", 0)),
            "pp_rank": int(getattr(self_worker, "pp_rank", 0)),
            "addr": addr,
        }
        resp = requests.post(f"http://{host}:{port}/register", json=payload, timeout=5)
        # Duplicate registration of the same tuple returns 400 in upstream; it
        # is harmless because the registry already contains the needed entry.
        if resp.status_code not in (200, 400):
            resp.raise_for_status()
    except Exception:
        pass


def _reshape_kv_caches_for_single_region(kv_caches: Any) -> Any:
    if not isinstance(kv_caches, dict):
        return kv_caches
    new_caches = {}
    for name, value in kv_caches.items():
        if isinstance(value, (list, tuple)):
            new_caches[name] = value
            continue
        if hasattr(value, "shape") and len(value.shape) == 5 and int(value.shape[0]) == 2 and value.is_contiguous():
            new_caches[name] = value.view((int(value.shape[0]) * int(value.shape[1]),) + tuple(value.shape[2:]))
        else:
            new_caches[name] = value
    return new_caches


def _patch_mooncake_scheduler(mc_module: Any) -> None:
    scheduler_cls = getattr(mc_module, "MooncakeConnectorScheduler", None)
    if scheduler_cls is None or getattr(scheduler_cls, "_vllm_omni_rf_patched", False):
        return

    original_init = getattr(scheduler_cls, "__init__", None)
    if original_init is not None:

        def _patched_scheduler_init(self_sched, *args, **kwargs):
            original_init(self_sched, *args, **kwargs)
            _ensure_producer_registered(self_sched)

        scheduler_cls.__init__ = _patched_scheduler_init  # type: ignore[assignment]

    original = scheduler_cls.request_finished

    def request_finished(self_sched, request, block_ids):
        result = original(self_sched, request, block_ids)
        if isinstance(result, tuple) and len(result) == 2:
            delay_free, kv_params = result
        else:
            delay_free, kv_params = False, result
        if getattr(self_sched, "is_kv_producer", False):
            _ensure_producer_registered(self_sched)
            kv_params = _fill_producer_kv_params(kv_params, request, self_sched)
        return delay_free, kv_params

    scheduler_cls.request_finished = request_finished  # type: ignore[assignment]
    scheduler_cls._vllm_omni_rf_patched = True  # type: ignore[attr-defined]


def _patch_mooncake_worker(mc_module: Any) -> None:
    worker_cls = getattr(mc_module, "MooncakeConnectorWorker", None)
    if worker_cls is None or getattr(worker_cls, "_vllm_omni_kv_patched", False):
        return

    original_send_kv = getattr(worker_cls, "send_kv_to_decode", None)

    # Upstream reads mooncake_protocol inside __init__ immediately before
    # TransferEngine.initialize(). Resolve our PD-facing selector first so the
    # chosen data plane is explicit and cannot be silently changed later.
    original_init = getattr(worker_cls, "__init__", None)
    if original_init is not None and not getattr(original_init, "_vllm_omni_transport_wrapped", False):

        def worker_init(self_worker, vllm_config, *args, **kwargs):
            _configure_pd_transport(vllm_config)
            return original_init(self_worker, vllm_config, *args, **kwargs)

        worker_init._vllm_omni_transport_wrapped = True  # type: ignore[attr-defined]
        worker_cls.__init__ = worker_init  # type: ignore[assignment]
    if original_send_kv is not None and not getattr(original_send_kv, "_vllm_omni_wrapped", False):

        async def send_kv_to_decode(self_worker, identity, sock, meta):
            _cache_remote_layout(self_worker, meta)
            return await original_send_kv(self_worker, identity, sock, meta)

        send_kv_to_decode._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls.send_kv_to_decode = send_kv_to_decode  # type: ignore[assignment]

    original_send_blocks = getattr(worker_cls, "_send_blocks", None)
    if original_send_blocks is not None and not getattr(original_send_blocks, "_vllm_omni_wrapped", False):

        def send_blocks(self_worker, remote_session, src_ptrs, dst_ptrs, lengths):
            try:
                local_num_blocks = int(getattr(self_worker, "num_blocks", 0) or 0)
                local_block_lens = list(getattr(self_worker, "block_len_per_layer", None) or [])
                nb_map = getattr(self_worker, "_vllm_omni_remote_num_blocks_map", None) or {}
                bl_map = getattr(self_worker, "_vllm_omni_remote_block_lens_map", None) or {}
                session_key = str(remote_session)
                remote_num_blocks = int(nb_map.get(session_key, 0) or 0)
                remote_block_lens = list(bl_map.get(session_key, []) or [])
                layout_source = "remote_cache"
                if remote_num_blocks <= 0 or not remote_block_lens:
                    remote_num_blocks = local_num_blocks
                    remote_block_lens = local_block_lens
                    layout_source = "local_fallback"
                if (
                    local_num_blocks >= 2
                    and remote_num_blocks >= 2
                    and local_num_blocks % 2 == 0
                    and remote_num_blocks % 2 == 0
                    and local_block_lens
                    and remote_block_lens
                    and len(src_ptrs) == len(dst_ptrs) == len(lengths)
                    and len(src_ptrs) > 0
                ):
                    src_offset = (local_num_blocks // 2) * int(local_block_lens[0])
                    dst_offset = (remote_num_blocks // 2) * int(remote_block_lens[0])
                    if _pd_trace_enabled():
                        logger.info(
                            "[PD_TRACE] mooncake_send_blocks session=%s layout_source=%s local_blocks=%d remote_blocks=%d src_offset=%d dst_offset=%d ptrs=%d",
                            session_key,
                            layout_source,
                            local_num_blocks,
                            remote_num_blocks,
                            src_offset,
                            dst_offset,
                            len(src_ptrs),
                        )
                    ret_k = original_send_blocks(
                        self_worker, remote_session, list(src_ptrs), list(dst_ptrs), list(lengths)
                    )
                    ret_v = original_send_blocks(
                        self_worker,
                        remote_session,
                        [int(p) + src_offset for p in src_ptrs],
                        [int(p) + dst_offset for p in dst_ptrs],
                        list(lengths),
                    )
                    if _pd_trace_enabled():
                        logger.info(
                            "[PD_TRACE] mooncake_send_blocks_done session=%s ret_k=%s ret_v=%s",
                            session_key,
                            ret_k,
                            ret_v,
                        )
                    return ret_k if ret_k != 0 else ret_v
            except Exception:
                logger.exception("[vllm-omni] Mooncake K/V split send failed; falling back to upstream transfer")
            if _pd_trace_enabled():
                logger.info(
                    "[PD_TRACE] mooncake_send_blocks_unsplit session=%s ptrs=%d",
                    remote_session,
                    len(src_ptrs),
                )
            return original_send_blocks(self_worker, remote_session, src_ptrs, dst_ptrs, lengths)

        send_blocks._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls._send_blocks = send_blocks  # type: ignore[assignment]

    original_register = getattr(worker_cls, "register_kv_caches", None)
    if original_register is not None and not getattr(original_register, "_vllm_omni_wrapped", False):

        def register_kv_caches(self_worker, kv_caches):
            topo = getattr(self_worker, "transfer_topo", None)
            original_flag = None
            kv_caches_to_register = kv_caches
            try:
                if topo is not None:
                    original_flag = getattr(topo, "_is_kv_layout_blocks_first", None)
                    topo._is_kv_layout_blocks_first = True  # type: ignore[attr-defined]
                kv_caches_to_register = _reshape_kv_caches_for_single_region(kv_caches)
            except Exception:
                kv_caches_to_register = kv_caches
            try:
                result = original_register(self_worker, kv_caches_to_register)
                _ensure_producer_registered(self_worker)
                return result
            finally:
                if topo is not None and original_flag is not None:
                    try:
                        topo._is_kv_layout_blocks_first = original_flag  # type: ignore[attr-defined]
                    except Exception:
                        pass
                _export_consumer_layout(self_worker)

        register_kv_caches._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls.register_kv_caches = register_kv_caches  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Decode (kv_consumer) side watchdog for stuck KV pulls.
    #
    # Upstream behaviour (see docstring at module top / PR discussion):
    #   - The producer side (`fetch_finished_sending_reqs`) has a timeout
    #     (VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT, default 480s): if it can't
    #     send in time it frees its blocks and, in `send_kv_to_decode`,
    #     replies to the decode side with a FINISH response whose
    #     `err_reqs` names the timed-out requests.
    #   - But `process_pulling_result` on the decode side only
    #     `logger.error(...)`s that `err_reqs` list — it never decrements
    #     `pull_tasks_count` / adds to `finished_recving_reqs`, so the
    #     scheduler never sees the request as "done" (successfully or not).
    #     The request is stuck in WAITING_FOR_REMOTE_KVS forever.
    #   - If the producer's own response is *also* lost (e.g. process died,
    #     partial network partition), the decode side never even gets an
    #     err_reqs signal and has literally no local timeout of its own
    #     (`PullReqMeta.expire_time` is defined but never populated/read
    #     upstream) -- it can hang indefinitely.
    #
    # Under load (many concurrent PD requests) this silently backs up the
    # whole pipeline: once one request is stuck, every request queued
    # behind it on the same decode replica never gets scheduled either.
    #
    # Fix: wire both failure signals into vLLM's *existing*, standard
    # `get_block_ids_with_load_errors()` / `invalid_block_ids` contract
    # (see vllm/v1/core/sched/scheduler.py `_handle_invalid_blocks`), which
    # vllm-omni's own OmniARScheduler already consumes correctly (recompute
    # by default, or hard-fail via kv_transfer_config.kv_load_failure_policy
    # = "fail"). This turns a silent multi-minute pipeline deadlock into a
    # bounded (<= VLLM_OMNI_MOONCAKE_PULL_TIMEOUT_S) recompute/abort of just
    # the affected request(s).
    # ------------------------------------------------------------------ #

    original_start_load_kv = getattr(worker_cls, "start_load_kv", None)
    if original_start_load_kv is not None and not getattr(original_start_load_kv, "_vllm_omni_wrapped", False):

        def start_load_kv(self_worker, metadata):
            reqs_to_recv = getattr(metadata, "reqs_to_recv", None)
            if reqs_to_recv:
                pull_started_at = time.perf_counter()
                deadline = pull_started_at + _kv_pull_timeout_s()
                pull_deadlines = getattr(self_worker, "_vllm_omni_pull_deadlines", None)
                if pull_deadlines is None:
                    pull_deadlines = {}
                    self_worker._vllm_omni_pull_deadlines = pull_deadlines  # type: ignore[attr-defined]
                pull_block_ids = getattr(self_worker, "_vllm_omni_pull_block_ids", None)
                if pull_block_ids is None:
                    pull_block_ids = {}
                    self_worker._vllm_omni_pull_block_ids = pull_block_ids  # type: ignore[attr-defined]
                pull_started = getattr(self_worker, "_vllm_omni_pull_started", None)
                if pull_started is None:
                    pull_started = {}
                    self_worker._vllm_omni_pull_started = pull_started  # type: ignore[attr-defined]
                for pull_metas in reqs_to_recv.values():
                    for req_id, pull_meta in pull_metas.items():
                        if _pd_trace_enabled():
                            pull_started[req_id] = pull_started_at
                        # PullReqMeta.expire_time exists upstream but is
                        # never populated/read for the decode side; setting
                        # it is harmless (a dead field otherwise) and lets
                        # future upstream versions pick it up naturally.
                        try:
                            pull_meta.expire_time = deadline
                        except Exception:
                            pass
                        pull_deadlines[req_id] = deadline
                        pull_block_ids[req_id] = _flatten_block_ids(getattr(pull_meta, "local_block_ids", None))
            return original_start_load_kv(self_worker, metadata)

        start_load_kv._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls.start_load_kv = start_load_kv  # type: ignore[assignment]

    original_process_pulling_result = getattr(worker_cls, "process_pulling_result", None)
    if original_process_pulling_result is not None and not getattr(
        original_process_pulling_result, "_vllm_omni_wrapped", False
    ):

        def process_pulling_result(self_worker, response, pull_metas):
            original_process_pulling_result(self_worker, response, pull_metas)

            pull_deadlines = getattr(self_worker, "_vllm_omni_pull_deadlines", None)
            pull_block_ids = getattr(self_worker, "_vllm_omni_pull_block_ids", None)
            pull_started = getattr(self_worker, "_vllm_omni_pull_started", None)

            ok_reqs = list(getattr(response, "ok_reqs", None) or [])
            for req_id in ok_reqs:
                pull_meta = pull_metas.get(req_id)
                # Only stop tracking once this pull is truly done (all
                # participating remote workers acked); a multi-rank pull
                # may still have other in-flight legs.
                if pull_meta is not None and int(getattr(pull_meta, "pull_tasks_count", 0) or 0) <= 0:
                    blocks = pull_block_ids.pop(req_id, None) if pull_block_ids is not None else None
                    if pull_deadlines is not None:
                        pull_deadlines.pop(req_id, None)
                    if pull_started is not None:
                        started_at = pull_started.pop(req_id, None)
                        if started_at is not None:
                            logger.info(
                                "[PD_TRACE] mooncake_pull_ok req=%s blocks=%d elapsed_ms=%.3f",
                                req_id,
                                len(blocks or ()),
                                (time.perf_counter() - started_at) * 1000.0,
                            )

            err_reqs = list(getattr(response, "err_reqs", None) or [])
            if err_reqs:
                error_block_ids = getattr(self_worker, "_vllm_omni_load_error_block_ids", None)
                if error_block_ids is None:
                    error_block_ids = set()
                    self_worker._vllm_omni_load_error_block_ids = error_block_ids  # type: ignore[attr-defined]

                failed_d_req_ids = set()
                for req_id in err_reqs:
                    pull_meta = pull_metas.get(req_id)
                    d_req_id = getattr(pull_meta, "d_req_id", req_id) if pull_meta is not None else req_id
                    blocks = pull_block_ids.pop(req_id, None) if pull_block_ids is not None else None
                    if blocks is None and pull_meta is not None:
                        blocks = _flatten_block_ids(getattr(pull_meta, "local_block_ids", None))
                    if blocks:
                        error_block_ids.update(blocks)
                    if pull_deadlines is not None:
                        pull_deadlines.pop(req_id, None)
                    if pull_started is not None:
                        started_at = pull_started.pop(req_id, None)
                        if started_at is not None:
                            logger.info(
                                "[PD_TRACE] mooncake_pull_error req=%s blocks=%d elapsed_ms=%.3f",
                                req_id,
                                len(blocks or ()),
                                (time.perf_counter() - started_at) * 1000.0,
                            )
                    failed_d_req_ids.add(d_req_id)

                if failed_d_req_ids:
                    logger.warning(
                        "[vllm-omni] Mooncake KV pull failed for %d request(s) "
                        "(producer-reported error, e.g. its own %ds send timeout). "
                        "Reporting as invalid_block_ids so the scheduler can "
                        "recompute/abort instead of hanging: %s",
                        len(failed_d_req_ids),
                        int(os.environ.get("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 480) or 480),
                        failed_d_req_ids,
                    )
                    # Contract (get_block_ids_with_load_errors docstring):
                    # even on failure the request id must still surface via
                    # get_finished() in the same or an earlier pass.
                    self_worker.finished_recving_reqs.update(failed_d_req_ids)

        process_pulling_result._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls.process_pulling_result = process_pulling_result  # type: ignore[assignment]

    original_fetch_finished_recving_reqs = getattr(worker_cls, "fetch_finished_recving_reqs", None)
    if original_fetch_finished_recving_reqs is not None and not getattr(
        original_fetch_finished_recving_reqs, "_vllm_omni_wrapped", False
    ):

        async def fetch_finished_recving_reqs(self_worker):
            result = await original_fetch_finished_recving_reqs(self_worker)

            pull_deadlines = getattr(self_worker, "_vllm_omni_pull_deadlines", None)
            if not pull_deadlines:
                return result

            now = time.perf_counter()
            expired = [req_id for req_id, deadline in pull_deadlines.items() if deadline < now]
            if not expired:
                return result

            pull_block_ids = getattr(self_worker, "_vllm_omni_pull_block_ids", None) or {}
            error_block_ids = getattr(self_worker, "_vllm_omni_load_error_block_ids", None)
            if error_block_ids is None:
                error_block_ids = set()
                self_worker._vllm_omni_load_error_block_ids = error_block_ids  # type: ignore[attr-defined]

            for req_id in expired:
                pull_deadlines.pop(req_id, None)
                blocks = pull_block_ids.pop(req_id, None)
                if blocks:
                    error_block_ids.update(blocks)

            logger.warning(
                "[vllm-omni] Mooncake KV pull watchdog: %d request(s) exceeded the local "
                "%.0fs pull timeout (VLLM_OMNI_MOONCAKE_PULL_TIMEOUT_S) with no producer "
                "response at all (not even a timeout notification). Reporting as "
                "invalid_block_ids so the scheduler can recompute/abort instead of hanging "
                "indefinitely: %s",
                len(expired),
                _kv_pull_timeout_s(),
                expired,
            )
            return set(result) | set(expired)

        fetch_finished_recving_reqs._vllm_omni_wrapped = True  # type: ignore[attr-defined]
        worker_cls.fetch_finished_recving_reqs = fetch_finished_recving_reqs  # type: ignore[assignment]

    if not hasattr(worker_cls, "get_block_ids_with_load_errors"):

        def get_block_ids_with_load_errors(self_worker) -> set:
            error_block_ids = getattr(self_worker, "_vllm_omni_load_error_block_ids", None)
            if not error_block_ids:
                return set()
            self_worker._vllm_omni_load_error_block_ids = set()  # type: ignore[attr-defined]
            return error_block_ids

        worker_cls.get_block_ids_with_load_errors = get_block_ids_with_load_errors  # type: ignore[assignment]

    worker_cls._vllm_omni_kv_patched = True  # type: ignore[attr-defined]


def apply_mooncake_connector_patch() -> None:
    global _patched
    if _patched:
        return

    _patch_bootstrap_server()
    mc_module = _import_mooncake_module()
    if mc_module is None:
        raise ImportError("Cannot import MooncakeConnector")

    original_cls = mc_module.MooncakeConnector
    patched_cls = _create_patched_mooncake_connector()
    mc_module.MooncakeConnector = patched_cls

    for path in (
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer",
    ):
        module = sys.modules.get(path)
        if module is not None and getattr(module, "MooncakeConnector", None) is original_cls:
            module.MooncakeConnector = patched_cls

    _patch_mooncake_scheduler(mc_module)
    _patch_mooncake_worker(mc_module)
    _patched = True

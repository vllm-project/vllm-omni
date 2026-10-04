# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Four-rank spatial Wan decoder used by the numerical validation tool."""

import os
import traceback
from pathlib import Path


def _rank_worker(rank, conn, devices, model, port):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    import copy
    import hashlib
    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from diffusers import AutoencoderKLWan

    from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import install_wan_spatial_shard_decode
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(4)
    current_omni_platform.set_device(torch.device("cuda:0"))
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=len(devices),
        timeout=timedelta(seconds=180),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=240))
    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32).cuda().eval()
    assert vae.config.patch_size is None
    # Preserve a plain decoder on CPU for a single-GPU reference, for numerical validation.
    reference_vae = copy.deepcopy(vae).cpu() if rank == 0 else None
    install_wan_spatial_shard_decode(vae, dist.group.WORLD, split_dim="width")
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    capability = current_omni_platform.get_device_capability(0)
    dist.barrier()
    if rank == 0:
        conn.send(
            dict(
                ready=True,
                name=current_omni_platform.get_device_name(0),
                memory=current_omni_platform.get_device_total_memory(0),
                cc=list(capability) if capability is not None else None,
                vae_parallel_size=4,
                vae_parallel_mode="spatial_shard_width",
                devices=devices,
            )
        )
    cache, first = None, True
    saved_z, saved_out = [], []

    def decode(z, state, first_frame):
        x = vae.post_quant_conv(z)
        frames = []
        for i in range(x.shape[2]):
            out = vae.decoder(x[:, :, i : i + 1], feat_cache=state, feat_idx=[0], first_chunk=first_frame)
            first_frame = False
            if rank == 0:
                frames.append(out)
        return (torch.cat(frames, 2).clamp(-1, 1) if rank == 0 else z.new_empty(0)), first_frame

    try:
        with torch.inference_mode():
            while True:
                messages = [conn.recv() if rank == 0 else None]
                dist.broadcast_object_list(messages, src=0, group=control)
                msg = messages[0]
                if msg["op"] == "stop":
                    break
                if msg["op"] == "reset":
                    vae.clear_cache()
                    cache = [None] * vae._conv_num
                    first = True
                    saved_z, saved_out = [], []
                    dist.barrier()
                    if rank == 0:
                        conn.send(dict(reset=True))
                    continue
                if msg["op"] == "validate":
                    z = torch.cat(saved_z, 2).cuda()
                    whole, _ = decode(z, [None] * vae._conv_num, True)
                    torch.accelerator.synchronize()
                    dist.barrier()
                    if rank == 0:
                        whole_cpu = whole.cpu()
                        streamed = torch.cat(saved_out, 2)
                        state_delta = (whole_cpu - streamed).abs()
                        del whole
                        reference_vae.cuda()
                        serial = reference_vae.decode(z, return_dict=False)[0].cpu()
                        reference_vae.cpu()
                        serial_delta = (serial - streamed).abs()
                        state_pass = torch.allclose(whole_cpu, streamed, atol=1e-5, rtol=1e-5)
                        serial_pass = bool(serial_delta.max() <= 0.03)
                        validation = dict(
                            exact=torch.equal(whole_cpu, streamed),
                            passed=bool(state_pass and serial_pass),
                            max_abs=float(state_delta.max()),
                            mean_abs=float(state_delta.mean()),
                            shape=list(streamed.shape),
                            streaming_tolerance=dict(atol=1e-5, rtol=1e-5),
                            serial_reference=dict(
                                max_abs=float(serial_delta.max()),
                                mean_abs=float(serial_delta.mean()),
                                rmse=float(serial_delta.square().mean().sqrt()),
                                tolerance_max_abs=0.03,
                                passed=serial_pass,
                            ),
                            note=(
                                "Stateful four-rank decode versus whole four-rank decode, "
                                "and versus original untiled single-GPU FP32 Wan VAE."
                            ),
                        )
                        if msg.get("output"):
                            folder = Path(msg["output"])
                            folder.mkdir(parents=True, exist_ok=True)
                            torch.save(
                                dict(latents=z.cpu(), streamed=streamed, whole_spatial=whole_cpu, serial=serial),
                                folder / "validation_tensors.pt",
                            )
                        conn.send(validation)
                    continue
                assert msg["op"] == "decode"
                z = torch.from_numpy(msg["latent"]).cuda() * std + mean
                torch.accelerator.synchronize()
                dist.barrier()
                pixels, first = decode(z, cache, first)
                torch.accelerator.synchronize()
                dist.barrier()
                if rank == 0:
                    cpu = ((pixels + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
                    finite = bool(torch.isfinite(pixels).all())
                    if msg.get("save_path"):
                        torch.save(cpu, msg["save_path"])
                    if msg.get("save"):
                        saved_out.append(pixels.cpu())
                if msg.get("save"):
                    saved_z.append(z.cpu())
                if rank == 0:
                    conn.send(
                        dict(
                            chunk_index=msg["index"],
                            frames=cpu.shape[2],
                            shape=list(cpu.shape),
                            finite=finite,
                            sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                        )
                    )
    except BaseException:
        # Do not enter collective teardown after a rank-local error: peers may
        # still be waiting for the next control broadcast. The supervisor
        # observes this exit and terminates the remaining ranks.
        traceback.print_exc()
        os._exit(1)
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


def worker(conn, device, model, port):
    devices = device.split(",")
    assert len(devices) == 4
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        import signal

        import torch.multiprocessing as mp

        def terminate(signum, frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, terminate)
        context = mp.spawn(_rank_worker, args=(conn, devices, model, port), nprocs=4, join=False)
        try:
            while not context.join(timeout=1):
                pass
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=10)
    except BaseException:
        try:
            conn.send(dict(error=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise

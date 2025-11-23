import multiprocessing as mp

from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.gpu_worker import WorkerProc

logger = init_logger(__name__)


def launch_engine(od_config: OmniDiffusionConfig, broadcast_handle, launch_http_server: bool = True):
    logger.info("Starting server...")

    num_gpus = od_config.num_gpus
    processes = []

    # Launch all worker processes
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        process = mp.Process(
            target=WorkerProc.worker_main,
            args=(
                i,  # rank
                od_config,
                writer,
                broadcast_handle,
            ),
            name=f"DiffusionWorker-{i}",
            daemon=True,
        )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    result_handle = None
    for writer in scheduler_pipe_writers:
        writer.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")

        if i == 0:
            result_handle = data.get("result_handle")

        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

    if launch_http_server:
        assert 0, "HTTP server launch is not supported yet."

    return processes, result_handle

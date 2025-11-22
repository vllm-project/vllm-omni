# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import multiprocessing as mp

from vllm.logger import init_logger
logger = init_logger(__name__)

from .data import EngineArgs
# from .schedule import SyncScheduler
from .gpu_worker import WorkerProc


def launch_engine(engine_args: EngineArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """

    # Start a new server with multiple worker processes
    print("log level: ",logger.level)
    logger.info("Starting server...")

    num_gpus = engine_args.num_gpus
    processes = []

    # Pipes for master to talk to slaves
    task_pipes_to_slaves_w = []
    task_pipes_to_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        task_pipes_to_slaves_r.append(r)
        task_pipes_to_slaves_w.append(w)

    # Pipes for slaves to talk to master
    result_pipes_from_slaves_w = []
    result_pipes_from_slaves_r = []
    for _ in range(num_gpus - 1):
        r, w = mp.Pipe(duplex=False)
        result_pipes_from_slaves_r.append(r)
        result_pipes_from_slaves_w.append(w)

    # Launch all worker processes
    master_port = engine_args.master_port or (engine_args.master_port + 100)
    scheduler_pipe_readers = []
    scheduler_pipe_writers = []

    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_writers.append(writer)
        process = mp.Process(
            target=WorkerProc.worker_main,
            args=(
                i,  # rank
                engine_args,
                writer,
            ),
            name=f"diffusionWorker-{i}",
            daemon=True,
        )
        scheduler_pipe_readers.append(reader)
        process.start()
        processes.append(process)

    # Wait for all workers to be ready
    scheduler_infos = []
    for writer in scheduler_pipe_writers:
        writer.close()

    # Close unused pipe ends in parent process
    for p in task_pipes_to_slaves_w:
        p.close()
    for p in task_pipes_to_slaves_r:
        p.close()
    for p in result_pipes_from_slaves_w:
        p.close()
    for p in result_pipes_from_slaves_r:
        p.close()

    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
        reader.close()

    logger.debug("All workers are ready")

    if launch_http_server:
        assert 0, "HTTP server launch is not supported yet."

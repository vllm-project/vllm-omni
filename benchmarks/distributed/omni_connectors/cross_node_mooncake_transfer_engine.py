# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Cross-Node RDMA Test Script (Automated Version)

This script enables testing RDMA transfers between two separate machines.
Supports three transfer modes:
  - copy:      Normal path - tensor copied to RDMA pool (default)
  - zerocopy:  Zero-copy path - data created directly in RDMA pool
  - gpu:       GPU transfer - RDMA pool on GPU, uses GPUDirect

Usage:
    # On Machine A (Producer) - start first:
    python cross_node_mooncake_transfer_engine.py --role producer --local-host hostname_A --remote-host hostname_B

    # On Machine B (Consumer) - start after producer:
    python cross_node_mooncake_transfer_engine.py --role consumer --local-host hostname_B --remote-host hostname_A

    # Zero-copy mode:
    python cross_node_mooncake_transfer_engine.py --role producer ... --mode zerocopy

    # GPU mode (requires GPUDirect RDMA support):
    python cross_node_mooncake_transfer_engine.py --role producer ... --mode gpu --gpu-id 0

    # Benchmark mode (skip random data generation and MD5 verification,
    # measures pure RDMA throughput):
    python cross_node_mooncake_transfer_engine.py --role producer ... --benchmark

Environment Variables:
    RDMA_DEVICE_NAME:              Specify RDMA device (e.g., mlx5_0)
    MC_IB_PCI_RELAXED_ORDERING:    Set to 1 to enable PCIe relaxed ordering
                                   for higher RDMA throughput
"""

import argparse
import hashlib
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

# Add parent path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
)

from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    ManagedBuffer,
    MooncakeTransferEngineConnector,
    TransferEngine,
)


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 checksum of a tensor."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.md5(data).hexdigest()


# Control channel message types
class CtrlMsg(msgspec.Struct):
    """Control channel message."""

    msg_type: str  # "READY", "TRANSFER", "ACK", "DONE", "ERROR"
    request_id: str = ""
    md5: str = ""
    data_size: int = 0
    error: str = ""


@dataclass
class TransferConfig:
    """Configuration for cross-node transfer test."""

    local_host: str
    remote_host: str
    local_port: int
    remote_port: int
    ctrl_port: int
    num_transfers: int
    tensor_size_mb: int
    mode: str  # "copy", "zerocopy", "gpu"
    gpu_id: int = 0
    benchmark: bool = False  # Skip MD5 verification for pure performance test


@dataclass
class TransferStats:
    """Statistics for transfer operations."""

    success_count: int = 0
    fail_count: int = 0
    total_bytes: int = 0
    elapsed_time: float = 0.0

    @property
    def throughput_mbps(self) -> float:
        if self.elapsed_time > 0:
            return (self.total_bytes / (1024 * 1024)) / self.elapsed_time
        return 0.0

    def print_summary(self, role: str):
        print(f"\n{'=' * 60}")
        print(f" {role.upper()} SUMMARY")
        print(f"  Successful: {self.success_count}/{self.success_count + self.fail_count}")
        print(f"  Failed:     {self.fail_count}/{self.success_count + self.fail_count}")
        print(f"  Total:      {self.total_bytes / (1024 * 1024):.2f} MB")
        print(f"  Time:       {self.elapsed_time:.2f} s")
        print(f"  Throughput: {self.throughput_mbps:.2f} MB/s")
        print(f"{'=' * 60}")


class CrossNodeTester(ABC):
    """Abstract base class for cross-node RDMA testing."""

    def __init__(self, config: TransferConfig):
        self.config = config
        self.connector: MooncakeTransferEngineConnector | None = None
        self.zmq_ctx: zmq.Context | None = None
        self.ctrl_socket: zmq.Socket | None = None
        self.stats = TransferStats()

    def get_connector_config(self) -> dict:
        """Get connector configuration based on mode."""
        pool_size = int(self.config.tensor_size_mb * 1.5) * 1024 * 1024
        pool_size = max(pool_size, 128 * 1024 * 1024)

        conn_config = {
            "host": self.config.local_host,
            "zmq_port": self.config.local_port,
            "protocol": "rdma",
            "memory_pool_size": pool_size,
        }

        # Set device based on mode
        if self.config.mode == "gpu":
            conn_config["memory_pool_device"] = f"cuda:{self.config.gpu_id}"
        else:
            conn_config["memory_pool_device"] = "cpu"

        # RDMA device name from environment
        device_name = os.environ.get("RDMA_DEVICE_NAME")
        if device_name:
            conn_config["device_name"] = device_name
            print(f"[CONFIG] Using RDMA device: {device_name}")

        return conn_config

    def initialize(self):
        """Initialize connector and ZMQ context."""
        print(f"[{self.role}] Initializing connector...")
        conn_config = self.get_connector_config()
        self.connector = MooncakeTransferEngineConnector(conn_config)
        self.zmq_ctx = zmq.Context()
        print(f"[{self.role}] Ready at {self.config.local_host}:{self.config.local_port}")

    def cleanup(self):
        """Cleanup resources."""
        if self.ctrl_socket:
            self.ctrl_socket.close()
        if self.zmq_ctx:
            self.zmq_ctx.term()
        if self.connector:
            self.connector.close()
        print(f"[{self.role}] Closed.")

    @property
    @abstractmethod
    def role(self) -> str:
        pass

    @abstractmethod
    def run(self):
        pass


class Producer(CrossNodeTester):
    """Producer node - sends data to consumer."""

    @property
    def role(self) -> str:
        return "PRODUCER"

    def print_header(self):
        print(f"\n{'=' * 60}")
        print(f" PRODUCER MODE ({self.config.mode.upper()})")
        print(f" Local:  {self.config.local_host}:{self.config.local_port}")
        print(f" Remote: {self.config.remote_host}:{self.config.remote_port}")
        print(f" Control Port: {self.config.ctrl_port}")
        print(f" Transfer Mode: {self.config.mode}")
        if self.config.mode == "gpu":
            print(f" GPU ID: {self.config.gpu_id}")
        print(f"{'=' * 60}\n")

    def setup_control_channel(self):
        """Setup ZMQ control channel as server (REP socket)."""
        self.ctrl_socket = self.zmq_ctx.socket(zmq.REP)
        self.ctrl_socket.bind(f"tcp://*:{self.config.ctrl_port}")
        print(f"[PRODUCER] Control channel listening on port {self.config.ctrl_port}")
        print("[PRODUCER] Waiting for consumer to connect...")

    def wait_for_consumer(self) -> bool:
        """Wait for consumer READY signal."""
        msg_data = self.ctrl_socket.recv()
        msg = msgspec.msgpack.decode(msg_data, type=CtrlMsg)
        if msg.msg_type != "READY":
            print(f"[PRODUCER] Unexpected message: {msg.msg_type}")
            return False
        print("[PRODUCER] Consumer connected!")
        self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="ACK")))
        return True

    def create_test_data(self, transfer_idx: int) -> tuple[Any, str, int]:
        """
        Create test data based on transfer mode.

        Returns:
            (data, md5, size) tuple
        """
        num_elements = (self.config.tensor_size_mb * 1024 * 1024) // 4
        data_size = num_elements * 4

        if self.config.mode == "zerocopy":
            # Zero-Copy Path: Allocate directly from connector's pool
            offset = self.connector.allocator.alloc(data_size)
            managed_buf = ManagedBuffer(self.connector.allocator, offset, data_size, self.connector.pool)

            if self.config.benchmark:
                # In benchmark mode, skip random data generation (use uninitialized memory)
                return managed_buf, "", data_size
            else:
                # Fill buffer with random data using tensor view
                tensor_view = managed_buf.as_tensor(dtype=torch.float32, shape=(num_elements,))
                random_data = torch.randn(num_elements, dtype=torch.float32)
                if tensor_view.is_cuda:
                    tensor_view.copy_(random_data.to(tensor_view.device))
                else:
                    tensor_view.copy_(random_data)
                md5 = compute_md5(tensor_view)
                return managed_buf, md5, data_size

        elif self.config.mode == "gpu":
            # GPU Path: Create tensor on GPU
            device = f"cuda:{self.config.gpu_id}"
            if self.config.benchmark:
                # In benchmark mode, use empty tensor (no random generation)
                gpu_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
                return gpu_tensor, "", data_size
            else:
                cpu_tensor = torch.randn(num_elements, dtype=torch.float32)
                md5 = compute_md5(cpu_tensor)
                gpu_tensor = cpu_tensor.to(device)
                return gpu_tensor, md5, data_size

        else:
            # Copy Path (default): Create regular CPU tensor
            if self.config.benchmark:
                # In benchmark mode, use empty tensor (no random generation)
                tensor = torch.empty(num_elements, dtype=torch.float32)
                return tensor, "", data_size
            else:
                tensor = torch.randn(num_elements, dtype=torch.float32)
                md5 = compute_md5(tensor)
                return tensor, md5, data_size

    def do_transfer(self, transfer_idx: int) -> bool:
        """Perform a single transfer."""
        req_id = f"cross_node_transfer_{transfer_idx}"

        if not self.config.benchmark:
            print(f"\n[PRODUCER] Transfer {transfer_idx + 1}/{self.config.num_transfers}")

        # Create test data
        t0 = time.time()
        data, md5, data_size = self.create_test_data(transfer_idx)
        t_create = time.time() - t0

        if not self.config.benchmark:
            print(f"  Mode: {self.config.mode}")
            print(f"  Size: {self.config.tensor_size_mb} MB")
            if md5:
                print(f"  MD5:  {md5[:16]}...")
            print(f"  Create time: {t_create * 1000:.1f} ms")

        # Put data
        t1 = time.time()
        success, size, metadata = self.connector.put("producer", "consumer", req_id, data)
        t_put = time.time() - t1

        if not success:
            print("  [FAIL] Put failed")
            return False

        if not self.config.benchmark:
            print(f"  [OK] Put successful, {size} bytes ({t_put * 1000:.1f} ms)")

        # Wait for consumer to request transfer info
        msg_data = self.ctrl_socket.recv()
        msg = msgspec.msgpack.decode(msg_data, type=CtrlMsg)

        if msg.msg_type != "READY":
            print(f"  [ERROR] Unexpected message: {msg.msg_type}")
            return False

        # Send transfer metadata to consumer
        transfer_msg = CtrlMsg(
            msg_type="TRANSFER",
            request_id=req_id,
            md5=md5,
            data_size=data_size,
        )
        self.ctrl_socket.send(msgspec.msgpack.encode(transfer_msg))

        # Wait for consumer ACK (this includes RDMA transfer time)
        t2 = time.time()
        msg_data = self.ctrl_socket.recv()
        t_rdma = time.time() - t2
        msg = msgspec.msgpack.decode(msg_data, type=CtrlMsg)

        success = msg.msg_type == "ACK"
        if success:
            if not self.config.benchmark:
                print(f"  [OK] RDMA transfer complete ({t_rdma * 1000:.1f} ms)")
            self.stats.success_count += 1
            self.stats.total_bytes += size
        else:
            print(f"  [WARN] Consumer reported error: {msg.error}")
            self.stats.fail_count += 1

        # Send ACK to allow consumer to continue
        self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="ACK")))

        # Cleanup buffer
        self.connector.cleanup(req_id)

        return success

    def run(self):
        """Run the producer."""
        self.print_header()
        self.initialize()
        self.setup_control_channel()

        try:
            if not self.wait_for_consumer():
                return

            if self.config.benchmark:
                print(
                    f"[BENCHMARK] Running {self.config.num_transfers} "
                    f"transfers of {self.config.tensor_size_mb} MB each..."
                )

            start_time = time.time()

            for i in range(self.config.num_transfers):
                self.do_transfer(i)
                if self.config.benchmark and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    current_throughput = (self.stats.total_bytes / (1024 * 1024)) / elapsed
                    print(f"  Progress: {i + 1}/{self.config.num_transfers}, Throughput: {current_throughput:.2f} MB/s")

            self.stats.elapsed_time = time.time() - start_time
            self.stats.print_summary("PRODUCER")

            # Wait for final consumer message and send DONE
            self.ctrl_socket.recv()
            self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="DONE")))

        finally:
            self.cleanup()


class Consumer(CrossNodeTester):
    """Consumer node - receives data from producer."""

    @property
    def role(self) -> str:
        return "CONSUMER"

    def print_header(self):
        print(f"\n{'=' * 60}")
        print(f" CONSUMER MODE ({self.config.mode.upper()})")
        print(f" Local:  {self.config.local_host}:{self.config.local_port}")
        print(f" Remote: {self.config.remote_host}:{self.config.remote_port}")
        print(f" Control Port: {self.config.ctrl_port}")
        print(f" Transfer Mode: {self.config.mode}")
        if self.config.mode == "gpu":
            print(f" GPU ID: {self.config.gpu_id}")
        print(f"{'=' * 60}\n")

    def setup_control_channel(self):
        """Setup ZMQ control channel as client (REQ socket)."""
        self.ctrl_socket = self.zmq_ctx.socket(zmq.REQ)
        ctrl_addr = f"tcp://{self.config.remote_host}:{self.config.ctrl_port}"
        print(f"[CONSUMER] Connecting to producer control channel at {ctrl_addr}...")
        self.ctrl_socket.connect(ctrl_addr)

    def connect_to_producer(self) -> bool:
        """Connect to producer and send READY signal."""
        self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="READY")))
        msg_data = self.ctrl_socket.recv()
        msg = msgspec.msgpack.decode(msg_data, type=CtrlMsg)
        if msg.msg_type != "ACK":
            print(f"[CONSUMER] Unexpected response: {msg.msg_type}")
            return False
        print("[CONSUMER] Connected to producer! Starting transfers...")
        return True

    def do_transfer(self, transfer_idx: int) -> bool:
        """Perform a single transfer."""
        if not self.config.benchmark:
            print(f"\n[CONSUMER] Transfer {transfer_idx + 1}/{self.config.num_transfers}")

        # Request next transfer info
        self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="READY")))
        msg_data = self.ctrl_socket.recv()
        msg = msgspec.msgpack.decode(msg_data, type=CtrlMsg)

        if msg.msg_type == "DONE":
            print("[CONSUMER] Producer signaled completion")
            return False

        if msg.msg_type != "TRANSFER":
            print(f"[CONSUMER] Unexpected message: {msg.msg_type}")
            return False

        req_id = msg.request_id
        expected_md5 = msg.md5
        data_size = msg.data_size
        num_elements = data_size // 4

        if not self.config.benchmark:
            print(f"  Request ID: {req_id}")
            if expected_md5:
                print(f"  Expected MD5: {expected_md5[:16]}...")
            print(f"  Data Size: {data_size / (1024 * 1024):.2f} MB")

        # Build metadata for get
        metadata = {
            "request_id": req_id,
            "source_host": self.config.remote_host,
            "source_port": self.config.remote_port,
            "data_size": data_size,
            "dtype": "float32",
            "shape": [num_elements],
            "is_fast_path": True,
        }

        if not self.config.benchmark:
            print(f"  [INFO] Requesting from {self.config.remote_host}:{self.config.remote_port}")

        # Get data with timing
        t0 = time.time()
        result = self.connector.get("producer", "consumer", req_id, metadata)
        t_get = time.time() - t0

        response_msg = CtrlMsg(msg_type="ERROR", error="Get failed")

        if result is not None:
            recv_buffer, recv_size = result
            if not self.config.benchmark:
                print(f"  [OK] Get successful, {recv_size} bytes ({t_get * 1000:.1f} ms)")

            if isinstance(recv_buffer, ManagedBuffer):
                # In benchmark mode, skip MD5 verification
                if self.config.benchmark or not expected_md5:
                    response_msg = CtrlMsg(msg_type="ACK")
                    self.stats.success_count += 1
                    self.stats.total_bytes += recv_size
                else:
                    # Verify data
                    t1 = time.time()
                    reconstructed = recv_buffer.as_tensor(dtype=torch.float32, shape=(num_elements,))
                    recv_md5 = compute_md5(reconstructed)
                    t_md5 = time.time() - t1
                    print(f"  MD5: {recv_md5[:16]}... ({t_md5 * 1000:.1f} ms)")

                    if recv_md5 == expected_md5:
                        print("  [PASS] MD5 checksum verified!")
                        response_msg = CtrlMsg(msg_type="ACK")
                        self.stats.success_count += 1
                        self.stats.total_bytes += recv_size
                    else:
                        print("  [FAIL] MD5 mismatch!")
                        response_msg = CtrlMsg(msg_type="ERROR", error="MD5 mismatch")
                        self.stats.fail_count += 1

                recv_buffer.release()
            else:
                response_msg = CtrlMsg(msg_type="ACK")
                self.stats.success_count += 1
                self.stats.total_bytes += recv_size
        else:
            print("  [FAIL] Get failed")
            self.stats.fail_count += 1

        # Send response to producer
        self.ctrl_socket.send(msgspec.msgpack.encode(response_msg))
        # Wait for ACK
        self.ctrl_socket.recv()

        return response_msg.msg_type == "ACK"

    def run(self):
        """Run the consumer."""
        self.print_header()
        self.initialize()
        self.setup_control_channel()

        try:
            if not self.connect_to_producer():
                return

            if self.config.benchmark:
                print(
                    f"[BENCHMARK] Running {self.config.num_transfers} "
                    f"transfers of {self.config.tensor_size_mb} MB each..."
                )

            start_time = time.time()

            for i in range(self.config.num_transfers):
                if not self.do_transfer(i):
                    break
                if self.config.benchmark and (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    current_throughput = (self.stats.total_bytes / (1024 * 1024)) / elapsed
                    print(f"  Progress: {i + 1}/{self.config.num_transfers}, Throughput: {current_throughput:.2f} MB/s")

            self.stats.elapsed_time = time.time() - start_time
            self.stats.print_summary("CONSUMER")

            # Send final READY and wait for DONE
            self.ctrl_socket.send(msgspec.msgpack.encode(CtrlMsg(msg_type="READY")))
            self.ctrl_socket.recv()

        finally:
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Cross-Node RDMA Test (Automated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transfer Modes:
  copy      - Normal path: tensor copied to RDMA pool (default)
  zerocopy  - Zero-copy path: data created directly in RDMA pool
  gpu       - GPU transfer: RDMA pool on GPU, uses GPUDirect

Examples:
  # Copy mode (default):
  python cross_node_mooncake_transfer_engine.py --role producer \
  --local-host hostA --remote-host hostB

  # Zero-copy mode:
  python cross_node_mooncake_transfer_engine.py --role producer \
  --local-host hostA --remote-host hostB --mode zerocopy

  # GPU mode:
  python cross_node_mooncake_transfer_engine.py --role producer \
  --local-host hostA --remote-host hostB --mode gpu --gpu-id 0

  # Benchmark mode (skip MD5, measure pure RDMA performance):
  python cross_node_mooncake_transfer_engine.py --role producer \
  --local-host hostA --remote-host hostB --benchmark

  # With specific RDMA device:
  RDMA_DEVICE_NAME=mlx5_0 python cross_node_mooncake_transfer_engine.py --role producer ...
        """,
    )

    parser.add_argument(
        "--role", required=True, choices=["producer", "consumer"], help="Role: producer (sends) or consumer (receives)"
    )
    parser.add_argument("--local-host", required=True, help="Local hostname or IP address")
    parser.add_argument("--remote-host", required=True, help="Remote hostname or IP address")
    parser.add_argument("--local-port", type=int, default=15500, help="Local ZMQ port for RDMA data (default: 15500)")
    parser.add_argument("--remote-port", type=int, default=15500, help="Remote ZMQ port for RDMA data (default: 15500)")
    parser.add_argument("--ctrl-port", type=int, default=15501, help="Control channel port (default: 15501)")
    parser.add_argument("--num-transfers", type=int, default=20, help="Number of transfers to perform (default: 3)")
    parser.add_argument("--tensor-size-mb", type=int, default=100, help="Tensor size in MB (default: 100)")
    parser.add_argument(
        "--mode",
        choices=["copy", "zerocopy", "gpu"],
        default="copy",
        help="Transfer mode: copy, zerocopy, or gpu (default: copy)",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for GPU mode (default: 0)")
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark mode: skip MD5 verification for pure performance test"
    )

    args = parser.parse_args()

    # Check Mooncake
    if TransferEngine is None:
        print("[ERROR] Mooncake TransferEngine is not available.")
        print("Install with: pip install mooncake")
        sys.exit(1)

    # Check CUDA for GPU mode
    if args.mode == "gpu":
        if not torch.cuda.is_available():
            print("[ERROR] CUDA is not available but GPU mode was requested.")
            sys.exit(1)
        if args.gpu_id >= torch.cuda.device_count():
            print(f"[ERROR] GPU {args.gpu_id} not available. Found {torch.cuda.device_count()} GPUs.")
            sys.exit(1)
        print(f"[INFO] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    config = TransferConfig(
        local_host=args.local_host,
        remote_host=args.remote_host,
        local_port=args.local_port,
        remote_port=args.remote_port,
        ctrl_port=args.ctrl_port,
        num_transfers=args.num_transfers,
        tensor_size_mb=args.tensor_size_mb,
        mode=args.mode,
        gpu_id=args.gpu_id,
        benchmark=args.benchmark,
    )

    if args.role == "producer":
        tester = Producer(config)
    else:
        tester = Consumer(config)

    tester.run()


if __name__ == "__main__":
    main()

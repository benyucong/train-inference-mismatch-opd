# Copyright 2026 The authors (see README.md)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SPECOPD: a CUDA weight-sync checkpoint engine that needs no cupy.

verl's v1 ``separate_async`` trainer (the fully asynchronous mode: standalone rollout replicas on their own GPUs) refuses
the ``naive`` backend and syncs weights through a checkpoint engine. Upstream's CUDA engine (``nccl``,
nccl_checkpoint_engine.py) imports cupy at module load for ray.util.collective, so in an environment without cupy that
engine does not register (the nixl and mooncake engines likewise need their own libraries). This
module is upstream's ``hccl_checkpoint_engine.py`` (the Ascend engine, which already talks through verl's
``stateless_init_process_group`` -> a vLLM ``PyNcclCommunicator`` over a ``StatelessProcessGroup``) adapted to CUDA, with
one protocol change: each bucket's broadcast covers only its used bytes (the length travels with the bucket metadata,
as in upstream's nccl engine), not the whole bucket.

Opt-in only. It is NOT imported by ``verl.checkpoint_engine`` (so no default registry changes); a run selects it with
    actor_rollout_ref.rollout.checkpoint_engine.backend=specopd_pynccl
    actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.checkpoint_engine.specopd_pynccl_checkpoint_engine
and every process that builds an engine (the trainer's actor workers, each rollout replica's CheckpointEngineWorker and
the driver's CheckpointEngineManager) imports it through ``import_external_libs`` first (upstream's plugin hook).

Topology (as the hccl engine): one sender, the actor's rank 0 (group rank 0); every rollout worker is a receiver
(group ranks 1..n); the other actor ranks only walk the weight generator (each weight is gathered collectively over
the actor group). Bucket metadata goes over ZeroMQ from the sender (an XPUB socket: before its first bucket the
sender waits for every receiver's subscription, so no metadata is lost to the slow-joiner window), the bytes over NCCL. The
receiver yields views of its bucket; the server adapter copies them (on the same CUDA stream) before the next
bucket's broadcast is waited on, exactly as upstream's two-buffer scheme.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import ray
import torch
import zmq

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BACKEND = "specopd_pynccl"
# every chunk starts at a multiple of ALIGN bytes in its bucket: the receiver views a small weight's bytes as its dtype in
# place (merge_weight_chunks: chunk.view(dtype)), which needs a dtype-aligned storage offset. The sender alone chooses the
# offsets (they travel in the bucket metadata), so the padding is invisible to the receiver.
ALIGN = 64


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


@dataclass
class MasterMetadata:
    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


def _default_init_process_group(dist_ip, dist_port, rank, world_size, device):
    """verl's helper: a TCPStore-backed vLLM StatelessProcessGroup + PyNcclCommunicator (no torch.distributed group)."""
    from verl.utils.distributed import stateless_init_process_group

    return stateless_init_process_group(dist_ip, dist_port, rank, world_size, device)


class BroadcastOperation:
    """One bucket: metadata over ZeroMQ (sender -> receivers), then the bucket's used bytes over NCCL.

    The NCCL call only ENQUEUES the broadcast on the current stream; callers synchronize before they reuse or read the
    buffer (as upstream's engines do).
    """

    def __init__(self, rank: int, comm, bucket: torch.Tensor, metadata: dict | None, socket, topic: str) -> None:
        self.rank = rank
        self.comm = comm
        self.bucket = bucket
        self.metadata = metadata
        self.socket = socket
        self.topic = topic
        self._run()

    def _run(self):
        if self.rank == 0:
            self.socket.send_string(self.topic, flags=zmq.SNDMORE)
            self.socket.send_pyobj(self.metadata)
        else:
            self.socket.recv_string()
            self.metadata = self.socket.recv_pyobj()
        length = int(self.metadata["length"])
        if length > 0:
            self.comm.broadcast(self.bucket[:length], src=0)

    async def wait_for_complete(self) -> dict:
        return self.metadata


@CheckpointEngineRegistry.register(BACKEND)
class SpecopdPyNcclCheckpointEngine(CheckpointEngine):
    """CUDA checkpoint engine over vLLM's PyNcclCommunicator.

    Args:
        bucket_size (int): bytes per bucket; two buckets are allocated on the device (send + receive).
        rebuild_group (bool): rebuild the communicator at every sync (default False: built once, reused).
        is_master (bool): the sender (the actor's rank 0).
        rollout_dtype (torch.dtype): accepted for interface parity with the other engines; weights travel as bytes.
    """

    wire_format = "named_tensors"

    def __init__(
        self,
        bucket_size: int,
        group_name: str = "default",
        rebuild_group: bool = False,
        is_master: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
        **_ignored,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.comm = None
        self.rank = None
        self.world_size = None
        self.send_buf = self.recv_buf = None
        # test seams: the device and the communicator factory (CPU tests use a fake communicator)
        self._device = None
        self._init_pg = _default_init_process_group
        self.is_master = is_master
        self.topic = "bucket_metadata"
        if self.is_master:
            self._start_zmq_server()
            self.dist_port, _ = get_free_port(self.ip)

    # ------------------------------------------------------------------ device helpers
    def _dev(self):
        if self._device is not None:
            return self._device
        return torch.device("cuda", torch.cuda.current_device())

    def _sync(self):
        if self._dev().type == "cuda":
            torch.cuda.synchronize()

    def _alloc(self):
        dev = self._dev()
        if self.is_master and dev.type == "cuda":
            # the trainer may run with expandable segments; upstream's nccl engine allocates the sender's buckets
            # outside them ("avoid memory register error"), as verl's naive sync path does for its transfer
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(False)
            try:
                return (torch.zeros(self.bucket_size, dtype=torch.uint8, device=dev),
                        torch.zeros(self.bucket_size, dtype=torch.uint8, device=dev))
            finally:
                set_expandable_segments(True)
        return (torch.zeros(self.bucket_size, dtype=torch.uint8, device=dev),
                torch.zeros(self.bucket_size, dtype=torch.uint8, device=dev))

    # ------------------------------------------------------------------ CheckpointEngine interface
    def prepare(self) -> MasterMetadata | None:
        self.send_buf, self.recv_buf = self._alloc()
        return (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.zmq_port, dist_ip=self.ip, dist_port=self.dist_port)
            if self.is_master
            else None
        )

    def finalize(self):
        if self.rebuild_group:
            self.comm = None
            self.rank = None
            self.world_size = None
        self.send_buf = None
        self.recv_buf = None
        if self._dev().type == "cuda":
            torch.cuda.empty_cache()

    @classmethod
    def build_topology(cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list):
        master = metadata[0]
        assert master is not None, "actor rank 0 must be the checkpoint engine master (is_master=True)"
        world_size = rollout_world_size + 1
        actor_wg_kwargs = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [world_size] * actor_wg_world_size,
            "master_metadata": [master] * actor_wg_world_size,
        }
        rollout_kwargs = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [world_size] * rollout_world_size,
            "master_metadata": [master] * rollout_world_size,
        }
        return actor_wg_kwargs, rollout_kwargs

    @staticmethod
    def _node_ip() -> str:
        return ray.util.get_node_ip_address().strip("[]")

    def _start_zmq_server(self):
        self.ip = self._node_ip()
        self.zmq_port, _ = get_free_port(self.ip)
        context = zmq.Context()
        # XPUB (a PUB that also delivers the subscriptions): the sender waits until every receiver's subscription has
        # arrived before its first bucket, so no metadata message is lost to ZeroMQ's slow-joiner window
        self.socket = context.socket(zmq.XPUB)
        self.socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        self._n_subs = 0
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.zmq_port}"
        self.socket.bind(address)

    def _await_subscribers(self, n: int, timeout_s: float = 600.0):
        """Block until n receivers have subscribed (counted once per receiver socket; sockets persist across syncs)."""
        deadline = time.time() + timeout_s
        while self._n_subs < n:
            left = deadline - time.time()
            if left <= 0:
                raise RuntimeError(f"[specopd_pynccl] only {self._n_subs} of {n} receivers subscribed within {timeout_s}s")
            if self.socket.poll(int(min(left, 1.0) * 1000), zmq.POLLIN):
                msg = self.socket.recv()
                if msg[:1] == b"\x01":
                    self._n_subs += 1
                elif msg[:1] == b"\x00":
                    self._n_subs -= 1

    def _connect_zmq_client(self, metadata: MasterMetadata):
        assert not self.is_master, "Master process should not connect to other processes."
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f"tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}"
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{metadata.zmq_ip}:{metadata.zmq_port}"
        self.socket.connect(address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        # actor ranks other than 0 take no part in the transfer
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return

        if self.rebuild_group or self.comm is None:
            self.comm = self._init_pg(master_metadata.dist_ip, master_metadata.dist_port, rank, world_size, self._dev())
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, f"world_size {world_size} != self.world_size {self.world_size}"

        if self.rank > 0 and not hasattr(self, "socket"):
            self._connect_zmq_client(master_metadata)

        # barrier: every member has its communicator (and its SUB socket connected) before the first bucket
        signal = torch.ones(1, dtype=torch.int8, device=self._dev())
        self.comm.all_reduce(signal)
        self._sync()
        if self.rank == 0:
            self._await_subscribers(world_size - 1)
        logger.info(f"[specopd_pynccl] init_process_group rank {self.rank} / world_size {self.world_size}")

    @torch.no_grad()
    async def send_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
    ):
        assert self.rank is not None and self.rank <= 0, "Only the actor's rank 0 sends weights."
        if self.rank < 0:
            # producing each weight is collective over the actor group: walk the generator, send nothing
            async for _name, _weight in _aiter(weights):
                pass
            return

        send_buf, recv_buf = self.send_buf, self.recv_buf
        broadcast_op = None
        start_time = time.time()
        bucket_meta: dict[str, TensorMeta] = {}
        offset = 0
        n_bytes = 0
        async for tensor_meta, chunk in split_weight_chunks(weights, self.bucket_size):
            start = _align(offset)
            if start + tensor_meta.chunk_size > self.bucket_size:
                self._sync()
                if broadcast_op is not None:
                    await broadcast_op.wait_for_complete()
                broadcast_op = BroadcastOperation(
                    rank=self.rank, comm=self.comm, bucket=send_buf,
                    metadata={"bucket_meta": bucket_meta, "is_last": False, "length": offset},
                    socket=self.socket, topic=self.topic,
                )
                n_bytes += offset
                send_buf, recv_buf = recv_buf, send_buf
                bucket_meta = {}
                offset = start = 0

            assert start + tensor_meta.chunk_size <= self.bucket_size
            assert tensor_meta.name not in bucket_meta, f"duplicate weight name {tensor_meta.name} in one bucket"
            tensor_meta.offset = start
            bucket_meta[tensor_meta.name] = tensor_meta
            send_buf[start : start + tensor_meta.chunk_size] = chunk.to(send_buf.device, non_blocking=False)
            offset = start + tensor_meta.chunk_size

        self._sync()
        if broadcast_op is not None:
            await broadcast_op.wait_for_complete()
        broadcast_op = BroadcastOperation(
            rank=self.rank, comm=self.comm, bucket=send_buf,
            metadata={"bucket_meta": bucket_meta, "is_last": True, "length": offset},
            socket=self.socket, topic=self.topic,
        )
        n_bytes += offset
        await broadcast_op.wait_for_complete()
        # the broadcast is only enqueued: wait for it before finalize() frees the buffer
        self._sync()
        dt = time.time() - start_time
        logger.info(f"[specopd_pynccl] rank 0 sent {n_bytes / 2**30:.2f} GiB in {dt:.2f}s")
        return {}

    @torch.no_grad()
    async def receive_weights(
        self,
        global_steps: int | None = None,
    ) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        async for name, weight in merge_weight_chunks(self._receive_weight_chunks(), self.bucket_size):
            yield name, weight

    async def _receive_weight_chunks(self) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        assert self.rank is not None and self.rank > 0, "Rank 0 does not receive weights."
        send_buf, recv_buf = self.send_buf, self.recv_buf
        total_bytes, total_params = 0, 0
        start_time = time.time()

        broadcast_op = BroadcastOperation(
            rank=self.rank, comm=self.comm, bucket=recv_buf, metadata=None, socket=self.socket, topic=self.topic
        )
        metadata = await broadcast_op.wait_for_complete()
        total_bytes += int(metadata["length"])
        total_params += len(metadata["bucket_meta"])
        # the broadcast must have landed before the views below are copied out
        self._sync()
        send_buf, recv_buf = recv_buf, send_buf
        while not metadata["is_last"]:
            broadcast_op = BroadcastOperation(
                rank=self.rank, comm=self.comm, bucket=recv_buf, metadata=None, socket=self.socket, topic=self.topic
            )
            for _name, tensor_meta in metadata["bucket_meta"].items():
                yield tensor_meta, send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]
            metadata = await broadcast_op.wait_for_complete()
            total_bytes += int(metadata["length"])
            total_params += len(metadata["bucket_meta"])
            self._sync()
            send_buf, recv_buf = recv_buf, send_buf

        for _name, tensor_meta in metadata["bucket_meta"].items():
            yield tensor_meta, send_buf[tensor_meta.offset : tensor_meta.offset + tensor_meta.chunk_size]

        dt = max(time.time() - start_time, 1e-9)
        logger.info(
            f"[specopd_pynccl] rank {self.rank} received {total_params} chunks, {total_bytes / 2**30:.2f} GiB in "
            f"{dt:.2f}s ({total_bytes / dt / 2**30:.2f} GiB/s)"
        )


async def _aiter(weights):
    from verl.workers.rollout.utils import ensure_async_iterator

    async for item in ensure_async_iterator(weights):
        yield item

# Copyright 2025 Individual Contributor: manncodes
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
"""Elastic rollout: grow, shrink and heal a pool of standalone rollout replicas as GPU nodes join and leave.

Capacity is governed by three layers:

1. Infrastructure (KubeRay or cluster autoscaler, ``kubectl``, spot reclaim) creates and deletes nodes.
2. A :class:`ScalingPolicy` moves the rollout GPU target once per training step. With ``autoscale`` the target
   is published to the Ray autoscaler through ``request_resources``.
3. :class:`ElasticRolloutPool` fills free GPUs on eligible nodes with replicas shaped for each node, drains
   whole nodes above the target and evicts replicas whose processes died.

The replica list the checkpoint manager shares only changes in :meth:`ElasticRolloutPool.commit`, which the
trainer calls right before a weight sync, so a replica never serves stale weights.
"""

import asyncio
import logging
import os
import sys
import threading
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.plugin.platform import get_platform
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.fate import FateMonitor
from verl.trainer.config import ElasticConfig
from verl.workers.rollout.replica import RolloutReplica

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# A replica is only launched on a node with free GPUs, so its placement group is placed at once unless
# another consumer won the race for them.
_PLACEMENT_TIMEOUT_S = 60.0


async def _gather(coros) -> None:
    await asyncio.gather(*coros, return_exceptions=True)


@dataclass(frozen=True)
class Node:
    """An accelerator node as the pool sees it."""

    node_id: str
    accelerator: Optional[str]
    total_gpus: int
    free_gpus: int

    @property
    def affinity(self) -> dict[str, float]:
        """Bundle resources that keep a placement group on nodes of this node's accelerator type."""
        return {f"accelerator_type:{self.accelerator}": 1e-3} if self.accelerator else {}


def list_nodes(include: Optional[str] = None, exclude: Optional[str] = None) -> list[Node]:
    """Alive accelerator nodes, restricted to those advertising ``include`` and lacking ``exclude``."""
    device = get_platform().ray_resource_name()
    available = ray._private.state.available_resources_per_node()
    nodes = []
    for info in ray.nodes():
        resources = info["Resources"]
        if not info["Alive"] or resources.get(device, 0) < 1:
            continue
        if (include and include not in resources) or (exclude and exclude in resources):
            continue
        accelerator = next((key.split(":", 1)[1] for key in resources if key.startswith("accelerator_type:")), None)
        free = available.get(info["NodeID"], {}).get(device, 0)
        nodes.append(Node(info["NodeID"], accelerator, int(resources[device]), int(free)))
    return nodes


@ray.remote
def _device_memory() -> int:
    from verl.utils.device import get_torch_device

    return get_torch_device().get_device_properties(0).total_memory


def probe_device_memory(node: Node) -> Optional[int]:
    """Memory of one accelerator on ``node``, read by a short task pinned there; None if it cannot run."""
    ref = _device_memory.options(
        **get_platform().ray_resource_options(1),
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node.node_id, soft=False),
    ).remote()
    try:
        return ray.get(ref, timeout=_PLACEMENT_TIMEOUT_S)
    except ray.exceptions.GetTimeoutError:
        ray.cancel(ref, force=True)
        return None


def model_footprint(path: str, trust_remote_code: bool = False) -> tuple[int, int]:
    """Weight bytes and attention heads of a Hugging Face checkpoint."""
    from transformers import AutoConfig

    from verl.utils.fs import copy_to_local

    local = Path(copy_to_local(path))
    files = list(local.glob("*.safetensors")) or list(local.glob("*.bin"))
    text_config = AutoConfig.from_pretrained(local, trust_remote_code=trust_remote_code).get_text_config()
    return sum(f.stat().st_size for f in files), getattr(text_config, "num_attention_heads", 0)


class ReplicaShaper:
    """Choose the tensor parallel size of a rollout replica for each kind of node.

    Rollout replicas only stream tokens, so any node that holds the model can serve: the smallest power of two
    that divides the node's accelerator count and the model's attention heads, and keeps one weight shard within
    ``max_weight_fraction`` of an accelerator's memory (the rest is left for KV cache), is picked. An entry of
    ``tp_by_accelerator``, or its ``"*"`` wildcard, overrides the choice.

    Args:
        weight_bytes: Total size of the model weights.
        num_heads: Attention heads of the model; 0 when unknown.
        max_weight_fraction: Largest fraction of one accelerator's memory a weight shard may use.
        tp_by_accelerator: Tensor parallel size per Ray accelerator type.
        device_memory: Memory of one accelerator of a node, None when unknown.
    """

    def __init__(
        self,
        weight_bytes: int,
        num_heads: int = 0,
        max_weight_fraction: float = 0.5,
        tp_by_accelerator: Optional[dict[str, int]] = None,
        device_memory: Callable[[Node], Optional[int]] = probe_device_memory,
    ):
        self.weight_bytes = weight_bytes
        self.num_heads = num_heads
        self.max_weight_fraction = max_weight_fraction
        self.tp_by_accelerator = dict(tp_by_accelerator or {})
        self.device_memory = device_memory
        self._memory: dict[str, int] = {}

    def __call__(self, node: Node) -> Optional[int]:
        tp = self.tp_by_accelerator.get(node.accelerator, self.tp_by_accelerator.get("*"))
        if tp is not None:
            return tp if tp <= node.total_gpus else None

        key = node.accelerator or node.node_id
        if key not in self._memory:
            memory = self.device_memory(node)
            if memory is None:
                return None
            self._memory[key] = memory
        budget = self.max_weight_fraction * self._memory[key]

        tp = 1
        while tp <= node.total_gpus:
            if node.total_gpus % tp == 0 and self.num_heads % tp == 0 and self.weight_bytes <= tp * budget:
                return tp
            tp *= 2
        return None


@dataclass
class StepStats:
    """What a scaling policy sees of one training step."""

    step_s: float
    wait_s: float
    backlog: int
    batch_size: int


_POLICIES: dict[str, type["ScalingPolicy"]] = {}


def register_policy(name: str):
    def decorator(cls):
        _POLICIES[name] = cls
        return cls

    return decorator


def build_policy(config: ElasticConfig) -> "ScalingPolicy":
    if config.policy not in _POLICIES:
        raise ValueError(f"Unknown elastic scaling policy {config.policy!r}, available: {sorted(_POLICIES)}")
    return _POLICIES[config.policy](config)


class ScalingPolicy:
    """Moves the rollout GPU target once per training step."""

    def __init__(self, config: ElasticConfig):
        self.config = config

    def initial(self, current_gpus: int) -> int:
        return self.config.max_rollout_gpus or sys.maxsize

    def target(self, current: int, stats: StepStats) -> int:
        return current


@register_policy("static")
class StaticPolicy(ScalingPolicy):
    """Hold the target at ``max_rollout_gpus``: use whatever capacity arrives, replace whatever is lost."""


@register_policy("backlog")
class BacklogPolicy(ScalingPolicy):
    """Follow the balance between producing and consuming samples.

    Scale up while the trainer starves, waiting for samples longer than ``up_wait_ratio`` of a step. Scale down
    while rollout runs a full batch ahead of a trainer that never waited, since those samples only grow stale.
    """

    def __init__(self, config: ElasticConfig):
        super().__init__(config)
        self._streak = 0

    def initial(self, current_gpus: int) -> int:
        return max(current_gpus, self.config.min_rollout_gpus)

    def target(self, current: int, stats: StepStats) -> int:
        if stats.wait_s > self.config.up_wait_ratio * stats.step_s:
            self._streak = max(self._streak, 0) + 1
        elif stats.wait_s == 0 and stats.backlog >= stats.batch_size:
            self._streak = min(self._streak, 0) - 1
        else:
            self._streak = 0
        if abs(self._streak) < self.config.patience:
            return current
        step = self.config.scale_step_gpus if self._streak > 0 else -self.config.scale_step_gpus
        self._streak = 0
        return current + step


@dataclass(eq=False)
class _Member:
    rank: int
    replica: RolloutReplica
    pool: RayResourcePool
    node_id: str
    accelerator: Optional[str]
    placed: bool = True

    @property
    def gpus(self) -> int:
        return self.replica.world_size

    @property
    def actors(self) -> list:
        return [*self.replica.servers, *self.replica.workers]


class ElasticRolloutPool:
    """Keep the standalone replicas of an ``LLMServerManager`` in step with the GPU nodes of the cluster.

    A background thread probes every replica, fills free GPUs on eligible nodes with replicas shaped for each node
    up to the target, and drains whole nodes above it. Replicas whose processes died are taken out of the load
    balancer at once. New replicas wait in a ready set until :meth:`commit` admits them.

    Args:
        manager: Manager whose ``rollout_replicas`` list, shared with the checkpoint manager, the pool owns.
        config: Elastic training configuration.
        shaper: Tensor parallel size of a replica for a node, None when the node cannot host the model.
        policy: Scaling policy that moves the target.
    """

    def __init__(
        self,
        manager,
        config: ElasticConfig,
        shaper: Callable[[Node], Optional[int]],
        policy: Optional[ScalingPolicy] = None,
    ):
        self.manager = manager
        self.config = config
        self.shaper = shaper
        self.policy = policy or build_policy(config)
        self.monitor = FateMonitor()
        self.counters: Counter = Counter()

        self._lock = threading.RLock()
        self._live: dict[int, _Member] = {}
        self._ready: dict[int, _Member] = {}
        self._launching: dict[int, _Member] = {}
        # Live members that died or were drained; they leave the replica list at the next commit.
        self._leaving: set[int] = set()
        self._paused = False
        self._requested: Optional[int] = None
        self._next_rank = 1 + max((r.replica_rank for r in manager.rollout_replicas), default=manager.start_rank - 1)

        self._executor = ThreadPoolExecutor(thread_name_prefix="elastic-rollout-launch")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        accelerators = {node.node_id: node.accelerator for node in list_nodes()}
        for replica in manager.rollout_replicas:
            pg = replica.resource_pool.get_placement_groups()[0]
            node_id = ray.util.placement_group_table(pg)["bundles_to_node_id"][0]
            member = _Member(replica.replica_rank, replica, replica.resource_pool, node_id, accelerators.get(node_id))
            self._live[member.rank] = member
            self._watch(member)
        self.target_gpus = self._clamp(self.policy.initial(self.gpus))

    # ------------------------------ background loop ------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="elastic-rollout", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background loop and free every replica the pool holds."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            members = [*self._live.values(), *self._ready.values(), *self._launching.values()]
            self._live, self._ready, self._launching, self._leaving = {}, {}, {}, set()
        for m in members:
            m.replica.shutdown()
            m.pool.release()

    def _loop(self) -> None:
        while not self._stop.wait(self.config.health_check_interval_s):
            try:
                self.tick()
            except Exception:
                logger.exception("[ElasticRollout] tick failed")

    def tick(self) -> None:
        """One round of the background loop: probe, then fill or drain toward the target."""
        self.monitor.tick()
        with self._lock:
            if self._paused:
                return
            surplus = self.gpus - self.target_gpus
        if surplus > 0:
            self._drain(surplus)
        elif surplus < 0:
            self._fill(-surplus)
        self._request()

    @property
    def gpus(self) -> int:
        """Accelerators held by serving, ready and launching replicas."""
        with self._lock:
            live = (m for rank, m in self._live.items() if rank not in self._leaving)
            return sum(m.gpus for m in (*live, *self._ready.values(), *self._launching.values()))

    def _fill(self, need: int) -> None:
        with self._lock:
            claimed = Counter()
            for m in self._launching.values():
                if not m.placed:
                    claimed[m.node_id] += m.gpus
        for node in list_nodes(self.config.rollout_node_resource, self.config.stable_node_resource):
            free = node.free_gpus - claimed[node.node_id]
            tp = self.shaper(node) if free > 0 else None
            while tp and need > 0 and free >= tp:
                self._launch(node, tp)
                free, need = free - tp, need - tp
            if need <= 0:
                return

    def _launch(self, node: Node, tp: int) -> None:
        manager = self.manager
        config = OmegaConf.merge(
            manager.rollout_config,
            {"tensor_model_parallel_size": tp, "data_parallel_size": 1, "pipeline_model_parallel_size": 1},
        )
        with self._lock:
            rank, self._next_rank = self._next_rank, self._next_rank + 1
        replica = manager.rollout_replica_class(
            replica_rank=rank, config=config, model_config=manager.model_config, gpus_per_node=node.total_gpus
        )
        pool = replica.make_resource_pool(bundle_extra=node.affinity)
        pool.reserve()
        member = _Member(rank, replica, pool, node.node_id, node.accelerator, placed=False)
        with self._lock:
            self._launching[rank] = member
        logger.info(f"[ElasticRollout] launching replica {rank} (tp={tp}) on a {node.accelerator} node")
        self._executor.submit(self._start, member)

    def _start(self, member: _Member) -> None:
        try:
            pg = member.pool.reserve()[0]
            ray.get(pg.ready(), timeout=_PLACEMENT_TIMEOUT_S)
            member.node_id = ray.util.placement_group_table(pg)["bundles_to_node_id"][0]
            member.placed = True
            asyncio.run(member.replica.init_standalone(resource_pool=member.pool))
        except Exception:
            with self._lock:
                cancelled = self._launching.pop(member.rank, None) is None
            if not cancelled:
                logger.exception(f"[ElasticRollout] replica {member.rank} failed to start on node {member.node_id}")
            member.pool.release()
            return
        with self._lock:
            started = self._launching.pop(member.rank, None) is not None
            if started:
                self._ready[member.rank] = member
        if started:
            self._watch(member)
        else:
            member.pool.release()

    def _drain(self, surplus: int) -> None:
        """Take whole nodes out of service, the least loaded first, without dropping below the target."""
        for members in sorted(self._by_node().values(), key=lambda ms: sum(m.gpus for m in ms)):
            gpus = sum(m.gpus for m in members)
            if gpus > surplus:
                return
            self._retire(members)
            surplus -= gpus

    def _request(self) -> None:
        if not self.config.autoscale or self._requested == self.target_gpus:
            return
        from ray.autoscaler.sdk import request_resources

        bundle = {get_platform().ray_resource_name(): 1, self.config.rollout_node_resource: 1e-3}
        request_resources(bundles=[bundle] * self.target_gpus)
        self._requested = self.target_gpus

    # ------------------------------ membership ------------------------------

    def _watch(self, member: _Member) -> None:
        self.monitor.watch(member.rank, member.actors, on_death=self._on_death)

    def _on_death(self, rank: int) -> None:
        with self._lock:
            member = self._ready.pop(rank, None)
            if member is None:
                member = self._live.get(rank)
                if member is None or rank in self._leaving:
                    return
                self._leaving.add(rank)
                self._unroute([member])
            self.counters["evicted"] += 1
        logger.warning(f"[ElasticRollout] replica {rank} on node {member.node_id} died")
        member.pool.release()

    def _unroute(self, members: list[_Member]) -> None:
        if members:
            addresses = [m.replica.server_address for m in members]
            ray.get(self.manager.global_load_balancer.remove_servers.remote(addresses))

    def _by_node(self) -> dict[str, list[_Member]]:
        with self._lock:
            by_node = defaultdict(list)
            live = (m for rank, m in self._live.items() if rank not in self._leaving)
            for m in (*live, *self._ready.values(), *self._launching.values()):
                by_node[m.node_id].append(m)
            return by_node

    def _retire(self, members: list[_Member], now: bool = False) -> None:
        """Gracefully take members out of service.

        A serving replica stops receiving requests and its in-flight requests resume elsewhere. Its processes are
        killed at the next :meth:`commit`, so a drain never tears down a replica in the middle of a weight sync,
        unless ``now`` is set by a caller on the trainer's thread, where no sync can be in flight.
        """
        with self._lock:
            serving = [m for m in members if m.rank in self._live and m.rank not in self._leaving]
            idle = [m for m in members if m not in serving]
            for m in idle:
                self._ready.pop(m.rank, None)
                self._launching.pop(m.rank, None)
            self._leaving.update(m.rank for m in serving)
            self._unroute(serving)
            self.counters["drained"] += len(members)
        for m in members:
            self.monitor.unwatch(m.rank)
        for m in idle:
            m.pool.release()
        if serving:
            asyncio.run(_gather(m.replica.abort_all_requests(reject_request=True) for m in serving))
        if now:
            for m in serving:
                m.replica.shutdown()
                m.pool.release()

    def commit(self) -> list[RolloutReplica]:
        """Apply membership changes to the replica list the checkpoint manager shares.

        Drops replicas that died or were drained and admits replicas that finished launching. Call it only from the
        trainer's thread right before a weight sync, then :meth:`publish` the returned replicas after it.

        Returns:
            Replicas admitted by this call.
        """
        with self._lock:
            leaving = [self._live.pop(rank) for rank in self._leaving]
            self._leaving.clear()
            joined = sorted(self._ready.values(), key=lambda m: m.rank)
            self._ready.clear()
            self._live.update((m.rank, m) for m in joined)
            self.counters["joined"] += len(joined)
            replicas = [m.replica for m in self._live.values()]
        for m in leaving:
            self.monitor.unwatch(m.rank)
            m.replica.shutdown()
            m.pool.release()

        manager = self.manager
        manager.rollout_replicas[:] = replicas
        manager.server_handles[:] = [r.server_handle for r in replicas]
        manager.server_addresses[:] = [r.server_address for r in replicas]
        return [m.replica for m in joined]

    def publish(self, replicas: list[RolloutReplica]) -> None:
        """Route requests to replicas admitted by :meth:`commit`, once they hold current weights."""
        with self._lock:
            servers = {
                r.server_address: r.server_handle
                for r in replicas
                if r.replica_rank in self._live and r.replica_rank not in self._leaving
            }
            if servers:
                ray.get(self.manager.global_load_balancer.add_servers.remote(servers))

    # ------------------------------ trainer facing ------------------------------

    def step(self, stats: StepStats) -> None:
        """Let the scaling policy move the target after a training step."""
        self.target_gpus = self._clamp(self.policy.target(self.target_gpus, stats))

    def pause(self) -> None:
        """Stop filling and give back the accelerators of replicas that are still launching."""
        with self._lock:
            self._paused = True
            launching = list(self._launching.values())
            self._launching.clear()
        for m in launching:
            m.pool.release()

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def vacate(self, count: int, gpus: int, accelerator: Optional[str] = None) -> int:
        """Free ``count`` nodes with at least ``gpus`` accelerators of the given type, to lend them to the trainer.

        Call it only from the trainer's thread: the replicas on those nodes are killed at once.

        Returns:
            Number of nodes freed.
        """
        shapes = {node.node_id: node for node in list_nodes(self.config.rollout_node_resource)}
        candidates = [
            members
            for node_id, members in self._by_node().items()
            if node_id in shapes
            and shapes[node_id].total_gpus >= gpus
            and accelerator in (None, shapes[node_id].accelerator)
        ]
        candidates.sort(key=lambda ms: sum(m.gpus for m in ms))
        for members in candidates[:count]:
            self._retire(members, now=True)
        return min(count, len(candidates))

    def metrics(self) -> dict[str, float]:
        with self._lock:
            live = [m for rank, m in self._live.items() if rank not in self._leaving]
        per_type = Counter()
        for m in live:
            per_type[m.accelerator or "unknown"] += m.gpus
        metrics = {
            "elastic/rollout_gpus": sum(per_type.values()),
            "elastic/rollout_replicas": len(live),
            **{f"elastic/{name}": count for name, count in self.counters.items()},
            **{f"elastic/gpus/{name}": count for name, count in per_type.items()},
        }
        if self.target_gpus < sys.maxsize:
            metrics["elastic/target_gpus"] = self.target_gpus
        return metrics

    def _clamp(self, gpus: int) -> int:
        return max(self.config.min_rollout_gpus, min(self.config.max_rollout_gpus or sys.maxsize, gpus))

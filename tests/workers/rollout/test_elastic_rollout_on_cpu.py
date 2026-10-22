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
"""Elastic rollout on a heterogeneous Ray cluster whose GPU nodes join and leave.

The cluster is real (``ray.cluster_utils.Cluster``) with logical GPUs; replicas are stand-ins whose processes are
dummy actors placed in the replica's placement group, so placement, eviction and draining run for real.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf
from ray.cluster_utils import Cluster
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from verl.trainer.config import ElasticConfig
from verl.workers.rollout import llm_server
from verl.workers.rollout.elastic import (
    BacklogPolicy,
    ElasticRolloutPool,
    Node,
    ReplicaShaper,
    StaticPolicy,
    StepStats,
    list_nodes,
)
from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.router import get_router_handle

GiB = 1 << 30
MEMORY = {"H100": 80 * GiB, "L4": 24 * GiB}


# ------------------------------ shaping and policies ------------------------------


def _node(accelerator, gpus=8, free=None):
    return Node(f"node-{accelerator}", accelerator, gpus, gpus if free is None else free)


@pytest.mark.parametrize(
    "accelerator, gpus, expected",
    [("H100", 8, 1), ("L4", 8, 4), ("L4", 2, None)],
)
def test_shaper_picks_smallest_tp_that_fits(accelerator, gpus, expected):
    # 70 GiB of weights: one H100 holds them in half its memory, an L4 needs four GPUs.
    shaper = ReplicaShaper(70 * GiB // 2, num_heads=32, device_memory=lambda n: MEMORY[n.accelerator])
    assert shaper(_node(accelerator, gpus)) == expected


def test_shaper_respects_heads_and_overrides():
    shaper = ReplicaShaper(140 * GiB, num_heads=6, device_memory=lambda n: MEMORY[n.accelerator])
    assert shaper(_node("H100")) is None, "4 GPUs fit the weights but neither 4 nor 8 divides 6 heads"
    assert ReplicaShaper(0, tp_by_accelerator={"*": 2})(_node("L4")) == 2
    assert ReplicaShaper(0, tp_by_accelerator={"L4": 4, "*": 1})(_node("L4")) == 4
    assert ReplicaShaper(0, tp_by_accelerator={"*": 16})(_node("L4")) is None


def test_static_policy_holds_the_cap():
    policy = StaticPolicy(ElasticConfig(max_rollout_gpus=32))
    assert policy.initial(8) == 32
    assert policy.target(32, StepStats(step_s=10, wait_s=9, backlog=0, batch_size=64)) == 32


def test_backlog_policy_follows_the_balance():
    policy = BacklogPolicy(ElasticConfig(policy="backlog", min_rollout_gpus=8, patience=2, scale_step_gpus=4))
    starving = StepStats(step_s=10, wait_s=5, backlog=0, batch_size=64)
    ahead = StepStats(step_s=10, wait_s=0, backlog=64, batch_size=64)
    assert policy.initial(0) == 8
    assert policy.target(8, starving) == 8
    assert policy.target(8, starving) == 12
    assert policy.target(12, ahead) == 12
    assert policy.target(12, starving) == 12, "a change of direction restarts the streak"
    assert policy.target(12, ahead) == 12
    assert policy.target(12, ahead) == 8


def test_elastic_config_validation():
    with pytest.raises(ValueError, match="autoscale"):
        ElasticConfig(autoscale=True, max_rollout_gpus=8)
    with pytest.raises(ValueError, match="max_rollout_gpus"):
        ElasticConfig(min_rollout_gpus=8, max_rollout_gpus=4)
    with pytest.raises(ValueError, match="backlog"):
        ElasticConfig(policy="backlog")


# ------------------------------ client failover ------------------------------


class _DyingServer:
    """Fails like a server whose node vanished, then resumes the request elsewhere."""

    def __init__(self, failures):
        self.failures = failures
        self.prompts = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        self.prompts.append(len(prompt_ids))
        if len(self.prompts) == 1:
            return TokenOutput(token_ids=[7] * 3, stop_reason="aborted")
        if self.failures:
            self.failures -= 1
            raise ray.exceptions.ActorDiedError()
        return TokenOutput(token_ids=[7] * 2, stop_reason="completed")


@pytest.fixture
def no_backoff(monkeypatch):
    real_sleep = asyncio.sleep

    async def no_wait(_delay, *args, **kwargs):
        return await real_sleep(0, *args, **kwargs)

    monkeypatch.setattr(llm_server.asyncio, "sleep", no_wait)


@pytest.mark.parametrize("retries, raises", [(5, False), (0, True)])
def test_client_resumes_after_server_death(monkeypatch, no_backoff, retries, raises):
    server = _DyingServer(failures=2)
    monkeypatch.setattr(llm_server.LLMServerClient, "generate", server.generate)
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"name": "vllm", "response_length": 16}},
            "trainer": {"elastic": {"enable": True, "request_retries": retries}},
        }
    )
    client = FullyAsyncLLMServerClient(config=config)

    async def generate():
        return await client.generate("req", prompt_ids=[1, 2], sampling_params={})

    if raises:
        with pytest.raises(ray.exceptions.RayActorError):
            asyncio.run(generate())
        return
    output = asyncio.run(generate())
    assert output.token_ids == [7] * 5, "tokens generated before the failure are kept"
    assert server.prompts == [2, 5, 5, 5], "each retry resumes from prompt plus the generated prefix"


# ------------------------------ elastic pool on a live cluster ------------------------------


@ray.remote(num_cpus=0)
class _Process:
    async def abort_all_requests(self, reject_request=False):
        return reject_request


class FakeReplica(RolloutReplica):
    """A replica whose server and workers are dummy actors held by its placement group."""

    async def init_standalone(self, resource_pool=None):
        self.rollout_mode = RolloutMode.STANDALONE
        self.resource_pool = resource_pool
        pg = resource_pool.get_placement_groups()[0]
        self.workers = [
            _Process.options(num_gpus=1, scheduling_strategy=PlacementGroupSchedulingStrategy(pg, i)).remote()
            for i in range(self.world_size)
        ]
        self.servers = [_Process.options(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, 0)).remote()]
        await asyncio.gather(*[a.__ray_ready__.remote() for a in self.workers + self.servers])
        self._server_handle, self._server_address = self.servers[0], f"replica-{self.replica_rank}"

    async def launch_servers(self):
        raise NotImplementedError


@pytest.fixture(scope="module")
def cluster():
    ray.shutdown()  # an earlier module in this process may have left Ray running
    cluster = Cluster(initialize_head=True, head_node_args={"num_cpus": 4, "resources": {"verl_stable": 1}})
    ray.init(address=cluster.address, include_dashboard=False)
    yield cluster
    ray.shutdown()
    cluster.shutdown()


def _add_node(cluster, accelerator, gpus=4):
    node = cluster.add_node(num_cpus=16, num_gpus=gpus, resources={f"accelerator_type:{accelerator}": 1})
    cluster.wait_for_nodes()
    return node


def _manager():
    rollout_config = OmegaConf.create(
        {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": "vllm",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
        }
    )
    return SimpleNamespace(
        rollout_config=rollout_config,
        model_config={},
        rollout_replica_class=FakeReplica,
        rollout_replicas=[],
        server_handles=[],
        server_addresses=[],
        start_rank=0,
        global_load_balancer=get_router_handle(servers={}),
    )


def _pool(**overrides):
    config = ElasticConfig(enable=True, stable_node_resource="verl_stable", **overrides)
    shaper = ReplicaShaper(int(35 * GiB), num_heads=32, device_memory=lambda n: MEMORY[n.accelerator])
    return ElasticRolloutPool(_manager(), config, shaper)


def _routed(pool):
    return set(ray.get(pool.manager.global_load_balancer.get_all_servers.remote()))


def _until(pool, condition, timeout_s=60.0):
    """Drive the background loop by hand until ``condition`` holds."""
    deadline = time.monotonic() + timeout_s
    while True:
        pool.tick()
        if condition():
            return
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.2)


def _settle(pool, gpus):
    """Admit replicas until ``gpus`` accelerators serve, the way the trainer does at each weight sync."""
    joined = []

    def serving():
        joined.extend(pool.commit())
        return pool.metrics()["elastic/rollout_gpus"] == gpus

    _until(pool, serving)
    return joined


def test_pool_fills_heterogeneous_nodes_and_heals(cluster):
    h100 = _add_node(cluster, "H100")
    l4 = _add_node(cluster, "L4")
    pool = _pool()
    try:
        joined = _settle(pool, 8)
        assert sorted(r.world_size for r in joined) == [1, 1, 1, 1, 4], "4 x tp1 on H100, 1 x tp4 on L4"
        assert pool.metrics()["elastic/gpus/H100"] == 4 and pool.metrics()["elastic/gpus/L4"] == 4
        assert not _routed(pool), "replicas take traffic only after their first weight sync"
        pool.publish(joined)
        assert _routed(pool) == {r.server_address for r in joined}
        assert pool.manager.rollout_replicas == joined, "the shared replica list is updated in place"

        # A replica crashes: it leaves the load balancer at once and is replaced on the same node.
        victim = joined[0]
        ray.kill(victim.servers[0])
        _until(pool, lambda: victim.server_address not in _routed(pool))
        replacements = _settle(pool, 8)
        assert victim not in pool.manager.rollout_replicas
        assert min(r.replica_rank for r in replacements) > max(r.replica_rank for r in joined), "ranks never repeat"
        pool.publish(replacements)

        # The L4 node is deleted: its replica is evicted, the H100 replicas keep serving.
        l4_replica = next(r for r in pool.manager.rollout_replicas if r.world_size == 4)
        cluster.remove_node(l4)
        _until(pool, lambda: l4_replica.server_address not in _routed(pool), timeout_s=120)
        _settle(pool, 4)
        assert pool.counters["evicted"] == 2
    finally:
        pool.stop()
        cluster.remove_node(h100)


def test_pool_drains_to_target_and_lends_nodes(cluster):
    nodes = [_add_node(cluster, "H100"), _add_node(cluster, "H100")]
    pool = _pool()
    try:
        pool.publish(_settle(pool, 8))

        # The target drops by a whole node: that node is drained gracefully and freed at the next commit.
        pool.target_gpus = 4
        _until(pool, lambda: len(_routed(pool)) == 4)
        pool.commit()
        assert len(pool.manager.rollout_replicas) == 4
        _until(pool, lambda: sum(n.free_gpus for n in list_nodes()) == 4)

        # A restarting trainer borrows the remaining node: it is freed at once and not refilled while paused.
        pool.pause()
        assert pool.vacate(1, gpus=4, accelerator="H100") == 1
        _until(pool, lambda: sum(n.free_gpus for n in list_nodes()) == 8)
        assert not _routed(pool)
        pool.commit()
        assert pool.manager.rollout_replicas == []

        pool.target_gpus = 8
        pool.resume()
        _settle(pool, 8)
    finally:
        pool.stop()
        for node in nodes:
            cluster.remove_node(node)

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
"""Building blocks of elastic training: resource pools that reserve without blocking, and fate sharing."""

import time

import pytest
import ray

from verl.single_controller.ray.base import RayResourcePool
from verl.single_controller.ray.fate import FateMonitor


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    ray.shutdown()  # an earlier module in this process may have left Ray running
    ray.init(num_cpus=8, num_gpus=2, include_dashboard=False)
    yield
    ray.shutdown()


@ray.remote(num_cpus=0)
class Member:
    def ping(self):
        return "pong"


def _wait(condition, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.1)


def test_reserve_does_not_block_and_release_frees_resources():
    blocker = RayResourcePool([2], max_colocate_count=1, name_prefix="blocker")
    blocker.get_placement_groups()

    pool = RayResourcePool([2], max_colocate_count=1, name_prefix="waiting")
    (pg,) = pool.reserve()
    assert pool.reserve() == [pg], "reserve is idempotent"
    ready, _ = ray.wait([pg.ready()], timeout=0.5)
    assert not ready, "the cluster has no free GPUs, so the group stays pending"

    blocker.release()
    ray.get(pg.ready(), timeout=30)
    assert pool.get_placement_groups() == [pg]

    pool.release()
    _wait(lambda: ray.available_resources().get("GPU", 0) == 2)
    with pytest.raises(RuntimeError, match="released"):
        pool.reserve()


def test_bundle_extra_constrains_placement():
    pool = RayResourcePool([1], max_colocate_count=1, bundle_extra={"missing_resource": 1e-3})
    (pg,) = pool.reserve()
    ready, _ = ray.wait([pg.ready()], timeout=0.5)
    assert not ready, "no node advertises the extra resource"
    pool.release()


def test_fate_monitor_kills_the_group_of_a_dead_member():
    monitor = FateMonitor()
    group, bystander = [Member.remote() for _ in range(3)], Member.remote()
    deaths = []
    monitor.watch("group", group, on_death=deaths.append)
    monitor.watch("bystander", [bystander])
    assert monitor.tick() == []

    ray.kill(group[0])
    _wait(lambda: monitor.tick() == ["group"] or deaths)

    assert deaths == ["group"]
    for member in group[1:]:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(member.ping.remote(), timeout=30)
    assert ray.get(bystander.ping.remote()) == "pong"
    assert monitor.tick() == [], "a dead group is forgotten"


def test_fate_monitor_watching_scope():
    monitor = FateMonitor()
    member = Member.remote()
    with monitor.watching("sync", [member]):
        assert monitor.tick() == []
    ray.kill(member)
    time.sleep(1)
    assert monitor.tick() == [], "the group is no longer watched after the scope"

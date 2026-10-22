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
import contextlib
import logging
import threading
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Optional

import ray
from ray.actor import ActorHandle

logger = logging.getLogger(__name__)


@dataclass
class _Group:
    actors: list[ActorHandle]
    on_death: Optional[Callable[[Hashable], None]]
    probes: dict[int, ray.ObjectRef] = field(default_factory=dict)


class FateMonitor:
    """Kill every member of a group as soon as one member dies.

    Members of a collective (FSDP ranks, the processes of one rollout replica, the participants of a
    weight sync) cannot finish without each other: survivors hang until NCCL times out. Killing them
    turns that hang into a prompt ``RayActorError`` for whoever is waiting on the group.

    Liveness is probed with the built-in ``__ray_ready__`` actor method. An actor busy with a long task
    keeps a single outstanding probe, which fails as soon as the actor dies.

    Args:
        timeout_s: How long one :meth:`tick` waits for probes to come back.
    """

    def __init__(self, timeout_s: float = 1.0):
        self.timeout_s = timeout_s
        self._groups: dict[Hashable, _Group] = {}
        self._lock = threading.Lock()

    def watch(
        self, key: Hashable, actors: Iterable[ActorHandle], on_death: Optional[Callable[[Hashable], None]] = None
    ) -> None:
        with self._lock:
            self._groups[key] = _Group(list(actors), on_death)

    def unwatch(self, key: Hashable) -> None:
        with self._lock:
            self._groups.pop(key, None)

    @contextlib.contextmanager
    def watching(self, key: Hashable, actors: Iterable[ActorHandle], on_death=None):
        self.watch(key, actors, on_death)
        try:
            yield
        finally:
            self.unwatch(key)

    def tick(self) -> list[Hashable]:
        """Probe every watched actor once. Kill and forget each group with a dead member.

        Returns:
            Keys of the groups that died.
        """
        with self._lock:
            groups = list(self._groups.items())

        pending = {}
        for key, group in groups:
            for i, actor in enumerate(group.actors):
                if i not in group.probes:
                    group.probes[i] = actor.__ray_ready__.remote()
                pending[group.probes[i]] = (key, group, i)
        if not pending:
            return []

        done, _ = ray.wait(list(pending), num_returns=len(pending), timeout=self.timeout_s, fetch_local=False)
        dead, killed = {}, []
        for ref in done:
            key, group, i = pending[ref]
            del group.probes[i]
            try:
                ray.get(ref)
            except ray.exceptions.RayActorError:
                dead[key] = group

        for key, group in dead.items():
            with self._lock:
                if self._groups.get(key) is not group:
                    continue
                del self._groups[key]
            killed.append(key)
            logger.warning(f"[FateMonitor] a member of {key!r} died, killing its {len(group.actors)} actors")
            kill(group.actors)
            if group.on_death is not None:
                group.on_death(key)
        return killed


def kill(actors: Iterable[ActorHandle]) -> None:
    """Kill actors without restart, ignoring ones that are already gone."""
    for actor in actors:
        with contextlib.suppress(Exception):
            ray.kill(actor, no_restart=True)

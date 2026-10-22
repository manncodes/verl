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
"""Elastic separate_async training: a lost trainer node restarts the trainer in place and the job carries on."""

from collections import Counter
from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf

import verl.trainer.ppo.v1.trainer_base as base_module
import verl.trainer.ppo.v1.trainer_separate_async as trainer_module
from verl.single_controller.ray.base import RayResourcePool
from verl.trainer.config import ElasticConfig
from verl.trainer.main_ppo import pin_transfer_queue_storage
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

# ------------------------------ validation ------------------------------


class _Engine:
    def __init__(self, bucket_size, rebuild_group=False):
        pass


def _construct(monkeypatch, *, backend="nccl", nnodes=1, enable_switch=False, elastic=True):
    monkeypatch.setattr(trainer_module.CheckpointEngineRegistry, "get", staticmethod(lambda name: _Engine))
    config = OmegaConf.create(
        {
            "data": {"train_batch_size": 64},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 16},
                "rollout": {
                    "nnodes": nnodes,
                    "n_gpus_per_node": 8,
                    "checkpoint_engine": {"backend": backend, "engine_kwargs": {}},
                    "disaggregation": {"enabled": False},
                },
            },
            "trainer": {
                "nnodes": 1,
                "n_gpus_per_node": 8,
                "elastic": {"_target_": "verl.trainer.config.ElasticConfig", "enable": elastic},
                "v1": {
                    "separate_async": {
                        "parameter_sync_step": 4,
                        "hybrid_rollout": {
                            "_target_": "verl.trainer.config.HybridRolloutSwitchConfig",
                            "enable_switch": enable_switch,
                        },
                    }
                },
            },
            "reward": {"reward_model": {"enable": False}},
        }
    )

    def base_init(trainer, trainer_config):
        trainer.config = trainer_config
        trainer.replay_buffer = SimpleNamespace(wait_for_sampleable=None, get_sampleable_count=None)

    monkeypatch.setattr(trainer_module.PPOTrainer, "__init__", base_init)
    return PPOTrainerSeparateAsync(config)


def test_elastic_rebuilds_the_sync_group_and_allows_an_empty_start(monkeypatch):
    trainer = _construct(monkeypatch, nnodes=0)
    assert trainer.elastic_config.enable
    assert trainer.config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group is True


def test_elastic_off_keeps_today_behaviour(monkeypatch):
    trainer = _construct(monkeypatch, elastic=False)
    assert not trainer.elastic_config.enable
    assert "nccl" not in trainer.config.actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs
    with pytest.raises(AssertionError, match="nnodes"):
        _construct(monkeypatch, nnodes=0, elastic=False)


@pytest.mark.parametrize(
    "kwargs, match",
    [({"backend": "delta_sharded"}, "full-weight"), ({"enable_switch": True}, "enable_switch")],
)
def test_elastic_rejects_unsupported_setups(monkeypatch, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _construct(monkeypatch, **kwargs)


@pytest.mark.parametrize(
    "enable, stable, preset, expected",
    [
        (True, "verl_stable", None, "verl_stable"),
        (True, "verl_stable", "mine", "mine"),
        (False, "verl_stable", None, None),
        (True, None, None, None),
    ],
)
def test_transfer_queue_storage_stays_on_stable_nodes(enable, stable, preset, expected):
    key = "transfer_queue.backend.SimpleStorage.required_node_resource"
    config = OmegaConf.create(
        {
            "trainer": {"elastic": {"enable": enable, "stable_node_resource": stable}},
            "transfer_queue": {"backend": {"SimpleStorage": {"num_data_storage_units": 8}}},
        }
    )
    if preset is not None:
        OmegaConf.update(config, key, preset)
    OmegaConf.set_struct(config, True)
    pin_transfer_queue_storage(config)
    assert OmegaConf.select(config, key) == expected


# ------------------------------ recovery ------------------------------


def _trainer(**elastic) -> PPOTrainerSeparateAsync:
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.elastic_config = ElasticConfig(enable=True, **elastic)
    trainer.elastic_rollout = SimpleNamespace(counters=Counter())
    trainer._consecutive_failures = 0
    trainer.global_steps = 5
    trainer.events = []
    trainer._sync_standalone = lambda steps: trainer.events.append(("sync", steps))
    trainer._restart_trainer = lambda: trainer.events.append("restart")
    return trainer


def test_dead_trainer_is_restarted_and_rollout_resynced():
    trainer = _trainer()
    trainer._trainer_alive = lambda: False
    assert trainer.on_worker_failure(ray.exceptions.ActorDiedError())
    assert trainer.events == ["restart", ("sync", 4)]
    assert trainer.elastic_rollout.counters["recovery_s"] > 0


def test_live_trainer_only_resyncs_rollout():
    trainer = _trainer()
    trainer._trainer_alive = lambda: True
    assert trainer.on_worker_failure(ray.exceptions.ActorDiedError())
    assert trainer.events == [("sync", 4)]


def test_recovery_gives_up_without_progress():
    trainer = _trainer(max_consecutive_failures=3)
    trainer._trainer_alive = lambda: False

    def failing_sync(steps):
        trainer.events.append(("sync", steps))
        raise ray.exceptions.ActorDiedError()

    trainer._sync_standalone = failing_sync
    assert not trainer.on_worker_failure(ray.exceptions.ActorDiedError())
    assert trainer.events.count("restart") == 3


def test_failures_are_not_handled_without_elastic():
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.elastic_rollout = None
    assert not trainer.on_worker_failure(ray.exceptions.ActorDiedError())


def test_restart_rebuilds_only_the_trainer_pools(monkeypatch):
    events = []

    class Pool:
        def __init__(self, name):
            self.name = name

        def release(self):
            events.append(f"release:{self.name}")

    old_actors, new_actors = ["a0", "a1"], ["b0", "b1"]
    global_pool, reward_pool = Pool("global_pool"), Pool("reward_pool")
    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create({"trainer": {"default_local_dir": "/ckpt", "resume_mode": "auto"}})
    trainer.elastic_rollout = SimpleNamespace(
        counters=Counter(),
        pause=lambda: events.append("pause"),
        resume=lambda: events.append("resume"),
        monitor=SimpleNamespace(
            unwatch=lambda key: events.append(f"unwatch:{key}"),
            watch=lambda key, actors: events.append(f"watch:{key}:{actors}"),
        ),
    )
    trainer.resource_pool_to_cls = {global_pool: {"actor": object()}, reward_pool: {}}

    def create_resource_pool(names, check):
        events.append(f"create:{names}:{check}")
        trainer.resource_pool_manager.resource_pool_dict["global_pool"] = Pool("new_global_pool")

    trainer.resource_pool_manager = SimpleNamespace(
        resource_pool_dict={"global_pool": global_pool, "reward_pool": reward_pool},
        create_resource_pool=create_resource_pool,
    )
    trainer.actor_rollout_wg = SimpleNamespace(workers=old_actors)
    trainer._trainer_actors = lambda: list(trainer.actor_rollout_wg.workers)
    trainer._wait_for_trainer_nodes = lambda pools: events.append(f"wait:{[p.name for p in pools]}")

    def init_workers():
        events.append("init")
        trainer.actor_rollout_wg = SimpleNamespace(workers=new_actors)

    trainer._init_trainer_workers = init_workers
    trainer._checkpoint_dir = lambda: "/ckpt"
    trainer._load_model_checkpoint = lambda folder: events.append(f"load:{folder}")
    trainer.standalone_checkpoint_manager = SimpleNamespace(actor_wg=None)
    trainer.checkpoint_manager = SimpleNamespace(actor_wg=None)
    monkeypatch.setattr(trainer_module, "kill", lambda actors: events.append(f"kill:{actors}"))
    monkeypatch.setattr(trainer_module, "find_latest_ckpt_path", lambda path: f"{path}/global_step_3")

    trainer._restart_trainer()

    assert events == [
        "pause",
        "unwatch:trainer",
        f"kill:{old_actors}",
        "release:global_pool",
        "create:['global_pool']:False",
        "wait:['new_global_pool']",
        "init",
        "load:/ckpt/global_step_3",
        f"watch:trainer:{new_actors}",
        "resume",
    ]
    assert trainer.standalone_checkpoint_manager.actor_wg is trainer.actor_rollout_wg
    assert trainer.checkpoint_manager.actor_wg is trainer.actor_rollout_wg
    assert trainer.elastic_rollout.counters["trainer_restarts"] == 1


def test_trainer_borrows_a_rollout_node_when_no_replacement_comes():
    ray.shutdown()  # an earlier module in this process may have left Ray running
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
    try:
        rollout_node = RayResourcePool([2], max_colocate_count=1, name_prefix="rollout")
        rollout_node.get_placement_groups()
        borrowed = []

        def vacate(count, gpus, accelerator):
            borrowed.append((count, gpus, accelerator))
            rollout_node.release()
            return 1

        trainer = object.__new__(PPOTrainerSeparateAsync)
        trainer.config = OmegaConf.create({"trainer": {"n_gpus_per_node": 2}})
        trainer.elastic_config = ElasticConfig(enable=True, borrow_rollout_after_s=0.0)
        trainer.elastic_rollout = SimpleNamespace(vacate=vacate)
        trainer._trainer_accelerator = "H100"

        trainer_pool = RayResourcePool([2], max_colocate_count=1, name_prefix="trainer")
        trainer._wait_for_trainer_nodes([trainer_pool])

        assert borrowed == [(1, 2, "H100")]
        assert trainer_pool.get_placement_groups(), "the trainer took the freed node"
        trainer_pool.release()
    finally:
        ray.shutdown()


# ------------------------------ the training loop ------------------------------


def _fit_trainer(monkeypatch, recover: bool):
    for name in ("Tracking", "ValidationGenerationsLogger", "DapoFilteredRewardTableLogger"):
        monkeypatch.setattr(base_module, name, lambda **kwargs: SimpleNamespace(log=lambda **kw: None))
    monkeypatch.setattr(base_module.SkipManager, "init", staticmethod(lambda config: None))
    monkeypatch.setattr(base_module.SkipManager, "set_step", staticmethod(lambda step: None))
    monkeypatch.setattr(base_module.tq, "kv_clear", lambda **kwargs: None)

    trainer = object.__new__(PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "p",
                "experiment_name": "e",
                "logger": [],
                "val_before_train": False,
                "total_epochs": 1,
                "save_freq": 0,
                "test_freq": 0,
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.global_steps, trainer.steps_per_epoch, trainer.total_training_steps = 0, 10, 3
    trainer.steps, trainer.failures = [], []
    for hook in ("_reissue_inflight_prompts", "on_train_begin", "on_step_begin", "on_step_end", "on_train_end"):
        setattr(trainer, hook, lambda: None)
    trainer._start_profiling = trainer._stop_profiling = trainer._shutdown_dump_executor = lambda: None
    trainer._consume_sync_metrics = lambda: {}
    trainer._compute_metrics = lambda *args, **kwargs: None

    def step(metrics, timing_raw):
        trainer.steps.append(trainer.global_steps)
        if trainer.steps == [1, 2]:
            raise ray.exceptions.ActorDiedError()
        return SimpleNamespace(keys=[], partition_id="train")

    def on_worker_failure(error):
        trainer.failures.append(type(error).__name__)
        return recover

    trainer.step, trainer.on_worker_failure = step, on_worker_failure
    return trainer


def test_fit_retries_the_step_after_recovery(monkeypatch):
    trainer = _fit_trainer(monkeypatch, recover=True)
    trainer.fit(agent_loop_manager=None)
    assert trainer.steps == [1, 2, 2, 3]
    assert trainer.failures == ["ActorDiedError"]


def test_fit_raises_when_the_failure_is_not_recovered(monkeypatch):
    trainer = _fit_trainer(monkeypatch, recover=False)
    with pytest.raises(ray.exceptions.ActorDiedError):
        trainer.fit(agent_loop_manager=None)
    assert trainer.steps == [1, 2]

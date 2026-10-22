# Elastic Training

Last updated: 10/22/2026.

Elastic training keeps a `separate_async` run going while the GPU nodes under it change. Rollout nodes can be
added or deleted at any time and may be of different GPU types. When a trainer node is lost, the trainer restarts
in place from its latest checkpoint. Throughout, the driver and the job keep running.

```bash
python3 -m verl.trainer.main_ppo \
    trainer.v1.trainer_mode=separate_async \
    trainer.elastic.enable=true \
    trainer.elastic.stable_node_resource=verl_stable \
    trainer.save_freq=5 \
    actor_rollout_ref.rollout.nnodes=0 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    ...
```

With `rollout.nnodes=0` the run starts with no rollout replicas. It fills every free GPU on the eligible nodes
and follows them as they come and go.

## Who adds and removes nodes

Three layers govern capacity. verl never creates or deletes pods itself.

| Layer | Owner | Role |
|---|---|---|
| Infrastructure | KubeRay or cluster autoscaler, `kubectl`, spot reclaim | Creates and deletes nodes. |
| Demand | verl scaling policy, once per step | Sets the rollout GPU target. With `autoscale=true` the target is published through `ray.autoscaler.sdk.request_resources`, so the autoscaler adds nodes to meet it and removes the ones that go idle. |
| Supply | verl `ElasticRolloutPool`, a background loop every `health_check_interval_s` | Fills free GPUs with replicas shaped for each node, drains whole nodes above the target, evicts dead replicas. |

Policies (`trainer.elastic.policy`):

- **`static`** (default): the target is `max_rollout_gpus`, which is unbounded when unset. Whatever capacity
  arrives is used, and whatever is lost is replaced.
- **`backlog`**: follows the balance between making and consuming samples.
  - It scales up by `scale_step_gpus` after `patience` steps in which the trainer waited for samples longer than
    `up_wait_ratio` of the step.
  - It scales down after `patience` steps in which the trainer never waited and a full batch was already queued,
    because further samples would only go stale.
  - It requires `min_rollout_gpus >= 1`.

Custom policies register through `verl.workers.rollout.elastic.register_policy`.

## Replica lifecycle

- **Joining.** A replica is launched in the background but takes no traffic until it holds current weights. It
  is admitted in one of two places:
  - at the end of each step, together with the regular weight sync;
  - while the trainer waits for samples, through a sync to the new replicas only. This is what lets a run with no
    rollout capacity, at startup or after losing every rollout node, recover instead of waiting forever.
- **Leaving gracefully** (scale-in, or a node lent to the trainer). The replica is removed from the load balancer
  and its in-flight requests are aborted. They resume on other replicas and keep the tokens generated so far.
  Its processes are killed at the next sync barrier, so a drain never interrupts a weight sync.
- **Leaving abruptly** (pod deleted, node failure, crash). A fate monitor probes every process. When one dies, the
  rest of its replica is killed and the replica is evicted from the load balancer. Requests that were running on
  it fail over to another replica, again keeping their tokens, up to `request_retries` times.

## Heterogeneous rollout

Rollout replicas only stream tokens, so any node that can hold the model can serve. Each node gets replicas
sized for its hardware:

- **Tensor parallel size:** the smallest power of two that divides the node's GPU count and the model's attention
  heads, and keeps one weight shard within `max_weight_fraction` of a GPU's memory. The rest of the memory is left
  for KV cache.
- **Overrides:** `tp_by_accelerator` sets the size per Ray accelerator type, with `"*"` as a wildcard:
  `trainer.elastic.tp_by_accelerator='{H100: 1, L4: 4}'`.
- **Placement:** a replica is placed on a node of the accelerator type it was shaped for.

For example, 35 GB of weights become eight tp=1 replicas on an 8×H100 node and two tp=4 replicas on an 8×L4
node. Weight sync needs no change, because every rollout process receives full tensors and keeps its own shard.

Requirements:

- One image built for every GPU architecture in use.
- NVIDIA only, because NCCL does not span vendors.
- Ray names the accelerator type after the first word that follows "NVIDIA" in the GPU name, so an RTX PRO 6000
  and every other RTX card are all `RTX`. Different RTX models in one pool look the same to the shaper: keep one
  RTX model per cluster.
- Logprob differences across hardware are handled by rollout correction or by recomputing `old_log_prob`.

## Failure semantics

| Event | What happens |
|---|---|
| Rollout replica crashes | It is evicted within one health-check interval; its requests resume elsewhere; a new replica fills its GPUs. |
| Rollout node deleted | Same as above, for every replica on the node. |
| Trainer node deleted | `fit` catches the error. The trainer worker groups are killed, rebuilt at the same size and restored from the newest completed checkpoint, then rollout gets their weights and the step is retried. Rollout, the replay buffer and the dataloader carry on, so only the updates since the last checkpoint are lost. |
| No replacement trainer node | After `borrow_rollout_after_s`, the trainer takes a rollout node with the same GPU type and count. A later replacement node is then filled by rollout. |
| A process dies during a weight sync | The fate monitor kills every sync participant, so the broadcast fails instead of hanging, and the trainer restarts as above. |
| Repeated failures without progress | After `max_consecutive_failures`, the error is raised. |
| Ray head node lost | Not tolerated by default: it hosts the GCS and the driver. See [Head node](#head-node). |

The trainer world size never changes, because FSDP checkpoints are written per rank. Keep `save_freq` small and
`default_local_dir` on storage that every node can reach.

## Kubernetes setup

Mark the nodes that never leave (the head, and optionally a CPU control group) with a Ray resource and name it in
`stable_node_resource`. The driver, the load balancer, the agent-loop workers, the reward-loop workers and the
TransferQueue storage units are pinned there. Storage is pinned by defaulting
`transfer_queue.backend.SimpleStorage.required_node_resource` to the stable resource. The TransferQueue
controller cannot be pinned. It is created from the pinned driver and normally lands on the same node, but that
is not enforced.

To keep rollout away from some GPU nodes, give the rollout groups a second marker and set `rollout_node_resource`.
`autoscale` requires it.

### KubeRay

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: verl-elastic
spec:
  entrypoint: >-
    python3 -m verl.trainer.main_ppo trainer.v1.trainer_mode=separate_async
    trainer.elastic.enable=true trainer.elastic.stable_node_resource=verl_stable
    trainer.elastic.rollout_node_resource=verl_rollout trainer.elastic.autoscale=true
    trainer.elastic.max_rollout_gpus=64 actor_rollout_ref.rollout.nnodes=0 ...
  rayClusterSpec:
    enableInTreeAutoscaling: true
    headGroupSpec:
      rayStartParams:
        num-gpus: "0"
        resources: '"{\"verl_stable\": 1000}"'
      template: {...}             # on an on-demand node pool
    workerGroupSpecs:
      - groupName: trainer        # fixed size, on-demand
        replicas: 2
        minReplicas: 2
        maxReplicas: 2
        template: {...}           # nvidia.com/gpu: 8
      - groupName: rollout-h100   # elastic, may be spot
        replicas: 2
        minReplicas: 0
        maxReplicas: 8
        rayStartParams:
          resources: '"{\"verl_rollout\": 1000}"'
        template: {...}
      - groupName: rollout-l4
        replicas: 0
        minReplicas: 0
        maxReplicas: 16
        rayStartParams:
          resources: '"{\"verl_rollout\": 1000}"'
        template: {...}
```

### Kubeflow PyTorchJob

- **Master:** runs `ray start --head --num-gpus=0 --resources='{"verl_stable": 1000}'` followed by the
  `main_ppo` command above.
- **Workers:** run `ray start --address=$MASTER_ADDR:6379 --block` with `restartPolicy: OnFailure`, so a deleted
  worker pod comes back and rejoins without ending the job.
- **Scaling:** change the Worker `replicas` to grow or shrink the cluster.
- **Node markers:** all workers are alike. The trainer takes its nodes first and rollout fills the rest, so no
  rollout marker is needed.

### Head node

By default, losing the Ray head pod ends the job. The head hosts the GCS, and Ray Jobs runs the driver on the head
unless the entrypoint requests resources. A worker-only topology, with no work on the head, does not change that
by itself. Together with the other two steps below, it turns a head restart into a pause:

1. Keep work off the head with `num-cpus: "0"` and `num-gpus: "0"` in its `rayStartParams`. Add a CPU control
   group that carries `verl_stable` for the control plane.
2. Run the driver on that group with RayJob `entrypointNumCpus: 1` and `entrypointResources: '{"verl_stable": 1}'`.
   Ray Jobs then schedules the job supervisor, a detached actor, there instead of on the head.
3. Enable GCS fault tolerance with an external Redis. Set `RAY_gcs_rpc_server_reconnect_timeout_s` on every pod
   above the time a head restart takes, because a raylet that cannot reach the GCS within it exits.

While the GCS is away, calls between running actors carry on, which covers most of a training step. Anything that
needs the GCS waits for it to come back: creating actors or placement groups, and building the communication group
of a weight sync. This setup has not been tested yet.

## Metrics

| Metric | Meaning |
|---|---|
| `elastic/rollout_gpus`, `elastic/rollout_replicas` | Rollout capacity serving now. |
| `elastic/gpus/<accelerator>` | Rollout GPUs per accelerator type. |
| `elastic/target_gpus` | Policy target, when bounded. |
| `elastic/joined`, `elastic/evicted`, `elastic/drained` | Cumulative replica arrivals and departures. |
| `elastic/trainer_restarts`, `elastic/recovery_s` | Cumulative trainer restarts and time spent recovering. |

## Trying it out

The following can be run by hand on a cluster:

1. Start with two rollout nodes, then scale a rollout group up. `elastic/rollout_gpus` rises once the new
   replicas have loaded.
2. Delete a rollout pod. Training continues and capacity refills when the pod returns.
3. Delete a trainer pod. `elastic/trainer_restarts` goes up by 1 and training resumes from the last checkpoint
   while rollout keeps generating.
4. Delete a trainer pod and keep its replacement from scheduling. After `borrow_rollout_after_s` the trainer takes
   a rollout node.

## Limitations

- Only the v1 `separate_async` trainer is supported, with `hybrid_rollout.enable_switch=false`.
- No PD disaggregation, and no `delta_sharded` weight sync, because a replica that joins late would only ever
  receive deltas.
- Each rollout replica runs on a single node.
- Reward-model and teacher pools are not elastic; keep them on stable nodes.
- The Ray head pod is a single point of failure unless it is set up as in [Head node](#head-node).

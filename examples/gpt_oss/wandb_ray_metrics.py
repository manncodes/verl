"""Forward Ray's per-node Prometheus metrics to wandb.

verl's wandb run only logs system metrics from rank 0 (the trainer process).
This sidecar discovers all Ray nodes via the cluster's service-discovery
file, scrapes their /metrics endpoints, and logs cluster-wide system metrics
(per-node GPU util, CPU, memory, network, object store) to a *parallel*
wandb run.

Run it from a second shell after training has launched:

    python examples/gpt_oss/wandb_ray_metrics.py \\
        --project verl_gpt_oss_20b \\
        --run-name gpt_oss_20b_grpo_gsm8k

It will create a wandb run named "<run-name>-cluster" in the same project
so you can view it side-by-side with the training run. Stop it with Ctrl-C
when training finishes; safe to run multiple times.

Why a separate run instead of writing to verl's run? wandb's run_id isn't
trivially exposed by verl, and using `wandb.init(resume="must", id=...)`
from a sidecar racing with the trainer's own writes is fragile. A parallel
run grouped by name in the UI is the cleanest tradeoff.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

try:
    from prometheus_client.parser import text_string_to_metric_families
except ImportError:
    print(
        "[wandb-ray-metrics] prometheus_client not found. install with:\n"
        "    uv pip install prometheus-client\n"
        "(it usually ships transitively with ray; missing means a stripped venv)",
        file=sys.stderr,
    )
    sys.exit(1)

import wandb


# Ray emits hundreds of metrics; we keep the ones with node-level cardinality
# that wandb's chart UI plots usefully. Add to this dict if you want more.
INTERESTING = {
    "ray_node_cpu_utilization": "cpu_util_pct",
    "ray_node_cpu_count": "cpu_count",
    "ray_node_mem_used": "mem_used_bytes",
    "ray_node_mem_available": "mem_available_bytes",
    "ray_node_mem_total": "mem_total_bytes",
    "ray_node_gpus_utilization": "gpu_util_pct",
    "ray_node_gram_used": "gpu_mem_used_bytes",
    "ray_node_gram_available": "gpu_mem_available_bytes",
    "ray_node_disk_usage": "disk_used_bytes",
    "ray_node_disk_io_read_speed": "disk_read_bytes_per_sec",
    "ray_node_disk_io_write_speed": "disk_write_bytes_per_sec",
    "ray_node_network_send_speed": "net_send_bytes_per_sec",
    "ray_node_network_receive_speed": "net_recv_bytes_per_sec",
    "ray_object_store_memory": "object_store_bytes",
    "ray_node_gpus_available": "gpus_available",
}


def discover_endpoints(ray_session_dir: Path) -> list[str]:
    """Read Ray's Prometheus service-discovery JSON; return ['host:port', ...]."""
    sd = ray_session_dir / "metrics/prometheus/prom_metrics_service_discovery.json"
    if not sd.exists():
        raise FileNotFoundError(
            f"ray service discovery not found at {sd}. is ray running?\n"
            f"check that {ray_session_dir} exists and contains metrics/prometheus/"
        )
    data = json.loads(sd.read_text())
    targets: list[str] = []
    for entry in data:
        targets.extend(entry.get("targets", []))
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def scrape_one(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Hit a /metrics endpoint, parse the Prometheus text format, filter to
    INTERESTING metrics, and return {short_name[/label]: value}."""
    try:
        with urlopen(f"http://{url}/metrics", timeout=timeout) as resp:
            body = resp.read().decode()
    except (URLError, TimeoutError, OSError) as exc:
        return {"_scrape_error": 1.0, "_scrape_error_str": str(exc)}

    out: dict[str, float] = {}
    for fam in text_string_to_metric_families(body):
        if fam.name not in INTERESTING:
            continue
        short = INTERESTING[fam.name]
        for sample in fam.samples:
            # GPU metrics carry a GpuIndex label per device; everything else
            # is per-node, so we drop the labels and keep the latest value.
            gpu_idx = sample.labels.get("GpuIndex")
            if gpu_idx is not None:
                key = f"{short}/gpu{gpu_idx}"
            else:
                key = short
            out[key] = float(sample.value)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True, help="wandb project (matches verl trainer.project_name)")
    p.add_argument(
        "--run-name",
        required=True,
        help="verl experiment name; this sidecar appends '-cluster' to keep them separate",
    )
    p.add_argument("--interval", type=float, default=15.0, help="scrape interval in seconds")
    p.add_argument(
        "--ray-session",
        default="/tmp/ray/session_latest",
        help="ray session dir (default points at the most recent ray cluster)",
    )
    p.add_argument(
        "--wandb-base-url",
        default=os.environ.get("WANDB_BASE_URL"),
        help="override wandb host (defaults to $WANDB_BASE_URL)",
    )
    args = p.parse_args()

    print(f"[wandb-ray-metrics] discovering ray nodes from {args.ray_session}")
    endpoints = discover_endpoints(Path(args.ray_session))
    if not endpoints:
        print("[wandb-ray-metrics] no scrape targets found; exiting", file=sys.stderr)
        sys.exit(1)
    print(f"[wandb-ray-metrics] found {len(endpoints)} endpoint(s):")
    for ep in endpoints:
        print(f"  - {ep}")

    init_kwargs: dict = dict(
        project=args.project,
        name=f"{args.run_name}-cluster",
        config={
            "sidecar": "ray-prometheus",
            "scrape_interval_s": args.interval,
            "endpoints": endpoints,
            "metrics": list(INTERESTING.values()),
        },
    )
    if args.wandb_base_url:
        init_kwargs["settings"] = wandb.Settings(base_url=args.wandb_base_url)
    wandb.init(**init_kwargs)

    stop = False

    def _handler(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    print(f"[wandb-ray-metrics] logging every {args.interval:.0f}s; Ctrl-C to stop")
    iteration = 0
    while not stop:
        flat: dict[str, float] = {"scrape_iteration": iteration}
        for ep in endpoints:
            # Use the host part as the node label (port can drift across restarts).
            node = ep.split(":")[0].replace(".", "_")
            for k, v in scrape_one(ep).items():
                if isinstance(v, (int, float)):
                    flat[f"node_{node}/{k}"] = v
        wandb.log(flat)
        iteration += 1
        # break the sleep into 100ms ticks so SIGINT lands quickly
        for _ in range(max(1, int(args.interval * 10))):
            if stop:
                break
            time.sleep(0.1)

    print("[wandb-ray-metrics] closing wandb run")
    wandb.finish()


if __name__ == "__main__":
    main()

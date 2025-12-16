#!/usr/bin/env bash
set -xeuo pipefail

NNODES=${1:-2}
trainer_n_gpus_per_node=${2:-8}
head_node_name=${3:-mann-verl-post-install-debug-master-0}

echo NNODES = $NNODES , GPUs per node = $trainer_n_gpus_per_node
echo Head Node: $head_node_name

# echo "INFO: Fixing environment for user-installed packages..."
# # The user's local bin directory for any command-line tools
# export PATH="/home/jovyan/.local/bin:${PATH}"

# # The user's local Python library path. Pip will install here with the --user flag.
# USER_SITE_PACKAGES=$(python3 -m site --user-site)
# export PYTHONPATH="${USER_SITE_PACKAGES}:${PYTHONPATH:-}"

# # CRITICAL: The user's local C++ shared library path (.so files).
# # This forces the system to load our user-compiled flash-attn .so file first.
# export LD_LIBRARY_PATH="${USER_SITE_PACKAGES}:${LD_LIBRARY_PATH:-}"

# echo "Corrected PATH: ${PATH}"
# echo "Corrected PYTHONPATH: ${PYTHONPATH}"
# echo "Corrected LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"

# export CACHE_DIR=/fsxp2/meq967/pip_cache/
# if [ ! -d "${CACHE_DIR}" ] ; then
# 	mkdir -p $CACHE_DIR
# fi

export NO_PROXY="${NO_PROXY:-""},127.0.0.1,localhost,${MASTER_ADDR},${POD_IP}"
export no_proxy="${no_proxy:-""},127.0.0.1,localhost,${MASTER_ADDR},${POD_IP}"
unset WORLD_SIZE TOTAL_WORLD_SIZE

# --- ROLE-BASED EXECUTION ---
# Because the PATH is now correct, we can use the simple 'ray' commands again.
if [[ "${RANK}" == "0" ]]; then
    # --- MASTER LOGIC (This pod will be the Ray Head) ---
    echo "This is the Master pod (RANK 0). Starting Ray Head and submitting job."

    RAY_ADDRESS=${RAY_ADDRESS:-"auto"}
    WORKING_DIR=${WORKING_DIR:-"${PWD}"}
    RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}

    rm -rf /tmp/ray

    # This will now work because '/home/jovyan/.local/bin' is in the PATH
    ray start --head --node-ip-address=${POD_IP} --port=6379 --dashboard-host=0.0.0.0 --num-gpus=8 --num-cpus=48 --object-store-memory=$((50*1024*1024*1024))

    # Wait for all worker nodes to successfully connect and verify NCCL versions.
    cat << 'EOF' > /tmp/wait_for_nodes.py
import ray
import time
import sys
import json
import logging
import subprocess
import os

expected_nodes = int(sys.argv[1])
expected_gpus_per_node = int(sys.argv[2]) if len(sys.argv) > 2 else 8
total_expected_gpus = expected_nodes * expected_gpus_per_node

print(f"--> [VERIFICATION SCRIPT] Waiting for {expected_nodes} nodes to join the cluster...")

# Connect to the local Ray cluster
ray.init(address='auto', logging_level=logging.DEBUG)


@ray.remote
def get_node_info():
    """Get NCCL version and other info from each node."""
    import socket
    import subprocess
    import os

    hostname = socket.gethostname()
    node_ip = os.environ.get('RAY_NODE_IP', 'unknown')

    # Get NCCL version - try multiple methods
    nccl_version = None
    nccl_details = {}

    # Method 1: Try to get from PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            nccl_version = torch.cuda.nccl.version()
            nccl_details['pytorch_nccl'] = nccl_version
            nccl_details['torch_version'] = torch.__version__
            nccl_details['cuda_version'] = torch.version.cuda
    except Exception as e:
        nccl_details['pytorch_error'] = str(e)

    # Method 2: Try nccl-tests or nccl_info if available
    try:
        result = subprocess.run(
            ['python', '-c', 'import torch; print(torch.cuda.nccl.version())'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            nccl_details['nccl_from_subprocess'] = result.stdout.strip()
    except Exception as e:
        pass

    # Method 3: Check NCCL shared library version
    try:
        result = subprocess.run(
            ['ldconfig', '-p'], capture_output=True, text=True, timeout=10
        )
        nccl_libs = [line for line in result.stdout.split('\n') if 'nccl' in line.lower()]
        if nccl_libs:
            nccl_details['nccl_libs'] = nccl_libs[:5]  # First 5 matches
    except Exception as e:
        pass

    # Method 4: Check environment variables
    nccl_env_vars = {k: v for k, v in os.environ.items() if 'NCCL' in k}
    if nccl_env_vars:
        nccl_details['nccl_env_vars'] = nccl_env_vars

    # Get GPU info
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['gpu_count'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            gpu_info['cuda_available'] = True
    except Exception as e:
        gpu_info['error'] = str(e)

    return {
        'hostname': hostname,
        'node_ip': node_ip,
        'nccl_version': nccl_version,
        'nccl_details': nccl_details,
        'gpu_info': gpu_info,
    }


def verify_nccl_versions(node_infos):
    """Verify all nodes have the same NCCL version."""
    print("\n" + "="*60)
    print("--> [NCCL VERIFICATION] Checking NCCL versions across all nodes...")
    print("="*60)

    nccl_versions = {}
    all_same = True
    reference_version = None

    for info in node_infos:
        hostname = info['hostname']
        nccl_ver = info['nccl_version']

        print(f"\n  Node: {hostname}")
        print(f"    NCCL Version: {nccl_ver}")
        print(f"    GPU Count: {info['gpu_info'].get('gpu_count', 'N/A')}")

        if info['nccl_details']:
            if 'torch_version' in info['nccl_details']:
                print(f"    PyTorch Version: {info['nccl_details']['torch_version']}")
            if 'cuda_version' in info['nccl_details']:
                print(f"    CUDA Version: {info['nccl_details']['cuda_version']}")

        if nccl_ver is not None:
            nccl_versions[hostname] = nccl_ver
            if reference_version is None:
                reference_version = nccl_ver
            elif nccl_ver != reference_version:
                all_same = False

    print("\n" + "-"*60)

    if not nccl_versions:
        print("--> [NCCL WARNING] Could not determine NCCL version on any node!")
        print("--> This may cause issues with distributed training.")
        return True  # Don't fail, just warn

    if all_same:
        print(f"--> [NCCL VERIFICATION PASSED] All nodes have NCCL version: {reference_version}")
        return True
    else:
        print("--> [NCCL VERIFICATION FAILED] Nodes have MISMATCHED NCCL versions!")
        print("--> Version summary:")
        for hostname, ver in nccl_versions.items():
            marker = " <-- MISMATCH" if ver != reference_version else ""
            print(f"      {hostname}: {ver}{marker}")
        print("\n--> [ERROR] NCCL version mismatch can cause hangs or crashes during distributed training!")
        print("--> Please ensure all nodes have the same NCCL version installed.")
        return False


while True:
    nodes = ray.nodes()
    num_ready_nodes = len(nodes)
    print(f"--> [VERIFICATION SCRIPT] Found {num_ready_nodes} / {expected_nodes} nodes.")

    cluster_resources = ray.cluster_resources()
    print(json.dumps(cluster_resources, indent=2))

    if num_ready_nodes >= expected_nodes:
        print("--> [VERIFICATION SCRIPT] All nodes have joined. Waiting 10s for resources to register...")
        time.sleep(10)  # Give a moment for resource heartbeats to consolidate

        print("\n" + "="*60)
        print("--> [VERIFICATION SCRIPT] VERIFYING CLUSTER RESOURCES...")
        print("="*60)

        cluster_resources = ray.cluster_resources()
        print(json.dumps(cluster_resources, indent=2))

        # Check if the total GPU count is correct
        total_gpus = cluster_resources.get("GPU", 0)

        print("-" * 60)
        if total_gpus < total_expected_gpus:
            print(f"--> [VERIFICATION FAILED] Ray sees only {total_gpus} GPUs, expected {total_expected_gpus}.")
            print(f"--> This means the worker node's 'ray start' command is still MISSING the '--num-gpus' flag.")
            sys.exit(1)
        else:
            print(f"--> [VERIFICATION PASSED] Ray sees all {total_gpus} GPUs. The cluster is formed correctly.")
            print("-" * 60 + "\n")

        # === NCCL VERSION CHECK ===
        print("--> [VERIFICATION SCRIPT] Collecting node information for NCCL verification...")

        # Get one task per node by using placement groups or node affinity
        # Simple approach: submit tasks and let Ray distribute them
        node_ips = list(set([node['NodeManagerAddress'] for node in ray.nodes() if node['Alive']]))

        # Submit tasks with node affinity to ensure we get info from each node
        futures = []
        for node in ray.nodes():
            if node['Alive']:
                node_id = node['NodeID']
                # Use scheduling_strategy to target specific nodes
                future = get_node_info.options(
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=False
                    )
                ).remote()
                futures.append(future)

        try:
            node_infos = ray.get(futures, timeout=60)

            # Deduplicate by hostname (in case of multiple tasks per node)
            seen_hosts = set()
            unique_infos = []
            for info in node_infos:
                if info['hostname'] not in seen_hosts:
                    seen_hosts.add(info['hostname'])
                    unique_infos.append(info)

            nccl_ok = verify_nccl_versions(unique_infos)

            if not nccl_ok:
                print("\n--> [FATAL] NCCL version mismatch detected. Halting cluster startup.")
                print("--> To bypass this check, set SKIP_NCCL_CHECK=1")
                if os.environ.get('SKIP_NCCL_CHECK', '0') != '1':
                    sys.exit(1)
                else:
                    print("--> [WARNING] SKIP_NCCL_CHECK=1 set, continuing despite mismatch...")

        except Exception as e:
            print(f"--> [NCCL WARNING] Failed to verify NCCL versions: {e}")
            print("--> Continuing without NCCL verification...")

        break
    time.sleep(5)

print("\n" + "="*60)
print("--> [SUCCESS] Cluster verification complete!")
print("="*60)
EOF

    # This script will now work because the PYTHONPATH is correct
    python3 /tmp/wait_for_nodes.py ${NNODES} ${trainer_n_gpus_per_node} || { echo "Verification failed. Halting execution."; exit 1; }
    echo "All nodes have joined the cluster and resources are verified."
else
    # --- WORKER LOGIC (This pod will be a Ray Worker) ---
    echo "This is a Worker pod (RANK ${RANK}). Waiting for head to become available."
    rm -rf /tmp/ray

    until getent hosts $head_node_name ; do
      echo "Head not resolvable yet, retrying in 5s..."
      sleep 5
    done
    echo "Head is resolvable. Connecting now."

    # This will now work because '/home/jovyan/.local/bin' is in the PATH
    ray start  --num-gpus=${trainer_n_gpus_per_node} --num-cpus=48 --address="${head_node_name}:6379" --object-store-memory=$((50*1024*1024*1024))

    echo "Worker connected -- waiting till master node completes."
    while getent hosts $head_node_name ; do
      echo "Master node is resolvable, continuing ..."
      sleep 60
    done
    # sleep infinity
fi

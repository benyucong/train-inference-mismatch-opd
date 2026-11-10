#!/bin/bash
# =============================================================================
# cluster_start_ray.sh -- one Ray node per SLURM task on the GH200 cluster.
#
# Launched as `srun cluster_start_ray.sh &` with --ntasks-per-node=1, so this runs
# once per node and uses SLURM_PROCID to pick head vs worker.
#
#   - The Python environment must be loaded in this step too (it sources
#     cluster_env.sh itself), because srun steps may not inherit it: a step
#     launched without --export=ALL, or on a site whose environment is not carried
#     by exported variables, would otherwise get the system python and fail
#     somewhere far from here.
#   - ROCR_VISIBLE_DEVICES is unset defensively: verl refuses it together with
#     CUDA_VISIBLE_DEVICES.
#
# Expects from the caller: head_node_ip, head_node_port, ip_head,
#                          GPUS_PER_TASK, RAY_TMPDIR
# =============================================================================
set -x
# Prefer the caller's REPO_ROOT; fall back to BASH_SOURCE for a direct run. srun passes the real
# path so BASH_SOURCE is fine here -- unlike the sbatch script, which sbatch copies to the spool
# dir. Belt and braces because the failure mode is a job that fails two seconds in.
source "${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/cluster_env.sh"

current_node=$(hostname -s)
unset ROCR_VISIBLE_DEVICES

# vLLM binds the wrong interface on multi-node without an explicit host IP.
export VLLM_HOST_IP=$(hostname --ip-address | awk '{print $1}')
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"

# CPUs handed to Ray: RAY_NUM_CPUS, else the task's --cpus-per-task, else 64. A GH200
# node has 288 CPUs; handing Ray all of them oversubscribes the node once vLLM and
# FSDP both spawn workers, so cap unless the caller says otherwise.
CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-64}}"
GPUS="${GPUS_PER_TASK:-4}"

# Disjoint per-rank port ranges, so that two Ray nodes of one job on one host do not collide.
PORT_BASE=$((11000 + (SLURM_PROCID % 8) * 1000))

if [ "$SLURM_PROCID" = "0" ]; then
    echo "Ray HEAD on $current_node ($head_node_ip:$head_node_port), rank $SLURM_PROCID"
    "$PYTHON_BIN" -m ray.scripts.scripts start --head \
        --node-ip-address="$head_node_ip" \
        --port="$head_node_port" \
        --num-cpus="$CPUS" --num-gpus="$GPUS" \
        --min-worker-port=$PORT_BASE --max-worker-port=$((PORT_BASE + 999)) \
        --temp-dir="$RAY_TMPDIR" \
        --disable-usage-stats --include-dashboard=false --block
else
    sleep $((SLURM_PROCID * 2 + 5))   # stagger; do not all hit the head at once
    echo "Ray WORKER on $current_node -> $ip_head, rank $SLURM_PROCID"
    "$PYTHON_BIN" -m ray.scripts.scripts start \
        --address "$ip_head" \
        --num-cpus="$CPUS" --num-gpus="$GPUS" \
        --min-worker-port=$PORT_BASE --max-worker-port=$((PORT_BASE + 999)) \
        --temp-dir="$RAY_TMPDIR" \
        --disable-usage-stats --block
fi
wait

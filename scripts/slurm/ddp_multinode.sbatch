#!/bin/bash
# Multi-node DDP pre-training benchmark: one torchrun per node, c10d rendezvous (nvlink QOS: 4 GPUs per job).
#   CONFIG=config/experiments/ssl_scaling.yml MODE=weak sbatch --qos=nvlink --partition=nvlink \
#     --nodes=2 --gres=gpu:2 scripts/slurm/ddp_multinode.sbatch [--key=value ...]
# NCCL settings come from the environment (--export=ALL), for example NCCL_SOCKET_IFNAME=ib0.
#
#SBATCH --job-name=detect_ddp_multinode
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
PYTHON="${DETECT_PYTHON:-$HOME/miniconda3/envs/detect-fe/bin/python}"
MODE="${MODE:-weak}"
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS_PER_NODE=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
fi
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
LABEL="${SLURM_NNODES}x${GPUS_PER_NODE}"
echo "nodes=$SLURM_JOB_NODELIST master=$MASTER gpus_per_node=$GPUS_PER_NODE mode=$MODE start=$(date -Is)"

srun --ntasks-per-node=1 bash -c 'echo "$(hostname): $(ls /sys/class/net | tr "\n" " ") memlock=$(ulimit -l)"'
env | grep -E "^NCCL_" || true

RUN_JSON=$("$PYTHON" scripts/pretraining/pretrain_scaling.py --config "$CONFIG" --prepare-only \
  --mode "$MODE" --label "$LABEL" "$@" | tail -n 1)
echo "run_json=$RUN_JSON"

srun --kill-on-bad-exit=1 "$PYTHON" -m torch.distributed.run \
  --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
  --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER:29500" \
  scripts/pretraining/pretrain_ssl.py --config-json "$RUN_JSON"
echo "end=$(date -Is)"

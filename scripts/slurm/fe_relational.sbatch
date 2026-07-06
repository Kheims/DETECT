#!/bin/bash
# Relational Feature Envy study: one array task per graph variant
# (0 = plain AST, 1 = relational level 0, 2 = relational level 1).
# Each task builds its own graphs, then trains the best ablation GCN on 3 seeds.
#
#   sbatch scripts/slurm/fe_relational.sbatch
#
#SBATCH --job-name=fe_relational
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fe_relational_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

CONFIGS=(
  config/experiments/fe_relational_base.yml
  config/experiments/fe_relational_l0.yml
  config/experiments/fe_relational_l1.yml
)
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

PYTHON="$HOME/miniconda3/envs/detect-fe/bin/python"
echo "host=$(hostname) config=$CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

"$PYTHON" main.py --config "$CONFIG" --training.device=cuda

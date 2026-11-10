#!/bin/bash
#SBATCH --job-name=rl_data_gen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.log
# The RL comparison's base pass (bench/rl_data.py gen): ONE block of order positions [POS_LO, POS_HI) on one GPU, no
# speculation. The base student samples 8 responses per prompt at the training settings (T 1, top_p 1, top_k -1,
# 16,384 tokens, bf16, seeds 300,000 + 8 pos + s), graded by the training verifier in a worker thread.
# bench/rl_data.py's docstring says exactly what is reproduced from training and what is not.
#   env: POS_LO / POS_HI (required; blocks of 1,000 positions, eight jobs for [0, 8000)), D (out dir; default
#        $CLUSTER_ROOT/rl_data/base_pass, shared by every block), DATA (default the JR file written by
#        `bench/rl_data.py jr`), ORDER_DATA (default the DAPO parquet, on which the order is computed),
#        DRY_RUN=1 (checks and prompts only: print the plan, exit).
#   --time is sized for a block of 1,000 positions; set it at submission for other block sizes.
# Submitting the same block again resumes: finished chunk files are kept and only missing chunks are generated.
# Run `bench/rl_data.py summarize` once every block has finished.
# Submit from the repository root (sbatch spools this script: never use BASH_SOURCE):
#   for k in 0 1 2 3 4 5 6 7; do POS_LO=$((1000*k)) POS_HI=$((1000*(k+1))) sbatch --parsable bench/rl_data.sbatch; done
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
unset SPECOPD_SAMPLER_NOISE SPECOPD_SAMPLER_NOISE_SEED SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL
export PYTHONDONTWRITEBYTECODE=1
POS_LO=${POS_LO:?first order position of this block}; POS_HI=${POS_HI:?one past the last order position of this block}
D=${D:-$CLUSTER_ROOT/rl_data/base_pass}
DATA=${DATA:-$CLUSTER_ROOT/workspace/datasets/dapo-math-17k-jr.parquet}
ORDER_DATA=${ORDER_DATA:-$CLUSTER_ROOT/workspace/datasets/dapo-math-17k.parquet}
_CMD=("$PYTHON_BIN" -u bench/rl_data.py gen --out-dir "$D" --data "$DATA" --order-data "$ORDER_DATA"
      --pos-lo "$POS_LO" --pos-hi "$POS_HI")
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[rl_data] DRY_RUN positions [$POS_LO, $POS_HI) D=$D DATA=$DATA"
    "${_CMD[@]}" --dry-run; exit $?
fi
mkdir -p "$D"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[rl_data] positions [$POS_LO, $POS_HI) D=$D DATA=$DATA job=${SLURM_JOB_ID:-local} start=$(date -Is)"
{
    echo "job=${SLURM_JOB_ID:-local} positions=$POS_LO:$POS_HI start=$(date -Is) host=$(hostname) repo_dir=$PWD"
    sha256sum bench/rl_data.py bench/rl_data.sbatch bench/mo_common.py refresh/t3_acc2.py \
        custom_reward/specopd_reward.py custom_reward/ttrl_math/*.py verl/verl/trainer/ppo/specopd_grpo_metrics.py \
        "$DATA" "$ORDER_DATA"
} > "$D/provenance_${POS_LO}_${POS_HI}_${SLURM_JOB_ID:-local}.txt" 2>&1
t=$SECONDS
"${_CMD[@]}"
rc=$?
echo "[rl_data] gen rc=$rc ($((SECONDS-t))s) end=$(date -Is)"
exit $rc

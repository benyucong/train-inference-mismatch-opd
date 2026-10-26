#!/bin/bash
#SBATCH --job-name=ho_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# Held-out evaluation of ONE checkpoint on ONE prompt set (one GPU, no speculation).
# bench/ho_eval.py's docstring says exactly what a sample, a grade and the per-checkpoint quantities are.
#   env: ARM (label; the out dir is <HO_ROOT>/<ARM>_s<STEP>_<PROMPT>), PROMPT (dapo|jr), OBJECTIVE (rl|opd|base),
#        either TRAIN_JOB + STEP (the checkpoint is found under RUNS_ROOT, exactly one match, as tim_eval2.sbatch:
#        $RUNS_ROOT/*/checkpoints/specopd/*_<job>/global_step_<STEP>/actor/huggingface; RUNS_ROOT holds the repository
#        checkouts the training jobs ran from, default the parent directory of this checkout, so checkpoints this
#        checkout wrote are found) or CKPT=base (STEP 0),
#        N (4|8), J0 (seeds 100,000 + 16k + j for j in [J0, J0+N)), optional BLOCK=1 (one of the base's four j-blocks;
#        `merge` runs when all four exist), FREEZE_HO (default $CLUSTER_ROOT/ho/ho_freeze.json, written by
#        `bench/ho_eval.py freeze`), HO_ROOT (default $CLUSTER_ROOT/ho), DRY_RUN=1 (resolve + guards, print, exit 0).
#   Time limits: --time 02:30:00 at N=4 (each base j-block is one), 04:00:00 at N=8 (set it at submission).
# Submit from the repository root (sbatch spools this script: never use BASH_SOURCE):
#   ARM=<arm label> PROMPT=dapo OBJECTIVE=opd TRAIN_JOB=<training job id> STEP=100 N=8 J0=0 \
#       sbatch --parsable --job-name=ho_<arm label>_s100_dapo --time 04:00:00 bench/ho_eval.sbatch
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
unset SPECOPD_SAMPLER_NOISE SPECOPD_SAMPLER_NOISE_SEED SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL
export PYTHONDONTWRITEBYTECODE=1
ARM=${ARM:?arm label}; PROMPT=${PROMPT:?dapo|jr}; OBJECTIVE=${OBJECTIVE:?rl|opd|base}
N=${N:?4|8}; J0=${J0:-0}
FREEZE_HO=${FREEZE_HO:-$CLUSTER_ROOT/ho/ho_freeze.json}
HO_ROOT=${HO_ROOT:-$CLUSTER_ROOT/ho}
RUNS_ROOT=${RUNS_ROOT:-$(dirname "$REPO_ROOT")}
_X=(); [ "${BLOCK:-0}" = 1 ] && _X+=(--block)
if [ "${CKPT:-}" = base ]; then
    STEP=0; CK=base
else
    TRAIN_JOB=${TRAIN_JOB:?training job id}; STEP=${STEP:?step}
    CK=$("$PYTHON_BIN" bench/ho_eval.py resolve --job "$TRAIN_JOB" --step "$STEP" --runs-root "$RUNS_ROOT") \
        || { echo "[ho] checkpoint lookup FAILED"; exit 2; }
fi
_CMD=("$PYTHON_BIN" -u bench/ho_eval.py gen --freeze-ho "$FREEZE_HO" --prompt "$PROMPT" --objective "$OBJECTIVE"
      --arm "$ARM" --step "$STEP" --ckpt "$CK" --n "$N" --j0 "$J0" --ho-root "$HO_ROOT" ${_X[@]+"${_X[@]}"})
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "[ho] DRY_RUN arm=$ARM prompt=$PROMPT objective=$OBJECTIVE step=$STEP ckpt=$CK n=$N j0=$J0 ${_X[*]:-}"
    "${_CMD[@]}" --dry-run; exit $?
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[ho] arm=$ARM prompt=$PROMPT objective=$OBJECTIVE step=$STEP n=$N j0=$J0 job=${SLURM_JOB_ID:-local} ${_X[*]:-}"
{
    echo "job=${SLURM_JOB_ID:-local} start=$(date -Is) host=$(hostname) repo_dir=$PWD ckpt=$CK"
    sha256sum bench/ho_eval.py bench/ho_eval.sbatch custom_reward/specopd_reward_g2.py custom_reward/specopd_reward.py \
        custom_reward/ttrl_math/__init__.py custom_reward/ttrl_math/math_utils.py "$FREEZE_HO"
} > "$HO_ROOT/provenance_${SLURM_JOB_ID:-local}.txt" 2>&1
t=$SECONDS
"${_CMD[@]}"
rc=$?
echo "[ho] gen rc=$rc ($((SECONDS-t))s) end=$(date -Is)"
if [ $rc -eq 0 ] && [ "${BLOCK:-0}" = 1 ]; then
    _M="$HO_ROOT/${ARM}_s${STEP}_${PROMPT}"
    if [ "$(ls -d "$_M"/blocks/j*/ 2>/dev/null | wc -l)" -eq 4 ] && \
       [ "$(grep -l '"status": "COMPLETE"' "$_M"/blocks/j*/summary.json 2>/dev/null | wc -l)" -eq 4 ]; then
        # two j-blocks that finish together both see four COMPLETE blocks: the merge runs under an exclusive lock
        # (mkdir is atomic), once; the lock (next to the output dir, never inside it) stays as the record of who merged
        if mkdir "$_M.merge.lock" 2>/dev/null; then
            echo "job=${SLURM_JOB_ID:-local}" > "$_M.merge.lock/owner"
            if [ -e "$_M/summary.json" ]; then
                echo "[ho] merge already done"
            else
                "$PYTHON_BIN" bench/ho_eval.py merge --freeze-ho "$FREEZE_HO" --prompt "$PROMPT" --arm "$ARM" \
                    --step "$STEP" --ho-root "$HO_ROOT"; rc=$?; echo "[ho] merge rc=$rc"
                [ $rc -ne 0 ] && echo "[ho] merge FAILED: run ho_eval.py merge by hand (the lock $_M.merge.lock stays)"
            fi
        else
            echo "[ho] merge: another j-block job holds $_M.merge.lock (it merges)"
        fi
    else
        echo "[ho] merge waits for the other j-blocks"
    fi
fi
exit $rc

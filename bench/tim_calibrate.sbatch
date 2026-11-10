#!/bin/bash
#SBATCH --job-name=tim_cal
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:15:00
#SBATCH --output=logs/slurm/%x_%j.log
# Sampler-trainer mismatch calibration (bench/tim_calibrate.py): per-token KL(q||p) between the clean student and each
# candidate SAMPLER on 16 of the student's own T=1 traces, one GPU of the type that trains (the CUDA noise draw depends
# on it). One process per engine configuration. Split so that each piece fits a short debug partition:
#   MODES=bf16 TARGETS=''      : clean reference + the sigma grid (default 0 .. 0.32)
#   MODES=bf16 SIGMAS=''       : the target sigmas (interpolated from the grid in D, +1 secant step, +-25% check)
#   MODES='fp8w_fp8kv fp8w'    : the FP8 weight + FP8 KV cache sampler and its weight-only half;  MODES=kvfp8 : the
#                                KV-only half
#   env: D (out dir, required), TRACES (the base student's T=1 samples, refresh/t3_acc.py gen's gen_T1_p1.jsonl,
#        required), MODES, SIGMAS, TARGETS ('' = none; unset = default), NATURAL (the measured rollout_corr/kl of the
#        BF16 engine against the trainer, which every target already contains; unset = 0)
# Exit status is non-zero if ANY mode failed (including a bf16 target whose measured KL missed by > 25%), so an
# afterok dependent never starts on a missing or rejected sigma.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
D=${D:?set D to the output directory}; mkdir -p "$D"
TRACES=${TRACES:?set TRACES to the T=1 samples of the base student (jsonl)}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[timcal] D=$D job=$SLURM_JOB_ID modes='${MODES:-bf16 fp8w_fp8kv fp8w kvfp8}' sigmas='${SIGMAS-default}' targets='${TARGETS-default}' natural='${NATURAL:-0}'"
RC=0
for M in ${MODES:-bf16 fp8w_fp8kv fp8w kvfp8}; do
    t=$SECONDS
    "$PYTHON_BIN" bench/tim_calibrate.py --mode $M --out-dir "$D" --traces "$TRACES" \
        --sigmas "${SIGMAS-0,0.01,0.02,0.04,0.08,0.16,0.32}" --targets "${TARGETS-0.01,0.05}" ${NATURAL:+--natural "$NATURAL"}
    rc=$?; echo "[timcal] mode $M load_format=auto rc=$rc ($((SECONDS-t))s)"
    # An fp8 engine that cannot init from the bf16 checkpoint (load_format=auto) is recorded, then the same sampler is
    # measured from verl's default dummy init.
    if [ $rc -ne 0 ] && [ "$M" != bf16 ]; then
        t=$SECONDS
        "$PYTHON_BIN" bench/tim_calibrate.py --mode $M --out-dir "$D" --traces "$TRACES" --load-format dummy
        rc=$?; echo "[timcal] mode $M load_format=dummy rc=$rc ($((SECONDS-t))s)"
    fi
    [ $rc -ne 0 ] && RC=$rc
done
"$PYTHON_BIN" bench/tim_calibrate.py --mode report --out-dir "$D"
ls -la "$D"
echo "[timcal] done rc=$RC"
exit $RC

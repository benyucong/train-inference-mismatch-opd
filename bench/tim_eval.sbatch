#!/bin/bash
#SBATCH --job-name=tim_eval
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# Held-out evaluation of ONE trained checkpoint of one arm (one GPU, no speculation):
#   1. refresh/t3_acc.py gen+grade at T=1.0/top_p=1.0 only: AIME24 30x8 + MATH-500 500x1, max 16384 new tokens, seeds
#      fixed per (problem, sample) -> identical across arms;
#   2. bench/tim_kl_eval.py: those same traces teacher-forced through the checkpoint and through the teacher ->
#      per-sample k1 = KL(p_theta || p_T) on the checkpoint's OWN samples (independent of the rollout engine).
#   env: TRAIN_JOB (training job id; the checkpoint is found by it, whatever the arm label), STEP (the checkpoint's step, e.g. 25 to 200),
#        ARM (name for the out dir), or CKPT (an explicit hf dir, e.g. the base student for step 0) + ARM.
#        Optional: RUNS_ROOT, the directory holding the repository checkouts the training jobs ran from (default: the
#        parent directory of this checkout, so checkpoints this checkout wrote are found), D (the out dir).
# The training job's checkpoint is resolved at RUN time, so this job can be submitted with
# --dependency=afterany:<training job id>. The lookup is tim_eval2.sbatch's:
# $RUNS_ROOT/*/checkpoints/specopd/sopd_*_<job>/global_step_<STEP>/actor/huggingface, exactly one match holding
# config.json; anything else (e.g. training stopped before STEP) fails fast and lists what the job did write.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
ARM=${ARM:?arm name}
RUNS_ROOT=${RUNS_ROOT:-$(dirname "$REPO_ROOT")}
BASE=$CLUSTER_ROOT/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60
TEACHER=$(ls -d $CLUSTER_ROOT/hf/hub/models--Skywork--Skywork-OR1-Math-7B/snapshots/*/ | head -1 | sed 's|/$||')
if [ -n "${CKPT:-}" ]; then
    STU=$CKPT; STEP=${STEP:-0}
else
    STEP=${STEP:?step}; TRAIN_JOB=${TRAIN_JOB:?training job id}
    # checkpoints live in the checkout the TRAINING ran from (the launcher writes checkpoints/specopd/sopd_<arm>_<job>
    # under it), not necessarily this one; job ids are unique, so exactly one match
    mapfile -t CK < <(ls -d "$RUNS_ROOT"/*/checkpoints/specopd/sopd_*_${TRAIN_JOB}/global_step_${STEP}/actor/huggingface 2>/dev/null)
    [ ${#CK[@]} -eq 1 ] && [ -f "${CK[0]}/config.json" ] || { echo "[timeval] need exactly 1 checkpoint for job $TRAIN_JOB step $STEP, found ${#CK[@]}: ${CK[*]:-none}"; \
        ls -d "$RUNS_ROOT"/*/checkpoints/specopd/sopd_*_${TRAIN_JOB}/* 2>/dev/null; exit 2; }
    STU=${CK[0]}
fi
D=${D:-$CLUSTER_ROOT/eval/${ARM}_s${STEP}}; mkdir -p "$D"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[timeval] arm=$ARM step=$STEP STU=$STU TEACHER=$TEACHER D=$D job=$SLURM_JOB_ID"
RC=0
t=$SECONDS
$CLUSTER_ROOT/venv-refresh/bin/python refresh/t3_acc.py gen --model "$STU" --tokenizer "$BASE" --out-dir "$D" --temps 1.0:1.0
rc=$?; echo "[timeval] gen rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
t=$SECONDS
$PYTHON_BIN refresh/t3_acc.py grade --out-dir "$D" --tokenizer "$BASE"
rc=$?; echo "[timeval] grade rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
for ROLE in student teacher; do
    [ -s "$D/lp_$ROLE.npz" ] && { echo "[timeval] lp_$ROLE exists"; continue; }
    M=$STU; [ $ROLE = teacher ] && M=$TEACHER
    t=$SECONDS
    $PYTHON_BIN bench/tim_kl_eval.py score --model "$M" --tokenizer "$BASE" --gen "$D/gen_T1_p1.jsonl" --out "$D/lp_$ROLE.npz"
    rc=$?; echo "[timeval] score $ROLE rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
done
$PYTHON_BIN bench/tim_kl_eval.py combine --dir "$D"; rc=$?; [ $rc -ne 0 ] && RC=$rc
ls -la "$D"
echo "[timeval] done arm=$ARM step=$STEP rc=$RC"
exit $RC

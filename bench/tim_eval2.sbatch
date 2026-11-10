#!/bin/bash
#SBATCH --job-name=tim_eval2
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# Accuracy composite of ONE checkpoint (one GPU, no speculation). Two stages, each skipped where its outputs exist:
#   A. EXACTLY bench/tim_eval.sbatch's pass, same scripts and arguments, into the same dir D:
#      refresh/t3_acc.py gen+grade (AIME24 30x8 + MATH-500 500x1, T=1.0/top_p=1.0, max 16384 new tokens) and
#      bench/tim_kl_eval.py (held-out KL(p_theta || p_T) on those traces). A checkpoint tim_eval.sbatch already did
#      costs nothing here; combine runs only when kl_summary.json is missing (tim_eval.sbatch always runs it again; it
#      is deterministic, so existing results are never rewritten).
#   B. refresh/t3_acc2.py into D16=D/avg16: AIME24 samples 8-15, AIME25 30x16, AMC23 83x16 (same prompts, sampling,
#      engine and grader as t3_acc.py; seeds extend its scheme -- see t3_acc2.py), then `summarize` merges AIME24
#      samples 0-7 from stage A (after checking seeds/sampling/model/tokenizer) -> D16/summary16_T1_p1.json with
#      avg@16 for AIME24, AIME25, AMC23 and D16/merged16_T1_p1.jsonl (one graded row per sample, problem-paired).
#   env: ARM (eval dir label: REUSE THE LABEL tim_eval.sbatch used, <arm label>, so stage A is found)
#        and either TRAIN_JOB + STEP (the checkpoint is found under RUNS_ROOT, the directory holding the repository
#        checkouts the training jobs ran from; default: the parent directory of this checkout, so checkpoints this
#        checkout wrote are found: $RUNS_ROOT/*/checkpoints/specopd/sopd_*_<job>, exactly one match)
#        or CKPT (an explicit hf dir, e.g. the base student; STEP defaults to 0). Optional: D, D16, PLAN,
#        FREEZE (the t3_acc2.py freeze record: benchmark sha256 pins, decontamination exclusion list, endpoint;
#        default refresh/t3_acc2_freeze.json, the record shipped with this repository; FREEZE= (empty) runs without a
#        record: those checks are skipped, no problem is excluded and t3_acc2.py's default endpoint is used), DRY_RUN=1
#        (resolve the checkpoint + run the checks below, print, exit 0: `DRY_RUN=1 ARM=.. TRAIN_JOB=.. STEP=.. bash ..`).
#   Submit from the repository root:
#     ARM=<arm label> TRAIN_JOB=<training job id> STEP=100 sbatch --parsable --job-name=tim_ev2_<arm label>_s100 \
#         bench/tim_eval2.sbatch
#   If a tim_eval.sbatch job for the same checkpoint is still queued/running, add --dependency=afterany:<that job>:
#   both would write stage A's files.
# Guard: if D already holds a t3_acc.py pass, its gen meta must name the SAME checkpoint path (a mislabeled ARM would
# otherwise silently merge another arm's AIME24 samples 0-7 and report another arm's KL).
# FREEZE CHECK: before ANY stage, `t3_acc2.py freeze-check` must pass: with a record, the record must be valid and the
# parquets on disk must match its pins (with FREEZE= it prints one line and passes). A failing check aborts the job
# (exit 2) before stage A; DRY_RUN=1 reports it and still exits 0. Stage B's logs print no accuracy (counts and
# completeness only); stage A is t3_acc.py's pass, unchanged, which prints its own AIME24 avg@8 / MATH-500 lines as
# tim_eval.sbatch does.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
ARM=${ARM:?arm name}
BASE=$CLUSTER_ROOT/hf/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60
TEACHER=$(ls -d $CLUSTER_ROOT/hf/hub/models--Skywork--Skywork-OR1-Math-7B/snapshots/*/ | head -1 | sed 's|/$||')
RUNS_ROOT=${RUNS_ROOT:-$(dirname "$REPO_ROOT")}
if [ -n "${CKPT:-}" ]; then
    STU=$CKPT; STEP=${STEP:-0}
else
    STEP=${STEP:?step}; TRAIN_JOB=${TRAIN_JOB:?training job id}
    # checkpoints live in the checkout the TRAINING ran from (the launcher writes checkpoints/specopd/sopd_<arm>_<job>
    # under it), not necessarily this one; job ids are unique, so exactly one match
    mapfile -t CK < <(ls -d "$RUNS_ROOT"/*/checkpoints/specopd/sopd_*_${TRAIN_JOB}/global_step_${STEP}/actor/huggingface 2>/dev/null)
    [ ${#CK[@]} -eq 1 ] && [ -f "${CK[0]}/config.json" ] || { echo "[timeval2] need exactly 1 checkpoint for job $TRAIN_JOB step $STEP, found ${#CK[@]}: ${CK[*]:-none}"; \
        ls -d "$RUNS_ROOT"/*/checkpoints/specopd/sopd_*_${TRAIN_JOB}/* 2>/dev/null; exit 2; }
    STU=${CK[0]}
fi
D=${D:-$CLUSTER_ROOT/eval/${ARM}_s${STEP}}; D16=${D16:-$D/avg16}
PLAN=${PLAN:-AIME24:8-15,AIME25:0-15,AMC23:0-15}
# ${FREEZE-...} (no colon): an unset FREEZE takes the shipped record, an empty FREEZE= means no record
FREEZE=${FREEZE-$REPO_ROOT/refresh/t3_acc2_freeze.json}
_FZ=(); [ -n "$FREEZE" ] && _FZ=(--freeze "$FREEZE")
if [ -s "$D/gen_T1_p1_meta.json" ]; then
    OLD=$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$D/gen_T1_p1_meta.json")
    [ "$(realpath -m "$OLD")" = "$(realpath -m "$STU")" ] || { echo "[timeval2] $D was made from $OLD, not $STU: wrong ARM label?"; exit 2; }
fi
$PYTHON_BIN refresh/t3_acc2.py freeze-check ${_FZ[@]+"${_FZ[@]}"}; FZ_RC=$?
if [ $FZ_RC -ne 0 ]; then
    [ "${DRY_RUN:-0}" = 1 ] || { echo "[timeval2] ABORT: freeze-check rc=$FZ_RC -- the freeze record is invalid or the data moved; nothing was run"; exit 2; }
    echo "[timeval2] DRY_RUN: freeze-check rc=$FZ_RC, a real run would ABORT here"
fi
[ "${DRY_RUN:-0}" = 1 ] && { echo "[timeval2] DRY_RUN arm=$ARM step=$STEP STU=$STU D=$D D16=$D16 plan=$PLAN freeze=${FREEZE:-none} freeze_rc=$FZ_RC"; exit 0; }
mkdir -p "$D" "$D16"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[timeval2] arm=$ARM step=$STEP STU=$STU TEACHER=$TEACHER D=$D D16=$D16 plan=$PLAN freeze=${FREEZE:-none} job=$SLURM_JOB_ID"
RC=0
# ---- stage A: tim_eval.sbatch's pass, unchanged (t3_acc.py gen builds its engine BEFORE checking for its output, so
#      skip the call when the traces exist: same result, no engine start)
t=$SECONDS
if [ -s "$D/gen_T1_p1.jsonl" ]; then echo "[timeval2] A gen exists"; else
    $CLUSTER_ROOT/venv-refresh/bin/python refresh/t3_acc.py gen --model "$STU" --tokenizer "$BASE" --out-dir "$D" --temps 1.0:1.0
    rc=$?; echo "[timeval2] A gen rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
fi
t=$SECONDS
$PYTHON_BIN refresh/t3_acc.py grade --out-dir "$D" --tokenizer "$BASE"
rc=$?; echo "[timeval2] A grade rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
for ROLE in student teacher; do
    [ -s "$D/lp_$ROLE.npz" ] && { echo "[timeval2] A lp_$ROLE exists"; continue; }
    M=$STU; [ $ROLE = teacher ] && M=$TEACHER
    t=$SECONDS
    $PYTHON_BIN bench/tim_kl_eval.py score --model "$M" --tokenizer "$BASE" --gen "$D/gen_T1_p1.jsonl" --out "$D/lp_$ROLE.npz"
    rc=$?; echo "[timeval2] A score $ROLE rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
done
if [ -s "$D/kl_summary.json" ]; then echo "[timeval2] A kl_summary exists"; else
    $PYTHON_BIN bench/tim_kl_eval.py combine --dir "$D"; rc=$?; [ $rc -ne 0 ] && RC=$rc
fi
# ---- stage B: avg@16 benches
t=$SECONDS
$CLUSTER_ROOT/venv-refresh/bin/python refresh/t3_acc2.py gen --model "$STU" --tokenizer "$BASE" --out-dir "$D16" --plan "$PLAN" \
    ${_FZ[@]+"${_FZ[@]}"}
rc=$?; echo "[timeval2] B gen rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
t=$SECONDS
$PYTHON_BIN refresh/t3_acc2.py grade --out-dir "$D16" --tokenizer "$BASE"
rc=$?; echo "[timeval2] B grade rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
REUSE=""; case ",$PLAN," in *,AIME24:0-*) ;; *,AIME24:*) REUSE="--reuse-dir $D";; esac
$PYTHON_BIN refresh/t3_acc2.py summarize --out-dir "$D16" $REUSE ${_FZ[@]+"${_FZ[@]}"}
rc=$?; echo "[timeval2] B summarize rc=$rc"; [ $rc -ne 0 ] && RC=$rc
ls -la "$D" "$D16"
echo "[timeval2] done arm=$ARM step=$STEP rc=$RC"
exit $RC

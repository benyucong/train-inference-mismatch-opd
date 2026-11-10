#!/bin/bash
#SBATCH --job-name=probe0
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.log
# The exact gradient probe (tools/probe0/probe0.py), HF transformers + torch only, one GH200. Submit from the
# repository root (Slurm writes the log into logs/slurm there, so create it once with `mkdir -p logs/slurm`), e.g.
#   FULL :  SIGMAS=nhigh=0.15 A_TRACES=<a.jsonl> B_TRACES=<dir>/gen_T1_p1.jsonl \
#               sbatch --parsable tools/probe0/probe.sbatch
#   SMOKE:  MODE=smoke SIGMAS=nhigh=0.15 A_TRACES=<a.jsonl> B_TRACES=<dir>/gen_T1_p1.jsonl \
#               sbatch --parsable --partition=PARTITION_DEBUG --time=00:15:00 tools/probe0/probe.sbatch
# env: SIGMAS (required: probe0.py --sigmas, comma-separated name=sigma pairs, one noisy sampler per pair; the bias gate
#      reads the last pair, and bench/prefix_shift.py repro compares with the level at sigma 0.15), A_TRACES
#      (probe0.py --a-traces), B_TRACES (--b-traces), EXPECTED_STEP0 (optional, --expected-step0: JSON of a training
#      run's step-0 [SPECOPD-TIM] line), OUT_DIR (optional; default $CLUSTER_ROOT/probe0/<job id>), EXTRA (more
#      probe0.py arguments, e.g. EXTRA="--tim-ref verl/verl/utils/specopd_tim.py" to check that the probe's copy of
#      the noise code is byte-identical to the training code's).
# A_TRACES (also bench/prefix_shift.sbatch's PS_REPRO_TRACES; no script in this repository writes it): JSONL, one
#   response of the base student per line, with trace_id (unique), prompt_idx, prompt_ids, response_ids,
#   finish_reason and seed. Recipe: 64 distinct DAPO-Math-17k prompts (a seeded uniform sample; prompt_idx = the row
#   in $CLUSTER_ROOT/workspace/datasets/dapo-math-17k.parquet); prompt_ids = the base student's chat template applied
#   to the row's prompt with add_generation_prompt=True, tokenized with add_special_tokens=False (they end with the
#   ids 151645 151648 198: the assistant tag, <think> and a newline); one response per prompt, sampled by vLLM from
#   the base student in bf16 at T=1.0, top_p=1.0, at most 16384 new tokens, request seed 1000 + prompt_idx (the seed
#   field), with EAGLE-3 speculative decoding and 3 draft tokens (lossless: plain sampling draws from the same
#   distribution, though not the same tokens); response_ids (the final EOS token included) and finish_reason
#   ("stop" or "length") as vLLM returns them.
# B_TRACES: the gen_T1_p1.jsonl that, on a GPU node from the repository root (after source scripts/cluster_env.sh),
#     $CLUSTER_ROOT/venv-refresh/bin/python refresh/t3_acc.py gen --model <base student dir> \
#         --tokenizer <base student dir> --out-dir <dir> --temps 1.0:1.0
#   writes into <dir>: AIME24 x 8 and MATH-500 x 1 samples of the base student, no speculative decoding, T=1.0,
#   top_p=1.0; at most 16384 new tokens and base seed 2000 are t3_acc.py's defaults.
# MODE=smoke: 2 part-(a) traces + 1 AIME problem x 2 samples, every response truncated to 1024 tokens, identity on
# 1 trace x 256 tokens; it is sized for --time=00:15:00. MODE=full: every part-(a) trace, AIME24 problems 0-7 x 8
# samples (part b), identity on 2 traces x 1024 tokens. Output: probe0.json (+ rows_*.npz) in the output dir.
# Memory: CPU fp32 gradient accumulators of 30.5 GB each, at most 4 alive (--groups level), plus the teacher and
# sampler hidden-state caches in host RAM.
# The CUDA noise draw depends on the GPU type: run on the GPU type that trains, so that the noise record can be
# checked against the training run's step-0 line (EXPECTED_STEP0).
set -uo pipefail
# sbatch runs a spooled copy of this script, so the submit dir (not the script's own path) locates the repository
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2; export REPO_ROOT="$PWD"
P=tools/probe0
[ -f $P/probe0.py ] && [ -f $P/probe_lib.py ] && [ -f $P/specopd_tim.py ] \
    || { echo "[probe0] FATAL: submit from the repository root ($PWD has no $P)"; exit 2; }
source scripts/cluster_env.sh
: "${SIGMAS:?set SIGMAS (probe0.py --sigmas: comma-separated name=sigma pairs, e.g. SIGMAS=nhigh=0.15)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
MODE=${MODE:-full}
OUT=${OUT_DIR:-$CLUSTER_ROOT/probe0/${SLURM_JOB_ID}}
mkdir -p "$OUT" || { echo "[probe0] FATAL: cannot create $OUT"; exit 2; }
sha256sum $P/probe0.py $P/probe_lib.py $P/specopd_tim.py > "$OUT/source_sha256.txt"
cp $P/probe0.py $P/probe_lib.py $P/specopd_tim.py "$OUT/" 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "[probe0] mode=$MODE out=$OUT host=$(hostname) sigmas=$SIGMAS extra=${EXTRA:-}"
echo "[probe0] a=${A_TRACES:-} b=${B_TRACES:-} expected_step0=${EXPECTED_STEP0:-}"
"$PYTHON_BIN" -u $P/probe0.py --mode "$MODE" --out-dir "$OUT" --sigmas "$SIGMAS" ${A_TRACES:+--a-traces "$A_TRACES"} \
    ${B_TRACES:+--b-traces "$B_TRACES"} ${EXPECTED_STEP0:+--expected-step0 "$EXPECTED_STEP0"} ${EXTRA:-}
rc=$?; echo "[probe0] done rc=$rc"; exit $rc

#!/bin/bash
#SBATCH --job-name=prefix_shift
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# The prefix-shift measurement (bench/prefix_shift.py): per-token mismatch against the prefix shift. One GPU, no
# training. Submit from the repository root (Slurm writes the log into logs/slurm there: mkdir -p logs/slurm once).
# Variables (exported; sbatch passes the submit environment to the job):
#   PS_JOB      smoke | vllm | hf
#   PS_CKPT     theta0 (the base student) | theta200 (the trained checkpoint); vllm and hf jobs
#   PS_ROOT     the experiment's output root. Each job writes into D = $PS_ROOT/smoke (smoke) or $PS_ROOT/$PS_CKPT
#               (vllm, hf; the hf job reads what the vllm job of its checkpoint wrote there), made here if absent.
#               Setting D overrides it, but the hf job's reader still looks in exactly those places. The hf job
#               writes the combined reading into $PS_ROOT.
#   PS_LIMIT_S  optional: this job's time limit in seconds, for the phases' deadline guards; unset = read from Slurm
#               (squeue's time limit of this job)
#   PS_GAP_DIR  (smoke, vllm) gap_probe's main output dir: prefixes.json, scales_base.json, fp8_base_auto.json
#   PS_HO_DIR   (smoke, vllm theta0) ho_eval's output dir for the base student: samples.jsonl, meta.json
#   PS_THETA200_CKPT  (smoke, vllm theta200) the trained checkpoint's HF dir
#   PS_REPRO_TRACES   (smoke) the exact gradient probe's (tools/probe0) part-(a) traces, the file given to its
#               --a-traces (A_TRACES in tools/probe0/probe.sbatch, which also documents it). No script in this
#               repository writes it: JSONL, one response of the base student per line, with trace_id, prompt_idx,
#               prompt_ids, response_ids, finish_reason and seed, for 64 distinct DAPO-Math-17k prompts (a seeded
#               uniform sample; prompt_idx = the row in $CLUSTER_ROOT/workspace/datasets/dapo-math-17k.parquet);
#               prompt_ids = the base student's chat template with add_generation_prompt=True, tokenized with
#               add_special_tokens=False (ending in the ids 151645 151648 198); one vLLM response per prompt, bf16,
#               T=1.0, top_p=1.0, at most 16384 new tokens, request seed 1000 + prompt_idx (the seed field), with
#               EAGLE-3 speculative decoding and 3 draft tokens (lossless: plain sampling draws from the same
#               distribution, though not the same tokens); response_ids and finish_reason as vLLM returns them
#   PS_PROBE0_JSON    (smoke, optional) probe0.json of a full-mode probe run on those same traces, with a noise
#               level at sigma 0.15 (e.g. SIGMAS=nhigh=0.15); absent = the reproduction is not compared with it
#   PS_EXPECTED_STEP0 (optional) JSON with the training log's step-0 [SPECOPD-TIM] fields; absent = not checked
# Jobs, in this order (each line from the repository root, with the time limit the job is sized for):
#   1. the smoke, both checkpoints at 4 prompts x 2 responses; the main reading gates on its result
#      ($PS_ROOT/smoke/reading.txt), so check that it passed before the other jobs:
#        PS_JOB=smoke PS_ROOT=<root> PS_GAP_DIR=<gap dir> PS_HO_DIR=<ho dir> PS_THETA200_CKPT=<ckpt dir> \
#            PS_REPRO_TRACES=<traces.jsonl> PS_PROBE0_JSON=<probe0.json> sbatch --parsable bench/prefix_shift.sbatch
#   2. the vllm jobs of theta0 and theta200 (they may run side by side):
#        J0=$(PS_JOB=vllm PS_CKPT=theta0 PS_ROOT=<root> PS_GAP_DIR=<gap dir> PS_HO_DIR=<ho dir> \
#            sbatch --parsable --time=03:00:00 bench/prefix_shift.sbatch)
#        J1=$(PS_JOB=vllm PS_CKPT=theta200 PS_ROOT=<root> PS_GAP_DIR=<gap dir> PS_THETA200_CKPT=<ckpt dir> \
#            sbatch --parsable --time=03:30:00 bench/prefix_shift.sbatch)
#   3. the hf jobs, each after its own checkpoint's vllm job:
#        PS_JOB=hf PS_CKPT=theta0 PS_ROOT=<root> sbatch --parsable --time=04:30:00 --dependency=afterok:$J0 \
#            bench/prefix_shift.sbatch
#        PS_JOB=hf PS_CKPT=theta200 PS_ROOT=<root> sbatch --parsable --time=04:30:00 --dependency=afterok:$J1 \
#            bench/prefix_shift.sbatch
#      The first hf job to finish reads the one checkpoint it has (exit 5 of the reader: the other is still
#      missing); the second writes the full reading, $PS_ROOT/reading_after_<job id>.txt and .json.
# What each job runs:
#   0. node record (driver, CUDA, GPU) and the CPU tests into $D
#   vllm: plan; gen std (theta200: P, P2; both: the noise reloads Qn015, Qn0325), gen f8all, gen f8s0;
#         fp8score f8all, f8s0, floor (caching on, then the caching-off check; a failed check rescores without caching)
#   hf:   the HF pass (noisy caches, sketches, per-row measures, Gram), then the reader
#   smoke: the exact gradient probe's reproduction (first, with a deadline guard), then every vllm and hf phase for
#         both checkpoints at 4 prompts x 2 responses, 1,024 tokens, the exact-vs-sketch check and the full-length
#         memory probe at theta0, the smoke reader
#   The f8s0 engines (gen and fp8score) install g0_kernels' FA3 skipped-layer patch in the worker and refuse unless it
#   reports installed; its counters go into the set and score records.
# Every phase is resumable: an output that exists and verifies is not recomputed.
# Exit: 0 = every phase ran; 2 bad env; else the first failing phase's rc.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 2; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${PS_JOB:?set PS_JOB=smoke|vllm|hf}" "${PS_ROOT:?set PS_ROOT}"
case "$PS_JOB" in
    smoke) ;;
    vllm|hf) : "${PS_CKPT:?set PS_CKPT=theta0|theta200}"
             case "$PS_CKPT" in theta0|theta200) ;; *) echo "[ps] PS_CKPT=$PS_CKPT"; exit 2 ;; esac ;;
    *) echo "[ps] PS_JOB=$PS_JOB"; exit 2 ;;
esac
case "$PS_JOB" in
    smoke) : "${PS_GAP_DIR:?set PS_GAP_DIR}" "${PS_HO_DIR:?set PS_HO_DIR}" "${PS_THETA200_CKPT:?set PS_THETA200_CKPT}" \
               "${PS_REPRO_TRACES:?set PS_REPRO_TRACES}" ;;
    vllm) : "${PS_GAP_DIR:?set PS_GAP_DIR}"
          if [ "$PS_CKPT" = theta0 ]; then : "${PS_HO_DIR:?set PS_HO_DIR}"; else : "${PS_THETA200_CKPT:?set PS_THETA200_CKPT}"; fi ;;
esac
# the job's output dir: the smoke's dir, or the checkpoint's dir that its vllm and hf jobs share (the reader below
# looks for them at exactly these places)
if [ -z "${D:-}" ]; then
    if [ "$PS_JOB" = smoke ]; then D=$PS_ROOT/smoke; else D=$PS_ROOT/$PS_CKPT; fi
fi
if [ "$PS_JOB" = hf ] && [ ! -f "$D/plan_record.json" ]; then
    echo "[ps] D=$D has no plan_record.json: run the vllm job of $PS_CKPT first"; exit 2
fi
mkdir -p "$D" || { echo "[ps] cannot create D=$D"; exit 2; }
slurm_seconds() {   # squeue's time format ([days-]hours:minutes:seconds or minutes:seconds) -> seconds
    local re='^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$'
    [[ $1 =~ $re ]] || return 1
    echo $(( ((10#${BASH_REMATCH[2]:-0} * 24 + 10#${BASH_REMATCH[4]:-0}) * 60 + 10#${BASH_REMATCH[5]}) * 60 \
             + 10#${BASH_REMATCH[6]} ))
}
if [ -z "${PS_LIMIT_S:-}" ]; then   # this job's own time limit
    tl=$(squeue -h -j "${SLURM_JOB_ID:?}" -o %l 2>/dev/null | head -1)
    PS_LIMIT_S=$(slurm_seconds "$tl") || { echo "[ps] cannot read this job's time limit ('$tl'): set PS_LIMIT_S"; exit 2; }
fi
END=$(( $(date +%s) + PS_LIMIT_S - 180 ))      # the job's end, less 3 min for the reader
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton" TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
unset SPECOPD_SAMPLER_NOISE SPECOPD_SAMPLER_NOISE_SEED SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL \
      VLLM_ATTENTION_BACKEND VLLM_DISABLE_COMPILE_CACHE KV_DTYPE
echo "[ps] job=$SLURM_JOB_ID node=$(hostname) job=$PS_JOB ckpt=${PS_CKPT:-both} D=$D limit=${PS_LIMIT_S}s repo=$REPO_ROOT"
md5sum bench/prefix_shift.py bench/grad_sketch.py bench/prefix_shift_read.py bench/prefix_shift.sbatch bench/gap_probe.py \
       bench/probe0_lib.py bench/tim_calibrate.py bench/g0_kernels.py verl/verl/utils/specopd_tim.py \
       verl/verl/utils/specopd_quant.py scripts/cluster_env.sh \
       | tee "$D/source_md5_${SLURM_JOB_ID}.txt"

# 0. the node's driver, CUDA and GPU, for the record; the newer nvidia-smi prints "CUDA UMD Version:"
cuda_of() { sed -nE 's/.*CUDA (UMD )?Version: *([0-9.]+).*/\2/p' | head -1; }
smi=$(nvidia-smi 2>&1)
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
vbios=$(nvidia-smi --query-gpu=vbios_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(printf '%s\n' "$smi" | cuda_of)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "vbios ${vbios:-?}"; echo "kernel $(uname -r)"
  echo "node $(hostname)"; echo "date $(date -Is)"; echo "gpu $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
} > "$D/node_gpu_${SLURM_JOB_ID}.txt"
cat "$D/node_gpu_${SLURM_JOB_ID}.txt"
# the CPU tests (closed forms against autograd); the readers gate on the line "CPU TESTS PASSED" in $D/cpu_tests.txt
{ "$PYTHON_BIN" bench/prefix_shift.py selfcheck && "$PYTHON_BIN" bench/gap_probe.py selfcheck \
    && echo "CPU TESTS PASSED"; } > "$D/cpu_tests.txt" 2>&1
cat "$D/cpu_tests.txt"

RC=0
note() { echo "[ps] $*"; }
run() {   # run <tag> <prefix_shift.py args...>; the first failure sets RC
    local tag=$1 t=$SECONDS rc; shift
    "$PYTHON_BIN" bench/prefix_shift.py "$@"; rc=$?
    note "$tag rc=$rc ($((SECONDS - t))s)"; [ $rc -ne 0 ] && [ $RC -eq 0 ] && RC=$rc
    return $rc
}
step0=(); [ -n "${PS_EXPECTED_STEP0:-}" ] && step0=(--expected-step0 "$PS_EXPECTED_STEP0")
vllm_phases() {   # <dir> <ckpt> <mode>
    local d=$1 ck=$2 mode=$3 e inputs=(--gap-dir "$PS_GAP_DIR")
    case "$ck" in theta0) inputs+=(--ho-dir "$PS_HO_DIR") ;; theta200) inputs+=(--ckpt-path "$PS_THETA200_CKPT") ;; esac
    mkdir -p "$d"
    run "plan $ck" plan --out-dir "$d" --mode "$mode" --ckpt "$ck" "${inputs[@]}" ${step0[@]+"${step0[@]}"} || return 1
    for e in std f8all f8s0; do run "gen $ck $e" gen --out-dir "$d" --mode "$mode" --engine "$e" --deadline "$((END - 60))"; done
    for e in f8all f8s0 floor; do
        run "fp8score $ck $e on" fp8score --out-dir "$d" --mode "$mode" --engine "$e" --cache on --deadline "$((END - 60))"
        run "fp8score $ck $e off" fp8score --out-dir "$d" --mode "$mode" --engine "$e" --cache off --deadline "$((END - 60))"
    done
}

case "$PS_JOB" in
    smoke)   # the exact gradient probe's reproduction first: it needs no vLLM output, and its own deadline guard
             # stops it in time
        ref=(); [ -n "${PS_PROBE0_JSON:-}" ] && ref=(--reference "$PS_PROBE0_JSON")
        run repro repro --out-dir "$D/repro" --traces "$PS_REPRO_TRACES" --gap-dir "$PS_GAP_DIR" \
            ${ref[@]+"${ref[@]}"} ${step0[@]+"${step0[@]}"} --deadline "$((END - 600))"
        vllm_phases "$D/theta0" theta0 smoke
        vllm_phases "$D/theta200" theta200 smoke
        run "hf theta0" hf --out-dir "$D/theta0" --mode smoke --exact-check 1 --mem-probe 1 --deadline "$((END - 600))"
        run "hf theta200" hf --out-dir "$D/theta200" --mode smoke --deadline "$((END - 420))"
        t=$SECONDS
        "$PYTHON_BIN" bench/prefix_shift_read.py smoke --smoke "$D" --json "$D/reading.json" > "$D/reading.txt" 2>&1; rc=$?
        note "read smoke rc=$rc ($((SECONDS - t))s)"; [ $rc -ne 0 ] && [ $RC -eq 0 ] && RC=$rc
        cat "$D/reading.txt" ;;
    vllm)
        vllm_phases "$D" "$PS_CKPT" main ;;
    hf)
        run "hf $PS_CKPT" hf --out-dir "$D" --mode main --deadline "$((END - 300))"
        args=(main --smoke "$PS_ROOT/smoke" --json "$PS_ROOT/reading_after_${SLURM_JOB_ID}.json")
        for ck in theta0 theta200; do [ -s "$PS_ROOT/$ck/gram.npz" ] && args+=(--"$ck" "$PS_ROOT/$ck"); done
        t=$SECONDS
        "$PYTHON_BIN" bench/prefix_shift_read.py "${args[@]}" > "$PS_ROOT/reading_after_${SLURM_JOB_ID}.txt" 2>&1; rc=$?
        note "read rc=$rc ($((SECONDS - t))s; 5 = one checkpoint still missing)"
        tail -80 "$PS_ROOT/reading_after_${SLURM_JOB_ID}.txt" ;;
esac
ls -la "$D"
note "done job=$PS_JOB rc=$RC"
exit $RC

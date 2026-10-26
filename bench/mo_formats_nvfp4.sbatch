#!/bin/bash
#SBATCH --job-name=mo_formats_nvfp4
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# The NVFP4 job: the three NVFP4 configurations of the formats job (mo_common.NVFP4_LABELS), measured on their own in
# their own D. The two emulated configurations run compiled without CUDA graphs, because vLLM's NVFP4 emulation cannot
# be captured in CUDA graphs; the Marlin configuration runs with vLLM's own backend selection, because verl's NVFP4
# patch needs the modelopt package. ONE GH200. It reuses the formats job's calibration set and NVFP4 export UNCHANGED
# (hashes checked first) and writes only to its own D:
#   0. node record -> D/node_gpu.txt
#   1. the calibration set's sha256 = its record; every file of BASE/ckpt/nvfp4_kv = the export record's md5; the export
#      record's calibration hash = the calibration set's
#   2. mo_common.NVFP4_LABELS, one process each (mo_measure.py q, --ckpt-root BASE/ckpt): the two emulated
#      configurations compiled WITHOUT CUDA graphs (cudagraph_mode NONE; torch.compile kept), Marlin with vLLM's own
#      backend selection (VLLM_NVFP4_GEMM_BACKEND=marlin, verl's NVFP4 patch replaced by the recorder); a configuration
#      starts only while >= 25 min of the job's 90 min remain
#   3. a clean bf16 engine scores them -> D/dec_<label>.json / .npz
#   4. the reader, with BASE filling in the floor and the unit-scale cache -> D/reading.json / reading.txt
# Submit from the repository root with D (created empty), BASE (the formats job's D) and CALSET exported, and
# optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the reader's anchors, see
# bench/mo_formats_read.py --anchors).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${CALSET:?set CALSET}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[nvfp4] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS; LIMIT_S=$((90 * 60)); RESERVE_S=$((25 * 60))
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_NOISE VLLM_DISABLE_COMPILE_CACHE \
      VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM SAMPLER_MODELOPT \
      SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL
echo "[nvfp4] job=$SLURM_JOB_ID node=$(hostname) D=$D BASE=$BASE CALSET=$CALSET repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_common.py bench/mo_measure.py bench/mo_formats_read.py bench/mo_formats_nvfp4.sbatch

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"

"$PYTHON_BIN" - "$CALSET" "$BASE/ckpt/nvfp4_kv" > "$D/inputs_check.txt" 2>&1 <<'PY'
import json, os, sys
sys.path.insert(0, "bench")
import mo_calset, mo_common as C
rows, crec = mo_calset.load_calset(sys.argv[1])                 # refuses on a sha256 mismatch
ex = sys.argv[2]
rec = json.load(open(os.path.join(ex, "export_record.json")))
bad = [f for f, h in rec["files"].items() if C.md5_file(os.path.join(ex, f)) != h]
assert not bad, f"export files differ from the record: {bad}"
assert rec["calset_sha256"] == crec["calset_sha256"], "the export was calibrated on another calibration set"
assert str(rec["quantization_config"]["quant_algo"]).upper() == "NVFP4"
print(f"[nvfp4] inputs ok: calibration set {crec['calset_sha256'][:16]} ({len(rows)} responses); nvfp4_kv export "
      f"{len(rec['files'])} files = its record; calibrated on that set")
PY
rc=$?; cat "$D/inputs_check.txt"
if [ $rc -ne 0 ]; then echo "[nvfp4] input check FAILED: NOTHING RUN"; exit 4; fi

RC=0; labels=""
for label in $("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(' '.join(C.NVFP4_LABELS))"); do
    left=$((LIMIT_S - (SECONDS - T0)))
    if [ $left -lt $RESERVE_S ]; then
        echo "skip: time (${left}s left)" > "$D/rc_$label.txt"; echo "[nvfp4] config $label SKIPPED (time)"; continue
    fi
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --ckpt-root "$BASE/ckpt" --out-dir "$D" > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|\[cal\] decode\|Traceback\|Error" "$D/log_$label.txt" | cut -c1-700 | tail -6
    echo "[nvfp4] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done

if [ -n "$labels" ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"
    grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    echo "[nvfp4] score $labels rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
fi
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" --base-dir "$BASE" ${ANCHORS:+--anchors "$ANCHORS"} \
    --json "$D/reading.json" > "$D/reading.txt" 2>&1
cat "$D/reading.txt"
du -sh "$D" 2>/dev/null
echo "[nvfp4] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

#!/bin/bash
#SBATCH --job-name=mo_cal1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.log
# The KV-cache calibration configurations (mo_common.CAL1). ONE GH200. This job measures cal_c2 (bf16 weights + Model
# Optimizer's calibrated per-layer FP8 cache), cal_c2_s0 (cal_c2 with layer 0's cache bf16: its first-layer share),
# cal_c3 (llm-compressor's per-head FP8 cache), cal_c4 (an FP8 cache with vLLM's own scales, calculate_kv_scales), cal_c8
# (cal_c2 with the first and last layers' cache bf16) and, descriptive, cal_c2_llmc (cal_c2 calibrated by llm-compressor
# per tensor); the floor, the unit-scale cache and the Model Optimizer FP8 configurations come from the formats job. It
# reuses the formats job's calibration set and its fp8_kv export's KV scales UNCHANGED (hashes checked first) and
# writes only to its own D:
#   0. node record -> D/node_gpu.txt
#   1. the calibration set's sha256 = its record; every file of BASE/ckpt/fp8_kv = the export record's md5
#   2. venv-llmc: bench/llmc_kv_export.py -- llm-compressor's KV-only FP8 calibration on the same set, per head
#      (cal_c3) -> D/ckpt/llmc_kv_head, then per tensor (cal_c2_llmc) -> D/ckpt/llmc_kv_tensor (only while >= 30 min
#      remain)
#   3. mo_common.CAL1, one process each (mo_measure.py q --ckpt-root D/ckpt --export-root BASE/ckpt), tim_calibrate's
#      engine (compiled, FULL_AND_PIECEWISE CUDA graphs); a configuration starts only while >= 8 min remain
#   4. a clean bf16 engine scores them -> D/dec_<label>.json / .npz
#   5. the reader with the formats job as the base run (the floor, the unit-scale cache, mo_fp8_kv and its layer-0
#      variant, mo_fp8_kv_fl) -> D/reading.json / reading.txt, including the summary of the KV-cache configurations
# Submit from the repository root with D (created empty), BASE (the formats job's D) and CALSET exported, and
# optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the reader's anchors, see
# bench/mo_formats_read.py --anchors).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${CALSET:?set CALSET}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[cal1] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS; LIMIT_S=$((60 * 60)); RESERVE_S=$((8 * 60))
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL
LLMC_PY=$CLUSTER_ROOT/venv-llmc/bin/python
echo "[cal1] job=$SLURM_JOB_ID node=$(hostname) D=$D BASE=$BASE CALSET=$CALSET repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_common.py bench/mo_measure.py bench/mo_formats_read.py bench/llmc_kv_export.py \
       bench/mo_cal1.sbatch

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"

"$PYTHON_BIN" - "$CALSET" "$BASE/ckpt/fp8_kv" > "$D/inputs_check.txt" 2>&1 <<'PY'
import json, os, sys
sys.path.insert(0, "bench")
import mo_calset, mo_common as C
rows, crec = mo_calset.load_calset(sys.argv[1])                 # refuses on a sha256 mismatch
ex = sys.argv[2]
rec = json.load(open(os.path.join(ex, "export_record.json")))
bad = [f for f, h in rec["files"].items() if C.md5_file(os.path.join(ex, f)) != h]
assert not bad, f"export files differ from the record: {bad}"
assert rec["calset_sha256"] == crec["calset_sha256"], "the export was calibrated on another calibration set"
assert len(rec["kv_scales"]) == C.N_LAYERS, "the export has no KV scale for every layer"
print(f"[cal1] inputs ok: calibration set {crec['calset_sha256'][:16]} ({len(rows)} responses); fp8_kv export "
      f"{len(rec['files'])} files = its record; {len(rec['kv_scales'])} layers' KV scales")
PY
rc=$?; cat "$D/inputs_check.txt"
if [ $rc -ne 0 ]; then echo "[cal1] input check FAILED: NOTHING RUN"; exit 4; fi

mkdir -p "$D/ckpt"
for name in llmc_kv_head llmc_kv_tensor; do
    left=$((LIMIT_S - (SECONDS - T0)))
    if [ "$name" = llmc_kv_tensor ] && [ $left -lt $((30 * 60)) ]; then
        echo "skip: time (${left}s left)" > "$D/rc_export_$name.txt"; echo "[cal1] export $name SKIPPED (time)"; continue
    fi
    strat=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(C.LLMC_EXPORTS['$name'])")
    t=$SECONDS
    env -u PYTHONPATH "$LLMC_PY" -I bench/llmc_kv_export.py export --strategy "$strat" --calset-dir "$CALSET" \
        --out-dir "$D/ckpt/$name" > "$D/log_export_$name.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_export_$name.txt"
    grep -h "\[llmc_kv_export\|Traceback\|Error" "$D/log_export_$name.txt" | cut -c1-600 | tail -4
    echo "[cal1] export $name ($strat) rc=$rc ($((SECONDS-t))s)"
done

RC=0; labels=""
for label in $("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(' '.join(C.CAL1_LABELS))"); do
    ck=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(C.formats_dict()['$label']['ckpt'] or '')")
    if [ -n "$ck" ] && [ "$(cat "$D/rc_export_$ck.txt" 2>/dev/null)" != 0 ]; then
        echo "skip: export $ck not made" > "$D/rc_$label.txt"; echo "[cal1] config $label SKIPPED (no $ck)"; continue
    fi
    left=$((LIMIT_S - (SECONDS - T0)))
    if [ $left -lt $RESERVE_S ]; then
        echo "skip: time (${left}s left)" > "$D/rc_$label.txt"; echo "[cal1] config $label SKIPPED (time)"; continue
    fi
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --ckpt-root "$D/ckpt" --export-root "$BASE/ckpt" --out-dir "$D" \
        > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|\[cal\] decode\|Traceback\|Error" "$D/log_$label.txt" | cut -c1-700 | tail -6
    echo "[cal1] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done

if [ -n "$labels" ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"
    grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    echo "[cal1] score $labels rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
fi
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" --base-dir "$BASE" ${ANCHORS:+--anchors "$ANCHORS"} \
    --json "$D/reading.json" > "$D/reading.txt" 2>&1
cat "$D/reading.txt"
du -sh "$D" 2>/dev/null
echo "[cal1] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

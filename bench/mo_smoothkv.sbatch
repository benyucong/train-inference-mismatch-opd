#!/bin/bash
#SBATCH --job-name=mo_smoothkv
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:25:00
#SBATCH --output=logs/slurm/%x_%j.log
# One emulated recipe (RECIPE = int8_sq | int4_awq) WITH Model Optimizer's calibrated FP8 cache, and with layer 0's
# cache bf16 (the first-layer share). ONE GH200, no training, no calibration: it reuses the recipe's smoothemu static
# file UNCHANGED (SD; its bit-exact check must have passed) and the formats job's fp8_kv export's k / v scales (the
# cache of cal_c2); writes only to its own D:
#   0. node record -> D/node_gpu.txt
#   1. SD's check.json verdict BIT-EXACT and its static file loadable (tensors sha256), then bench/mo_export.py
#      smoothkv -> D/static_kv.json (the same smoothing + the export's cache scales, bf16)
#   2. the two configurations (mo_common.SMOOTH_KV[RECIPE]: mo_sqemu_kv + mo_sqemu_kv_s0, or the awqemu pair) as
#      nvfp4emu's: the bf16 student, the grid re-streamed through SamplerQuant, the file's k / v scales re-applied after
#      the re-stream and every KV wake-up, FULL_AND_PIECEWISE CUDA graphs
#   3. a clean bf16 engine scores them -> D/dec_<label>.json / .npz
#   4. the reader over D + SD + the formats job (floor, unit cache) -> D/reading.json / reading.txt (the bf16-cache row
#      too; the first-layer share of the FP8-cache configuration)
# Submit once per recipe from the repository root with D (created empty), BASE (the formats job's D), SD (the recipe's
# mo_smoothemu.sbatch D) and RECIPE exported, and optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95%
# intervals for the reader's anchors, see bench/mo_formats_read.py --anchors); set --job-name (mo_sqkv | mo_awqkv)
# and --time at submission.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${SD:?set SD}" "${RECIPE:?set RECIPE (int8_sq | int4_awq)}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[smoothkv] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS; LIMIT_S=$((25 * 60)); RESERVE_S=$((6 * 60))
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL \
      SPECOPD_FA3_KVSKIP_PATCH
echo "[smoothkv] job=$SLURM_JOB_ID node=$(hostname) RECIPE=$RECIPE D=$D BASE=$BASE SD=$SD repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_common.py bench/mo_export.py bench/mo_measure.py bench/mo_formats_read.py \
       bench/mo_smoothkv.sbatch verl/verl/utils/specopd_smooth.py verl/verl/utils/specopd_quant.py

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"

for r in "$RECIPE"; do
    "$PYTHON_BIN" - "$SD" <<'PY' >> "$D/inputs_check.txt" 2>&1 || { cat "$D/inputs_check.txt"; echo "[smoothkv] $r: input check FAILED: NOTHING RUN"; exit 4; }
import json, os, sys
sys.path.insert(0, "verl")
from verl.utils import specopd_quant as sq
d = sys.argv[1]
chk = json.load(open(os.path.join(d, "check.json")))
assert chk["verdict"] == "BIT-EXACT", f"{d}: check verdict {chk['verdict']}"
assert open(os.path.join(d, "rc_check.txt")).read().strip() == "0"
st = sq.load_static(os.path.join(d, "static.json"))       # format, grid, narrow range, tensors sha256
assert not st.get("kv"), "the smoothemu static file already has a cache"
print(f"[smoothkv] {d}: check BIT-EXACT (Model Optimizer path {chk.get('modelopt_int_path')}), static file "
      f"{st['_sha256'][:16]} grid {st['grid']}")
PY
    "$PYTHON_BIN" bench/mo_export.py smoothkv --static "$SD/static.json" --export-dir "$BASE/ckpt/fp8_kv" \
        --static-out "$D/static_kv.json" >> "$D/inputs_check.txt" 2>&1 \
        || { cat "$D/inputs_check.txt"; echo "[smoothkv] $r: smoothkv FAILED: NOTHING RUN"; exit 5; }
done
cat "$D/inputs_check.txt"

RC=0; labels=""
for r in "$RECIPE"; do
    for label in $("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(' '.join(C.SMOOTH_KV['$r']))"); do
        left=$((LIMIT_S - (SECONDS - T0)))
        if [ $left -lt $RESERVE_S ]; then
            echo "skip: time (${left}s left)" > "$D/rc_$label.txt"; echo "[smoothkv] config $label SKIPPED (time)"; continue
        fi
        t=$SECONDS
        "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --static "$D/static_kv.json" --out-dir "$D" > "$D/log_$label.txt" 2>&1
        rc=$?; echo "$rc" > "$D/rc_$label.txt"
        grep -h "\[mo_measure\|\[cal\] decode\|\[cal\] weights\|SPECOPD-QUANT-CFG\] INT\|SPECOPD-QUANT-AUDIT\|Traceback\|Error" \
            "$D/log_$label.txt" | cut -c1-700 | tail -8
        echo "[smoothkv] config $label rc=$rc ($((SECONDS-t))s)"
        if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
    done
done

if [ -n "$labels" ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"
    grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    echo "[smoothkv] score $labels rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
fi
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" --base-dir "$SD,$BASE" ${ANCHORS:+--anchors "$ANCHORS"} \
    --json "$D/reading.json" > "$D/reading.txt" 2>&1
cat "$D/reading.txt"
du -sh "$D" 2>/dev/null
echo "[smoothkv] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

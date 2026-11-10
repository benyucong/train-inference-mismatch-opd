#!/bin/bash
#SBATCH --job-name=mo_nvfp4emu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.log
# The EMULATED NVFP4 sampler. ONE GH200, no training. It reuses the formats job's calibration set and NVFP4 export
# UNCHANGED (hashes checked first) and writes only to its own D:
#   0. node record -> D/node_gpu.txt
#   1. the calibration set's sha256 = its record; every file of BASE/ckpt/nvfp4_kv = the export record's md5
#   2. venv-modelopt: bench/mo_export.py amax -- Model Optimizer's NVFP4 calibration again (same model load, set, loop),
#      no export -> D/nvfp4_static.json (the step-0 activation amax + the export's KV scales as vLLM serves them, and a
#      cross-check of this calibration against the export) and D/nvfp4_capture.pt (inputs + Model Optimizer's own
#      fake-quantized outputs for layers 0, 13, 27)
#   3. venv-specopd (the training stack): bench/mo_nvfp4_check.py -- the vendored kernel, the custom op, the sync code
#      path, torch.compile and CUDA graphs == Model Optimizer bit for bit on the identical tensors; vLLM 0.19.1's own
#      NVFP4 emulation diagnosed on the same layers -> D/nvfp4_check.json. NOT BIT-EXACT -> nothing is measured
#   4. mo_common.NVFP4_EMU, one process each (mo_measure.py q --static D/nvfp4_static.json): the bf16 student, the
#      w=nvfp4 grid re-streamed through SamplerQuant as at a training sync, compiled with FULL_AND_PIECEWISE CUDA graphs;
#      a configuration starts only while >= 8 min of the job's 45 min remain
#   5. a clean bf16 engine scores them -> D/dec_<label>.json / .npz
#   6. the reader, with NV (mo_formats_nvfp4.sbatch's D) and BASE (the formats job's D) filling in the other
#      configurations
# Submit from the repository root with D (created empty), BASE (the formats job's D), NV (mo_formats_nvfp4.sbatch's D)
# and CALSET exported, and optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the
# reader's anchors, see bench/mo_formats_read.py --anchors). With D=$CLUSTER_ROOT/mo/nvfp4emu the job's static file
# D/nvfp4_static.json is the launcher's default SAMPLER_QUANT_STATIC for the emulated NVFP4 sampler
# ($CLUSTER_ROOT/mo/nvfp4emu/nvfp4_static.json, scripts/cluster_opd_sync.sh).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${NV:?set NV}" "${CALSET:?set CALSET}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[nvfp4emu] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS; LIMIT_S=$((45 * 60)); RESERVE_S=$((8 * 60))
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL
MO_PY=$CLUSTER_ROOT/venv-modelopt/bin/python
EXPORT=$BASE/ckpt/nvfp4_kv
echo "[nvfp4emu] job=$SLURM_JOB_ID node=$(hostname) D=$D BASE=$BASE NV=$NV CALSET=$CALSET repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_common.py bench/mo_export.py bench/mo_nvfp4_check.py bench/mo_measure.py \
       bench/mo_formats_read.py bench/mo_nvfp4emu.sbatch verl/verl/utils/specopd_nvfp4.py verl/verl/utils/specopd_quant.py

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"

"$PYTHON_BIN" - "$CALSET" "$EXPORT" > "$D/inputs_check.txt" 2>&1 <<'PY'
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
print(f"[nvfp4emu] inputs ok: calibration set {crec['calset_sha256'][:16]} ({len(rows)} responses); nvfp4_kv export "
      f"{len(rec['files'])} files = its record; calibrated on that set")
PY
rc=$?; cat "$D/inputs_check.txt"
if [ $rc -ne 0 ]; then echo "[nvfp4emu] input check FAILED: NOTHING RUN"; exit 4; fi

t=$SECONDS
env -u PYTHONPATH "$MO_PY" -I bench/mo_export.py amax --name nvfp4_kv --calset-dir "$CALSET" --export-dir "$EXPORT" \
    --static-out "$D/nvfp4_static.json" --capture-out "$D/nvfp4_capture.pt" > "$D/log_amax.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_amax.txt"
grep -h "\[mo_export\|Traceback\|Error" "$D/log_amax.txt" | grep -v "calibration [0-9]" | cut -c1-900 | tail -6
echo "[nvfp4emu] amax + capture rc=$rc ($((SECONDS-t))s)"
if [ $rc -ne 0 ]; then echo "[nvfp4emu] no static file: NOTHING MEASURED"; exit 5; fi

t=$SECONDS
"$PYTHON_BIN" bench/mo_nvfp4_check.py --capture "$D/nvfp4_capture.pt" --static "$D/nvfp4_static.json" \
    --export-dir "$EXPORT" --json "$D/nvfp4_check.json" > "$D/log_check.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_check.txt"
grep -h "\[mo_nvfp4_check\]\|Traceback\|Error" "$D/log_check.txt" | cut -c1-600 | tail -26
echo "[nvfp4emu] bit-exact check rc=$rc ($((SECONDS-t))s)"
if [ $rc -ne 0 ]; then echo "[nvfp4emu] the emulated grid is NOT bit-exact with Model Optimizer: NOTHING MEASURED"; exit 6; fi

RC=0; labels=""
for label in $("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(' '.join(C.NVFP4_EMU_LABELS))"); do
    left=$((LIMIT_S - (SECONDS - T0)))
    if [ $left -lt $RESERVE_S ]; then
        echo "skip: time (${left}s left)" > "$D/rc_$label.txt"; echo "[nvfp4emu] config $label SKIPPED (time)"; continue
    fi
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --static "$D/nvfp4_static.json" --out-dir "$D" > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|\[cal\] decode\|\[cal\] weights\|SPECOPD-QUANT-CFG\] NVFP4\|SPECOPD-QUANT-AUDIT\|Traceback\|Error" \
        "$D/log_$label.txt" | cut -c1-700 | tail -8
    echo "[nvfp4emu] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done

if [ -n "$labels" ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"
    grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    echo "[nvfp4emu] score $labels rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
fi
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" --base-dir "$NV,$BASE" --ckpt-root "$BASE/ckpt" \
    ${ANCHORS:+--anchors "$ANCHORS"} --json "$D/reading.json" > "$D/reading.txt" 2>&1
cat "$D/reading.txt"
du -sh "$D" 2>/dev/null
echo "[nvfp4emu] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

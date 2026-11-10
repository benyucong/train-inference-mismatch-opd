#!/bin/bash
#SBATCH --job-name=mo_smoothemu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# The EMULATED INT8 SmoothQuant / INT4 AWQ sampler, one recipe per job (RECIPE = int8_sq | int4_awq; set --job-name and
# --time per recipe at submission). ONE GH200, no training. It reuses the formats job's calibration set UNCHANGED (hash
# checked first) and writes only to its own D:
#   0. node record -> D/node_gpu.txt
#   1. the calibration set's sha256 = its record
#   2. venv-modelopt: bench/mo_export.py smooth -- Model Optimizer's preset (mtq.INT8_SMOOTHQUANT_CFG / INT4_AWQ_CFG)
#      calibrated on the set -> D/static.json + D/static.safetensors (the fixed smoothing, SmoothQuant's activation
#      amax) and D/capture.pt; records whether Model Optimizer ran its CUDA extension or the torch path
#   3. venv-specopd: bench/mo_smooth_check.py -- the emulation == Model Optimizer bit for bit on the identical tensors
#      (activation, smoothed weight, fresh-scale weight quantization, the sync code path, torch.compile, CUDA graphs)
#      -> D/check.json. NOT BIT-EXACT -> nothing is measured
#   4. the recipe's configuration (mo_common.SMOOTH_EMU: mo_sqemu | mo_awqemu; mo_measure.py q --static D/static.json):
#      the bf16 student, the grid re-streamed through SamplerQuant as at a training sync, FULL_AND_PIECEWISE CUDA graphs
#   5. a clean bf16 engine scores it -> D/dec_<label>.json / .npz
#   6. the reader with the formats job as the base run (floor, unit cache) -> D/reading.json / reading.txt
# Submit from the repository root with D (created empty), BASE (the formats job's D), CALSET and RECIPE exported, and
# optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the reader's anchors, see
# bench/mo_formats_read.py --anchors).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${CALSET:?set CALSET}" "${RECIPE:?set RECIPE (int8_sq | int4_awq)}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[smooth] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL \
      SPECOPD_FA3_KVSKIP_PATCH
MO_PY=$CLUSTER_ROOT/venv-modelopt/bin/python
LABEL=$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(C.SMOOTH_LABEL['$RECIPE'])") \
    || { echo "[smooth] unknown RECIPE $RECIPE"; exit 2; }
echo "[smooth] job=$SLURM_JOB_ID node=$(hostname) RECIPE=$RECIPE ($LABEL) D=$D BASE=$BASE CALSET=$CALSET repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_common.py bench/mo_export.py bench/mo_smooth_check.py bench/mo_measure.py \
       bench/mo_formats_read.py bench/mo_smoothemu.sbatch verl/verl/utils/specopd_smooth.py verl/verl/utils/specopd_quant.py

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"

"$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_calset; r, c = mo_calset.load_calset('$CALSET'); print('[smooth] calibration set', c['calset_sha256'][:16], len(r), 'responses = its record')" \
    > "$D/inputs_check.txt" 2>&1
rc=$?; cat "$D/inputs_check.txt"
if [ $rc -ne 0 ]; then echo "[smooth] input check FAILED: NOTHING RUN"; exit 4; fi

t=$SECONDS
env -u PYTHONPATH "$MO_PY" -I bench/mo_export.py smooth --recipe "$RECIPE" --calset-dir "$CALSET" \
    --static-out "$D/static.json" --tensors-out "$D/static.safetensors" --capture-out "$D/capture.pt" > "$D/log_smooth.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_smooth.txt"
grep -h "\[mo_export\|Loading extension\|Loaded extension\|could not\|Traceback\|Error" "$D/log_smooth.txt" | grep -v "calibration [0-9]" | cut -c1-900 | tail -8
echo "[smooth] calibration + capture rc=$rc ($((SECONDS-t))s)"
if [ $rc -ne 0 ]; then echo "[smooth] no static file: NOTHING MEASURED"; exit 5; fi

t=$SECONDS
"$PYTHON_BIN" bench/mo_smooth_check.py --capture "$D/capture.pt" --static "$D/static.json" --json "$D/check.json" \
    > "$D/log_check.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_check.txt"
grep -h "\[mo_smooth_check\]\|Traceback\|Error" "$D/log_check.txt" | cut -c1-500 | tail -26
echo "[smooth] bit-exact check rc=$rc ($((SECONDS-t))s)"
if [ $rc -ne 0 ]; then echo "[smooth] the emulated grid is NOT bit-exact with Model Optimizer: NOTHING MEASURED"; exit 6; fi

RC=0
t=$SECONDS
"$PYTHON_BIN" bench/mo_measure.py q --label "$LABEL" --static "$D/static.json" --out-dir "$D" > "$D/log_$LABEL.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_$LABEL.txt"
grep -h "\[mo_measure\|\[cal\] decode\|\[cal\] weights\|SPECOPD-QUANT-CFG\] INT\|SPECOPD-QUANT-AUDIT\|Traceback\|Error" \
    "$D/log_$LABEL.txt" | cut -c1-700 | tail -8
echo "[smooth] config $LABEL rc=$rc ($((SECONDS-t))s)"
if [ $rc -eq 0 ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$LABEL" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"
    grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    echo "[smooth] score $LABEL rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
else RC=$rc; fi
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" --base-dir "$BASE" ${ANCHORS:+--anchors "$ANCHORS"} \
    --json "$D/reading.json" > "$D/reading.txt" 2>&1
cat "$D/reading.txt"
du -sh "$D" 2>/dev/null
echo "[smooth] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

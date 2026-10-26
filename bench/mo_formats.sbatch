#!/bin/bash
#SBATCH --job-name=mo_formats
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.log
# The Model Optimizer formats job: which Model Optimizer formats vLLM 0.19.1 loads and serves on a GH200, and each
# one's offline mismatch on the student (responses sampled by the configuration's engine, re-scored by a clean bf16
# engine). ONE GH200, no training. Submit from the repository root with D (created empty), CALSET and CALSET_MODE
# (build | reuse) exported, and optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the
# reader's anchors, see bench/mo_formats_read.py --anchors).
#   0. node record: driver, CUDA, VBIOS and kernel versions -> D/node_gpu.txt
#   1. the calibration set (mo_calset.py): CALSET_MODE=build -> select (CPU) + generate (GPU, clean bf16 vLLM);
#      CALSET_MODE=reuse -> its sha256 is checked against its record
#   2. Model Optimizer exports (venv-modelopt, one process each): mo_common.EXPORTS (fp8_kv, fp8_kv_fl, nvfp4_kv)
#   3. the engine configurations (venv-specopd, one process each, tim_calibrate's decode path): mo_common.FORMATS, in
#      order; a configuration whose export failed is skipped; one is started only while >= 35 min remain. Model
#      Optimizer FP8 with a bf16 cache (mo_fp8_bf16kv) is not one of them: vLLM serves the fp8_kv export with the FP8
#      cache its config declares, so bench/mo_bf16kv.sbatch measures it on a copy of the export without that entry
#   3b. the sleep / wake probe (mo_measure.py sleepwake; descriptive, sleepwake.json): the fp8_kv engine's scales after
#      the build and after each step of verl's sleep / wake order
#   4. a CLEAN bf16 engine (separate process) scores every configuration's samples -> dec_<label>.json / .npz
#   5. the reader (CPU) -> reading.json / reading.txt (descriptive; validity per configuration)
# Exit: 0 = every step ran and the reading is READ; 2 bad D; 4 calset; else the failing step's rc.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${CALSET:?set CALSET}" "${CALSET_MODE:?set CALSET_MODE=build|reuse}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[mo] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS; LIMIT_S=$((4 * 3600)); RESERVE_S=$((35 * 60))
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_ALLOW_SKIP SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND KV_DTYPE KV_SKIP_LAYERS VLLM_USE_NVFP4_CT_EMULATIONS \
      VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM SAMPLER_MODELOPT
MO_PY=$CLUSTER_ROOT/venv-modelopt/bin/python
echo "[mo] job=$SLURM_JOB_ID node=$(hostname) D=$D CALSET=$CALSET mode=$CALSET_MODE repo=$REPO_ROOT"
md5sum bench/tim_calibrate.py bench/mo_*.py bench/mo_formats.sbatch bench/share_stats.py bench/ho_select.py 2>/dev/null

# 0. node record
smi=$(nvidia-smi 2>&1)
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
vbios=$(nvidia-smi --query-gpu=vbios_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(printf '%s\n' "$smi" | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "vbios ${vbios:-?}"; echo "kernel $(uname -r)"
  echo "node $(hostname)"; echo "date $(date -Is)"; echo "nvrm $(head -1 /proc/driver/nvidia/version 2>/dev/null)"; } > "$D/node_gpu.txt"
cat "$D/node_gpu.txt"

# 1. the calibration set
case "$CALSET_MODE" in
    build)
        if [ -e "$CALSET/calset_record.json" ] || [ -e "$CALSET/problems.json" ]; then echo "[mo] CALSET exists, mode build"; exit 4; fi
        mkdir -p "$CALSET"
        "$PYTHON_BIN" bench/mo_calset.py select --out-dir "$CALSET" > "$D/log_calset_select.txt" 2>&1 || { tail -20 "$D/log_calset_select.txt"; exit 4; }
        tail -2 "$D/log_calset_select.txt"
        t=$SECONDS
        "$PYTHON_BIN" bench/mo_calset.py generate --out-dir "$CALSET" > "$D/log_calset_generate.txt" 2>&1 || { tail -20 "$D/log_calset_generate.txt"; exit 4; }
        grep "\[mo_calset" "$D/log_calset_generate.txt" | tail -2 | cut -c1-1200
        echo "[mo] calset generated ($((SECONDS-t))s)" ;;
    reuse)
        "$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_calset; r, c = mo_calset.load_calset('$CALSET'); print('[mo] calset reused', c['calset_sha256'], len(r))" \
            || { echo "[mo] CALSET check failed"; exit 4; } ;;
    *) echo "[mo] CALSET_MODE must be build or reuse"; exit 4 ;;
esac
cp "$CALSET/calset_record.json" "$D/calset_record.json"

# 2. Model Optimizer exports
RC=0
declare -A EXPORT_RC
for name in $("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; print(' '.join(C.EXPORTS))"); do
    t=$SECONDS
    env -u PYTHONPATH "$MO_PY" -I bench/mo_export.py export --name "$name" --calset-dir "$CALSET" --out-dir "$D/ckpt/$name" \
        > "$D/log_export_$name.txt" 2>&1
    rc=$?; EXPORT_RC[$name]=$rc; echo "$rc" > "$D/rc_export_$name.txt"
    grep -h "\[mo_export\|Traceback\|Error" "$D/log_export_$name.txt" | grep -v "calibration [0-9]" | cut -c1-900 | tail -6
    echo "[mo] export $name rc=$rc ($((SECONDS-t))s)"
    [ $rc -ne 0 ] && RC=$rc
done

# 3. the engine configurations
labels=""
while IFS='|' read -r label ckpt; do
    [ -z "$label" ] && continue
    if [ -n "$ckpt" ] && [ "${EXPORT_RC[$ckpt]:-1}" != 0 ]; then
        echo "skip: export $ckpt failed" > "$D/rc_$label.txt"; echo "[mo] config $label SKIPPED (export $ckpt failed)"; continue
    fi
    left=$((LIMIT_S - (SECONDS - T0)))
    if [ $left -lt $RESERVE_S ]; then
        echo "skip: time (${left}s left)" > "$D/rc_$label.txt"; echo "[mo] config $label SKIPPED (time: ${left}s left)"; continue
    fi
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --ckpt-root "$D/ckpt" --out-dir "$D" > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|\[cal\] decode\|Traceback\|Error" "$D/log_$label.txt" | cut -c1-700 | tail -6
    echo "[mo] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done < <("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; [print(f\"{l}|{c['ckpt'] or ''}\") for l, c in C.FORMATS]")

# 3b. the export's scales across verl's sleep / wake order on the GPU, no weight stream
if [ "${EXPORT_RC[fp8_kv]:-1}" = 0 ] && [ $((LIMIT_S - (SECONDS - T0))) -ge $((20 * 60)) ]; then
    t=$SECONDS
    "$PYTHON_BIN" bench/mo_measure.py sleepwake --export fp8_kv --ckpt-root "$D/ckpt" --out-dir "$D" > "$D/log_sleepwake.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_sleepwake.txt"
    grep -h "sleepwake\|Traceback\|Error" "$D/log_sleepwake.txt" | cut -c1-600 | tail -5
    echo "[mo] sleep/wake probe rc=$rc ($((SECONDS-t))s) (descriptive; does not change RC)"
else
    echo "skip" > "$D/rc_sleepwake.txt"; echo "[mo] sleep/wake probe SKIPPED"
fi

# 4. clean bf16 scoring engine, separate process
t=$SECONDS
"$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_score.txt"
grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -12
echo "[mo] score $labels rc=$rc ($((SECONDS-t))s)"
[ $rc -ne 0 ] && RC=$rc

# 5. the reading (CPU)
"$PYTHON_BIN" bench/mo_formats_read.py --out-dir "$D" ${ANCHORS:+--anchors "$ANCHORS"} --json "$D/reading.json" \
    > "$D/reading.txt" 2>&1
rcr=$?
cat "$D/reading.txt"
[ $rcr -ne 0 ] && [ $RC -eq 0 ] && RC=$rcr
du -sh "$D/cache_${SLURM_JOB_ID}" "$D/ckpt" "$D" 2>/dev/null
echo "[mo] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

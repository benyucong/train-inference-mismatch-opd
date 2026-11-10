#!/bin/bash
#SBATCH --job-name=mo_grid
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/%x_%j.log
# The W8A16 and W4A8 weight / activation combinations, each with the three KV caches of the other formats (Model
# Optimizer's calibrated FP8 cache on every layer, the same with layer 0's cache bf16, bf16), plus two controls (the
# floor; cal_c2 = bf16 weights + the calibrated FP8 cache, the kv_from path the grid reuses). The W4A8 configurations
# install the fix for vLLM 0.19.1's W4A8 scale view (mo_measure.install_w4a8_view_fix).
# 1. exports (bench/mo_grid_export.py): w8a16_fp8 = fp8_kv dequantized to bf16; w4a8_awq = llm-compressor AWQ + W4AFP8
# 2. one process per label (mo_measure.py q), then a clean bf16 engine scores them (score)
# 3. bench/mo_grid_read.py checks every engine and the controls. Writes only to its own D.
# Submit from the repository root with D (created empty), BASE (the formats job's D) and CALSET exported, and
# optionally ANCHORS (a JSON file {label: [lo, hi]} of reference 95% intervals for the controls, see
# bench/mo_grid_read.py --anchors).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${CALSET:?set CALSET}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[grid] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL
LLMC_PY=$CLUSTER_ROOT/venv-llmc/bin/python
echo "[grid] job=$SLURM_JOB_ID node=$(hostname) D=$D BASE=$BASE CALSET=$CALSET repo=$REPO_ROOT"
nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader > "$D/node.txt" 2>&1; uname -r >> "$D/node.txt"
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; C.selftest(); print('[grid] selftest ok')" || exit 2
"$PYTHON_BIN" - "$CALSET" "$BASE/ckpt/fp8_kv" > "$D/inputs_check.txt" 2>&1 <<'PY2'
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
print(f"[grid] inputs ok: calibration set {crec['calset_sha256'][:16]} ({len(rows)} responses); fp8_kv export "
      f"{len(rec['files'])} files = its record; {len(rec['kv_scales'])} layers' KV scales")
PY2
rc=$?; cat "$D/inputs_check.txt"; [ $rc -eq 0 ] || { echo "[grid] inputs check FAILED"; exit 2; }
mkdir -p "$D/ckpt"
t=$SECONDS
timeout 1200 "$PYTHON_BIN" bench/mo_grid_export.py dequant --src "$BASE/ckpt/fp8_kv" --out "$D/ckpt/w8a16_fp8" > "$D/log_export_w8a16.txt" 2>&1
rc=$?; grep -h "\[mo_grid_export\]\|Traceback\|Error" "$D/log_export_w8a16.txt" | cut -c1-500 | tail -4
echo "[grid] export w8a16_fp8 rc=$rc ($((SECONDS-t))s)"; W8OK=$rc
t=$SECONDS
timeout 3600 env -u PYTHONPATH "$LLMC_PY" -I bench/mo_grid_export.py w4afp8 --calset-dir "$CALSET" --out "$D/ckpt/w4a8_awq" > "$D/log_export_w4a8.txt" 2>&1
rc=$?; grep -h "\[mo_grid_export\]\|Traceback\|Error" "$D/log_export_w4a8.txt" | cut -c1-500 | tail -4
echo "[grid] export w4a8_awq rc=$rc ($((SECONDS-t))s)"; W4OK=$rc
RC=0; labels=""; [ "$W8OK" -eq 0 ] && [ "$W4OK" -eq 0 ] || RC=3
for label in mo_z_none cal_c2 mo_w8a16_bf16kv mo_w8a16_kv mo_w8a16_kv_s0 mo_w4a8_bf16kv mo_w4a8_kv mo_w4a8_kv_s0; do
    if [ $((SECONDS-T0)) -gt $((150*60 - 25*60)) ]; then echo "[grid] skip $label: under 25 min left"; RC=4; continue; fi
    case "$label" in mo_w8a16_*) [ "$W8OK" -eq 0 ] || { echo "[grid] skip $label: its export failed"; continue; } ;;
                      mo_w4a8_*) [ "$W4OK" -eq 0 ] || { echo "[grid] skip $label: its export failed"; continue; } ;; esac
    t=$SECONDS; CKR="$D/ckpt"; [ "$label" = mo_z_none ] || [ "$label" = cal_c2 ] && CKR="$BASE/ckpt"
    timeout 900 "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --ckpt-root "$CKR" --export-root "$BASE/ckpt" --out-dir "$D" \
        > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|Traceback\|Error" "$D/log_$label.txt" | cut -c1-700 | tail -6
    echo "[grid] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done
if [ -n "$labels" ]; then
    timeout 1200 "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"; grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    [ $rc -ne 0 ] && RC=$rc
fi
python3 bench/mo_grid_read.py "$D" "$D/ckpt" ${ANCHORS:+--anchors "$ANCHORS"}; rc=$?; [ $rc -ne 0 ] && RC=$rc
echo "[grid] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

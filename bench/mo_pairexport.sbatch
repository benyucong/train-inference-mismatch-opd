#!/bin/bash
#SBATCH --job-name=mo_pairexport
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.log
# Another student / teacher pair's inputs for its Model Optimizer FP8 sampler: MO_PAIR = r1_1p5b | llama_3b (set
# --job-name mo_pairexport_<pair> at submission). ONE GH200. Writes only to the pair's directory (its formats_dir in
# mo_common.PAIRS):
#   0. node record -> D/node_gpu.txt; the student's config must have 28 decoder layers
#   1. the calibration set: the main pair's problems (problems.json, the same draw: pair-independent), its sha256 = the
#      main calibration record's; 512 responses sampled by THIS pair's student (mo_calset.py generate, the same
#      sampling: T 1, top-p 1, per-request seeds, 16,384 tokens)
#   2. venv-modelopt: Model Optimizer's FP8 recipe (general/ptq/fp8_default-kv_fp8) calibrated on it and exported
#      (mo_export.py export --name fp8_kv) -> <pair dir>/ckpt/fp8_kv, rc_export_fp8_kv.txt
# Then make the pair's scale file on CPU from <pair dir>/ckpt/fp8_kv (bench/mo_scales.py make --recipe fp8, written
# to $CLUSTER_ROOT/mo/scales/fp8<tag>.json with the pair's tag from mo_common.PAIRS and passed to the launcher as
# SAMPLER_MODELOPT_SCALES) and run its refit check (mo_refit.sbatch).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${MO_PAIR:?set MO_PAIR}"
export MO_PAIR
eval "$("$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; assert C.PAIR != 'main'; print(f'D={C.FORMATS_DIR} CALSET={C.CALSET_DIR} MAIN_CALSET={C.MAIN_CALSET_DIR} STUDENT_PATH={C.STUDENT}')")" \
    || { echo "[pairexport] MO_PAIR=$MO_PAIR is not one of the other pairs of mo_common.PAIRS"; exit 2; }
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[pairexport] D=$D missing or not empty"; exit 2; fi
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES \
      SPECOPD_SAMPLER_MODELOPT_MODEL SPECOPD_FA3_KVSKIP_PATCH
MO_PY=$CLUSTER_ROOT/venv-modelopt/bin/python
T0=$SECONDS
echo "[pairexport] job=$SLURM_JOB_ID node=$(hostname) MO_PAIR=$MO_PAIR student=$STUDENT_PATH D=$D repo=$REPO_ROOT"
md5sum bench/mo_common.py bench/mo_calset.py bench/mo_export.py bench/mo_pairexport.sbatch

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"
nl=$(python3 -c "import json; print(json.load(open('$STUDENT_PATH/config.json'))['num_hidden_layers'])")
[ "$nl" = 28 ] || { echo "[pairexport] the student has $nl decoder layers (the bench/mo_* tools assume 28): NOTHING RUN"; exit 3; }

# 1. the calibration set: the main draw's problems, this student's responses
mkdir -p "$CALSET"
python3 - "$MAIN_CALSET" "$CALSET" <<'PY' || { echo "[pairexport] problems copy failed: NOTHING RUN"; exit 4; }
import hashlib, json, os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]
rec = json.load(open(os.path.join(src, "calset_record.json")))
p = os.path.join(src, "problems.json")
assert hashlib.sha256(open(p, "rb").read()).hexdigest() == rec["problems_sha256"], "main problems.json != its record"
shutil.copy2(p, os.path.join(dst, "problems.json"))
print(f"[pairexport] problems: the main draw ({rec['n_problems']} problems, sha256 {rec['problems_sha256'][:16]})")
PY
t=$SECONDS
"$PYTHON_BIN" bench/mo_calset.py generate --out-dir "$CALSET" > "$D/log_calset_generate.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_calset.txt"
grep "\[mo_calset" "$D/log_calset_generate.txt" | tail -2 | cut -c1-1200
echo "[pairexport] calibration set rc=$rc ($((SECONDS-t))s)"
[ $rc -ne 0 ] && { tail -20 "$D/log_calset_generate.txt"; exit 4; }
cp "$CALSET/calset_record.json" "$D/calset_record.json"

# 2. Model Optimizer's FP8 export
t=$SECONDS
env -u PYTHONPATH "$MO_PY" -I bench/mo_export.py export --name fp8_kv --calset-dir "$CALSET" --out-dir "$D/ckpt/fp8_kv" \
    > "$D/log_export_fp8_kv.txt" 2>&1
rc=$?; echo "$rc" > "$D/rc_export_fp8_kv.txt"
grep -h "\[mo_export\|Traceback\|Error" "$D/log_export_fp8_kv.txt" | grep -v "calibration [0-9]" | cut -c1-900 | tail -6
echo "[pairexport] export fp8_kv rc=$rc ($((SECONDS-t))s)"
du -sh "$D" 2>/dev/null
echo "[pairexport] done rc=$rc ($((SECONDS-T0))s)"
exit $rc

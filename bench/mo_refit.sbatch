#!/bin/bash
#SBATCH --job-name=mo_refit_fp8
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.log
# The refit check of SAMPLER_MODELOPT=fp8. ONE GH200, no training. Exit 0 = PASS, so a training job can be chained on it
# (afterok). Submit from the repository root with D (created empty), EXPORT (the Model Optimizer fp8_kv export of the
# formats job) and SCALES (its scale file, bench/mo_scales.py) exported.
#   0. node record -> D/node_gpu.txt
#   1. Model Optimizer numerics on THIS GPU: mo_fp8_weight == _export_quantized_weight, bit for bit (venv-modelopt)
#   2. vLLM engine from the export -> digest_export.json
#   3. vLLM engine as a training rollout server + two syncs through SamplerModelOpt and verl's sleep / wake order
#      -> digest_refit.json
#   3b. a control engine (the export, layer 0's cache in bf16), descriptive only
#   4. the check (CPU) -> refit_check.json: PASS iff the refit engine equals the export engine byte for byte after each
#      sync and after the KV wake-up, the KV scales are restored, and both engines generate the same greedy tokens
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${EXPORT:?set EXPORT}" "${SCALES:?set SCALES}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[mo_refit] D=$D missing or not empty"; exit 2; fi
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_NOISE SPECOPD_SAMPLER_MODELOPT \
      SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND
MO_PY=$CLUSTER_ROOT/venv-modelopt/bin/python
echo "[mo_refit] job=$SLURM_JOB_ID node=$(hostname) D=$D EXPORT=$EXPORT SCALES=$SCALES ($(sha256sum < "$SCALES" | cut -c1-16)) repo=$REPO_ROOT"
md5sum verl/verl/utils/specopd_modelopt.py verl/verl/workers/rollout/vllm_rollout/utils.py bench/mo_refit_check.py bench/mo_numerics_check.py
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"
RC=0
env -u PYTHONPATH "$MO_PY" -I bench/mo_numerics_check.py fp8 --device cuda --real --json "$D/fp8_numerics_cuda.json" \
    > "$D/log_numerics.txt" 2>&1; rc=$?
grep -h "mo_numerics\] fp8:" "$D/log_numerics.txt"; echo "[mo_refit] numerics rc=$rc"; [ $rc -ne 0 ] && RC=$rc
# fp8fl: the scale file's bf16-cache layers (kv_cache_dtype_skip_layers) for the export engine too; empty for fp8
KVSKIP=$("$PYTHON_BIN" -c 'import json,sys; print(",".join(str(i) for i in json.load(open(sys.argv[1])).get("kv_bf16_layers", [])))' "$SCALES")
echo "[mo_refit] recipe $("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["recipe"])' "$SCALES"), bf16-cache layers '${KVSKIP}'"
for step in "export-digest --export-dir $EXPORT${KVSKIP:+ --kv-skip $KVSKIP} --out $D/digest_export.json" "refit-digest --scales $SCALES --out $D/digest_refit.json"; do
    name=${step%% *}; t=$SECONDS
    # shellcheck disable=SC2086
    "$PYTHON_BIN" bench/mo_refit_check.py $step > "$D/log_$name.txt" 2>&1; rc=$?
    grep -h "\[mo_refit\]\|SPECOPD-MO\|Traceback\|Error" "$D/log_$name.txt" | cut -c1-800 | tail -8
    echo "[mo_refit] $name rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
done
# control (descriptive, never changes RC): the export with layer 0's cache in bf16 -- shows whether the prompts and the
# template give fluent text once the first layer's cache is exempt
if [ -z "$KVSKIP" ]; then
    "$PYTHON_BIN" bench/mo_refit_check.py export-digest --export-dir "$EXPORT" --kv-skip 0 --out "$D/digest_control_s0.json" \
        > "$D/log_control_s0.txt" 2>&1; echo "[mo_refit] control (layer-0 cache bf16) rc=$?"
    grep -h "\[mo_refit\]" "$D/log_control_s0.txt" | cut -c1-400 | tail -2
fi
"$PYTHON_BIN" bench/mo_refit_check.py compare --export "$D/digest_export.json" --refit "$D/digest_refit.json" \
    --scales "$SCALES" --control "$D/digest_control_s0.json" --json "$D/refit_check.json"; rc=$?
[ $rc -ne 0 ] && RC=$rc
du -sh "$D" 2>/dev/null
echo "[mo_refit] done rc=$RC"
exit $RC

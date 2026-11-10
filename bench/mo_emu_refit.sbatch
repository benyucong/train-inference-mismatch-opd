#!/bin/bash
#SBATCH --job-name=mo_refit_nvfp4
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/%x_%j.log
# The refit check of the EMULATED NVFP4 sampler in training (bench/mo_emu_refit.py). ONE GH200, no training. Exit 0 =
# PASS, so a training job can be chained on it (afterok). Submit from the repository root with D (created empty), STATIC
# (the bit-exact NVFP4 static file, mo_nvfp4emu.sbatch's D/nvfp4_static.json), KVSKIP ('' for recipe nvfp4, 0 for
# nvfp4l0: layer 0's cache bf16) and OFFLINE (the offline measurement's engine record, engine_<label>.json) exported.
#   0. node record -> D/node_gpu.txt; the static file is the one the BIT-EXACT nvfp4emu check ran on (nvfp4_check.json
#      in its directory)
#   1. the reference engine: a training rollout server's, one sync through SamplerQuant -> digest_ref.json
#   2. the refit engine: two syncs and verl's sleep / wake order -> digest_refit.json (+ the expected weights)
#   3. the check (CPU) -> refit_check.json
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${STATIC:?set STATIC}" "${OFFLINE:?set OFFLINE}"
KVSKIP=${KVSKIP:-}
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[mo_emu_refit] D=$D missing or not empty"; exit 2; fi
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL SPECOPD_FA3_KVSKIP_PATCH \
      VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_ATTENTION_BACKEND
echo "[mo_emu_refit] job=$SLURM_JOB_ID node=$(hostname) D=$D STATIC=$STATIC ($(sha256sum < "$STATIC" | cut -c1-16)) KVSKIP='$KVSKIP' repo=$REPO_ROOT"
md5sum verl/verl/utils/specopd_quant.py verl/verl/utils/specopd_nvfp4.py verl/verl/utils/specopd_kvskip.py \
       verl/verl/workers/rollout/vllm_rollout/utils.py bench/mo_emu_refit.py bench/mo_refit_check.py
drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
cuda=$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
{ echo "driver ${drv:-?}"; echo "cuda ${cuda:-?}"; echo "kernel $(uname -r)"; echo "node $(hostname)"; echo "date $(date -Is)"; } > "$D/node_gpu.txt"
"$PYTHON_BIN" - "$STATIC" <<'PY' || { echo "[mo_emu_refit] the static file is not the bit-exact one: NOTHING RUN"; exit 4; }
import json, os, sys
st = sys.argv[1]
chk = json.load(open(os.path.join(os.path.dirname(st), "nvfp4_check.json")))
assert chk["verdict"] == "BIT-EXACT", chk["verdict"]
print(f"[mo_emu_refit] {os.path.dirname(st)}/nvfp4_check.json: {chk['verdict']}")
PY
RC=0
for step in "ref-digest --out $D/digest_ref.json" "refit-digest --out $D/digest_refit.json"; do
    name=${step%% *}; t=$SECONDS
    # shellcheck disable=SC2086
    "$PYTHON_BIN" bench/mo_emu_refit.py $step --static "$STATIC" ${KVSKIP:+--kv-skip $KVSKIP} > "$D/log_$name.txt" 2>&1; rc=$?
    grep -h "\[mo_emu_refit\]\|SPECOPD-QUANT\|SPECOPD-KVSKIP\|Traceback\|Error" "$D/log_$name.txt" | cut -c1-700 | tail -10
    echo "[mo_emu_refit] $name rc=$rc ($((SECONDS-t))s)"; [ $rc -ne 0 ] && RC=$rc
done
"$PYTHON_BIN" bench/mo_emu_refit.py compare --ref "$D/digest_ref.json" --refit "$D/digest_refit.json" --static "$STATIC" \
    ${KVSKIP:+--kv-skip $KVSKIP} --offline-engine "$OFFLINE" --json "$D/refit_check.json"; rc=$?
[ $rc -ne 0 ] && RC=$rc
du -sh "$D" 2>/dev/null
echo "[mo_emu_refit] done rc=$RC"
exit $RC

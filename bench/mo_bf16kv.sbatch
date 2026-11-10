#!/bin/bash
#SBATCH --job-name=mo_bf16kv
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=PARTITION_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm/%x_%j.log
# Model Optimizer FP8 weights and activations (export fp8_kv) served with a bf16 KV cache (mo_fp8_bf16kv,
# mo_common.BF16KV), with the floor as the control, on the formats job's calibration set. The bf16-cache configuration
# loads $CKPT2/fp8_kv, a copy of the fp8_kv export whose configs declare no KV-cache quantization: vLLM resolves
# kv_cache_dtype 'auto' to the cache dtype a checkpoint's quantization config declares, so the export itself would be
# served with its FP8 cache. Make the copy first (CPU, stdlib; it refuses a non-empty output directory):
#     python3 bench/mo_grid_export.py nokv --src "$BASE/ckpt/fp8_kv" --out "$CKPT2/fp8_kv"
# One process per label (mo_measure.py q), then a clean bf16 engine scores them (score); the engine check reads the
# bf16-cache engine's log for layer 0's cache dtype and the Model Optimizer FP8 linear config. Writes only to its own D.
# Submit from the repository root with D (created empty), BASE (the formats job's D), CALSET and CKPT2 exported.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; export REPO_ROOT="$PWD"
source scripts/cluster_env.sh
: "${D:?set D}" "${BASE:?set BASE}" "${CALSET:?set CALSET}" "${CKPT2:?set CKPT2}"
if [ ! -d "$D" ] || [ -n "$(ls -A "$D")" ]; then echo "[bf16kv] D=$D missing or not empty"; exit 2; fi
T0=$SECONDS
export VLLM_CACHE_ROOT="$D/cache_${SLURM_JOB_ID}/vllm" TORCHINDUCTOR_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/inductor"
export TORCH_EXTENSIONS_DIR="$D/cache_${SLURM_JOB_ID}/torch_ext" TRITON_CACHE_DIR="$D/cache_${SLURM_JOB_ID}/triton"
export VERL_LOGGING_LEVEL=WARN PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC SPECOPD_SAMPLER_NOISE \
      VLLM_DISABLE_COMPILE_CACHE VLLM_ATTENTION_BACKEND VLLM_USE_NVFP4_CT_EMULATIONS VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FBGEMM \
      SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL
echo "[bf16kv] job=$SLURM_JOB_ID node=$(hostname) D=$D BASE=$BASE CALSET=$CALSET repo=$REPO_ROOT"
nvidia-smi --query-gpu=driver_version,vbios_version --format=csv,noheader > "$D/node.txt" 2>&1; uname -r >> "$D/node.txt"
"$PYTHON_BIN" -c "import sys; sys.path.insert(0, 'bench'); import mo_common as C; C.selftest(); print('[bf16kv] selftest ok')" || exit 2
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
print(f"[bf16kv] inputs ok: calibration set {crec['calset_sha256'][:16]} ({len(rows)} responses); fp8_kv export "
      f"{len(rec['files'])} files = its record; {len(rec['kv_scales'])} layers' KV scales")
PY2
rc=$?; cat "$D/inputs_check.txt"; [ $rc -eq 0 ] || { echo "[bf16kv] inputs check FAILED"; exit 2; }
RC=0; labels=""
for label in mo_z_none mo_fp8_bf16kv; do
    t=$SECONDS; CKR="$BASE/ckpt"; [ "$label" = mo_fp8_bf16kv ] && CKR="$CKPT2"
    "$PYTHON_BIN" bench/mo_measure.py q --label "$label" --ckpt-root "$CKR" --export-root "$CKR" --out-dir "$D" \
        > "$D/log_$label.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_$label.txt"
    grep -h "\[mo_measure\|Traceback\|Error" "$D/log_$label.txt" | cut -c1-700 | tail -6
    echo "[bf16kv] config $label rc=$rc ($((SECONDS-t))s)"
    if [ $rc -eq 0 ]; then labels="$labels${labels:+,}$label"; else RC=$rc; fi
done
if [ -n "$labels" ]; then
    "$PYTHON_BIN" bench/mo_measure.py score --labels "$labels" --out-dir "$D" > "$D/log_score.txt" 2>&1
    rc=$?; echo "$rc" > "$D/rc_score.txt"; grep -h "\[cal\]\|Traceback\|Error" "$D/log_score.txt" | cut -c1-400 | tail -6
    [ $rc -ne 0 ] && RC=$rc
fi
"$PYTHON_BIN" - "$D" <<'PY3'
import json, os, sys
D = sys.argv[1]
import re
log = open(os.path.join(D, "log_mo_fp8_bf16kv.txt"), errors="ignore").read()
m = re.search(r"layer-0 attn=\{[^}]*'kv_cache_dtype': '([^']+)'", log)
lin = "ModelOptFp8Config" in log
kv0 = m.group(1) if m else "not found"
ok = m is not None and not kv0.startswith("fp8") and lin
print(f"[bf16kv] engine check: layer-0 cache dtype {kv0}; ModelOptFp8Config {lin}; {'VALID' if ok else 'NOT VALID'}")
for lab in ("mo_z_none", "mo_fp8_bf16kv"):
    p = os.path.join(D, f"dec_{lab}.json")
    if not os.path.exists(p):
        print(f"[bf16kv] {lab}: no result"); continue
    d = json.load(open(p))
    print(f"[bf16kv] {lab}: kl_samp_train {d['kl_samp_train']:.5f} ci95 {d['kl_samp_train_ci95']} n_tok {d['n_tok']} trunc {d['frac_truncated']}")
PY3
echo "[bf16kv] done rc=$RC ($((SECONDS-T0))s)"
exit $RC

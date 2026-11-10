"""Shared constants of the offline Model Optimizer (MO) tools (stdlib only at import).

Users: mo_calset.py (the calibration set), mo_export.py (Model Optimizer quantize + export, venv-modelopt),
mo_measure.py (tim_calibrate.py driven for one engine configuration, venv-specopd), mo_formats_read.py (the reader)
and bench/mo_formats.sbatch.
"""
import hashlib
import json
import os

R = os.environ.get("CLUSTER_ROOT", "/path/to/workdir")
HUB = R + "/hf/hub"
# ---- the pairs (each pair's sampler needs its own calibration set, FP8 export and scale file). MO_PAIR in
# the environment selects one (default: the main pair); every tool reads STUDENT / CALSET_DIR / FORMATS_DIR from it.
# The calibration set's PROBLEMS are pair-independent (the same draw); the responses are the pair's student's own.
# tag: the suffix of the pair's file names (e.g. scales/fp8<tag>.json); launch: the environment knobs that make the
# launcher (scripts/cluster_opd_sync.sh) run the pair, besides the main recipe's.
PAIRS = {
    "main": dict(student=HUB + "/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/"
                         "916b56a44061fd5cd7d6a8fb632557ed4f724f60",
                 calset_dir=R + "/mo/calset", formats_dir=R + "/mo/formats", tag="",
                 launch={}),
    "r1_1p5b": dict(student=HUB + "/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/"
                            "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
                    teacher=HUB + "/models--hbx--JustRL-DeepSeek-1.5B/snapshots/0637e4096c789c67f9eecbe8355e0bdeddede1c2",
                    calset_dir=R + "/mo/pairs/r1_1p5b/calset",
                    formats_dir=R + "/mo/pairs/r1_1p5b", tag="_r1_1p5b",
                    launch=dict(SPEC="none", VERL_CONTINUOUS_TOKEN_MODEL_FAMILY="deepseek")),
    "llama_3b": dict(student=HUB + "/models--unsloth--Llama-3.2-3B-Instruct/snapshots/"
                             "006f5dcd1393c3add266de40994ba96225e9689d",
                     teacher=HUB + "/models--TMLR-Group-HF--GT-Llama-3.2-3B-Instruct-MATH/snapshots/"
                             "b6a0ab79ab4802ef77334532dd1936bb1d219cf9",
                     calset_dir=R + "/mo/pairs/llama_3b/calset",
                     formats_dir=R + "/mo/pairs/llama_3b", tag="_llama_3b",
                     launch=dict(SPEC="none", VERL_CONTINUOUS_TOKEN_MODEL_FAMILY="default")),
}
PAIR = os.environ.get("MO_PAIR", "main") or "main"
if PAIR not in PAIRS:
    raise SystemExit(f"MO_PAIR={PAIR!r} not in {sorted(PAIRS)}")
STUDENT = PAIRS[PAIR]["student"]
STUDENT_REV = STUDENT.rstrip("/").rsplit("/", 1)[1]
N_LAYERS = 28                       # all three students have 28 decoder layers (checked from config.json by the tools)
DAPO = R + "/workspace/datasets/dapo-math-17k.parquet"
EVAL_DATA = R + "/workspace/datasets/test_data"
CALSET_DIR = PAIRS[PAIR]["calset_dir"]
FORMATS_DIR = PAIRS[PAIR]["formats_dir"]
MAIN_CALSET_DIR = PAIRS["main"]["calset_dir"]


def student_tensor(name, device="cpu", student=None):
    """One tensor of the student's checkpoint (sharded with an index, or a single model.safetensors)."""
    from safetensors import safe_open
    d = student or STUDENT
    ip = os.path.join(d, "model.safetensors.index.json")
    f = json.load(open(ip))["weight_map"][name] if os.path.exists(ip) else "model.safetensors"
    with safe_open(os.path.join(d, f), framework="pt", device=device) as fh:
        return fh.get_tensor(name)


# ---- the calibration set (mo_calset.py) ----------------------------------------------------------------------------
CAL_SEED = 20261008
CAL_N_PROBLEMS = 256
CAL_N_PER_PROBLEM = 2
CAL_TEMPERATURE = 1.0
CAL_TOP_P = 1.0
CAL_MAX_TOKENS = 16384          # the training response cap (MAX_RESP)
CAL_EXCL_FILE = "mo_calset_excl.json"   # the 1,000 held-out pool indices (selected by bench/ho_select.py)
CAL_EXCL_LIST_SHA256 = "5fdc07dfa28e9501145dfea007160717d45a927f9890346fa65e5f164d10e6a8"  # sha256(json.dumps(sorted))
CAL_EVAL_BENCHES = ("AIME24", "AIME25", "AMC23", "MATH-500")
CAL_DROP_TIERS = ("auto", "flag")


def request_seed(k, j):
    """Per-request sampling seed of calibration response j of problem k (k = rank in the drawn order)."""
    return CAL_SEED + 10 * k + j


# ---- Model Optimizer exports (venv-modelopt) -------------------------------------------------------------------------
# recipe = a built-in Model Optimizer PTQ recipe (modelopt.recipe.load_recipe), unchanged; disable_layers = decoder layers
# whose every quantizer (weights, activations, K/V) is disabled after the recipe's own entries.
EXPORTS = {
    "fp8_kv": dict(recipe="general/ptq/fp8_default-kv_fp8", disable_layers=()),
    "fp8_kv_fl": dict(recipe="general/ptq/fp8_default-kv_fp8", disable_layers=(0, N_LAYERS - 1)),
    "nvfp4_kv": dict(recipe="general/ptq/nvfp4_default-kv_fp8", disable_layers=()),
}

# ---- the engine configurations of the formats job, in run order -----------------------------------------------------
# ckpt: an EXPORTS name (the engine loads that export) or None (the bf16 student). args: extra tim_calibrate --mode q
# arguments. env: set in the measuring process before vLLM is imported. kind: how the arithmetic is done. expect: what
# the engine audit must show (checked by the reader; a configuration that fails it is INVALID, never read).
#   kv: 'auto' (bf16 cache) | 'unit' (FP8 cache, every k/v scale 1.0) | 'calibrated' (FP8 cache, every k/v scale = the
#       export's k_scale / v_scale) ; kv_bf16_layers: layers whose cache must be bf16 (vLLM kv_cache_dtype_skip_layers)
#   linear: the quant-method class every decoder linear layer must have, except linear_bf16_layers (Unquantized)
#   nvfp4_backend: the NVFP4 GEMM backend vLLM must report
#   verl_nvfp4_patch: whether verl's ModelOpt NVFP4 rollout patch (Marlin-only, verl/utils/modelopt) may be installed
#   cudagraph_mode: the engine's compilation_config.cudagraph_mode (default FULL_AND_PIECEWISE, tim_calibrate's); the
#       engine is always torch.compile'd (vLLM's default compilation mode) -- only CUDA-graph capture can be switched off
# cudagraph: set only where capture is impossible: vLLM 0.19.1's NVFP4 emulation backend fails inside graph capture (an
# Inductor-generated copy from a CPU constant, "operation not permitted when stream is capturing"), so its
# configurations run compiled WITHOUT CUDA graphs, keeping the compiled rounding.
FORMATS = (
    ("mo_z_none", dict(ckpt=None, args="", env={}, kind="floor (clean bf16)",
                       expect=dict(kv="auto", linear="UnquantizedLinearMethod"))),
    ("mo_kv8_unit", dict(ckpt=None, args="--kv-cache-dtype fp8", env={}, kind="real kernels",
                         expect=dict(kv="unit", linear="UnquantizedLinearMethod"))),
    ("mo_fp8_kv", dict(ckpt="fp8_kv", args="--kv-cache-dtype fp8", env={}, kind="real kernels",
                       expect=dict(kv="calibrated", linear="ModelOptFp8LinearMethod"))),
    ("mo_fp8_kv_fl", dict(ckpt="fp8_kv_fl", args="--kv-cache-dtype fp8 --kv-skip-layers 0,27", env={},
                          kind="real kernels",
                          expect=dict(kv="calibrated", kv_bf16_layers=(0, 27), linear="ModelOptFp8LinearMethod",
                                      linear_bf16_layers=(0, 27)))),
    ("mo_nvfp4_kv_emu", dict(ckpt="nvfp4_kv", args="--kv-cache-dtype fp8", cudagraph="NONE",
                             env={"VLLM_USE_NVFP4_CT_EMULATIONS": "1"},
                             kind="emulated (vLLM's NVFP4 emulation backend; compiled, no CUDA graphs)",
                             expect=dict(kv="calibrated", linear="ModelOptNvFp4LinearMethod", nvfp4_backend="emulation",
                                         verl_nvfp4_patch=False, cudagraph_mode="NONE"))),
    ("mo_nvfp4_kv_marlin", dict(ckpt="nvfp4_kv", args="--kv-cache-dtype fp8", env={"VLLM_NVFP4_GEMM_BACKEND": "marlin"},
                                kind="real kernels, weight-only FP4 (Marlin): NOT the W4A4 recipe",
                                expect=dict(kv="calibrated", linear="ModelOptNvFp4LinearMethod", nvfp4_backend="marlin",
                                            verl_nvfp4_patch=False))),
    ("mo_fp8_kv_s0", dict(ckpt="fp8_kv", args="--kv-cache-dtype fp8 --kv-skip-layers 0", env={}, kind="real kernels",
                          expect=dict(kv="calibrated", kv_bf16_layers=(0,), linear="ModelOptFp8LinearMethod"))),
    ("mo_nvfp4_kv_emu_s0", dict(ckpt="nvfp4_kv", args="--kv-cache-dtype fp8 --kv-skip-layers 0", cudagraph="NONE",
                                fa3_kvskip_patch=True, env={"VLLM_USE_NVFP4_CT_EMULATIONS": "1"},
                                kind="emulated (vLLM's NVFP4 emulation backend; compiled, no CUDA graphs)",
                                expect=dict(kv="calibrated", kv_bf16_layers=(0,), linear="ModelOptNvFp4LinearMethod",
                                            nvfp4_backend="emulation", verl_nvfp4_patch=False,
                                            cudagraph_mode="NONE", fa3_kvskip_patch=True))),
    ("mo_kv8_unit_s0", dict(ckpt=None, args="--kv-cache-dtype fp8 --kv-skip-layers 0", env={}, kind="real kernels",
                            expect=dict(kv="unit", kv_bf16_layers=(0,), linear="UnquantizedLinearMethod"))),
)
# ---- Model Optimizer FP8 weights and activations with a bf16 cache (bench/mo_bf16kv.sbatch). Not a formats-job
# configuration: vLLM resolves kv_cache_dtype 'auto' to the cache dtype a checkpoint's quantization config declares, so
# the fp8_kv export itself is served with its FP8 cache. The job loads ckpt from a copy of the export whose configs
# declare no KV-cache quantization (bench/mo_grid_export.py nokv).
BF16KV = (
    ("mo_fp8_bf16kv", dict(ckpt="fp8_kv", args="", env={},
                           kind="real kernels: Model Optimizer FP8 weights and activations, bf16 cache",
                           expect=dict(kv="auto", linear="ModelOptFp8LinearMethod"))),
)
BF16KV_LABELS = tuple(lab for lab, _ in BF16KV)
# ---- the EMULATED NVFP4 sampler (bench/mo_nvfp4emu.sbatch): the bf16 student, specopd_quant's
# w=nvfp4 grid (Model Optimizer's own fake-quantization kernel, vendored in verl/utils/specopd_nvfp4.py) re-streamed
# through SamplerQuant as at a training sync, its static file (env value STATIC_PLACEHOLDER = mo_measure q --static:
# the step-0 activation amax and the KV scales), compiled with tim_calibrate's FULL_AND_PIECEWISE CUDA graphs -- the
# training engine. expect: kv 'static' = FP8 cache with the static file's k / v scales; sampler_quant = the spec;
# n_weight_tensors = the HF weights the sync must quantize (7 per decoder layer); op_dtype = what the activation
# quantizer was fed (None for weight-only).
STATIC_PLACEHOLDER = "@STATIC"
NVFP4_EMU_LINEAR = "SpecopdNvfp4ActUnquantizedLinearMethod"
_EMU_ENV = {"SPECOPD_SAMPLER_QUANT_STATIC": STATIC_PLACEHOLDER}
_EMU_W = 7 * N_LAYERS
NVFP4_EMU = (
    ("mo_nvfp4emu_kv", dict(ckpt=None, args="--kv-cache-dtype fp8 --quant w=nvfp4,a=nvfp4", env=_EMU_ENV,
                            kind="emulated W4A4 (Model Optimizer's kernel) + real FP8 cache: the recipe",
                            expect=dict(kv="static", linear=NVFP4_EMU_LINEAR, sampler_quant="w=nvfp4,a=nvfp4",
                                        n_weight_tensors=_EMU_W, op_dtype="torch.bfloat16",
                                        cudagraph_mode="FULL_AND_PIECEWISE"))),
    ("mo_nvfp4emu_kv_s0", dict(ckpt=None, args="--kv-cache-dtype fp8 --quant w=nvfp4,a=nvfp4 --kv-skip-layers 0",
                               env=_EMU_ENV, kind="emulated W4A4 + real FP8 cache, layer 0's cache bf16",
                               expect=dict(kv="static", kv_bf16_layers=(0,), linear=NVFP4_EMU_LINEAR,
                                           sampler_quant="w=nvfp4,a=nvfp4", n_weight_tensors=_EMU_W,
                                           op_dtype="torch.bfloat16", cudagraph_mode="FULL_AND_PIECEWISE"))),
    ("mo_nvfp4emu_bf16kv", dict(ckpt=None, args="--quant w=nvfp4,a=nvfp4", env=_EMU_ENV,
                                kind="emulated W4A4, bf16 cache (diagnostic: the linear layers alone)",
                                expect=dict(kv="auto", linear=NVFP4_EMU_LINEAR, sampler_quant="w=nvfp4,a=nvfp4",
                                            n_weight_tensors=_EMU_W, op_dtype="torch.bfloat16",
                                            cudagraph_mode="FULL_AND_PIECEWISE"))),
    ("mo_nvfp4emu_w_kv", dict(ckpt=None, args="--kv-cache-dtype fp8 --quant w=nvfp4", env=_EMU_ENV,
                              kind="emulated weight-only NVFP4 + real FP8 cache (diagnostic: vs mo_nvfp4_kv_marlin)",
                              expect=dict(kv="static", linear="UnquantizedLinearMethod", sampler_quant="w=nvfp4",
                                          n_weight_tensors=_EMU_W, op_dtype=None,
                                          cudagraph_mode="FULL_AND_PIECEWISE"))),
)
NVFP4_EMU_LABELS = tuple(lab for lab, _ in NVFP4_EMU)
# ---- the EMULATED INT8 SmoothQuant / INT4 AWQ samplers (bench/mo_smoothemu.sbatch): Model
# Optimizer's presets (mtq.INT8_SMOOTHQUANT_CFG, mtq.INT4_AWQ_CFG) calibrated at step 0 on the calibration set (fixed
# smoothing / activation amax), weights re-quantized at the sync; bf16 cache (neither preset quantizes the cache).
# smooth = the bench/mo_export.py smooth --recipe; op_seen = the audit key of the activation op's input dtypes;
# act_attach = the activation smoothing is attached (AWQ's too, though it quantizes no activation).
SMOOTH_EMU = (
    ("mo_sqemu", dict(ckpt=None, args="--quant w=int8sq,a=int8sq", env=_EMU_ENV, smooth="int8_sq",
                      kind="emulated INT8 SmoothQuant W8A8 (Model Optimizer's preset, alpha 1.0), bf16 cache",
                      expect=dict(kv="auto", linear="SpecopdSmoothint8sqUnquantizedLinearMethod",
                                  sampler_quant="w=int8sq,a=int8sq", n_weight_tensors=_EMU_W, op_dtype="torch.bfloat16",
                                  op_seen="smooth_op_seen", act_attach=True, cudagraph_mode="FULL_AND_PIECEWISE"))),
    ("mo_awqemu", dict(ckpt=None, args="--quant w=int4awq", env=_EMU_ENV, smooth="int4_awq",
                       kind="emulated INT4 AWQ W4A16 (Model Optimizer's preset, awq_lite), bf16 cache",
                       expect=dict(kv="auto", linear="SpecopdSmoothint4awqUnquantizedLinearMethod",
                                   sampler_quant="w=int4awq", n_weight_tensors=_EMU_W, op_dtype="torch.bfloat16",
                                   op_seen="smooth_op_seen", act_attach=True, cudagraph_mode="FULL_AND_PIECEWISE"))),
)


def _smooth_kv(label, s0):
    """The same emulated recipe with Model Optimizer's calibrated FP8 cache (bench/mo_smoothkv.sbatch: the smoothemu
    job's static file + the formats job's fp8_kv export's k / v scales as vLLM serves them = the cache of cal_c2);
    s0: layer 0's cache bf16 (the first-layer share's partner)."""
    base = dict(SMOOTH_EMU)[label]
    e = dict(base["expect"], kv="static", **({"kv_bf16_layers": (0,)} if s0 else {}))
    return (f"{label}_kv" + ("_s0" if s0 else ""),
            dict(base, args="--kv-cache-dtype fp8 " + base["args"] + (" --kv-skip-layers 0" if s0 else ""),
                 kind=base["kind"].replace("bf16 cache", "Model Optimizer's calibrated FP8 cache")
                 + (", layer 0's cache bf16" if s0 else ""), expect=e))


SMOOTH_LABEL = {cfg["smooth"]: lab for lab, cfg in SMOOTH_EMU}       # the smoothemu job's one configuration per recipe
SMOOTH_EMU = SMOOTH_EMU + tuple(_smooth_kv(lab, s0) for lab, _ in SMOOTH_EMU for s0 in (False, True))
SMOOTH_KV = {cfg["smooth"]: (f"{lab}_kv", f"{lab}_kv_s0") for lab, cfg in SMOOTH_EMU if lab in SMOOTH_LABEL.values()}
SMOOTH_EMU_LABELS = tuple(lab for lab, _ in SMOOTH_EMU)
# ---- the KV-cache calibration comparison (bench/mo_cal1.sbatch): numbered configurations (CAL1_MAP), of which the
# formats job covers the floor, 1 (mo_kv8_unit), 5 (mo_fp8_kv, with its layer-0 variant) and 7 (mo_fp8_kv_fl); CAL1
# holds the others. kv_from = an EXPORTS name whose calibrated k / v scales are written into the bf16 student's FP8
# cache right after the engine is built, as vLLM serves a checkpoint's (rounded to bf16, see bf16()): Model Optimizer's
# KV-only checkpoint is not loadable by vLLM 0.19.1, and its KV scales are identical in all three of the formats job's
# exports (the calibration forward is unquantized). llm_kwargs = extra vLLM engine arguments. ckpt in LLMC_EXPORTS = an
# llm-compressor KV-only export the calibration job builds (bench/llmc_kv_export.py). kv 'per_head' = one k / v scale
# per KV head, = the export's; 'computed' = vLLM's own scales (calculate_kv_scales), not 1.0. cal = the configuration's
# number.
LLMC_EXPORTS = {"llmc_kv_head": "attn_head", "llmc_kv_tensor": "tensor"}
CAL1 = (
    ("cal_c2", dict(ckpt=None, kv_from="fp8_kv", args="--kv-cache-dtype fp8", env={}, cal="2",
                    kind="bf16 weights + Model Optimizer's calibrated per-layer FP8 cache",
                    expect=dict(kv="calibrated", linear="UnquantizedLinearMethod"))),
    ("cal_c3", dict(ckpt="llmc_kv_head", args="--kv-cache-dtype fp8", env={}, cal="3",
                    kind="bf16 weights + llm-compressor's per-head FP8 cache (Model Optimizer: per-tensor only)",
                    expect=dict(kv="per_head", linear="UnquantizedLinearMethod"))),
    ("cal_c2_s0", dict(ckpt=None, kv_from="fp8_kv", args="--kv-cache-dtype fp8 --kv-skip-layers 0", env={},
                       cal="2, layer 0's cache bf16", kind="configuration 2, layer 0's cache in bf16 (first-layer share)",
                       expect=dict(kv="calibrated", kv_bf16_layers=(0,), linear="UnquantizedLinearMethod"))),
    ("cal_c8", dict(ckpt=None, kv_from="fp8_kv", args="--kv-cache-dtype fp8 --kv-skip-layers 0,27", env={}, cal="8",
                    kind="configuration 2 with the first and last layers' cache in bf16",
                    expect=dict(kv="calibrated", kv_bf16_layers=(0, 27), linear="UnquantizedLinearMethod"))),
    ("cal_c4", dict(ckpt=None, args="--kv-cache-dtype fp8", env={}, llm_kwargs={"calculate_kv_scales": True}, cal="4",
                    kind="bf16 weights + FP8 cache with vLLM's own scales (calculate_kv_scales: its first forward)",
                    expect=dict(kv="computed", linear="UnquantizedLinearMethod"))),
    ("cal_c2_llmc", dict(ckpt="llmc_kv_tensor", args="--kv-cache-dtype fp8", env={}, cal="2 (llm-compressor)",
                         kind="configuration 2 calibrated by llm-compressor per tensor (descriptive)",
                         expect=dict(kv="calibrated", linear="UnquantizedLinearMethod"))),
)
CAL1_LABELS = tuple(lab for lab, _ in CAL1)
CAL1_MAP = (("floor", "mo_z_none"), ("1", "mo_kv8_unit"), ("2", "cal_c2"), ("3", "cal_c3"), ("4", "cal_c4"),
            ("5", "mo_fp8_kv"), ("7", "mo_fp8_kv_fl"), ("8", "cal_c8"), ("2 (llm-compressor)", "cal_c2_llmc"))
CAL1_SHARES = (("cal_c2", "cal_c2_s0"), ("mo_fp8_kv", "mo_fp8_kv_s0"))     # configurations 2 and 5
# the NVFP4 job (bench/mo_formats_nvfp4.sbatch): the NVFP4 configurations of FORMATS measured on their own, the two
# emulated ones compiled without CUDA graphs (cudagraph NONE); read together with the formats job's other outputs
# (floor, unit cache) by mo_formats_read.py --base-dir
NVFP4_LABELS = ("mo_nvfp4_kv_emu", "mo_nvfp4_kv_marlin", "mo_nvfp4_kv_emu_s0")
# stock vLLM 0.19.1's FlashAttention-3 metadata fails for a bf16-cache layer inside an FP8-cache engine on the path
# without CUDA graphs ("scheduler_metadata must have shape (metadata_size)"), so mo_nvfp4_kv_emu_s0 installs the FA3
# kv-skip patch (fa3_kvskip_patch; bench/g0_kernels.install_fa3_kvskip_patch: it changes the metadata only where stock
# vLLM would raise, and counts those calls).
FLOOR = "mo_z_none"
UNIT = "mo_kv8_unit"
# first-layer share: (A - B) / (A - F), B = A with layer 0's cache in bf16, F = the floor
SHARE_PAIRS = (("mo_kv8_unit", "mo_kv8_unit_s0"), ("mo_fp8_kv", "mo_fp8_kv_s0"), ("mo_nvfp4emu_kv", "mo_nvfp4emu_kv_s0"),
               ("mo_nvfp4_kv_emu", "mo_nvfp4_kv_emu_s0"), ("cal_c2", "cal_c2_s0"),
               ("mo_sqemu_kv", "mo_sqemu_kv_s0"), ("mo_awqemu_kv", "mo_awqemu_kv_s0"))
DECODE_ARGS = "--no-tf --decode-n 2 --decode-max-tokens 16384 --decode-seed 777 --n-per-bench 8"
N_SAMPLES = 32
SHORT_GEN_TOKENS = 48


def bf16(x):
    """float32(x) rounded to bfloat16 (round to nearest even), as a python float. vLLM 0.19.1 creates the attention
    layer's k_scale / v_scale checkpoint parameters with the model's default dtype (bf16; BaseKVCacheMethod.create_weights
    uses torch.tensor(-1.0) inside set_default_torch_dtype), so a checkpoint's float32 KV scale is served rounded to
    bf16 (for example 0.3 is served as 0.30078125)."""
    import struct
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


# ---- the weight / activation grid (bench/mo_grid.sbatch): W8A16 and W4A8, the two combinations of W4A4, W4A8, W4A16,
# W8A8 and W8A16 that the configurations above do not cover. ckpt = a GRID_EXPORTS name under the grid job's own
# --ckpt-root (bench/mo_grid_export.py); kv_from reads the formats job's fp8_kv export (--export-root), as
# configuration 2 (cal_c2) does for the bf16 student.
#   w8a16_fp8: fp8_kv's FP8 weights dequantized to bf16 -> mo_fp8_kv's weights with bf16 activations (bf16 GEMMs)
#   w4a8_awq:  llm-compressor AWQ + its W4AFP8 preset (INT4 weights in groups of 128, FP8 activations per token), real
#              kernels (vLLM's CompressedTensorsW4A8Fp8; vLLM 0.19.1 loads no Model Optimizer W4A8 checkpoint)
GRID_EXPORTS = {"w8a16_fp8": "dequant", "w4a8_awq": "w4afp8"}
_GRID_LIN = {"w8a16_fp8": dict(linear="UnquantizedLinearMethod"),
             "w4a8_awq": dict(linear="CompressedTensorsLinearMethod", scheme="CompressedTensorsW4A8Fp8")}
_GRID_KIND = {"w8a16_fp8": "W8A16: Model Optimizer FP8 weights (fp8_kv) dequantized to bf16, bf16 activations",
              "w4a8_awq": "W4A8: llm-compressor AWQ INT4 weights (groups of 128; vLLM's CUTLASS kernel stores the group "
                          "scales in FP8), FP8 activations per token"}

# vLLM 0.19.1's W4A8 kernel needs the scale-view fix (bench/mo_measure.install_w4a8_view_fix): its
# convert_bf16_scales_to_fp8 calls view(orig_shape[:-1], -1), which torch rejects, so the CUTLASS W4A8 kernel loads no
# checkpoint; the fix calls view(*orig_shape[:-1], -1) and counts its calls
_GRID_FIX = {"w4a8_awq": dict(vllm_w4a8_view_fix=True)}


def _grid(ckpt, tag, kv):
    if kv == "bf16":
        return (f"mo_{tag}_bf16kv", dict(ckpt=ckpt, args="", env={}, kind=_GRID_KIND[ckpt] + ", bf16 cache",
                                         expect=dict(kv="auto", **_GRID_LIN[ckpt]), **_GRID_FIX.get(ckpt, {})))
    s0 = kv == "fp8_s0"
    return (f"mo_{tag}_kv" + ("_s0" if s0 else ""),
            dict(ckpt=ckpt, kv_from="fp8_kv", args="--kv-cache-dtype fp8" + (" --kv-skip-layers 0" if s0 else ""),
                 env={}, kind=_GRID_KIND[ckpt] + ", Model Optimizer's calibrated FP8 cache"
                 + (", layer 0's cache bf16" if s0 else ""),
                 expect=dict(kv="calibrated", **({"kv_bf16_layers": (0,)} if s0 else {}), **_GRID_LIN[ckpt]),
                 **_GRID_FIX.get(ckpt, {})))


GRID = tuple(_grid(c, t, kv) for c, t in (("w8a16_fp8", "w8a16"), ("w4a8_awq", "w4a8"))
             for kv in ("fp8", "fp8_s0", "bf16"))
GRID_LABELS = tuple(lab for lab, _ in GRID)
# the grid job's controls: configurations measured again in the grid job; mo_grid_read.py --anchors gives the reference
# 95% interval each must fall in
GRID_CONTROLS = ("mo_z_none", "cal_c2")


def formats_dict():
    return dict(FORMATS + BF16KV + NVFP4_EMU + CAL1 + SMOOTH_EMU + GRID)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    obj = json.loads(json.dumps(obj, default=str))   # mixed int/str keys (e.g. block_sizes {-1: 16, 'type': ..}) -> str
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_anchors(path):
    """Reference 95% intervals {label: (lo, hi)} from a JSON file {label: [lo, hi]} (e.g. the kl_samp_train_ci95 of a
    measurement with the same harness); {} when path is empty."""
    if not path:
        return {}
    with open(path) as f:
        rec = json.load(f)
    out = {}
    for lab, ci in rec.items():
        lo, hi = (float(x) for x in ci)
        assert lo <= hi, f"anchor {lab}: interval {ci}"
        out[lab] = (lo, hi)
    return out


def load_excl(bench_dir):
    """The 1,000 held-out pool indices, checked against the pinned list hash."""
    with open(os.path.join(bench_dir, CAL_EXCL_FILE)) as f:
        rec = json.load(f)
    idx = rec["pool_indices"]
    assert sha256_bytes(json.dumps(sorted(idx)).encode()) == CAL_EXCL_LIST_SHA256, "exclusion list hash"
    assert len(idx) == len(set(idx)) == 1000
    return set(int(i) for i in idx)


def selftest():
    labels = [lab for lab, _ in FORMATS + BF16KV + NVFP4_EMU + CAL1 + SMOOTH_EMU + GRID]
    assert len(labels) == len(set(labels))
    for lab, cfg in GRID:
        e = cfg["expect"]
        assert cfg["ckpt"] in GRID_EXPORTS and cfg["env"] == {}, lab
        assert cfg.get("kv_from") in (None, "fp8_kv") and (cfg.get("kv_from") is None) == (e["kv"] == "auto"), lab
        assert e["kv"] in ("auto", "calibrated") and (e["kv"] == "auto") == ("--kv-cache-dtype fp8" not in cfg["args"]), lab
        skip = tuple(e.get("kv_bf16_layers", ()))
        assert (f"--kv-skip-layers {','.join(str(i) for i in skip)}" in cfg["args"]) if skip \
            else ("--kv-skip-layers" not in cfg["args"]), lab
        assert e["linear"] == _GRID_LIN[cfg["ckpt"]]["linear"], lab
    assert len(GRID_LABELS) == 6 and all(lab in formats_dict() for lab in GRID_CONTROLS)
    assert [lab for lab, c in GRID if c.get("vllm_w4a8_view_fix")] == [l for l in GRID_LABELS if "w4a8" in l]
    fd = formats_dict()
    for lab, cfg in FORMATS + BF16KV + NVFP4_EMU + CAL1 + SMOOTH_EMU:
        assert cfg["ckpt"] is None or cfg["ckpt"] in EXPORTS or cfg["ckpt"] in LLMC_EXPORTS, lab
        e = cfg["expect"]
        assert e["kv"] in ("auto", "unit", "calibrated", "static", "per_head", "computed"), lab
        assert cfg.get("kv_from") is None or (cfg["kv_from"] in EXPORTS and cfg["ckpt"] is None), lab
        assert (e["kv"] == "computed") == bool((cfg.get("llm_kwargs") or {}).get("calculate_kv_scales")), lab
        assert (e["kv"] == "auto") == ("--kv-cache-dtype fp8" not in cfg["args"]), lab
        skip = tuple(e.get("kv_bf16_layers", ()))
        if skip:
            assert f"--kv-skip-layers {','.join(str(i) for i in skip)}" in cfg["args"], lab
        else:
            assert "--kv-skip-layers" not in cfg["args"], lab
        if e["kv"] in ("calibrated", "per_head"):
            assert cfg["ckpt"] is not None or cfg.get("kv_from"), lab
        if lab in NVFP4_EMU_LABELS + SMOOTH_EMU_LABELS:
            assert cfg["ckpt"] is None and cfg["env"].get("SPECOPD_SAMPLER_QUANT_STATIC") == STATIC_PLACEHOLDER, lab
            assert f"--quant {e['sampler_quant']}" in cfg["args"] and e["kv"] in ("static", "auto"), lab
        if lab in NVFP4_EMU_LABELS:
            assert (e["op_dtype"] is None) == ("a=" not in e["sampler_quant"]), lab
        if lab in SMOOTH_EMU_LABELS:
            assert e["act_attach"] and e["op_seen"] == "smooth_op_seen" and cfg["smooth"] in ("int8_sq", "int4_awq"), lab
    assert SMOOTH_LABEL == {"int8_sq": "mo_sqemu", "int4_awq": "mo_awqemu"}
    assert SMOOTH_KV == {"int8_sq": ("mo_sqemu_kv", "mo_sqemu_kv_s0"), "int4_awq": ("mo_awqemu_kv", "mo_awqemu_kv_s0")}
    for r, (a, b) in SMOOTH_KV.items():
        assert fd[a]["expect"]["kv"] == fd[b]["expect"]["kv"] == "static" and fd[a]["smooth"] == fd[b]["smooth"] == r
        assert fd[a]["args"] == "--kv-cache-dtype fp8 " + fd[SMOOTH_LABEL[r]]["args"] and (a, b) in SHARE_PAIRS
    for a, b in SHARE_PAIRS:
        assert a in fd and b in fd and fd[b]["expect"].get("kv_bf16_layers") == (0,)
        assert fd[a]["ckpt"] == fd[b]["ckpt"] and fd[a]["env"] == fd[b]["env"]
        assert fd[a].get("kv_from") == fd[b].get("kv_from")
    assert set(CAL1_SHARES) <= set(SHARE_PAIRS) and all(lab in fd for _, lab in CAL1_MAP)
    assert {fd[lab]["cal"] for lab in CAL1_LABELS} >= {"2", "3", "4", "8"}
    assert FLOOR in fd and UNIT in fd
    assert all(fd[lab]["expect"]["kv"] == "auto" and fd[lab]["ckpt"] in EXPORTS for lab in BF16KV_LABELS)
    assert set(NVFP4_LABELS) <= {lab for lab, _ in FORMATS}
    for lab in NVFP4_LABELS:
        assert fd[lab]["ckpt"] == "nvfp4_kv" and fd[lab]["expect"].get("verl_nvfp4_patch") is False, lab
        assert fd[lab].get("cudagraph") == fd[lab]["expect"].get("cudagraph_mode"), lab
    assert request_seed(0, 0) == 20261008 and request_seed(255, 1) == 20261008 + 2551
    for name, ex in EXPORTS.items():
        assert ex["recipe"].startswith("general/ptq/") and all(0 <= i < N_LAYERS for i in ex["disable_layers"]), name
    assert bf16(0.3) == 0.30078125 and bf16(0.1) == 0.10009765625 and bf16(1.0) == 1.0
    assert bf16(1.00390625) == 1.0 and bf16(1.01171875) == 1.015625     # ties round to even
    assert load_anchors("") == {}
    print("[mo_common] SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(selftest())

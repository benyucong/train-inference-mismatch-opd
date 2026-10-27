# Mismatch Budgets: Cheaper Rollouts for On-Policy Distillation

Code for the paper of the same name (under double-blind review). It holds the code and the dataset preparation only:
no experiment logs, outputs or results.

## Overview

On-policy distillation (OPD) trains a student on its own rollouts, scored token by token by a fixed teacher. Cheaper
rollout engines (quantized inference, asynchronous rollouts from stale weights) make the engine differ from the model
being trained. This repository measures how much of that training–inference mismatch OPD tolerates, estimates an
engine's mismatch offline before any training run, and compares OPD with GRPO and REINFORCE under the same engines.

```
verl/            the training framework, with our changes (Apache-2.0; see verl/LICENSE)
scripts/         the training launcher and its environment
bench/           the offline mismatch estimate, the rollout-speed benchmark, the prefix-shift measurement,
                 held-out evaluation and the RL comparison's data preparation
refresh/         answer sampling and grading for held-out evaluation, and the evaluation sets' record
custom_reward/   answer checking for the RL comparison and for accuracy
tools/probe0/    the exact gradient probe
logs/slurm/      where every job writes its Slurm log
```

Our changes to verl are mostly in `verl/verl/utils/specopd_*.py`:

| File | What it adds |
|---|---|
| `specopd_tim.py` | weight noise on the rollout engine: a fixed Gaussian offset after every weight sync |
| `specopd_modelopt.py` | NVIDIA Model Optimizer recipes on vLLM's real kernels (FP8, and FP8 with the first and last layers in BF16) |
| `specopd_quant.py`, `specopd_nvfp4.py`, `specopd_smooth.py` | emulated formats (NVFP4, INT8 SmoothQuant, INT4 AWQ): each tensor rounded to the format's grid with Model Optimizer's scales, then computed in BF16 |
| `specopd_kvskip.py`, `specopd_kvaudit.py` | keeping chosen layers' KV cache in BF16, and an audit of what each engine's cache stores |
| `specopd_stale.py` | rollouts from stale weights, emulated inside the synchronous trainer |

Score centering for OPD, GRPO and REINFORCE is in `verl/verl/trainer/ppo/score_centering.py`; truncated importance
sampling uses verl's rollout correction.

## Getting Started

### Environment Setup

We used vLLM 0.19.1 and PyTorch 2.10 on GH200 (aarch64) nodes with Slurm, four GPUs per node, plus NVIDIA Model
Optimizer 0.47 and llm-compressor 0.11 for the quantized exports. `scripts/cluster_env.sh` loads the Python stack and
sets the paths; edit its module lines and `CLUSTER_ROOT` for your site, and the `#SBATCH` lines of the job scripts
(`YOUR_ACCOUNT`, `PARTITION_*`). Python scripts read the same root from the `CLUSTER_ROOT` environment variable. Run
the commands below from the repository root after `source scripts/cluster_env.sh`.

The scripts expect four Python environments under `$CLUSTER_ROOT`:

| Environment | Used for |
|---|---|
| `venv-specopd` | training and the offline estimate: vLLM, PyTorch, `verl/requirements.txt` (the vendored `verl/` is put on `PYTHONPATH`, not installed) and the math verifier's packages (latex2sympy2_extended, math_verify, sympy) |
| `venv-refresh` | answer sampling for held-out evaluation (`refresh/`): vLLM 0.19.1, transformers, pandas |
| `venv-modelopt` | NVIDIA Model Optimizer exports (`bench/mo_export.py`) |
| `venv-llmc` | llm-compressor exports (`bench/llmc_kv_export.py`, the W4A8 export in `bench/mo_grid_export.py`) |

Jobs run offline by default (`HF_HUB_OFFLINE=1`): download models and datasets beforehand. Submit every job from the
repository root; the repository ships an empty `logs/slurm/`, where every job writes its log (Slurm does not create
`--output` directories).

### Models

The default pair is the DeepSeek-R1-Distill-Qwen-7B student (revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`) and
the Skywork-OR1-Math-7B teacher (revision `fcef970f1d10682f13c182b280146a194ed57d1e`). Download both into the
Hugging Face cache at `$CLUSTER_ROOT/hf` (`scripts/cluster_env.sh` sets `HF_HOME` to it). The bench, probe and
evaluation scripts read these revisions' snapshot directories; the launcher and the evaluation jobs' teacher lookup
take the first snapshot of each model, so keep only these revisions in the cache.

Most of the paper's training runs on this pair also decode rollouts with an EAGLE-3 draft head (`SPEC=eagle3`). For it,
download `thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3`, add `"max_position_embeddings": 131072` to its
`config.json`, and put it at `$CLUSTER_ROOT/models/r1-7b-eagle3-maxpos131072` (or set `DRAFTER=<dir>`).

### Data

`$CLUSTER_ROOT/workspace/datasets` holds, in verl's RL parquet format (a chat `prompt`, `reward_model.ground_truth`,
`data_source`):

- `dapo-math-17k.parquet`: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
  (revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`), whose 1,791,700 rows repeat each problem 100 times, reduced
  to the first row of each problem ID (17,917 rows):

  ```python
  import pandas as pd
  df = pd.read_parquet("data/dapo-math-17k.parquet")   # the Hugging Face file
  df = df[~df["extra_info"].map(lambda e: e["index"]).duplicated()].reset_index(drop=True)
  df.to_parquet("dapo-math-17k.parquet", index=False)
  ```

  `bench/ho_select.py`, `bench/mo_calset.py` and `bench/rl_data.py` check that the file holds exactly these rows in
  this order (by its sha256, or by a hash of its rows when another pandas or pyarrow version wrote it), because the
  training order and every derived set index its rows; `ho_select.py` and `rl_data.py` accept another file with
  `--allow-other-data`.
- `test_data/<BENCH>/test.parquet` for AIME24 (30 problems, from
  [NuminaMath](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)), AIME25
  (30), AMC23 (the 83-problem AMC12 2022 and 2023 set) and MATH-500. `refresh/t3_acc2_freeze.json` records the sha256
  and size of AIME24, AIME25 and AMC23; `bench/tim_eval2.sbatch` checks them and runs without the check when given
  `FREEZE=`.

Held-out KL uses AIME24 and MATH-500 in full (530 problems); accuracy uses AIME24, AIME25 and AMC23. The rest is
derived from these files:

- `bench/ho_select.py` selects the RL comparison's held-out problems: `order`, `neardup`, `select`, `check`, then
  `write --ho-dir $CLUSTER_ROOT/ho` (its docstring gives the rule); `bench/ho_eval.py freeze` then writes the
  `ho_freeze.json` that `bench/ho_eval.sbatch` reads.
- `bench/mo_calset.py` builds the offline estimate's calibration set: 256 training problems drawn outside the held-out
  pool listed in `bench/mo_calset_excl.json`, two responses each (`bench/mo_formats.sbatch` with `CALSET_MODE=build`
  runs it).
- `bench/rl_data.py` builds the RL comparison's training subset (see below).
- `refresh/t3_acc2_freeze.json` lists the AMC23 problems excluded for overlap with DAPO-Math-17k, with a note for every
  problem the overlap check flagged.

### Training

`scripts/cluster_opd_sync.sh` launches every run; environment variables choose the method and the rollout engine.

```bash
# OPD with the standard BF16 rollout engine
STEPS=200 sbatch scripts/cluster_opd_sync.sh

# weight noise on the rollout engine (sigma times each matrix's standard deviation)
SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# Model Optimizer FP8 rollouts: every layer, or with the first and last layers in BF16
SAMPLER_MODELOPT=fp8 STEPS=200 sbatch scripts/cluster_opd_sync.sh
SAMPLER_MODELOPT=fp8fl STEPS=200 sbatch scripts/cluster_opd_sync.sh

# vLLM's FP8 KV cache, with the first layer's cache in BF16
KV_DTYPE=fp8 KV_SKIP_LAYERS=0 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# emulated NVFP4 weights and activations with an FP8 cache
SAMPLER_QUANT=w=nvfp4,a=nvfp4 KV_DTYPE=fp8 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# fully asynchronous rollouts from weights k versions old after the pipeline fills (k + 1 for a group trained out of
# order; one answer can span two versions)
ASYNC_MODE=fully_async ASYNC_STALENESS=4 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# per-token corrections: score centering (SC=1) or truncated importance sampling (RC_IS=token)
SC=1 SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh
```

Add `SPEC=eagle3` to match the paper's runs, and `SAVE_FREQ=<n>` to keep checkpoints for evaluation (the default
saves none). The paper's runs without weight noise also set `SAMPLER_NOISE=0`, which adds nothing and logs the
engine's projection on the noise direction. `SAMPLER_MODELOPT=fp8|fp8fl` reads `$CLUSTER_ROOT/mo/scales/<recipe>.json` and
`SAMPLER_QUANT=...nvfp4...` reads `$CLUSTER_ROOT/mo/nvfp4emu/nvfp4_static.json`; the offline-estimate section below
makes both, and `SAMPLER_MODELOPT_SCALES` / `SAMPLER_QUANT_STATIC` point elsewhere. The comments in
`scripts/cluster_opd_sync.sh` document every knob.

### Rollout Speed

`bench/speed_read.py` computes updates per hour from training logs. It reads steps 3 to N (an asynchronous run from
its step-2 to its step-N time stamps, which leaves out the pipeline fill and the end-of-run drain) and gives each
run's ratio to the first log; the paper's 200-step runs are read with `--last 200`. The comparison at the training
batch without the draft head runs 30 steps of each configuration:

```bash
STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
KV_DTYPE=fp8 KV_SKIP_LAYERS=0 STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
SAMPLER_MODELOPT=fp8fl STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
ASYNC_MODE=fully_async ASYNC_STALENESS=4 STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh

# BF16 first: each run's updates per hour over steps 3-30 and its ratio to BF16
python3 bench/speed_read.py <bf16>.log <fp8 cache>.log <fp8fl>.log <async>.log --last 30
```

### RL Comparison

GRPO and REINFORCE train the same student without a teacher, on the DAPO problem followed by JustRL's instruction
(" Please reason step by step, and put your final answer within \boxed{}."). The training problems are the DAPO
problems, in OPD's training order, on which the untrained student answered between 1 and 7 of 8 samples correctly.
`bench/rl_data.py` builds them; its docstring gives every rule.

```bash
# 1. the reformatted training file, row-aligned with the DAPO parquet (CPU)
$PYTHON_BIN bench/rl_data.py jr          # -> $CLUSTER_ROOT/workspace/datasets/dapo-math-17k-jr.parquet (+ .json)
# 2. (optional) the training order (data.shuffle=True, data.seed=1) as a JSON list
$PYTHON_BIN bench/rl_data.py order       # -> $CLUSTER_ROOT/rl_data/order_seed1_skip1.json
# 3. the base student's 8 samples per prompt for order positions 0-7,999: eight one-GPU jobs of 1,000 positions
for k in 0 1 2 3 4 5 6 7; do
    POS_LO=$((1000*k)) POS_HI=$((1000*(k+1))) sbatch bench/rl_data.sbatch
done
# 4. per-prompt correct counts, once every job has finished
$PYTHON_BIN bench/rl_data.py summarize   # -> $CLUSTER_ROOT/rl_data/base_pass/{per_prompt.csv,summary.json}
# 5. the training subset: 1-7 of 8 correct, in increasing order position, the first 16 x 200 problems
$PYTHON_BIN bench/rl_data.py subset      # -> $CLUSTER_ROOT/workspace/datasets/dapo-math-17k-jr.subset.parquet (+ .json)
```

`jr` and `subset` report whether their files equal the ones the paper's runs used. A job submitted again resumes from
its last finished chunk, and `subset` refuses an incomplete pass. If fewer than 3,200 problems qualify, the subset's
record gives the number of 16-problem steps it holds (train with that as `STEPS`); below 1,600 problems (100 steps)
`subset` refuses, and `--min-steps` moves that floor. The base pass depends on vLLM's numerics, so the subset can
differ on other hardware.

```bash
SUB=$CLUSTER_ROOT/workspace/datasets/dapo-math-17k-jr.subset.parquet
# GRPO: two optimizer steps per training step (PPO mini-batch of 8), PPO ratio clipped to [0.8, 1.28]
ALGO=grpo TRAIN_FILES=$SUB SHUFFLE=False BATCH=16 PPO_MINI=8 CLIP_RATIO_HIGH=0.28 STEPS=200 SAVE_FREQ=50 \
    SAMPLER_NOISE=0 SPEC=eagle3 sbatch --nodes=1 scripts/cluster_opd_sync.sh
# REINFORCE: the raw 0/1 reward, no baseline, one optimizer step per training step
ALGO=reinforce TRAIN_FILES=$SUB SHUFFLE=False BATCH=16 PPO_MINI=16 CLIP_RATIO_HIGH=0.28 STEPS=200 SAVE_FREQ=50 \
    SAMPLER_NOISE=0 SPEC=eagle3 sbatch --nodes=1 scripts/cluster_opd_sync.sh
```

`SHUFFLE=False` reads the subset in its row order, 16 problems per step. To test a rollout engine, replace
`SAMPLER_NOISE=0` with that engine's knobs from the Training section. The learning rate (1e-6), 8 samples per problem,
16,384 response tokens and temperature 1 are the launcher's defaults.

### Offline Mismatch Estimate and Admission Check

`bench/mo_common.py` lists every engine configuration; `bench/mo_measure.py q --label <configuration>` samples
responses with that engine, and `bench/mo_measure.py score` re-scores them with a BF16 engine. The `bench/mo_*.sbatch`
jobs run whole groups, and the `*_read.py` scripts apply the validity checks. Each job's header lists the variables
it needs; every job writes only to its own output directory `D`, which must exist and be empty, and the later jobs
read the formats job's output directory as `BASE`. In order:

1. `bench/mo_formats.sbatch` (D, CALSET, `CALSET_MODE=build`): the calibration set, the Model Optimizer exports and
   their engines.
2. `bench/mo_formats_nvfp4.sbatch` (D, BASE, CALSET): the NVFP4 configurations on their own; the two emulated ones
   run without CUDA graphs.
3. `bench/mo_nvfp4emu.sbatch` (D, BASE, NV = step 2's D, CALSET) and `bench/mo_smoothemu.sbatch` (D, BASE, CALSET,
   `RECIPE=int8_sq|int4_awq`): the emulated formats. Running `mo_nvfp4emu.sbatch` with
   `D=$CLUSTER_ROOT/mo/nvfp4emu` writes the static file the emulated NVFP4 training engine reads by default.
4. `bench/mo_bf16kv.sbatch`: Model Optimizer FP8 with a BF16 cache. vLLM serves the `fp8_kv` export with the FP8 cache
   its config declares, so this job runs on a copy without that entry: first
   `python3 bench/mo_grid_export.py nokv --src $BASE/ckpt/fp8_kv --out $CKPT2/fp8_kv`, then submit with D, BASE,
   CALSET and CKPT2.
5. `bench/mo_grid.sbatch` (D, BASE, CALSET): the W8A16 and W4A8 combinations.

The training launcher's scale files come from the formats job's exports:
`$PYTHON_BIN bench/mo_scales.py make --export-dir $BASE/ckpt/fp8_kv --recipe fp8 --out $CLUSTER_ROOT/mo/scales/fp8.json`,
and the same with `fp8_kv_fl`, `fp8fl` and `fp8fl.json`. The readers also take an optional JSON file of reference 95%
intervals for their anchors and controls, `{label: [lo, hi]}`, which the jobs pass when `ANCHORS=<file>` is exported;
without it they check validity only and say that the interval check was skipped. `bench/share_stats.py` holds the
first-layer share statistics `bench/mo_formats_read.py` uses: (A - B) / (A - F), with a paired bootstrap over prompts.
`bench/mo_export.py` and `bench/llmc_kv_export.py` build the quantized checkpoints from the calibration set.

`bench/g0_kernels.py` benchmarks rollout speed and memory of the lossy configurations on real vLLM kernels; its
`report` needs the admissibility bar, as `--bar <nats/token>` or as the tolerance of a `--dtol-source` file.

### Prefix Shift

`bench/prefix_shift.py` (job `bench/prefix_shift.sbatch`) splits the update's error into the per-token error at the
engine's own prefixes and the prefix shift. Its inputs come from earlier steps:

- `bench/gap_probe.py build`, `measure` and `fp8` (one process each; its docstring describes them): the frozen
  prefixes, the weight scales and the engine floor;
- the base student's held-out samples from `bench/ho_eval.py`, and a trained checkpoint;
- for the smoke job, the exact gradient probe's part-(a) traces (below).

The header of `bench/prefix_shift.sbatch` lists the variables that point to them and the job order (the smoke job; the
vllm jobs for both checkpoints; then each hf job after its own vllm job), with one example line per job.

### Exact Gradient Probe

`tools/probe0/probe0.py` (job `tools/probe0/probe.sbatch`) computes in closed form, in parameter space, the gradient
error that a noisy sampler causes in OPD's reverse-KL update. It runs on the base student's own responses
(`A_TRACES`) and on AIME24 samples (`B_TRACES`: the `gen_T1_p1.jsonl` that `refresh/t3_acc.py gen --model <base
student> --tokenizer <base student> --out-dir <dir> --temps 1.0:1.0` writes). No script in this repository writes the
part-(a) traces: sample them with vLLM following the format and recipe in the header of `tools/probe0/probe.sbatch`.
Example: `SIGMAS=nhigh=0.15 A_TRACES=<a.jsonl> B_TRACES=<dir>/gen_T1_p1.jsonl sbatch tools/probe0/probe.sbatch`.

### Validation

`bench/tim_eval.sbatch` evaluates one checkpoint: `refresh/t3_acc.py` samples and grades answers on AIME24 (8 per
problem) and MATH-500 (1 per problem), and `bench/tim_kl_eval.py` scores the same answers with the checkpoint and the
teacher to give held-out KL. `bench/tim_eval2.sbatch` adds the accuracy composite (`refresh/t3_acc2.py`): avg@16,
weighted by benchmark, over the problems `refresh/t3_acc2_freeze.json` keeps. The job first checks the benchmark files
against that record and stops if they differ; `FREEZE=` runs without it, on the full benchmarks.
`$PYTHON_BIN refresh/t3_acc2.py decontam --freeze refresh/t3_acc2_freeze.json --check` recomputes the overlap check and
fails unless it reproduces the record's exclusions.

For the RL comparison, `bench/ho_eval.py` (job `bench/ho_eval.sbatch`) computes held-out accuracy; RL checkpoints are
evaluated on the prompt format they train on.

The evaluation jobs find a training job's checkpoint from `TRAIN_JOB` and `STEP` under
`$RUNS_ROOT/*/checkpoints/specopd/`, where `RUNS_ROOT` defaults to the parent directory of the checkout.
`bench/tim_eval.sbatch` and `bench/tim_eval2.sbatch` also take `CKPT=<hf dir>`; `bench/ho_eval.sbatch` takes
`CKPT=base` for the untrained student.

## Citation

To be added after the review.

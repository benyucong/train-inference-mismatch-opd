# How Much Training–Inference Mismatch Can On-Policy Distillation Tolerate?

**Paper:** [arXiv:2611.11450](https://arxiv.org/abs/2611.11450) · **Authors:** Cong Yu, Ziting Wang, Songlin Jiang,
Sitong Zhang, Chengliang Chai, Youhe Jiang, Eiko Yoneki and Tuo Shi · **License:** [Apache-2.0](LICENSE)

On-policy distillation (OPD) trains a student language model on its own responses, with a fixed teacher scoring every
token. Cheaper rollout engines (quantized inference, asynchronous rollouts from older weights) save time and memory,
but the engine that writes the responses then differs from the model being trained. We measure this training–inference
mismatch as a per-token KL divergence between the rollout engine and the trained model. This repository contains the
code and the dataset preparation for the paper's experiments: it trains OPD under each rollout engine, estimates an
engine's mismatch offline before any training run, and compares OPD with GRPO and REINFORCE under the same engines.
The experiment plan, the training logs, the analysis outputs and the scripts behind the paper's tables and figures will
follow.

Main results:

- On a 7B student, OPD keeps its training quality (its KL to the teacher on held-out problems) up to over 200 times the
  mismatch of a standard BF16 engine, and fails at about twice that level. We call the largest tolerated mismatch OPD's
  mismatch budget. Under the same engines, REINFORCE breaks at about 12 times and GRPO at about 60 times.
- An offline admission check estimates a rollout configuration's mismatch and compares it with the budget in minutes
  on one GPU, without a training run.
- Within the budget, FP8 KV caches hold at least 1.93 times as many tokens as BF16, and asynchronous rollouts with FP8
  weights (first and last layers in BF16) reach the same training quality 1.62 times sooner than synchronous BF16
  training.

## Contents

| Path | What it holds |
|---|---|
| `verl/` | the training framework: verl at upstream commit `fbb4b3a8` with our changes (Apache-2.0; see `verl/LICENSE`) |
| `scripts/` | the training launcher and its environment |
| `bench/` | the offline mismatch estimate, the rollout-speed benchmark, the prefix-shift measurement, held-out evaluation and the RL comparison's data preparation |
| `refresh/` | answer sampling and grading for held-out evaluation, and `refresh/t3_acc2_freeze.json`, the record of the evaluation sets |
| `custom_reward/` | answer checking for the RL comparison and for accuracy |
| `tools/probe0/` | the exact gradient probe: the gradient error that a noisy rollout engine causes in OPD's update, computed in closed form |
| `logs/slurm/` | where every job writes its Slurm log |

Most of our changes to verl are in `verl/verl/utils/specopd_*.py`:

| File | What it adds |
|---|---|
| `specopd_tim.py` | weight noise on the rollout engine: a fixed Gaussian offset after every weight sync |
| `specopd_modelopt.py` | NVIDIA Model Optimizer recipes on vLLM's real kernels (FP8, and FP8 with the first and last layers in BF16) |
| `specopd_quant.py`, `specopd_nvfp4.py`, `specopd_smooth.py` | emulated formats (NVFP4, INT8 SmoothQuant, INT4 AWQ): each tensor rounded to the format's grid with Model Optimizer's scales, then computed in BF16 |
| `specopd_kvskip.py`, `specopd_kvaudit.py` | keeping chosen layers' KV cache in BF16, and an audit of what each engine's cache stores |
| `specopd_stale.py` | stale-weight rollouts emulated inside the synchronous trainer (the paper's stale-weight results use verl's asynchronous modes, `ASYNC_MODE=fully_async` and `ASYNC_MODE=colocate`) |

The weight sync of fully asynchronous runs is in `verl/verl/checkpoint_engine/specopd_pynccl_checkpoint_engine.py`.
Two optional per-token corrections for mismatch are included: score centering
([arXiv:2609.20807](https://arxiv.org/abs/2609.20807); `SC=1`, for OPD, GRPO and REINFORCE) in
`verl/verl/trainer/ppo/score_centering.py`, and truncated importance sampling (`RC_IS=token`) through verl's rollout
correction.

## Setup

### Hardware and Software

The paper's runs used GH200 (aarch64) nodes with Slurm and four GPUs per node (GPU driver 595.91.07, CUDA 13.2).

| Job | GPUs |
|---|---|
| OPD | two nodes: on the first, the trainer and four rollout replicas share the node's four GPUs (one replica per GPU); the second serves the teacher |
| OPD, fully asynchronous | the same two nodes, with two of the first node's GPUs for the trainer and two for rollout replicas (`ASYNC_ROLLOUT_GPUS=2`, the default) |
| GRPO, REINFORCE | one node (no teacher) |
| every job in `bench/` and `tools/probe0/` | one GPU |

### Configure Your Cluster

1. Create four Python environments under `$CLUSTER_ROOT` (table below). The paper used vLLM 0.19.1 and PyTorch 2.10,
   NVIDIA Model Optimizer 0.47 in `venv-modelopt` and llm-compressor 0.11 in `venv-llmc`.
2. In `scripts/cluster_env.sh`, which loads the Python stack and sets the paths, edit the module lines and
   `CLUSTER_ROOT`. Python scripts read the same root from the `CLUSTER_ROOT` environment variable, and the script sets
   `PYTHON_BIN` to the `venv-specopd` interpreter used by the `$PYTHON_BIN` commands below.
3. In the job scripts, edit the `#SBATCH` lines (`YOUR_ACCOUNT`, `PARTITION_*`).
4. Download the models and datasets beforehand: jobs run offline by default (`HF_HUB_OFFLINE=1`).
5. Run every command, and submit every job, from the repository root after `source scripts/cluster_env.sh`. Each job
   writes its log to `logs/slurm/`, which the repository ships empty because Slurm does not create `--output`
   directories.

| Environment | Used for |
|---|---|
| `venv-specopd` | training and the offline estimate: vLLM, PyTorch, `verl/requirements.txt` (the vendored `verl/` is put on `PYTHONPATH`, not installed) and the math verifier's packages (latex2sympy2_extended, math_verify, sympy) |
| `venv-refresh` | answer sampling for held-out evaluation (`refresh/`): vLLM 0.19.1, transformers, pandas |
| `venv-modelopt` | NVIDIA Model Optimizer exports (`bench/mo_export.py`) |
| `venv-llmc` | llm-compressor exports (`bench/llmc_kv_export.py`, the W4A8 export in `bench/mo_grid_export.py`) |

### Models

The main pair, used throughout unless a section says otherwise:

| Role | Model | Revision |
|---|---|---|
| student | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` |
| teacher | `Skywork/Skywork-OR1-Math-7B` | `fcef970f1d10682f13c182b280146a194ed57d1e` |
| draft head (EAGLE-3) | `thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3` | `10c244711d720e375401f490326133f135c411fa` |

Download the student and the teacher at these revisions into the Hugging Face cache at `$CLUSTER_ROOT/hf`
(`scripts/cluster_env.sh` sets `HF_HOME` to it), and keep no other revision of them there. The bench, probe and
evaluation scripts read these revisions' snapshot directories, but the launcher and the evaluation jobs' teacher lookup
take the first snapshot of each model, which could otherwise be a different revision.

With `SPEC=eagle3`, rollouts are decoded with the draft head, which proposes 3 tokens (`SPEC_K` defaults to 3); the
launcher's default is `SPEC=none`. The paper's OPD, GRPO and REINFORCE
training on the main pair sets `SPEC=eagle3`, except the memory tests at 1,024 and 1,536 sequences per step and the
30-step speed runs, which set `SPEC=none`. The offline estimates, the evaluations and the smaller pairs' runs use no
draft head. To set it up, download the draft head, add `"max_position_embeddings": 131072` to its `config.json`
(without the key, vLLM builds the head's position table for only 2,048 positions), and put it at
`$CLUSTER_ROOT/models/r1-7b-eagle3-maxpos131072`, or set `DRAFTER=<dir>`.

**Smaller pairs.** `PAIRS` in `bench/mo_common.py` records the paper's two smaller pairs:

| Pair | Student | Teacher | Launcher settings |
|---|---|---|---|
| `r1_1p5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` at `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` | `hbx/JustRL-DeepSeek-1.5B` at `0637e4096c789c67f9eecbe8355e0bdeddede1c2` | `SPEC=none VERL_CONTINUOUS_TOKEN_MODEL_FAMILY=deepseek` |
| `llama_3b` | `unsloth/Llama-3.2-3B-Instruct` at `006f5dcd1393c3add266de40994ba96225e9689d` | `TMLR-Group-HF/GT-Llama-3.2-3B-Instruct-MATH` at `b6a0ab79ab4802ef77334532dd1936bb1d219cf9` | `SPEC=none VERL_CONTINUOUS_TOKEN_MODEL_FAMILY=default` |

To train one, set `STUDENT` and `TEACHER` to the two snapshot directories and add the pair's launcher settings. A Model
Optimizer FP8 engine needs the pair's own calibration set, FP8 export and scale file. The export job builds the first
two (its header lists the steps, including the refit check in `bench/mo_refit.sbatch`), and the scale file comes from
its export; pass that file to the launcher as `SAMPLER_MODELOPT_SCALES`:

```bash
MO_PAIR=<pair> sbatch --job-name mo_pairexport_<pair> bench/mo_pairexport.sbatch
$PYTHON_BIN bench/mo_scales.py make --export-dir $CLUSTER_ROOT/mo/pairs/<pair>/ckpt/fp8_kv --recipe fp8 \
    --out $CLUSTER_ROOT/mo/scales/fp8_<pair>.json
```

### Data

The training set and the benchmarks live in `$CLUSTER_ROOT/workspace/datasets`, in verl's RL parquet format (a chat
`prompt`, `reward_model.ground_truth` and `data_source`).

**Training set.** `dapo-math-17k.parquet` is
[DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (revision
`65877096c24ffa7abc4e4fa5edb95cf3413a5674`). Its 1,791,700 rows repeat each problem 100 times; keep the first row of
each problem ID (17,917 rows):

```python
import os, pandas as pd
df = pd.read_parquet("<download dir>/data/dapo-math-17k.parquet")   # the Hugging Face file
df = df[~df["extra_info"].map(lambda e: e["index"]).duplicated()].reset_index(drop=True)
df.to_parquet(os.path.join(os.environ["CLUSTER_ROOT"], "workspace/datasets/dapo-math-17k.parquet"), index=False)
```

The training order and every derived set refer to these rows by position. `bench/ho_select.py`, `bench/mo_calset.py`
and `bench/rl_data.py` therefore check that the file holds exactly these 17,917 rows in this order, by its sha256 or,
if another pandas or pyarrow version wrote it, by a hash of its rows. `ho_select.py` and `rl_data.py` accept another
file with `--allow-other-data`.

**Benchmarks.** Put each benchmark at `$CLUSTER_ROOT/workspace/datasets/test_data/<BENCH>/test.parquet`, downloaded
from Hugging Face in verl's RL parquet format, with each prompt ending in " Please reason step by step, and put your
final answer within \boxed{}.":

| Benchmark | Problems | Checksum |
|---|---|---|
| AIME24 | 30, from [NuminaMath](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf) | sha256 and size in `refresh/t3_acc2_freeze.json` |
| AIME25 | 30 | sha256 and size in `refresh/t3_acc2_freeze.json` |
| AMC23 | 83, the AMC12 2022 and 2023 set | sha256 and size in `refresh/t3_acc2_freeze.json` |
| MATH-500 | 500 | sha256 `2e4c43a3d52175a773816f99f327d3ed5e3d15af6f479d87196a702cdb4f5411` (77,936 bytes) |

`bench/tim_eval2.sbatch` checks the three files recorded in `refresh/t3_acc2_freeze.json` (it runs without the check
when given `FREEZE=`). Held-out KL (see [Evaluation](#evaluation)) uses AIME24 and MATH-500 in full (530 problems); the
accuracy composite uses AIME24, AIME25 and AMC23.

**Derived sets.**

- `bench/ho_select.py` selects the RL comparison's held-out problems. Run its subcommands in order: `order`, `neardup`,
  `select`, `check`, then `write --ho-dir $CLUSTER_ROOT/ho` (its docstring gives the rule). `bench/ho_eval.py freeze`
  then writes the `ho_freeze.json` that `bench/ho_eval.sbatch` reads.
- `bench/mo_calset.py` builds the calibration set from which the quantized exports take their scales: two responses of
  the untrained student to each of 256 training problems drawn outside the held-out pool listed in
  `bench/mo_calset_excl.json` (`bench/mo_formats.sbatch` with `CALSET_MODE=build` runs it). The offline estimate itself
  samples other prompts: two responses to each of AIME24 problems 0-7 and MATH-500 problems 0-7
  (`bench/mo_measure.py`).
- `bench/rl_data.py` builds the RL comparison's training subset (see [RL Comparison](#rl-comparison)).
- `refresh/t3_acc2_freeze.json` lists the AMC23 problems excluded for overlap with DAPO-Math-17k, with a note for every
  problem the overlap check flagged.

## Training

`scripts/cluster_opd_sync.sh` launches every run. Environment variables choose the method and the rollout engine, and
the comments in the script document each one. The paper's synchronous BF16 OPD run on the main pair:

```bash
SPEC=eagle3 SAVE_FREQ=100 SAMPLER_NOISE=0 STEPS=200 sbatch scripts/cluster_opd_sync.sh
```

- `STEPS` sets the run length; always pass it, since the launcher's default is 20.
- `SAVE_FREQ=100` keeps checkpoints at steps 100 and 200 for evaluation (the default saves none). The time spent saving
  counts toward each run's measured speed, and the time-to-quality runs save every 25 steps instead.
- `SAMPLER_NOISE=0` adds no noise, but it makes the run log the engine's projection on the noise direction and print
  the `[SPECOPD-TIM] step=0` line that the time-to-quality clock starts from. The paper's runs without weight noise set
  it.
- Everything else is a launcher default: 8 problems with 8 responses each per step, temperature 1, at most 16,384
  response tokens, a constant learning rate of 1e-6, data seed 1 (`SEED`) and noise seed 20261001
  (`SAMPLER_NOISE_SEED`).

The examples below show only the variables that choose the engine or the update; add
`SPEC=eagle3 SAVE_FREQ=100` (and `SAMPLER_NOISE=0` when the run has no weight noise) to match the paper's runs.

The FP8 and NVFP4 engines read files that the [offline estimate](#offline-mismatch-estimate-and-admission-check) makes:
`SAMPLER_MODELOPT=fp8|fp8fl` reads `$CLUSTER_ROOT/mo/scales/<recipe>.json`, and `SAMPLER_QUANT=...nvfp4...` reads
`$CLUSTER_ROOT/mo/nvfp4emu/nvfp4_static.json`. To use other files, set `SAMPLER_MODELOPT_SCALES` or
`SAMPLER_QUANT_STATIC`.

Rollout engines:

```bash
# weight noise (sigma times each matrix's standard deviation)
SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# Model Optimizer FP8: every layer, or with the first and last layers in BF16
SAMPLER_MODELOPT=fp8 STEPS=200 sbatch scripts/cluster_opd_sync.sh
SAMPLER_MODELOPT=fp8fl STEPS=200 sbatch scripts/cluster_opd_sync.sh

# vLLM's FP8 KV cache, with the first layer's cache in BF16
KV_DTYPE=fp8 KV_SKIP_LAYERS=0 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# emulated NVFP4 weights and activations with an FP8 cache
SAMPLER_QUANT=w=nvfp4,a=nvfp4 KV_DTYPE=fp8 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# fully asynchronous rollouts: once the pipeline fills, rollouts use weights ASYNC_STALENESS (here 4) versions old,
# or one more for a group trained out of order; a single answer can span two weight versions
ASYNC_MODE=fully_async ASYNC_STALENESS=4 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# stale weights and Model Optimizer FP8 together
ASYNC_MODE=fully_async ASYNC_STALENESS=8 SAMPLER_MODELOPT=fp8fl STEPS=200 sbatch scripts/cluster_opd_sync.sh

# verl's colocated asynchronous mode
ASYNC_MODE=colocate STEPS=200 sbatch scripts/cluster_opd_sync.sh
```

Changes to the update, and repeated runs:

```bash
# per-token corrections: score centering (SC=1) or truncated importance sampling (RC_IS=token); SC=probe computes
# the centering term but leaves the loss unchanged (the control)
SC=1 SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# ADV_OFFSET: a constant added to every token's advantage
ADV_OFFSET=0.15 SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh

# a second run of a configuration changes only the rollout sampling seed (default 42; replica r uses 42 + r)
ROLLOUT_SEED=43 SAMPLER_NOISE=0.15 STEPS=200 sbatch scripts/cluster_opd_sync.sh
```

### Reading a Training Log

Every step prints one line of verl's metrics (`step:<n> - key:value - ...`). The paper's per-run quantities come from
these fields:

| Quantity | Field |
|---|---|
| mismatch at a step | `rollout_corr/kl`; a run's mismatch is its mean over steps 1-200 (the admission check and Table 8 use steps 1-100) |
| training accuracy | `critic/score/mean`, the share of rollouts whose final answer is correct (OPD logs it but does not train on it) |
| the trainer's wait for rollouts | `timing_s/gen` (generation and teacher scoring in a synchronous run) |
| realized staleness of an asynchronous run | `training/off_policy/trajectory_staleness_worst/max` |
| cache use and preemptions (with `KV_METRICS=1`) | `kvmem/usage_max` and `kvmem/preemptions`; the capacity is `cap_tokens` on the `[SPECOPD-KV-CFG]` lines |

## Evaluation

Training writes checkpoints under `checkpoints/specopd/` in the checkout. The evaluation jobs find a training job's
checkpoint from `TRAIN_JOB` and `STEP` under `$RUNS_ROOT/*/checkpoints/specopd/`, where `RUNS_ROOT` defaults to the
parent directory of the checkout:

```bash
ARM=<run name> TRAIN_JOB=<job id> STEP=<step> sbatch bench/tim_eval.sbatch
```

`bench/tim_eval.sbatch` evaluates one checkpoint: `refresh/t3_acc.py` samples and grades answers on AIME24 (8 per
problem) and MATH-500 (1 per problem), and `bench/tim_kl_eval.py` scores the same answers with the checkpoint and the
teacher to give held-out KL.

`bench/tim_eval2.sbatch` adds the accuracy composite (`refresh/t3_acc2.py`): each benchmark's avg@16 (accuracy over 16
samples per problem), averaged over AIME24, AIME25 and AMC23 with each benchmark counting equally, on the problems
`refresh/t3_acc2_freeze.json` keeps. The job first checks the benchmark files against that record and stops if they
differ; `FREEZE=` runs it without the check, on the full benchmarks. To recompute the overlap check, run
`$PYTHON_BIN refresh/t3_acc2.py decontam --freeze refresh/t3_acc2_freeze.json --check`; it fails unless it reproduces
the record's exclusions.

Both jobs evaluate the main pair: they fix the tokenizer (`BASE`) and the teacher to its models, and `CKPT=<hf dir>`
replaces the student checkpoint.

For the RL comparison, `bench/ho_eval.py` (job `bench/ho_eval.sbatch`) computes held-out accuracy; RL checkpoints are
evaluated on the prompt format they train on, and `CKPT=base` evaluates the untrained student.

## Reproducing the Paper's Measurements

Figure and table numbers refer to the [arXiv paper](https://arxiv.org/abs/2611.11450).

### Rollout Speed

Paper: Figure 6, Tables 8 and 10. `bench/speed_read.py` computes updates per hour from training logs and each run's
ratio to the first log. With `--last N` it reads steps 3 to N; for an asynchronous run it measures from the step-2 to
the step-N time stamps, which leaves out the pipeline fill and the end-of-run drain.

**Staleness (Figure 6, Table 8).** Seven 200-step runs: the synchronous BF16 run, fully asynchronous runs with weights
up to 1, 2, 4 and 8 versions old, and Model Optimizer FP8 (first and last layers in BF16) at 4 and 8 versions old. The
4-version FP8 run is the [time-to-quality](#time-to-quality) run, which saves every 25 steps.

```bash
C="SPEC=eagle3 SAMPLER_NOISE=0 SAVE_FREQ=100 STEPS=200"
env $C sbatch scripts/cluster_opd_sync.sh                                    # synchronous BF16
for K in 1 2 4 8; do
    env $C ASYNC_MODE=fully_async ASYNC_STALENESS=$K sbatch scripts/cluster_opd_sync.sh
done
env $C ASYNC_MODE=fully_async ASYNC_STALENESS=8 SAMPLER_MODELOPT=fp8fl sbatch scripts/cluster_opd_sync.sh

# synchronous run first
python3 bench/speed_read.py <sync>.log <k1>.log <k2>.log <k4>.log <k8>.log <k4 fp8fl>.log <k8 fp8fl>.log --last 200
```

**The training batch without the draft head (Table 10).** 30 steps of each configuration:

```bash
STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
KV_DTYPE=fp8 KV_SKIP_LAYERS=0 STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
SAMPLER_MODELOPT=fp8fl STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh
ASYNC_MODE=fully_async ASYNC_STALENESS=4 STEPS=30 SAMPLER_NOISE=0 SPEC=none sbatch scripts/cluster_opd_sync.sh

# BF16 first: each run's updates per hour over steps 3-30 and its ratio to BF16
python3 bench/speed_read.py <bf16>.log <fp8 cache>.log <fp8fl>.log <async>.log --last 30
```

### Memory Tests

Paper: Figure 7, Table 9. The memory tests run 8 steps without the draft head, with `KV_METRICS=1` and one run per
engine, at 1,536 sequences per step (`BATCH=192 ROLLOUT_MAX_SEQS=384`, below) and at 1,024 (`BATCH=128` in place of
those two settings, with `SAMPLER_MODELOPT=fp8` as a fourth engine):

```bash
M="BATCH=192 ROLLOUT_MAX_SEQS=384 STEPS=8 SAMPLER_NOISE=0 SPEC=none KV_METRICS=1"
env $M sbatch scripts/cluster_opd_sync.sh
env $M KV_DTYPE=fp8 KV_SKIP_LAYERS=0 sbatch scripts/cluster_opd_sync.sh
env $M SAMPLER_MODELOPT=fp8fl sbatch scripts/cluster_opd_sync.sh
python3 bench/speed_read.py <bf16>.log <fp8 cache>.log <fp8fl>.log --last 8
```

`bench/speed_read.py` reports throughput. Read peak cache use and preemptions from the step lines of the training log:
peak cache use is the largest `kvmem/usage_max` over steps 3-8, and preemptions are the sum of `kvmem/preemptions` over
steps 3-8.

### Time to Quality

Paper: Table 11.

1. Train three 200-step runs that save a checkpoint every 25 steps:

   ```bash
   SAMPLER_NOISE=0 SPEC=eagle3 SAVE_FREQ=25 STEPS=200 sbatch scripts/cluster_opd_sync.sh
   for K in 8 4; do
       ASYNC_MODE=fully_async ASYNC_STALENESS=$K SAMPLER_MODELOPT=fp8fl SAMPLER_NOISE=0 SPEC=eagle3 SAVE_FREQ=25 \
           STEPS=200 sbatch scripts/cluster_opd_sync.sh
   done
   ```

2. Evaluate every checkpoint of each run, and the untrained student once (step 0), with `bench/tim_eval.sbatch`:

   ```bash
   for S in 25 50 75 100 125 150 175 200; do
       ARM=<run name> TRAIN_JOB=<job id> STEP=$S sbatch bench/tim_eval.sbatch
   done
   ARM=base CKPT=<the student's snapshot directory> sbatch bench/tim_eval.sbatch
   ```

   Each job writes `$CLUSTER_ROOT/eval/<ARM>_s<STEP>/kl_summary.json`, whose `ALL.kl_problem_mean` is the held-out KL.

3. Read the clock from the training log: the wall-clock time between two time-stamped lines.

   ```bash
   grep '\[SPECOPD-TIM\] step=0 ' logs/slurm/<run>.log              # start: the earliest of these lines
   grep 'Saving checkpoint to .*/global_step_' logs/slurm/<run>.log   # t(N): the line ending in global_step_N
   ```

   The start is the rollout engine's first weight load. `t(N)` includes the earlier checkpoints' saves and leaves out
   checkpoint N's own save and the start-up before training.

4. Compare with the two targets, both from the synchronous run. With KL(N) its held-out KL at step N and KL(0) the
   untrained student's, the targets are KL(200) and 90% of its improvement, KL(0) - 0.9 × (KL(0) - KL(200)). A run
   reaches a target at its first checkpoint at or below it, and the speed-up is the synchronous run's time divided by
   the run's time to the same target.

### RL Comparison

Paper: Figure 8. GRPO and REINFORCE train the same student without a teacher, on each DAPO problem followed by the
instruction JustRL uses, the same ending as the benchmark prompts (" Please reason step by step, and put your final
answer within \boxed{}."). The training problems are the DAPO problems, in OPD's training order, on which the untrained
student answered between 1 and 7 of 8 samples correctly. `bench/rl_data.py` builds them; its docstring gives every
rule.

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

`jr` and `subset` report whether their files equal the ones the paper's runs used. A `bench/rl_data.sbatch` job
submitted again resumes from its last finished chunk, and `subset` refuses an incomplete base pass. If fewer than 3,200
problems qualify, the subset's record gives the number of 16-problem steps it holds; train with that as `STEPS`. Below
1,600 problems (100 steps) `subset` refuses, and `--min-steps` moves that floor. The base pass depends on vLLM's
numerics, so the subset can differ on other hardware.

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
`SAMPLER_NOISE=0` with that engine's settings from [Training](#training). The learning rate (1e-6), 8 samples per
problem, 16,384 response tokens and temperature 1 are the launcher's defaults.

### Offline Mismatch Estimate and Admission Check

Paper: Figure 4, Table 7. `bench/mo_common.py` lists every engine configuration. To estimate one configuration's
mismatch, `bench/mo_measure.py q --label <configuration>` samples responses with that engine, and
`bench/mo_measure.py score` re-scores them with a BF16 engine. The `bench/mo_*.sbatch` jobs run whole groups of
configurations, and the `*_read.py` scripts check that each reading is valid.

Each job's header lists the variables it needs. `D` is the job's own output directory: it must exist and be empty, and
the job writes nowhere else. `CALSET` is the calibration set's directory, and `BASE` is the formats job's `D`, which
the later jobs read. Run the jobs in this order:

1. `bench/mo_formats.sbatch` (D, CALSET, `CALSET_MODE=build`): the calibration set, the Model Optimizer exports and
   their engines.
2. `bench/mo_formats_nvfp4.sbatch` (D, BASE, CALSET): the NVFP4 configurations on their own; the two emulated ones run
   without CUDA graphs.
3. `bench/mo_nvfp4emu.sbatch` (D, BASE, NV = step 2's D, CALSET) and `bench/mo_smoothemu.sbatch` (D, BASE, CALSET,
   `RECIPE=int8_sq|int4_awq`): the emulated formats. Running `mo_nvfp4emu.sbatch` with `D=$CLUSTER_ROOT/mo/nvfp4emu`
   writes the static file the emulated NVFP4 training engine reads by default.
4. `bench/mo_bf16kv.sbatch`: Model Optimizer FP8 with a BF16 cache. vLLM serves the `fp8_kv` export with the FP8 cache
   its config declares, so this job runs on a copy without that entry: first
   `python3 bench/mo_grid_export.py nokv --src $BASE/ckpt/fp8_kv --out $CKPT2/fp8_kv`, then submit with D, BASE,
   CALSET and CKPT2.
5. `bench/mo_grid.sbatch` (D, BASE, CALSET): the W8A16 and W4A8 combinations.

The training launcher's scale files come from the formats job's exports:

```bash
$PYTHON_BIN bench/mo_scales.py make --export-dir $BASE/ckpt/fp8_kv --recipe fp8 --out $CLUSTER_ROOT/mo/scales/fp8.json
$PYTHON_BIN bench/mo_scales.py make --export-dir $BASE/ckpt/fp8_kv_fl --recipe fp8fl --out $CLUSTER_ROOT/mo/scales/fp8fl.json
```

More on these scripts:

- The readers also take an optional JSON file `{label: [lo, hi]}` of reference 95% intervals for their anchors, the
  validity checks of a whole reading (for the formats job: the BF16 floor `mo_z_none` and the unit-scale FP8 cache
  `mo_kv8_unit`). The jobs pass it when `ANCHORS=<file>` is exported. Without it, the readers check validity and report
  that the interval check was skipped.
- `bench/share_stats.py` holds the first-layer share statistics that `bench/mo_formats_read.py` uses: (A - B) / (A - F),
  where A is a configuration's mismatch, B the same with the first decoder layer's KV cache in BF16 and F the BF16
  floor, with a paired bootstrap over prompts.
- `bench/mo_export.py` and `bench/llmc_kv_export.py` build the quantized checkpoints from the calibration set.
- `bench/g0_kernels.py` benchmarks the rollout speed and memory of the lossy configurations on real vLLM kernels. Its
  `report` needs the admissibility bar, as `--bar <nats/token>` or as the tolerance of a `--dtol-source` file.

### Prefix Shift

Paper: Figure 9. `bench/prefix_shift.py` (job `bench/prefix_shift.sbatch`) splits the update's error into the per-token
error at the engine's own prefixes and the prefix shift. Its inputs come from earlier steps:

- `bench/gap_probe.py build` (the frozen prefixes), `measure` (the weight scales) and `fp8` (the engine floor), one
  process each; its docstring describes them;
- the base student's held-out samples from `bench/ho_eval.py`, and a trained checkpoint;
- for the smoke job, the exact gradient probe's `A_TRACES` (see [Exact Gradient Probe](#exact-gradient-probe)).

The header of `bench/prefix_shift.sbatch` lists the variables that point to them and the job order (the smoke job; the
vllm jobs for both checkpoints; then each hf job after its own vllm job), with one example line per job.

### Exact Gradient Probe

Paper: Table 5. `tools/probe0/probe0.py` (job `tools/probe0/probe.sbatch`) computes in closed form, in parameter space,
the gradient error that a noisy sampler causes in OPD's reverse-KL update. It runs on two sets of traces:

- `A_TRACES`: the base student's own responses. Sample them with vLLM, following the format and recipe in the header
  of `tools/probe0/probe.sbatch`.
- `B_TRACES`: AIME24 samples, the `gen_T1_p1.jsonl` that `refresh/t3_acc.py gen` writes.

```bash
$CLUSTER_ROOT/venv-refresh/bin/python refresh/t3_acc.py gen --model <base student> --tokenizer <base student> \
    --out-dir <dir> --temps 1.0:1.0
SIGMAS=nhigh=0.15 A_TRACES=<a.jsonl> B_TRACES=<dir>/gen_T1_p1.jsonl sbatch tools/probe0/probe.sbatch
```

## Citation

```bibtex
@misc{yu2026mismatch,
  title         = {How Much Training--Inference Mismatch Can On-Policy Distillation Tolerate?},
  author        = {Cong Yu and Ziting Wang and Songlin Jiang and Sitong Zhang and Chengliang Chai and Youhe Jiang and Eiko Yoneki and Tuo Shi},
  year          = {2026},
  eprint        = {2611.11450},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.11450}
}
```

## License

The whole repository is released under the Apache License 2.0 (see [LICENSE](LICENSE)). `verl/` is a copy of verl,
also Apache-2.0, with its own [license](verl/LICENSE) and [notice](verl/Notice.txt).

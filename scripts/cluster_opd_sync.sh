#!/bin/bash
# =============================================================================
# Vanilla (logit-scoring) OPD on a GH200 (aarch64) cluster, SYNCHRONOUS trainer, UPSTREAM verl.
#
# Runs from the repository root:
#     verl/            upstream verl + minimal patches (git diff there)
#     custom_reward/   the math verifier reward (standalone package)
#     scripts/         this launcher, cluster_env.sh (PYTHONPATH -> this tree ONLY), cluster_start_ray.sh
#
#   cd <repository root>
#   sbatch scripts/cluster_opd_sync.sh                                     # veRL sequential OPD
#   SPEC=ngram_gpu sbatch scripts/cluster_opd_sync.sh                      # speculative decoding, model-free GPU n-gram drafter
#   REF_KL=1 SAVE_FREQ=25 STEPS=200 sbatch scripts/cluster_opd_sync.sh     # student drift
#   ALGO=grpo sbatch --nodes=1 scripts/cluster_opd_sync.sh                 # matched GRPO control (no teacher node)
#   ALGO=reinforce sbatch --nodes=1 scripts/cluster_opd_sync.sh            # matched plain-REINFORCE control (no teacher)
#
# ALGO=opd (default) | grpo | reinforce. grpo = the RL contrast to OPD on the SAME student, prompts and shape: distillation OFF,
# the task reward (custom_reward/specopd_reward.py, math verifier) optimized with algorithm.adv_estimator=grpo over
# n=ROLLOUT_N (8) samples per prompt, lr / batch / lengths / temperature / PPO_MINI=BATCH / 1 epoch as OPD, no KL
# loss, no entropy bonus. Every sampler-mismatch knob (SAMPLER_NOISE, SAMPLER_QUANT, ROLLOUT_QUANT, KV_DTYPE,
# STALENESS/STALE_*) acts on the rollout exactly as in OPD; SC unset|0|1 and RC_IS as in OPD (SC=1 = verl#8010's
# native bypass-REINFORCE score centering for RL; SC=probe is OPD-only on this bypass path -- RL's probe is
# SC_PATH=decoupled, below). One node: TRAINER_NNODES only.
# ALGO=reinforce = ALGO=grpo with the advantage replaced by PLAIN REINFORCE: A_t = the response's verifier score (0/1)
# broadcast over every response token, no baseline, no group mean, no std normalization
# (algorithm.adv_estimator=specopd_reinforce, verl/trainer/ppo/specopd_reinforce.py; E_q[A] >= 0 by construction).
# Everything else -- n=ROLLOUT_N samples per prompt, lr, batch, lengths, temperature, PPO_MINI, 1 epoch, no KL loss,
# no entropy bonus, every mismatch knob, SC unset|0|1, RC_IS, the refusals -- is ALGO=grpo's (ROLLOUT_N >= 2 is not
# required: no group baseline). Logged per step: reinforce/adv_mean (token-weighted mean advantage = E_q[A]),
# reinforce/adv_seq_mean, reinforce/correct_frac, reinforce/nonzero_adv_frac, upstream's critic/advantages/{mean,min,max}
# and critic/score/mean, and the grpo/* group metrics (descriptive only here: the REINFORCE advantage does not use them).
# Every run prints its full Hydra override list ([HYDRA-OVERRIDE] lines) before training starts.
#
# RL recipe knobs (JustRL-style). Opt-in; unset = the command line without them, byte for byte:
#   CLIP_RATIO_HIGH=<v> (ALGO=grpo|reinforce; v in (0, 1]): actor_rollout_ref.actor.clip_ratio_high=v, label _ch<v>.
#              clip_ratio_low stays verl's 0.2 and the dual clip c stays 3.0, so v=0.28 is the ratio range [0.8, 1.28].
#              It acts only where the PPO ratio departs from 1 (GRPO's second mini-batch at PPO_MINI=BATCH/2).
#   BATCH / PPO_MINI (existing): BATCH must be a multiple of PPO_MINI (BATCH/PPO_MINI optimizer updates per step).
#   SC_PATH=unset|bypass|decoupled: unset and bypass = the SC block below unchanged (the OPD SC arms and the bypass
#              REINFORCE arms, PPO_MINI=BATCH). decoupled (ALGO=grpo|reinforce, SC=1|probe): score centring added to
#              the run's OWN loss -- the decoupled PPO-clip loss on the trainer's recompute (loss_mode stays vanilla) --
#              in every PPO mini-batch, on every response token (verl/workers/utils/losses.py ppo_loss);
#              SC=probe computes and logs the term but adds it to no loss. Label _dsc1k<k> / _dscprobek<k>.
#
# The teacher only SCORES (prompt_logprobs); every sample is scored at its OWN EOS
# (agent_loop._run_agent_loop -> _agent_loop_postprocess), so the only barrier is the gather before
# the update. OPD_CP_TELEMETRY=1 prints one [OPD-CP] line per sample (t0/eos/done, one clock), from
# which the per-step critical path follows.
#
# REF_KL=1: a frozen reference (the initial student) is loaded and actor.use_kl_loss=True with
# kl_loss_coef=0 -- the ONLY purpose is the logged actor/kl_loss = estimate of KL(p_k || p_0) on the
# step's own rollout tokens. It never reaches the gradient: with use_task_rewards=False the whole
# ppo_loss term is zeroed (distillation_ppo_loss in verl/verl/trainer/distillation/losses.py) and the
# coef is 0 besides.
# =============================================================================

#SBATCH --job-name=sopd_sync
#SBATCH --account=YOUR_ACCOUNT
# A multi-node partition: the default layout uses two nodes of four GPUs.
#SBATCH --partition=PARTITION_GPU_MULTINODE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.log

set -xeuo pipefail
trap 'rc=$?; [ $rc -ne 0 ] && echo "[FATAL] ${BASH_SOURCE[0]}:${LINENO} rc=$rc cmd: ${BASH_COMMAND}" >&2' ERR

# $SLURM_SUBMIT_DIR, never $(dirname $BASH_SOURCE): sbatch spools the script, so BASH_SOURCE points
# at /var/spool/slurmd. Submit from the repository root.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}}"
cd "$REPO_ROOT"
[ -f "$REPO_ROOT/verl/verl/__init__.py" ] && [ -d "$REPO_ROOT/custom_reward" ] \
    || { echo "CONFIG ERROR: REPO_ROOT=$REPO_ROOT is not the repository root (submit from it)" >&2; exit 1; }
mkdir -p logs/slurm
source "$REPO_ROOT/scripts/cluster_env.sh"
export REPO_ROOT
W="${W:-$CLUSTER_ROOT}"
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/${USER}/ray_${SLURM_JOB_ID:-local}}"
mkdir -p "$RAY_TMPDIR"
export RAY_DEDUP_LOGS=0      # per-sample lines differ only in numbers; dedup would sample them
export OPD_CP_TELEMETRY=1
export SPECOPD_SPEC_METRICS=1   # per-step spec-decode counters from the rollout servers' /metrics (zeros w/o SPEC)
# Which verl is this? Refuse to run any verl other than this tree's by accident.
_VERL_FILE=$("$PYTHON_BIN" -c 'import verl; print(verl.__file__)')
case "$_VERL_FILE" in
    "$REPO_ROOT"/verl/*) ;;
    *) echo "CONFIG ERROR: verl imports from $_VERL_FILE, not $REPO_ROOT/verl" >&2; exit 1 ;;
esac

# --- experiment shape ------------------------------------------------------------
# DRY_RUN=1 (login node or local): every check runs, no Ray / srun is started, the full Hydra override list is printed
# ([HYDRA-OVERRIDE] lines, ray address = the placeholder DRY_RUN:0) and the script exits 0 before training. Unset or 0
# = unchanged behaviour.
DRY_RUN="${DRY_RUN:-0}"
case "$DRY_RUN" in 0|1) ;; *) echo "CONFIG ERROR: DRY_RUN must be 0 or 1 (got '$DRY_RUN')" >&2; exit 1 ;; esac
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
ALGO="${ALGO:-opd}"
case "$ALGO" in opd|grpo|reinforce) ;; *) echo "CONFIG ERROR: ALGO must be opd, grpo or reinforce (got '$ALGO')" >&2; exit 1 ;; esac
# the driver-side advantage estimator: OPD keeps grpo on the command line (its loss reads the teacher, not this)
_ADV_EST=grpo
if [ "$ALGO" = reinforce ]; then _ADV_EST=specopd_reinforce; fi
# The R1-Distill models need a model-family override on upstream verl: its agent loop picks the Qwen (ChatML)
# builder from model_type "qwen2", which requires <|im_end|> (r1_7b sets VERL_CONTINUOUS_TOKEN_MODEL_FAMILY below).
# TRIPLE presets (student <- teacher, default drafter):
#   r1_7b : DeepSeek-R1-Distill-Qwen-7B <- Skywork-OR1-Math-7B, EAGLE-3 (the DEFAULT)
#   q3_8b : Qwen3-8B <- Qwen3-32B, RedHat thinking EAGLE-3
#   q35_4b: Qwen3.5-4B <- Qwen3.5-27B, native MTP head
#   q3_4b_a3b: Qwen3-4B (thinking OFF) <- Qwen3-30B-A3B-Instruct-2507, no drafter (SPEC=none)
# Every preset value is overridable by its own env var.
TRIPLE="${TRIPLE:-r1_7b}"
_snap() { ls -d "$HF_HOME"/hub/models--"$1"/snapshots/*/ 2>/dev/null | head -1 | sed 's|/$||'; }
case "$TRIPLE" in
    r1_7b)
        STUDENT="${STUDENT:-$(_snap deepseek-ai--DeepSeek-R1-Distill-Qwen-7B)}"
        TEACHER="${TEACHER:-$(_snap Skywork--Skywork-OR1-Math-7B)}"      # RL fine-tune of the same base: shared tokenizer
        # thoughtworks/DeepSeek-R1-Distill-Qwen-7B-Eagle3 with ONE config change: max_position_embeddings
        # 131072 (the student's). The published config omits the key, LlamaConfig defaults it to 2048, and
        # vLLM sizes the drafter's rotary cache by it -> past position 2048 the drafter is blind. The weights
        # are the published ones, unchanged; only config.json gains the key.
        DRAFTER="${DRAFTER:-$CLUSTER_ROOT/models/r1-7b-eagle3-maxpos131072}"
        SPEC_METHOD_DEFAULT=eagle3
        THINKING="${THINKING:-none}"                  # R1 template opens <think> unconditionally; no flag
        LM_ONLY="${LM_ONLY:-0}"
        ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-True}"
        # model_type is "qwen2" but the chat template is DeepSeek-R1's: select upstream's DeepSeek builder.
        export VERL_CONTINUOUS_TOKEN_MODEL_FAMILY="${VERL_CONTINUOUS_TOKEN_MODEL_FAMILY:-deepseek}"
        # A SMALL (7B) teacher on a 95 GiB GH200 wants a LOW util: the prompt_logprobs transient comes out
        # of what the reservation leaves over.
        TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.5}"
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.65}"
        ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-True}" ;;     # 7B actor: same hybrid-engine memory wall as the 8B
    q3_8b)
        STUDENT="${STUDENT:-$(_snap Qwen--Qwen3-8B)}"
        TEACHER="${TEACHER:-$(_snap Qwen--Qwen3-32B)}"
        DRAFTER="${DRAFTER:-$(_snap RedHatAI--Qwen3-8B-Thinking-speculator.eagle3)}"
        SPEC_METHOD_DEFAULT=eagle3
        LM_ONLY="${LM_ONLY:-0}"                       # Qwen3 is text-only
        ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-True}"   # standard attention: #51113 does not apply
        # 32B bf16 = 65.5 GB on a 95 GiB GH200 at TP1: util 0.85 reserves 80.7 GiB (~15 GiB KV =
        # ~60k tokens at 256 KiB/token) and leaves ~14 GiB for the prompt_logprobs transient (2.5 GB).
        TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.85}"
        # 8B rollout: 16 GB weights + 144 KiB/token KV; 16 seqs x 18.4k = 295k tokens needs ~0.65.
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.65}"
        # Hybrid engine: without offload the 8B actor keeps ~33 GiB/GPU resident (params+grads, plus Adam
        # once step 1 creates it) next to vLLM's 62 GiB pool on a 95 GiB GH200 -> vLLM's wake-up
        # allocation OOMs. GH200's NVLink-C2C makes offload cheap,
        # and its time falls outside the generation window, the same in every arm.
        ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-True}" ;;
    q35_4b)
        STUDENT="${STUDENT:-$(_snap Qwen--Qwen3.5-4B)}"
        TEACHER="${TEACHER:-$(_snap Qwen--Qwen3.5-27B)}"
        DRAFTER="${DRAFTER:-$STUDENT}"             # native MTP head lives in the student checkpoint
        SPEC_METHOD_DEFAULT=mtp
        LM_ONLY="${LM_ONLY:-1}"
        ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-False}"
        TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.85}"
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.55}" ;;
    q3_4b_a3b)
        # Pinned Hugging Face revisions: a defaulted path must resolve to exactly these shas.
        STUDENT="${STUDENT:-$(_snap Qwen--Qwen3-4B)}"
        TEACHER="${TEACHER:-$(_snap Qwen--Qwen3-30B-A3B-Instruct-2507)}"
        case "$STUDENT" in */models--Qwen--Qwen3-4B/snapshots/*)
            [ "$(basename "$STUDENT")" = 1cfa9a7208912126459214e8b04321603b3df60c ] \
                || { echo "CONFIG ERROR: Qwen3-4B snapshot $(basename "$STUDENT") is not the pinned 1cfa9a72" >&2; exit 1; } ;; esac
        case "$TEACHER" in */models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*)
            [ "$(basename "$TEACHER")" = 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe ] \
                || { echo "CONFIG ERROR: Qwen3-30B-A3B-Instruct-2507 snapshot $(basename "$TEACHER") is not the pinned 0d7cf239" >&2; exit 1; } ;; esac
        DRAFTER="${DRAFTER:-}"                        # no drafter: SPEC=default resolves to none
        SPEC_METHOD_DEFAULT=none
        # Thinking OFF: the student template renders `<|im_start|>assistant\n<think>\n\n</think>\n\n`;
        # the teacher scores the student's ids, so its own template (which ignores the kwarg) is never used.
        THINKING="${THINKING:-False}"
        LM_ONLY="${LM_ONLY:-0}"                       # Qwen3 / Qwen3-MoE are text-only
        ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-True}"   # standard attention: #51113 does not apply
        # FORCED, never inherited: model_type qwen3 -> upstream's Qwen/ChatML builder (<|im_end|> present); an
        # inherited r1_7b "deepseek" would pick the wrong boundary rules.
        export VERL_CONTINUOUS_TOKEN_MODEL_FAMILY=auto
        # 30B-A3B bf16 = 61 GB on a 95 GiB GH200 at TP1 (4 replicas on the teacher node): util 0.85 reserves
        # 80.7 GiB, ~20 GiB KV (96 KiB/token -> ~200k tokens), the same rule as q3_8b's 65.5 GB teacher. TP2 (2
        # replicas) only if TP1 does not fit: TEACHER_TP=2.
        TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.85}"
        # 4B rollout: 8 GB weights + 144 KiB/token KV; 16 seqs x 18.4k = 41 GiB.
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.65}"
        ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-True}" ;;     # as r1_7b (offload is numerically neutral)
    q3_4b_a3bh)
        # = q3_4b_a3b with ONE change, the teacher: the ORIGINAL hybrid Qwen/Qwen3-30B-A3B (its template has
        # enable_thinking) at the pinned ad44e777, thinking off. Instruct-2507's template ignores the kwarg, so that
        # teacher scores the student's ids in a frame it never emits; this one's own no-think render equals the student
        # frame id for id, so it scores the student's ids in a frame it emits itself. Same Qwen3MoeForCausalLM shape and
        # bf16 size as Instruct-2507 (rope_theta 1e6, 40,960 positions >= CTX): the q3_4b_a3b memory rule carries over.
        # Every other line is q3_4b_a3b's.
        STUDENT="${STUDENT:-$(_snap Qwen--Qwen3-4B)}"
        TEACHER="${TEACHER:-$(_snap Qwen--Qwen3-30B-A3B)}"
        case "$STUDENT" in */models--Qwen--Qwen3-4B/snapshots/*)
            [ "$(basename "$STUDENT")" = 1cfa9a7208912126459214e8b04321603b3df60c ] \
                || { echo "CONFIG ERROR: Qwen3-4B snapshot $(basename "$STUDENT") is not the pinned 1cfa9a72" >&2; exit 1; } ;; esac
        case "$TEACHER" in */models--Qwen--Qwen3-30B-A3B/snapshots/*)
            [ "$(basename "$TEACHER")" = ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 ] \
                || { echo "CONFIG ERROR: Qwen3-30B-A3B (hybrid) snapshot $(basename "$TEACHER") is not the pinned ad44e777" >&2; exit 1; } ;; esac
        DRAFTER="${DRAFTER:-}"                        # no drafter: SPEC=default resolves to none
        SPEC_METHOD_DEFAULT=none
        THINKING="${THINKING:-False}"
        LM_ONLY="${LM_ONLY:-0}"                       # Qwen3 / Qwen3-MoE are text-only
        ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-True}"   # standard attention: #51113 does not apply
        export VERL_CONTINUOUS_TOKEN_MODEL_FAMILY=auto   # FORCED, never inherited (as q3_4b_a3b)
        TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.85}"
        ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.65}"
        # the rollout trace dump is ON by default in this preset only
        TRACE_DUMP="${TRACE_DUMP:-1}"
        ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-True}" ;;
    custom) : "${STUDENT:?}" "${TEACHER:?}"; SPEC_METHOD_DEFAULT=none; DRAFTER="${DRAFTER:-}" ;;
    *) echo "CONFIG ERROR: TRIPLE must be r1_7b, q3_8b, q35_4b, q3_4b_a3b, q3_4b_a3bh or custom (got '$TRIPLE')" >&2; exit 1 ;;
esac
# --- rollout trace dump (opt-in) ----------------------------------------------------------------------------------------
# TRACE_DUMP=1: verl's v1 trainer writes every step's rollouts as exact, unpadded token ids (trainer_base.py
# _specopd_dump_traces: <dir>/step_NNNNN.jsonl, one row {uid, step, data_source, prompt_ids, response_ids} per sample,
# after generation and before the update, temp name + os.replace; own timer timing_s/specopd_trace_dump) to
# $REPO_ROOT/traces/sopd_<arm>_<job>. Diagnostic only: no Hydra override, no label change. Default 0; preset
# q3_4b_a3bh defaults it to 1. TRACE_DUMP=0 unsets an inherited SPECOPD_TRACE_DIR (sbatch --export=ALL). The export
# itself is below, once the arm label is final and before the Ray start.
TRACE_DUMP="${TRACE_DUMP:-0}"
case "$TRACE_DUMP" in 0|1) ;; *) echo "CONFIG ERROR: TRACE_DUMP must be 0 or 1 (got '$TRACE_DUMP')" >&2; exit 1 ;; esac
for _m in "$STUDENT" "$TEACHER"; do
    [ -f "$_m/config.json" ] || { echo "CONFIG ERROR: model not found: $_m" >&2; exit 1; }
done
# Qwen3.5 templates take enable_thinking (4B+ default on, 0.8B/2B default off) -> pass it explicitly.
# none = omit the flag (templates that open <think> unconditionally, e.g. R1-Distill).
THINKING="${THINKING:-True}"
case "$THINKING" in True|False|none) ;; *) echo "CONFIG ERROR: THINKING=$THINKING" >&2; exit 1 ;; esac
_TEMPLATE_ARGS=()
[ "$THINKING" != "none" ] && _TEMPLATE_ARGS+=("+data.apply_chat_template_kwargs.enable_thinking=$THINKING")
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESP="${MAX_RESP:-16384}"
CTX=$(( MAX_PROMPT + MAX_RESP + 1 ))
TEACHER_CTX="${TEACHER_CTX:-$CTX}"

BATCH="${BATCH:-8}"
# SHUFFLE=False keeps the train file's ORDER, for a train file whose row order is meant as the prompt schedule.
# Reward: custom_reward/specopd_reward.py = the math verifier; a prompt whose data_source marks it as a code problem
# scores 0.0 (logged only; OPD trains on the teacher signal).
SHUFFLE="${SHUFFLE:-True}"
PPO_MINI="${PPO_MINI:-$BATCH}"
# BATCH/PPO_MINI optimizer updates per step (ppo_epochs is pinned to 1 below); refused before any side effect
[[ "$BATCH" =~ ^[1-9][0-9]*$ && "$PPO_MINI" =~ ^[1-9][0-9]*$ ]] \
    || { echo "CONFIG ERROR: BATCH and PPO_MINI must be positive integers (got '$BATCH' / '$PPO_MINI')" >&2; exit 1; }
(( BATCH % PPO_MINI == 0 )) || { echo "CONFIG ERROR: BATCH ($BATCH) must be a multiple of PPO_MINI ($PPO_MINI)" >&2; exit 1; }
ROLLOUT_N="${ROLLOUT_N:-8}"
LR="${LR:-1e-6}"
STEPS="${STEPS:-20}"
SEED="${SEED:-1}"
TOK_PER_GPU="${TOK_PER_GPU:-16384}"
ULYSSES="${ULYSSES:-2}"
LOSS_AGG="${LOSS_AGG:-token-mean}"
ACTOR_OFFLOAD="${ACTOR_OFFLOAD:-False}"
case "$ACTOR_OFFLOAD" in True|False) ;; *) echo "CONFIG ERROR: ACTOR_OFFLOAD=$ACTOR_OFFLOAD" >&2; exit 1 ;; esac
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.55}"
TEACHER_TP="${TEACHER_TP:-1}"
# GPUs per teacher node that serve the teacher (default: all of them); must be a multiple of TEACHER_TP (guards below).
TEACHER_GPUS="${TEACHER_GPUS:-$GPUS_PER_NODE}"
# The prompt_logprobs transient (batched_tokens x vocab x 4 B) comes out of what the reservation LEAVES
# OVER: a big teacher wants a high util, a SMALL teacher a LOW one. Presets set it.
TEACHER_GPU_UTIL="${TEACHER_GPU_UTIL:-0.85}"
TEACHER_MAX_SEQS="${TEACHER_MAX_SEQS:-16}"
TEACHER_MAX_BATCHED_TOKENS="${TEACHER_MAX_BATCHED_TOKENS:-4096}"
TRAINER_NNODES="${TRAINER_NNODES:-1}"
TEACHER_NNODES="${TEACHER_NNODES:-1}"
# Qwen3.5 checkpoints are Qwen3_5ForConditionalGeneration with a vision tower; this is text-only math,
# so both vLLM engines skip the vision encoder (language_model_only). LM_ONLY=0 for non-multimodal models.
LM_ONLY="${LM_ONLY:-1}"
_LM_ARGS=()
[ "$LM_ONLY" = 1 ] && _LM_ARGS=(+actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
                                +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.language_model_only=True)
# GRPO / REINFORCE have no teacher engine: keep only the rollout half
if [ "$LM_ONLY" = 1 ] && [ "$ALGO" != opd ]; then _LM_ARGS=("${_LM_ARGS[0]}"); fi
# Student prefix caching OFF by default: vLLM 0.19.1 predates #51113 (prefix-cache poisoning on hybrid
# Gated-DeltaNet models under EAGLE/MTP). Off in EVERY arm, so the A/B stays one variable.
ROLLOUT_PREFIX_CACHE="${ROLLOUT_PREFIX_CACHE:-False}"
# Rollout prefix caching is set by the preset and must be IDENTICAL across the arms of one comparison.

# --- speculative decoding of the STUDENT rollout -----------------------------------
# vLLM 0.19.1: ngram (CPU) forces async scheduling OFF engine-wide; ngram_gpu keeps it ON.
SPEC="${SPEC:-none}"
[ "$SPEC" = "default" ] && SPEC="$SPEC_METHOD_DEFAULT"   # the preset's own drafter
SPEC_K="${SPEC_K:-3}"        # draft tokens proposed per decode step
SPEC_LOOKUP_MAX="${SPEC_LOOKUP_MAX:-4}"
SPEC_LOOKUP_MIN="${SPEC_LOOKUP_MIN:-2}"
case "$SPEC" in
    none) _SPEC_ARGS=() ;;
    ngram|ngram_gpu)
        _SPEC_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"${SPEC}\", \"num_speculative_tokens\": ${SPEC_K}, \"prompt_lookup_max\": ${SPEC_LOOKUP_MAX}, \"prompt_lookup_min\": ${SPEC_LOOKUP_MIN}}'") ;;
    # The student's NATIVE MTP head as drafter (Qwen3.5: 1 layer, reused per depth). Its own layers
    # are pushed after every weight sync from DRAFTER_WEIGHTS (default: the student's checkpoint, i.e.
    # the published head, fixed for the whole run) by vllm_rollout/utils.py:_load_external_drafter_weights;
    # the actor stream cannot carry them (transformers drops mtp.* on load). Audit line: [SPECOPD-DRAFTER].
    mtp)
        _SPEC_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"qwen3_5_mtp\", \"num_speculative_tokens\": ${SPEC_K}}'")
        export SPECOPD_DRAFTER_WEIGHTS="${DRAFTER_WEIGHTS:-$STUDENT}" ;;
    # EAGLE-3: a separate drafter model (e.g. RedHat's thinking-trained head for Qwen3-8B). vLLM builds
    # it under the same load_format=dummy and level-2 sleep wipes it, so its real weights are pushed
    # after every sync from DRAFTER_WEIGHTS (default: the drafter checkpoint itself, fixed for the whole run).
    eagle3)
        [ -f "$DRAFTER/config.json" ] || { echo "CONFIG ERROR: SPEC=eagle3 needs DRAFTER (got '$DRAFTER')" >&2; exit 1; }
        _SPEC_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config='{\"method\": \"eagle3\", \"model\": \"${DRAFTER}\", \"num_speculative_tokens\": ${SPEC_K}}'")
        export SPECOPD_DRAFTER_WEIGHTS="${DRAFTER_WEIGHTS:-$DRAFTER}" ;;
    *) echo "CONFIG ERROR: SPEC must be none, ngram, ngram_gpu, mtp or eagle3 (got '$SPEC')" >&2; exit 1 ;;
esac
ARM_LBL="$SPEC"; [ "$SPEC" != none ] && ARM_LBL="${SPEC}_k${SPEC_K}"
# A LEARNED drafter needs the rollout engine to load REAL weights at init (load_format=auto): vLLM
# decides at load time whether the drafter shares the target's embed_tokens/lm_head, from flags and
# weight equality that are meaningless under load_format=dummy -- under dummy weights it can share the 152k
# target lm_head into the 64k-vocab EAGLE-3 drafter, and loading the drafter then fails. Init-only: the actor
# sync overwrites the target before the first rollout, so step timings are unaffected.
case "$SPEC" in mtp|eagle3) _LOADFMT_ARGS=(actor_rollout_ref.rollout.load_format=auto) ;; *) _LOADFMT_ARGS=() ;; esac
# ROLLOUT_LOAD_FORMAT=auto (opt-in): a run WITHOUT a learned drafter also builds its rollout engines from
# the student's real weights (actor_rollout_ref.rollout.load_format=auto, the key the drafter runs carry; verl's default
# without it is dummy = random weights until the first weight sync). Init only, as above: the first sync overwrites the
# target before the first rollout. It puts a no-drafter run's engines on the same start-up path as a run with a
# learned drafter. No label change. Unset = unchanged command line.
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-}"
if [ -n "$ROLLOUT_LOAD_FORMAT" ]; then
    [ "$ROLLOUT_LOAD_FORMAT" = auto ] || { echo "CONFIG ERROR: ROLLOUT_LOAD_FORMAT must be auto (got '$ROLLOUT_LOAD_FORMAT')" >&2; exit 1; }
    _LOADFMT_ARGS=(actor_rollout_ref.rollout.load_format=auto)
fi
ASYNC_SCHED="${ASYNC_SCHED:-auto}"
case "$ASYNC_SCHED" in
    auto) _ASYNC_ARGS=() ;;
    off)  _ASYNC_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.no_async_scheduling=True"); ARM_LBL="${ARM_LBL}_noasync" ;;
    *) echo "CONFIG ERROR: ASYNC_SCHED must be auto or off" >&2; exit 1 ;;
esac

# --- training-inference mismatch (TIM) on the SAMPLER only -------------------------------------------
# The trainer, its log-probs, the advantages and the checkpoints stay clean; only the rollout engine's sampler q
# moves away from the trainer p. rollout_corr/kl (= E_q[log q - log p] per token) measures the mismatch per step.
#   SAMPLER_NOISE=<sigma>: after every weight sync, W += sigma * std(W_0) * eps_name on every transformer-block
#     projection weight (not embeddings/lm_head/norms/biases), eps_name a FIXED per-name Gaussian from
#     SAMPLER_NOISE_SEED -- a systematic offset like quantization error (verl/utils/specopd_tim.py). One
#     [SPECOPD-TIM] line per sync per replica with the projection d of the CLEAN weights on the noise direction (the
#     feedback-loop diagnostic). SAMPLER_NOISE=0 adds nothing and logs the same projection (scale reference).
#   ROLLOUT_QUANT=fp8: verl's rollout fp8 (blockwise 128x128 e4m3 weights, dynamic activations; re-quantized from
#     the bf16 stream every sync). KV_DTYPE=fp8: vLLM fp8 KV cache (rollout engine only; the teacher stays bf16).
# Unset (the defaults) = unchanged command line and no SPECOPD_SAMPLER_NOISE in the environment.
SAMPLER_NOISE="${SAMPLER_NOISE:-}"
# SAMPLER_NOISE_FILE: read sigma at job START from a file the calibration job writes (bench/tim_calibrate.py
# sigma_tgt*.txt), so a run can be queued with --dependency=afterok:<calibration> before the number exists.
if [ -z "$SAMPLER_NOISE" ] && [ -n "${SAMPLER_NOISE_FILE:-}" ]; then
    [ -s "$SAMPLER_NOISE_FILE" ] || { echo "CONFIG ERROR: SAMPLER_NOISE_FILE=$SAMPLER_NOISE_FILE missing/empty" >&2; exit 1; }
    SAMPLER_NOISE=$(tr -d '[:space:]' < "$SAMPLER_NOISE_FILE")
    echo "sampler noise sigma=$SAMPLER_NOISE read from $SAMPLER_NOISE_FILE"
fi
SAMPLER_NOISE_SEED="${SAMPLER_NOISE_SEED:-20261001}"
ROLLOUT_QUANT="${ROLLOUT_QUANT:-none}"
KV_DTYPE="${KV_DTYPE:-auto}"
_TIM_ARGS=()
if [ -n "$SAMPLER_NOISE" ]; then
    [[ "$SAMPLER_NOISE" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] \
        || { echo "CONFIG ERROR: SAMPLER_NOISE must be a sigma >= 0 (got '$SAMPLER_NOISE')" >&2; exit 1; }
    [[ "$SAMPLER_NOISE_SEED" =~ ^[0-9]+$ ]] || { echo "CONFIG ERROR: SAMPLER_NOISE_SEED=$SAMPLER_NOISE_SEED" >&2; exit 1; }
    # Exported BEFORE the Ray start below: the rollout servers' EngineCore workers inherit it.
    export SPECOPD_SAMPLER_NOISE="$SAMPLER_NOISE" SPECOPD_SAMPLER_NOISE_SEED="$SAMPLER_NOISE_SEED"
    ARM_LBL="${ARM_LBL}_tim${SAMPLER_NOISE}"
else
    # sbatch exports the submit environment: an inherited SPECOPD_SAMPLER_NOISE would silently noise the sampler.
    unset SPECOPD_SAMPLER_NOISE SPECOPD_SAMPLER_NOISE_SEED
fi
case "$ROLLOUT_QUANT" in
    none) ;;
    fp8) _TIM_ARGS+=(actor_rollout_ref.rollout.quantization=fp8); ARM_LBL="${ARM_LBL}_wfp8" ;;
    *) echo "CONFIG ERROR: ROLLOUT_QUANT must be none or fp8 (got '$ROLLOUT_QUANT')" >&2; exit 1 ;;
esac
# ROLLOUT_SEED: the rollout engines' sampling seed (verl rollout.seed, default 42; replica r uses seed + r). Set ONLY
# for a run-to-run REPLICATE of an arm (same config, an independent sampling stream); unset = unchanged command line.
ROLLOUT_SEED="${ROLLOUT_SEED:-}"
if [ -n "$ROLLOUT_SEED" ]; then
    [[ "$ROLLOUT_SEED" =~ ^[0-9]+$ ]] || { echo "CONFIG ERROR: ROLLOUT_SEED=$ROLLOUT_SEED" >&2; exit 1; }
    _TIM_ARGS+=(actor_rollout_ref.rollout.seed=$ROLLOUT_SEED); ARM_LBL="${ARM_LBL}_rs${ROLLOUT_SEED}"
fi
case "$KV_DTYPE" in
    auto) ;;
    fp8) _TIM_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8"); ARM_LBL="${ARM_LBL}_kvfp8" ;;
    *) echo "CONFIG ERROR: KV_DTYPE must be auto or fp8 (got '$KV_DTYPE')" >&2; exit 1 ;;
esac
# SAMPLER_QUANT=<spec>: the Score Centering paper's fake-quant SAMPLER on the student's rollout engine only
# (verl/utils/specopd_quant.py): w= weights re-quantized every sync, a= per-token activations, kv= KV writes after
# bias+RoPE. Paper settings: w=e4m3,a=e4m3,kv=e4m3_unit | w=e4m3,a=e4m3,kv=e2m1 | w=int8,a=int8,kv=int4.
# Exported before the Ray start; vLLMReplica blanks it for teacher servers and sets VLLM_DISABLE_COMPILE_CACHE=1 for
# the rollout servers; the worker guard (SPECOPD_SAMPLER_QUANT_MODEL=$STUDENT) makes any OTHER model that still sees
# the variable refuse to start (fail-closed). kvskip=0 (layer-0 K/V bf16) is a deviation, not a paper setting.
# LAYER-0 KV-HEAD LADDER: ...,kvl0heads=<S>,
# S = the layer-0 KV heads that ARE snapped, strictly ascending (e.g. w=e4m3,a=e4m3,kv=e4m3_unit,kvl0heads=1:3), written
# WITHOUT kvskip=0; S={} is ...,kvskip=0 and S=all is the spec without kvl0heads. Needs rollout TP=1 (default). Audit:
# [SPECOPD-QUANT-CFG] per engine, [SPECOPD-QUANT] per sync, [SPECOPD-QUANT-AUDIT] after the first sync.
# EMULATED NVFP4 (NVIDIA Model Optimizer's NVFP4 recipe with its own fake-quantization kernel; NOT a Score Centering
# setting): SAMPLER_QUANT=w=nvfp4,a=nvfp4 with KV_DTYPE=fp8 (the recipe's real FP8 cache) and
# SAMPLER_QUANT_STATIC=<static file> (default $CLUSTER_ROOT/mo/nvfp4emu/nvfp4_static.json: the step-0 activation amax +
# KV scales; running bench/mo_nvfp4emu.sbatch with D=$CLUSTER_ROOT/mo/nvfp4emu produces it), exported as
# SPECOPD_SAMPLER_QUANT_STATIC; layer 0's cache bf16 with KV_SKIP_LAYERS=0 (below). The emulated INT8 SmoothQuant /
# INT4 AWQ grids (w=int8sq,a=int8sq | w=int4awq) take their static file the same way (no default:
# SAMPLER_QUANT_STATIC=<.../static.json> required).
# SAMPLER_QUANT_STATIC with any other spec is refused.
# Unset (the default) = unchanged command line and no SPECOPD_SAMPLER_QUANT in the environment.
SAMPLER_QUANT="${SAMPLER_QUANT:-}"
if [ -n "$SAMPLER_QUANT" ]; then
    [ "$ROLLOUT_QUANT" = none ] || { echo "CONFIG ERROR: SAMPLER_QUANT with ROLLOUT_QUANT=$ROLLOUT_QUANT" >&2; exit 1; }
    case "$SAMPLER_QUANT" in
        *kv=*) [ "$KV_DTYPE" = auto ] || { echo "CONFIG ERROR: SAMPLER_QUANT kv= needs KV_DTYPE=auto" >&2; exit 1; } ;;
    esac
    export SPECOPD_SAMPLER_QUANT="$SAMPLER_QUANT" SPECOPD_SAMPLER_QUANT_MODEL="$STUDENT"
    case "$SAMPLER_QUANT" in
        *nvfp4*|*int8sq*|*int4awq*)
            case "$SAMPLER_QUANT" in
                *nvfp4*) SAMPLER_QUANT_STATIC="${SAMPLER_QUANT_STATIC:-$CLUSTER_ROOT/mo/nvfp4emu/nvfp4_static.json}" ;;
            esac
            [ -s "${SAMPLER_QUANT_STATIC:-}" ] || { echo "CONFIG ERROR: SAMPLER_QUANT=$SAMPLER_QUANT needs SAMPLER_QUANT_STATIC (missing: ${SAMPLER_QUANT_STATIC:-unset})" >&2; exit 1; }
            export SPECOPD_SAMPLER_QUANT_STATIC="$SAMPLER_QUANT_STATIC" ;;
        *)
            [ -z "${SAMPLER_QUANT_STATIC:-}" ] || { echo "CONFIG ERROR: SAMPLER_QUANT_STATIC is for the emulated grids nvfp4 / int8sq / int4awq only (got $SAMPLER_QUANT)" >&2; exit 1; }
            unset SPECOPD_SAMPLER_QUANT_STATIC ;;
    esac
    ARM_LBL="${ARM_LBL}_sq$(printf '%s' "$SAMPLER_QUANT" | tr ',=:' '-_.')"
else
    # sbatch exports the submit environment: an inherited SPECOPD_SAMPLER_QUANT would silently quantize the sampler.
    [ -z "${SAMPLER_QUANT_STATIC:-}" ] || { echo "CONFIG ERROR: SAMPLER_QUANT_STATIC without SAMPLER_QUANT" >&2; exit 1; }
    unset SPECOPD_SAMPLER_QUANT SPECOPD_SAMPLER_QUANT_MODEL SPECOPD_SAMPLER_QUANT_STATIC
fi
# SAMPLER_MODELOPT=<recipe>: NVIDIA Model Optimizer quantization of the student's rollout engine only, with vLLM's real
# kernels (verl/utils/specopd_modelopt.py). Recipe fp8 = Model Optimizer's FP8
# per-tensor W8A8 (static activation scales) + its calibrated FP8 KV cache. The step-0 calibration is a Model Optimizer
# export whose static scales are pinned in SAMPLER_MODELOPT_SCALES (default $CLUSTER_ROOT/mo/scales/<recipe>.json,
# written by bench/mo_scales.py). Every sync re-quantizes the bf16 stream with fresh weight scales and re-supplies the
# static activation and KV scales; every KV wake-up re-applies the KV scales vLLM resets. Renders the FP8 cache itself
# (rollout engine_kwargs.vllm.kv_cache_dtype=fp8; the teacher stays bf16) and the label suffix _mo<recipe>. Refused with
# SAMPLER_QUANT, ROLLOUT_QUANT, KV_DTYPE != auto, or a sampler noise other than 0. Unset = unchanged command line.
# Recipe fp8fl = the same with the first and last decoder layers unquantized and their cache in
# bf16: its scale file (default .../scales/fp8fl.json) names those layers (kv_bf16_layers), rendered as the rollout's
# engine_kwargs.vllm.kv_cache_dtype_skip_layers (a list of strings), with SPECOPD_FA3_KVSKIP_PATCH=1 (the FA3
# kv-skip patch, verl/utils/specopd_kvskip.py, installed by every rollout worker at its first sync); label _mofp8fl.
SAMPLER_MODELOPT="${SAMPLER_MODELOPT:-}"
if [ -n "$SAMPLER_MODELOPT" ]; then
    case "$SAMPLER_MODELOPT" in
        fp8|fp8fl) ;;
        *) echo "CONFIG ERROR: SAMPLER_MODELOPT must be fp8 or fp8fl (got '$SAMPLER_MODELOPT')" >&2; exit 1 ;;
    esac
    [ -z "$SAMPLER_QUANT" ] || { echo "CONFIG ERROR: SAMPLER_MODELOPT with SAMPLER_QUANT=$SAMPLER_QUANT" >&2; exit 1; }
    [ "$ROLLOUT_QUANT" = none ] || { echo "CONFIG ERROR: SAMPLER_MODELOPT with ROLLOUT_QUANT=$ROLLOUT_QUANT" >&2; exit 1; }
    [ "$KV_DTYPE" = auto ] || { echo "CONFIG ERROR: SAMPLER_MODELOPT sets the FP8 cache itself (got KV_DTYPE=$KV_DTYPE)" >&2; exit 1; }
    case "${SAMPLER_NOISE:-}" in
        ""|0|0.0) ;;
        *) echo "CONFIG ERROR: SAMPLER_MODELOPT with SAMPLER_NOISE=$SAMPLER_NOISE (only 0 composes)" >&2; exit 1 ;;
    esac
    SAMPLER_MODELOPT_SCALES="${SAMPLER_MODELOPT_SCALES:-$CLUSTER_ROOT/mo/scales/$SAMPLER_MODELOPT.json}"
    [ -s "$SAMPLER_MODELOPT_SCALES" ] || { echo "CONFIG ERROR: SAMPLER_MODELOPT_SCALES=$SAMPLER_MODELOPT_SCALES missing" >&2; exit 1; }
    _TIM_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8")
    _mo_skip=$(python3 -c 'import json,sys; print(",".join(str(i) for i in json.load(open(sys.argv[1])).get("kv_bf16_layers", [])))' \
               "$SAMPLER_MODELOPT_SCALES" 2>/dev/null) || { echo "CONFIG ERROR: cannot read $SAMPLER_MODELOPT_SCALES" >&2; exit 1; }
    case "$SAMPLER_MODELOPT:$_mo_skip" in
        fp8:) unset SPECOPD_FA3_KVSKIP_PATCH ;;
        fp8fl:?*)
            _mo_list=""; for _l in ${_mo_skip//,/ }; do _mo_list="${_mo_list:+$_mo_list,}'$_l'"; done
            _TIM_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype_skip_layers=[$_mo_list]")
            export SPECOPD_FA3_KVSKIP_PATCH=1 ;;
        *) echo "CONFIG ERROR: SAMPLER_MODELOPT=$SAMPLER_MODELOPT with a scale file whose bf16-cache layers are '$_mo_skip'" >&2; exit 1 ;;
    esac
    export SPECOPD_SAMPLER_MODELOPT="$SAMPLER_MODELOPT" SPECOPD_SAMPLER_MODELOPT_SCALES="$SAMPLER_MODELOPT_SCALES" \
           SPECOPD_SAMPLER_MODELOPT_MODEL="$STUDENT"
    ARM_LBL="${ARM_LBL}_mo${SAMPLER_MODELOPT}"
else
    # sbatch exports the submit environment: an inherited SPECOPD_SAMPLER_MODELOPT would silently quantize the sampler.
    unset SPECOPD_SAMPLER_MODELOPT SPECOPD_SAMPLER_MODELOPT_SCALES SPECOPD_SAMPLER_MODELOPT_MODEL SPECOPD_FA3_KVSKIP_PATCH
fi
# KV_SKIP_LAYERS=<i[,j...]>: the real FP8 cache (KV_DTYPE=fp8) kept in bf16 on these decoder layers -- vLLM's own
# kv_cache_dtype_skip_layers on the rollout engine (a list of strings, as SAMPLER_MODELOPT=fp8fl renders it), with
# SPECOPD_FA3_KVSKIP_PATCH=1 (the FA3 kv-skip patch, verl/utils/specopd_kvskip.py, installed by every rollout worker at
# its first sync). Emulated NVFP4 with layer 0's cache in bf16: SAMPLER_QUANT=w=nvfp4,a=nvfp4 KV_DTYPE=fp8
# KV_SKIP_LAYERS=0. Only with KV_DTYPE=fp8 and without SAMPLER_MODELOPT (whose scale file names its own). Label suffix
# _kvs<i>[-<j>...].
KV_SKIP_LAYERS="${KV_SKIP_LAYERS:-}"
if [ -n "$KV_SKIP_LAYERS" ]; then
    [[ "$KV_SKIP_LAYERS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "CONFIG ERROR: KV_SKIP_LAYERS must be layer indices i[,j...] (got '$KV_SKIP_LAYERS')" >&2; exit 1; }
    [ "$KV_DTYPE" = fp8 ] || { echo "CONFIG ERROR: KV_SKIP_LAYERS needs KV_DTYPE=fp8 (got $KV_DTYPE)" >&2; exit 1; }
    [ -z "$SAMPLER_MODELOPT" ] || { echo "CONFIG ERROR: KV_SKIP_LAYERS with SAMPLER_MODELOPT (its scale file names the bf16-cache layers)" >&2; exit 1; }
    _kv_list=""; for _l in ${KV_SKIP_LAYERS//,/ }; do _kv_list="${_kv_list:+$_kv_list,}'$_l'"; done
    _TIM_ARGS+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype_skip_layers=[$_kv_list]")
    export SPECOPD_FA3_KVSKIP_PATCH=1
    ARM_LBL="${ARM_LBL}_kvs${KV_SKIP_LAYERS//,/-}"
fi
# KV_METRICS=1: the rollout cache's memory readings, for ANY cache, the bf16 one included. Exports SPECOPD_KV_METRICS=1
# before the Ray start, which adds
#   - to every step's console line: kvmem/usage_max (the cache-use high-water mark of the step's generation, polled every
#     2 s from each rollout server's vllm:kv_cache_usage_perc; max over replicas), kvmem/running_max, kvmem/poll_ok and
#     kvmem/preemptions (the step's delta of vllm:num_preemptions summed over the replicas; NaN, never a silent zero,
#     when a server does not answer) -- verl/utils/specopd_metrics.py, verl/trainer/ppo/v1/trainer_base.py;
#   - one [SPECOPD-KV-CFG] line per rollout replica at every weight sync: each layer's cache dtype and storage, the
#     k / v / q / prob scales and the capacity in tokens -- verl/utils/specopd_kvaudit.py.
# Memory readings only: none of them is a latency or throughput measurement. No Hydra override and no label change.
# Unset or 0 = SPECOPD_KV_METRICS unset (an inherited one too: sbatch exports the submit environment), unchanged
# command line, and neither module is imported.
KV_METRICS="${KV_METRICS:-0}"
case "$KV_METRICS" in
    0) unset SPECOPD_KV_METRICS ;;
    1) export SPECOPD_KV_METRICS=1 ;;
    *) echo "CONFIG ERROR: KV_METRICS must be 0 or 1 (got '$KV_METRICS')" >&2; exit 1 ;;
esac
# ROLLOUT_MAX_SEQS=<n> (opt-in; for shapes above 256 sequences per rollout replica): the rollout engines'
# max_num_seqs (verl's default 256 = the most sequences one replica decodes at once; further ones wait in its queue).
# Renders actor_rollout_ref.rollout.max_num_seqs=<n> and the label suffix _mns<n>. Unset = unchanged command line.
ROLLOUT_MAX_SEQS="${ROLLOUT_MAX_SEQS:-}"
if [ -n "$ROLLOUT_MAX_SEQS" ]; then
    [[ "$ROLLOUT_MAX_SEQS" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: ROLLOUT_MAX_SEQS must be a positive integer (got '$ROLLOUT_MAX_SEQS')" >&2; exit 1; }
    _TIM_ARGS+=(actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_SEQS); ARM_LBL="${ARM_LBL}_mns${ROLLOUT_MAX_SEQS}"
fi

# --- score centering on vanilla k1 OPD (verl-project/verl#8010; Marek & Ryabinin, arXiv 2609.20807) ----------------
# Needs the verl tree with #8010 + the OPD wiring (both in the vendored verl/). Unset SC = unchanged command line.
#   SC unset : CURRENT OPD loss -- PPO clip on the trainer's recomputed old_log_probs (decoupled mode), no rollout IS.
#   SC=0     : MATCHED BASELINE -- bypass REINFORCE, old_log_probs = the sampler's rollout_log_probs, no centering.
#              Gradient bitwise equal to SC unset at ratio 1, which needs 1 epoch AND PPO_MINI=BATCH (enforced below:
#              with several mini-batches the current arm takes PPO-clipped steps around its recompute while bypass
#              REINFORCE has no trust region -- a different algorithm).
#   SC=probe : SC=0's loss + the sampler top-k head requested, the centering term computed and LOGGED but applied to
#              no loss -- the exact control of SC=1 (same rollout request, same trainer kernels, bitwise SC=0 gradient).
#   SC=1     : score centering: L_t += A_t * sc_t, A_t = log p_T - log p_theta (detached), after the IS weight.
#   RC_IS=none|token: token-level truncated IS (TIS) at RC_IS_THRESHOLD (default 2.0), w_t = min(pi/q, thr) on the
#              sampled token, detached:
#              * SC unset (current OPD, decoupled mode): algorithm.rollout_correction.rollout_is=token -> the driver
#                computes w_t = min(exp(old_log_prob - rollout_log_prob), thr) from the trainer's RECOMPUTE (= the
#                pre-update trainer p) and the sampler q, and the PPO-clip loss is multiplied by w_t (distillation
#                losses.py pads the driver's nested weights). Label _tis<thr>.
#              * bypass arms (SC=0|probe|1): the actor computes w_t = min(exp(log p_theta - log q), thr) on its own
#                training forward (bypass REINFORCE: L = -w A log p_theta) AND, under SC, applies it to the head and
#                the modeled tail (tail coefficient rho * f(1/rho), rho = sampler/trainer tail mass).
#   TOPK_LOGPROBS (SC=probe|1): the sampler head size per generated token (paper: 128). It comes from
#              the SAMPLER engine, so under SAMPLER_NOISE / SAMPLER_QUANT / ROLLOUT_QUANT it is the perturbed q, as
#              intended; teacher scoring is untouched (no logprobs request on teacher servers).
#   SC_CHUNK (SC=probe|1, default 4096): rows per chunk of the trainer's (chunk x V) fp32 head workspace
#              (VERL_SC_TOPK_LOG_PROB_CHUNK_SIZE; 4096 -> 2.5 GB, 1024 -> 0.6 GB at V=152064). Math unchanged.
#   SC_ENTROPY=1 (bypass arms, default 0): actor.calculate_entropy=True (checkpointed, chunked) so actor/entropy_loss
#              = token-mean H(p_theta), the actor/entropy that bypass drops with the old_log_prob forward. Costs the
#              in-place CE backward in SC=0 (+1 rows x V buffer, ~5 GB); never in the gradient (task loss is zeroed).
# Metrics under ANY bypass arm (SC=0|probe|1) -- compare across arms on these, never on distillation/loss:
#   distillation/k1_mean        token-mean k1 = log p_theta - log p_T (all arms, the current one included): the reverse-
#                               KL curve. distillation/loss becomes token-mean(-A log p_theta) (+ the SC term in SC=1).
#   distillation/mismatch_kl    token-weighted log q - log p_theta (= the driver's rollout_corr/kl of current runs);
#   distillation/mismatch_k3_kl the loss-side actor/rollout_corr/* are unweighted per-micro-batch means. distillation/
#                               ppo_kl = mean(log q - log p_theta), the mismatch, not ~0 as in current runs.
#   distillation/sc_*           SC=probe|1 only: sc_correction, sc_{sampler,train}_head_mass (>= 0.999 expected) and
#                               sc_sampler_head_entropy = -sum_H q log q (lower bound on H(q); the SC=1-vs-probe
#                               entropy-collapse diagnostic). Lost under bypass: actor/entropy (see SC_ENTROPY),
#                               training/rollout_probs_diff_*, distillation/pg_clipfrac*.
# Cost of SC=probe|1: k+1 log-probs per generated token through vLLM (~10.8 KB/token of Python objects in the
# frontend), Ray and TQ ((P+R) x k x 8 B per sample, up to ~1.2 GB/step at 64 x 18k tokens x k=128); trainer peak
# ~+5 GB (probe: no in-place CE backward) / ~+12.5 GB (SC=1: + the head's own rows x V gradient + workspace) over SC=0
# at TOK_PER_GPU=16384, ULYSSES=2, V=152064, bf16 logits. Never use these arms for SD throughput numbers; compare
# SC=1 against SC=probe, not against SC=0.
SC="${SC:-}"
RC_IS="${RC_IS:-none}"
RC_IS_THRESHOLD="${RC_IS_THRESHOLD:-2.0}"
TOPK_LOGPROBS="${TOPK_LOGPROBS:-128}"
SC_CHUNK="${SC_CHUNK:-4096}"
SC_ENTROPY="${SC_ENTROPY:-0}"
_SC_ARGS=()
case "$RC_IS" in none) _rc_is=null ;; token) _rc_is=token ;;
    *) echo "CONFIG ERROR: RC_IS must be none or token (got '$RC_IS')" >&2; exit 1 ;; esac
case "$SC_ENTROPY" in 0|1) ;; *) echo "CONFIG ERROR: SC_ENTROPY must be 0 or 1 (got '$SC_ENTROPY')" >&2; exit 1 ;; esac
# SC_PATH: unset or bypass = the case block below, unchanged; decoupled =
# score centring for RL on the run's own decoupled PPO-clip loss (see the header). Every refusal comes before any side
# effect. RL on the bypass path is the case block below: SC_PATH unset, SC=0|1, SC_ENTROPY=1.
SC_PATH="${SC_PATH:-}"
_SC_PATH_BANNER=""
case "$SC_PATH" in
    ""|bypass|decoupled) ;;
    *) echo "CONFIG ERROR: SC_PATH must be unset, bypass or decoupled (got '$SC_PATH')" >&2; exit 1 ;;
esac
if [ "$SC_PATH" = decoupled ]; then
    # >>> SC_PATH=decoupled
    [ "$ALGO" != opd ] \
        || { echo "CONFIG ERROR: SC_PATH=decoupled is the RL path (ALGO=grpo|reinforce); OPD's SC arms use the bypass path (SC_PATH unset)" >&2; exit 1; }
    case "$SC" in
        1|probe) ;;
        *) echo "CONFIG ERROR: SC_PATH=decoupled needs SC=1 or SC=probe (got SC='$SC'); the run without centring is the plain RL run (SC_PATH and SC unset)" >&2; exit 1 ;;
    esac
    [ "$SC_ENTROPY" = 0 ] \
        || { echo "CONFIG ERROR: SC_ENTROPY=1 with SC_PATH=decoupled: the trainer's recompute already logs actor/entropy" >&2; exit 1; }
    [ "$RC_IS" = none ] \
        || { echo "CONFIG ERROR: SC_PATH=decoupled with RC_IS=$RC_IS is not built (rollout_is=null only)" >&2; exit 1; }
    [[ "$RC_IS_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "CONFIG ERROR: RC_IS_THRESHOLD=$RC_IS_THRESHOLD" >&2; exit 1; }
    [[ "$TOPK_LOGPROBS" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: TOPK_LOGPROBS=$TOPK_LOGPROBS" >&2; exit 1; }
    [[ "$SC_CHUNK" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: SC_CHUNK=$SC_CHUNK" >&2; exit 1; }
    # No policy_loss.loss_mode override (it stays vanilla: the PPO clip on the recompute) and no bypass_mode=True: the
    # driver keeps the old_log_prob recompute (actor/entropy, rollout_corr/kl), and verl's ppo_loss adds
    # agg_loss(A * sc_correction) to the policy term (SC=1) or only logs it (SC=probe). The five keys are mirrored on the
    # actor (validate_config requires key-by-key equality, score_centering_probe included); the probe key is in neither
    # yaml, hence Hydra's + form on both sides. No PPO_MINI=BATCH requirement: the term is per mini-batch.
    _SC_ARGS=(algorithm.rollout_correction.score_centering=True
              algorithm.rollout_correction.bypass_mode=False
              algorithm.rollout_correction.loss_type=ppo_clip
              algorithm.rollout_correction.rollout_is=null
              algorithm.rollout_correction.rollout_is_threshold=$RC_IS_THRESHOLD
              +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=True
              +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=False
              +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=ppo_clip
              +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=null
              +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=$RC_IS_THRESHOLD)
    if [ "$SC" = probe ]; then
        _SC_ARGS+=(+algorithm.rollout_correction.score_centering_probe=True
                   +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering_probe=True)
    fi
    # the sampler head (top-k log-probs of the perturbed sampler q); SC needs the full logits (no fused kernels)
    _SC_ARGS+=(actor_rollout_ref.rollout.topk_log_probs=$TOPK_LOGPROBS
               actor_rollout_ref.model.use_fused_kernels=False)
    # read at import by verl/trainer/ppo/score_centering.py in the trainer workers (Ray starts below)
    export VERL_SC_TOPK_LOG_PROB_CHUNK_SIZE="$SC_CHUNK"
    if [ "$SC" = 1 ]; then ARM_LBL="${ARM_LBL}_dsc1k${TOPK_LOGPROBS}"; else ARM_LBL="${ARM_LBL}_dscprobek${TOPK_LOGPROBS}"; fi
    _SC_PATH_BANNER=" path=decoupled"
    # <<< SC_PATH=decoupled
else
case "$SC" in
    "") [ "$SC_ENTROPY" = 0 ] || { echo "CONFIG ERROR: SC_ENTROPY=1 needs a bypass arm (SC=0|probe|1)" >&2; exit 1; }
        if [ "$RC_IS" = token ]; then
            [[ "$RC_IS_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "CONFIG ERROR: RC_IS_THRESHOLD=$RC_IS_THRESHOLD" >&2; exit 1; }
            # decoupled-mode TIS: the driver's weights from the recompute vs the sampler's rollout_log_probs
            _SC_ARGS=(algorithm.rollout_correction.rollout_is=token
                      algorithm.rollout_correction.rollout_is_threshold=$RC_IS_THRESHOLD
                      actor_rollout_ref.rollout.calculate_log_probs=True)
            ARM_LBL="${ARM_LBL}_tis${RC_IS_THRESHOLD}"
        fi ;;
    0|probe|1)
        if [ "$SC" = probe ] && [ "$ALGO" != opd ]; then
            echo "CONFIG ERROR: SC=probe is an OPD control (distillation.score_centering_probe); ALGO=$ALGO takes SC=0 or 1" >&2; exit 1
        fi
        [[ "$RC_IS_THRESHOLD" =~ ^[0-9]*\.?[0-9]+$ ]] || { echo "CONFIG ERROR: RC_IS_THRESHOLD=$RC_IS_THRESHOLD" >&2; exit 1; }
        [[ "$TOPK_LOGPROBS" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: TOPK_LOGPROBS=$TOPK_LOGPROBS" >&2; exit 1; }
        [[ "$SC_CHUNK" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: SC_CHUNK=$SC_CHUNK" >&2; exit 1; }
        # the matched-baseline equivalence (and SC=1 vs probe at the same policy) holds for ONE optimizer step per
        # batch only; the launcher pins ppo_epochs=1, PPO_MINI is the remaining knob
        [ "$PPO_MINI" = "$BATCH" ] || { echo "CONFIG ERROR: SC arms need PPO_MINI=BATCH (got $PPO_MINI vs $BATCH)" >&2; exit 1; }
        # if/fi, not `[ ] && x`: as the LAST command of a case branch a false test fails the case under set -e
        _sc=False
        if [ "$SC" != 0 ]; then _sc=True; fi
        # driver side (old_log_probs := rollout_log_probs) and the actor-side mirror the loss, the centering hook and
        # the v1 trainer's micro-batch flag read; validate_config requires the two to agree key by key.
        # use_fused_kernels: the FSDP engine reads the MODEL key (the actor key only interpolates it); SC needs the
        # full logits, and the engine raises for SC + fused kernels.
        _SC_ARGS=(algorithm.rollout_correction.bypass_mode=True
                  algorithm.rollout_correction.loss_type=reinforce
                  algorithm.rollout_correction.rollout_is=$_rc_is
                  algorithm.rollout_correction.rollout_is_threshold=$RC_IS_THRESHOLD
                  algorithm.rollout_correction.score_centering=$_sc
                  actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode
                  +actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=True
                  +actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=reinforce
                  +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=$_rc_is
                  +actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=$RC_IS_THRESHOLD
                  +actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering=$_sc
                  actor_rollout_ref.model.use_fused_kernels=False
                  actor_rollout_ref.rollout.calculate_log_probs=True)
        # the OPD policy gradient reads the actor mirror too; GRPO's is the actor's own policy loss (no distillation)
        if [ "$ALGO" = opd ]; then _SC_ARGS+=(distillation.distillation_loss.policy_loss_mode=bypass_mode); fi
        ARM_LBL="${ARM_LBL}_sc${SC}"
        if [ "$SC" != 0 ]; then
            _SC_ARGS+=(actor_rollout_ref.rollout.topk_log_probs=$TOPK_LOGPROBS)
            ARM_LBL="${ARM_LBL}k${TOPK_LOGPROBS}"
            # read at import by verl/trainer/ppo/score_centering.py in the trainer workers (Ray starts below, so
            # srun --export=ALL carries it)
            export VERL_SC_TOPK_LOG_PROB_CHUNK_SIZE="$SC_CHUNK"
        fi
        if [ "$SC" = probe ]; then _SC_ARGS+=(distillation.distillation_loss.score_centering_probe=True); fi
        if [ "$SC_ENTROPY" = 1 ]; then
            _SC_ARGS+=(actor_rollout_ref.actor.calculate_entropy=True
                       actor_rollout_ref.actor.entropy_checkpointing=True
                       actor_rollout_ref.actor.entropy_from_logits_with_chunking=True)
            ARM_LBL="${ARM_LBL}_ent"
        fi
        if [ "$RC_IS" = token ]; then ARM_LBL="${ARM_LBL}_tis${RC_IS_THRESHOLD}"; fi ;;
    *) echo "CONFIG ERROR: SC must be unset, 0, probe or 1 (got '$SC')" >&2; exit 1 ;;
esac
fi

# --- PPO clip-higher -----------------------------
# CLIP_RATIO_HIGH=<v>: actor_rollout_ref.actor.clip_ratio_high=v (verl's PPO-clip upper epsilon); clip_ratio_low stays
# verl's 0.2 and clip_ratio_c 3.0. Unset = no override, no label (unchanged command line). It is an RL knob: an OPD run
# keeps verl's clip range, so ALGO=opd refuses it.
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-}"
_CLIP_ARGS=()
if [ -n "$CLIP_RATIO_HIGH" ]; then
    [ "$ALGO" != opd ] \
        || { echo "CONFIG ERROR: CLIP_RATIO_HIGH is an RL knob (ALGO=grpo|reinforce); OPD keeps verl's clip range" >&2; exit 1; }
    { [[ "$CLIP_RATIO_HIGH" =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v v="$CLIP_RATIO_HIGH" 'BEGIN { exit !(v + 0 > 0 && v + 0 <= 1) }'; } \
        || { echo "CONFIG ERROR: CLIP_RATIO_HIGH must be a number in (0, 1] (got '$CLIP_RATIO_HIGH')" >&2; exit 1; }
    _CLIP_ARGS=(actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH)
    ARM_LBL="${ARM_LBL}_ch${CLIP_RATIO_HIGH}"
fi

# --- advantage-offset causal test (OPD only; distillation.distillation_loss.adv_offset) -----------------------------
# ADV_OFFSET=<c> (any float, may be negative): A_t = log p_T - log p_theta + c on every response token, in the current
# (PPO clip) path and in the bypass / SC paths (where the SC term is scaled by the same A_t + c). On-policy
# c * E_p[grad log p] = 0 exactly (pure variance); under q != p it adds the drift c * (q - p) in logit space, so c above
# ~KL flips the sign of E_q[A] (expected: drift TOWARD the sampler, RL-like) and c < 0 strengthens the push away. Clean
# sampler + c = the variance-only control. Logged: distillation/adv_offset, distillation/adv_mean_with_offset
# (token-weighted mean of A + c; distillation/k1_mean stays the reverse-KL curve). Unset = unchanged command line.
ADV_OFFSET="${ADV_OFFSET:-}"
_ADV_ARGS=()
if [ -n "$ADV_OFFSET" ]; then
    [ "$ALGO" = opd ] || { echo "CONFIG ERROR: ADV_OFFSET is an OPD knob (distillation advantage); ALGO=$ALGO" >&2; exit 1; }
    [[ "$ADV_OFFSET" =~ ^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]] \
        || { echo "CONFIG ERROR: ADV_OFFSET must be a number (got '$ADV_OFFSET')" >&2; exit 1; }
    _ADV_ARGS=(distillation.distillation_loss.adv_offset=$ADV_OFFSET)
    ARM_LBL="${ARM_LBL}_ao${ADV_OFFSET}"
fi

# --- verl v1 trainer mode: asynchronous rollout ---------------------------------------
# ASYNC_MODE unset (the default) = verl's own trainer.v1.trainer_mode=sync (ppo_trainer.yaml): nothing below adds an
# override, a banner line or a label, so the command line is byte-identical.
# ASYNC_MODE=colocate: verl/trainer/ppo/v1/trainer_colocate_async.py on the SAME pools as sync (the hybrid actor/rollout
#   node + the teacher node). Generation is NOT overlapped with training: at every sample point the rollout replicas are
#   aborted and slept (level 2, as in sync), and after the weight sync (backend naive, the same path as sync, incl. the
#   EAGLE-3 drafter restore) they resume. An aborted request is resubmitted as prompt + tokens so far (full re-prefill
#   under the new weights; max_tokens = what is left of MAX_RESP, llm_server.py FullyAsyncLLMServerClient), so one
#   trajectory may mix weight versions. What changes vs sync: ASYNC_WARMUP_BATCHES extra batches stay in flight
#   (trainer.v1.colocate_async.num_warmup_batches), each step trains the OLDEST finished groups and the unfinished ones
#   carry over. Still one weight sync per step (parameter_sync_step is not a colocate_async key; trainer_base.py resolves
#   it to 1). verl forces data.gen_batch_size=1 in every async mode AFTER printing the resolved config (INFO line
#   'data.gen_batch_size defaulted to 1.'); the prompt ORDER is the sync one (seeded sampler, independent of batch size).
#   Off-policy control (replay_buffer.py ReplayBufferAsync): a prompt submitted at step s has span g - s + 1 when trained
#   at step g. ASYNC_MAX_OFF_POLICY = trainer.v1.sampler.max_off_policy_threshold (T), ASYNC_OFF_POLICY_STRATEGY =
#   max_off_policy_strategy: drop evicts finished groups with span > T and refills (biased toward SHORT responses), then
#   asserts span <= T; wait (dropless) blocks sampling while any IN-FLIGHT prompt has span >= T but never caps finished
#   groups, so span <= T (version lag <= T-1) holds only while the carried pool is <= one batch: wait with
#   ASYNC_WARMUP_BATCHES > 1 is refused. Defaults 1 / 3 / wait = lag <= 2.
#   Recompute, no rollout IS: algorithm.rollout_correction.bypass_mode / rollout_is stay verl's defaults (false / null),
#   NOT overridden, so the list is the synchronous run's + the 4 keys below; SC and RC_IS are refused. Refused too (exit 2), as not
#   built: the SIMULATED stale sampler (STALENESS / STALE_PERIOD / STALE_LAG; real lag would stack on it),
#   DEPTH=telemetry (its engine pull lives in trainer_sync.on_sample_end only) and ALGO != opd. The per-sync sampler
#   knobs (SAMPLER_NOISE, SAMPLER_QUANT, ROLLOUT_QUANT, KV_DTYPE, ROLLOUT_SEED) compose: they act inside the weight sync
#   both modes share.
#   Logged per step: verl's training/off_policy/trajectory_{spans,staleness,staleness_worst}/{mean,max,min} (worst-case
#   lag = trajectory_staleness_worst; trajectory_staleness is its LOWER bound); specopd's training/async/* (trainer wait
#   for finished groups, polls/seconds the wait rule held a full batch back) and timing_s/{abort_replicas,sleep_replicas,
#   resume_generation}; one driver-side '[STEP-WALL] step=<n> t=<unix>' line per completed step plus a step=0 clock start
#   before the warmup submission (the console logger prints no time). In async modes timing_s/gen = waiting for finished
#   groups + abort + sleep, NOT generation.
# ASYNC_MODE=fully_async: verl's FULLY asynchronous mode, trainer.v1.trainer_mode=
#   separate_async with actor_rollout_ref.hybrid_engine=False (docs/advance/v1_async_trainer.md: "v0 fully-async
#   semantics"): the student's rollout runs on its OWN GPUs (standalone vLLM replicas, TP1, never slept) while the trainer
#   trains on the rest of the same node, so generation and training overlap. GPU split (the same for every k): the
#   first node = trainer (GPUS_PER_NODE - ASYNC_ROLLOUT_GPUS GPUs, a multiple of ULYSSES) + rollout
#   (ASYNC_ROLLOUT_GPUS, default 2); OPD's teacher pool is the synchronous recipe's, unchanged, on its own node. Same
#   nodes as sync (--nodes=2 OPD, 1 GRPO/REINFORCE). Every other knob keeps its synchronous meaning.
#   ASYNC_STALENESS=k (required, integer >= 1) = the most model versions the generating model may lag the trainer:
#     trainer.v1.separate_async.num_warmup_batches=k (the rollout runs k train batches ahead of the trainer) and
#     trainer.v1.sampler.max_off_policy_threshold=k+1, max_off_policy_strategy=wait (dropless: sampling blocks while a
#     prompt submitted k steps before the current one is unfinished; nothing is evicted, so no length bias). The
#     sampler trains the oldest finished groups first, so after the k-step fill each step trains the batch submitted k
#     steps earlier (lag k); a group trained out of submission order (fill phase, refills) can carry one step (lag
#     k+1). Partial rollout (verl's FullyAsyncLLMServerClient): every weight sync aborts the in-flight requests and
#     resumes them under the new weights (prompt + tokens so far re-prefilled; max_tokens = what is left), so one
#     answer may mix versions. Logged per step (trainer_base): training/off_policy/trajectory_staleness_worst/* (the
#     oldest version any token of a trajectory came from vs the trainer's), trajectory_staleness/* (the newest),
#     trajectory_spans/*; per token the lag lies between the two. rollout_corr/kl = E_q[log q - log p] with q the
#     generating (stale, possibly mixed-version) model and p the trainer's weights (recompute path: bypass_mode /
#     rollout_is stay verl's defaults false / null, not overridden), i.e. the staleness mismatch. '[STEP-WALL]' lines per
#     step (trainer_base, every async mode) give the wall clock.
#   One weight sync per step: parameter_sync_step = BATCH / PPO_MINI controller mini-batches per step (separate_async's
#   invariant train_batch_size = parameter_sync_step x ppo_mini_batch_size; GRPO's 2 mini-batches use decoupled PPO with
#   pi_old restored from CPU, OPD and REINFORCE 1). Weight sync: the specopd_pynccl checkpoint engine
#   (verl/checkpoint_engine/specopd_pynccl_checkpoint_engine.py, vLLM's PyNcclCommunicator; verl's own nccl engine needs
#   cupy, which the environment does not provide), loaded through checkpoint_engine.custom_backend_module.
#   Refused (exit 2, not built): STALENESS / STALE_* (simulated staleness), SC, RC_IS, SC_PATH, DEPTH=telemetry,
#   REF_KL=1, TRAINER_NNODES != 1, the colocate knobs (ASYNC_WARMUP_BATCHES / ASYNC_MAX_OFF_POLICY /
#   ASYNC_OFF_POLICY_STRATEGY). Unset ASYNC_MODE: nothing here runs.
ASYNC_MODE="${ASYNC_MODE:-}"
_TRAINER_MODE_ARGS=()
_TRAINER_GPN=$GPUS_PER_NODE
if [ -z "$ASYNC_MODE" ]; then
    # sbatch exports the submit environment: an async knob without ASYNC_MODE would be silently ignored.
    [ -z "${ASYNC_WARMUP_BATCHES:-}${ASYNC_MAX_OFF_POLICY:-}${ASYNC_OFF_POLICY_STRATEGY:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_WARMUP_BATCHES / ASYNC_MAX_OFF_POLICY / ASYNC_OFF_POLICY_STRATEGY need ASYNC_MODE" >&2; exit 2; }
    [ -z "${ASYNC_STALENESS:-}${ASYNC_ROLLOUT_GPUS:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_STALENESS / ASYNC_ROLLOUT_GPUS need ASYNC_MODE=fully_async" >&2; exit 2; }
elif [ "$ASYNC_MODE" = fully_async ]; then
    # >>> ASYNC_MODE=fully_async
    [ -z "${ASYNC_WARMUP_BATCHES:-}${ASYNC_MAX_OFF_POLICY:-}${ASYNC_OFF_POLICY_STRATEGY:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_MODE=fully_async sets its window from ASYNC_STALENESS alone (ASYNC_WARMUP_BATCHES / ASYNC_MAX_OFF_POLICY / ASYNC_OFF_POLICY_STRATEGY are the colocate knobs)" >&2; exit 2; }
    ASYNC_STALENESS="${ASYNC_STALENESS:-}"
    ASYNC_ROLLOUT_GPUS="${ASYNC_ROLLOUT_GPUS:-2}"
    [[ "$ASYNC_STALENESS" =~ ^[1-9][0-9]*$ ]] \
        || { echo "CONFIG ERROR: ASYNC_MODE=fully_async needs ASYNC_STALENESS=k, an integer >= 1 (got '$ASYNC_STALENESS')" >&2; exit 2; }
    [[ "$ASYNC_ROLLOUT_GPUS" =~ ^[1-9][0-9]*$ ]] && (( ASYNC_ROLLOUT_GPUS < GPUS_PER_NODE )) \
        || { echo "CONFIG ERROR: ASYNC_ROLLOUT_GPUS must be an integer in [1, GPUS_PER_NODE=$GPUS_PER_NODE) (got '$ASYNC_ROLLOUT_GPUS')" >&2; exit 2; }
    _TRAINER_GPN=$(( GPUS_PER_NODE - ASYNC_ROLLOUT_GPUS ))
    (( _TRAINER_GPN % ULYSSES == 0 )) \
        || { echo "CONFIG ERROR: fully_async trainer GPUs ($_TRAINER_GPN = $GPUS_PER_NODE - $ASYNC_ROLLOUT_GPUS) must be a multiple of ULYSSES=$ULYSSES" >&2; exit 2; }
    (( (PPO_MINI * ROLLOUT_N) % (_TRAINER_GPN / ULYSSES) == 0 )) \
        || { echo "CONFIG ERROR: PPO_MINI*ROLLOUT_N not divisible by the fully_async trainer's dp=$(( _TRAINER_GPN / ULYSSES ))" >&2; exit 2; }
    [ "$TRAINER_NNODES" = 1 ] \
        || { echo "CONFIG ERROR: ASYNC_MODE=fully_async is built for TRAINER_NNODES=1 (trainer + rollout on one node; got $TRAINER_NNODES)" >&2; exit 2; }
    # raw reads: the stale-sampler, DEPTH and REF_KL knobs are resolved further down (same defaults)
    [ -z "${STALENESS:-}${STALE_PERIOD:-}${STALE_LAG:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_MODE is REAL staleness; STALENESS / STALE_PERIOD / STALE_LAG simulate it (got '${STALENESS:-}' / '${STALE_PERIOD:-}' / '${STALE_LAG:-}')" >&2; exit 2; }
    [ -z "$SC_PATH" ] || { echo "CONFIG ERROR: ASYNC_MODE=fully_async with SC_PATH=$SC_PATH is not built" >&2; exit 2; }
    [ -z "$SC" ] || { echo "CONFIG ERROR: ASYNC_MODE=fully_async with SC=$SC (bypass arms) is not built; fully_async is the recompute path" >&2; exit 2; }
    [ "$RC_IS" = none ] || { echo "CONFIG ERROR: ASYNC_MODE=fully_async with RC_IS=$RC_IS is not built; fully_async uses no rollout IS" >&2; exit 2; }
    [ "${DEPTH:-off}" = off ] || { echo "CONFIG ERROR: ASYNC_MODE=fully_async with DEPTH=${DEPTH:-} is not built (the telemetry pull is sync-only)" >&2; exit 2; }
    [ "${REF_KL:-0}" = 0 ] || { echo "CONFIG ERROR: ASYNC_MODE=fully_async with REF_KL=${REF_KL:-} is not built (no reference worker beside the pure actor)" >&2; exit 2; }
    _TRAINER_MODE_ARGS=(trainer.v1.trainer_mode=separate_async
                        actor_rollout_ref.hybrid_engine=False
                        actor_rollout_ref.rollout.nnodes=1
                        actor_rollout_ref.rollout.n_gpus_per_node=$ASYNC_ROLLOUT_GPUS
                        actor_rollout_ref.rollout.checkpoint_engine.backend=specopd_pynccl
                        actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.checkpoint_engine.specopd_pynccl_checkpoint_engine
                        trainer.v1.separate_async.num_warmup_batches=$ASYNC_STALENESS
                        trainer.v1.separate_async.parameter_sync_step=$(( BATCH / PPO_MINI ))
                        trainer.v1.sampler.max_off_policy_threshold=$(( ASYNC_STALENESS + 1 ))
                        trainer.v1.sampler.max_off_policy_strategy=wait)
    ARM_LBL="${ARM_LBL}_fa_k${ASYNC_STALENESS}"
    # <<< ASYNC_MODE=fully_async
else
    [ -z "${ASYNC_STALENESS:-}${ASYNC_ROLLOUT_GPUS:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_STALENESS / ASYNC_ROLLOUT_GPUS need ASYNC_MODE=fully_async (got ASYNC_MODE=$ASYNC_MODE)" >&2; exit 2; }
    case "$ASYNC_MODE" in
        colocate) _TRAINER_MODE=colocate_async ;;
        *) echo "CONFIG ERROR: ASYNC_MODE must be unset, colocate or fully_async (got '$ASYNC_MODE')" >&2; exit 2 ;;
    esac
    ASYNC_WARMUP_BATCHES="${ASYNC_WARMUP_BATCHES:-1}"
    ASYNC_MAX_OFF_POLICY="${ASYNC_MAX_OFF_POLICY:-3}"
    ASYNC_OFF_POLICY_STRATEGY="${ASYNC_OFF_POLICY_STRATEGY:-wait}"
    # raw reads: the stale-sampler and DEPTH knobs are resolved further down (same defaults)
    [ -z "${STALENESS:-}${STALE_PERIOD:-}${STALE_LAG:-}" ] \
        || { echo "CONFIG ERROR: ASYNC_MODE is REAL staleness; STALENESS / STALE_PERIOD / STALE_LAG simulate it (got '${STALENESS:-}' / '${STALE_PERIOD:-}' / '${STALE_LAG:-}')" >&2; exit 2; }
    [ "$ALGO" = opd ] || { echo "CONFIG ERROR: ASYNC_MODE is built for ALGO=opd only (got $ALGO)" >&2; exit 2; }
    [ -z "$SC" ] || { echo "CONFIG ERROR: ASYNC_MODE with SC=$SC (bypass arms) is not built; colocate is the recompute path" >&2; exit 2; }
    [ "$RC_IS" = none ] || { echo "CONFIG ERROR: ASYNC_MODE with RC_IS=$RC_IS is not built; colocate uses no rollout IS" >&2; exit 2; }
    [ "${DEPTH:-off}" = off ] || { echo "CONFIG ERROR: ASYNC_MODE with DEPTH=${DEPTH:-} is not built (the telemetry pull is sync-only)" >&2; exit 2; }
    [[ "$ASYNC_WARMUP_BATCHES" =~ ^[0-9]+(\.[0-9]+)?$ ]] \
        || { echo "CONFIG ERROR: ASYNC_WARMUP_BATCHES must be a number >= 0 (got '$ASYNC_WARMUP_BATCHES')" >&2; exit 2; }
    [[ "$ASYNC_MAX_OFF_POLICY" =~ ^[1-9][0-9]*$ ]] \
        || { echo "CONFIG ERROR: ASYNC_MAX_OFF_POLICY must be an integer >= 1 (got '$ASYNC_MAX_OFF_POLICY')" >&2; exit 2; }
    case "$ASYNC_OFF_POLICY_STRATEGY" in
        wait|drop) ;;
        *) echo "CONFIG ERROR: ASYNC_OFF_POLICY_STRATEGY must be wait or drop (got '$ASYNC_OFF_POLICY_STRATEGY')" >&2; exit 2 ;;
    esac
    if [ "$ASYNC_OFF_POLICY_STRATEGY" = wait ] && ! awk -v w="$ASYNC_WARMUP_BATCHES" 'BEGIN { exit !(w + 0 <= 1) }'; then
        echo "CONFIG ERROR: ASYNC_OFF_POLICY_STRATEGY=wait with ASYNC_WARMUP_BATCHES=$ASYNC_WARMUP_BATCHES > 1: wait never caps a finished group, so lag <= ASYNC_MAX_OFF_POLICY-1 would not hold (use drop for a larger window)" >&2
        exit 2
    fi
    _TRAINER_MODE_ARGS=(trainer.v1.trainer_mode=$_TRAINER_MODE
                        trainer.v1.colocate_async.num_warmup_batches=$ASYNC_WARMUP_BATCHES
                        trainer.v1.sampler.max_off_policy_threshold=$ASYNC_MAX_OFF_POLICY
                        trainer.v1.sampler.max_off_policy_strategy=$ASYNC_OFF_POLICY_STRATEGY)
    ARM_LBL="${ARM_LBL}_ca_t${ASYNC_MAX_OFF_POLICY}${ASYNC_OFF_POLICY_STRATEGY}_w${ASYNC_WARMUP_BATCHES}"
fi

# --- STALE sampler (verl/utils/specopd_stale.py): the rollout engine serves the trainer's weights from EARLIER syncs --
# One sync = one global step. The trainer, its log-probs and the OPD advantages stay CURRENT; only the tokens the sampler
# emits change (rollout_corr/kl measures the mismatch). Substituted BEFORE SAMPLER_NOISE / SAMPLER_QUANT, so it composes
# with them as a stale perturbed / quantized sampler.
#   STALENESS=s     Score Centering's sampler.staleness (its train_rl.py) == STALE_PERIOD=s+1; the paper's stale
#                   setting is s=64.
#   STALE_PERIOD=K  sampler refreshed every K syncs (lag cycles 0..K-1); caches ONE version.
#   STALE_LAG=k     fixed lag (AsyncOPD): sync t serves sync max(t-k,0) exactly; caches k versions.
#   STALE_CACHE_DTYPE=exact|bf16  STALE_SPILL_DIR=<node-local disk>  STALE_MAX_GB_PER_NODE (default 300)
# The 7B stream is fp32: 30.5 GB per cached version per rollout GPU (x GPUS_PER_NODE). exact -> lag <= 2 under
# 300 GB/node, bf16 -> lag <= 4; above that preflight (below) refuses before Ray starts. One [SPECOPD-STALE] line per
# sync per rollout worker; verl's training/off_policy/trajectory_staleness then reads the served lag too.
# Unset (the default) = unchanged command line and no SPECOPD_STALE_* in the environment.
STALENESS="${STALENESS:-}"; STALE_PERIOD="${STALE_PERIOD:-}"; STALE_LAG="${STALE_LAG:-}"
if [ -n "$STALENESS" ]; then
    [[ "$STALENESS" =~ ^[0-9]+$ ]] || { echo "CONFIG ERROR: STALENESS must be an integer >= 0 (got '$STALENESS')" >&2; exit 1; }
    [ -z "$STALE_PERIOD$STALE_LAG" ] || { echo "CONFIG ERROR: STALENESS means STALE_PERIOD=STALENESS+1; set one" >&2; exit 1; }
    STALE_PERIOD=$(( STALENESS + 1 ))
fi
# sbatch exports the submit environment: an inherited SPECOPD_STALE_* would silently stale the sampler.
unset SPECOPD_STALE_PERIOD SPECOPD_STALE_LAG SPECOPD_STALE_WORKERS_PER_NODE SPECOPD_STALE_CACHE_DTYPE \
    SPECOPD_STALE_SPILL_DIR SPECOPD_STALE_MAX_GB_PER_NODE SPECOPD_STALE_RESERVE_B_PER_PARAM
if [ -n "$STALE_PERIOD$STALE_LAG" ]; then
    [ -z "$STALE_PERIOD" ] || [ -z "$STALE_LAG" ] || { echo "CONFIG ERROR: STALE_PERIOD and STALE_LAG are exclusive" >&2; exit 1; }
    if [ -n "$STALE_PERIOD" ]; then
        [[ "$STALE_PERIOD" =~ ^[1-9][0-9]*$ ]] || { echo "CONFIG ERROR: STALE_PERIOD must be >= 1 (got '$STALE_PERIOD')" >&2; exit 1; }
        export SPECOPD_STALE_PERIOD="$STALE_PERIOD"; ARM_LBL="${ARM_LBL}_stp${STALE_PERIOD}"
    else
        [[ "$STALE_LAG" =~ ^[0-9]+$ ]] || { echo "CONFIG ERROR: STALE_LAG must be >= 0 (got '$STALE_LAG')" >&2; exit 1; }
        export SPECOPD_STALE_LAG="$STALE_LAG"; ARM_LBL="${ARM_LBL}_stl${STALE_LAG}"
    fi
    # Exported BEFORE the Ray start below: the rollout workers inherit them. Every rollout worker caches the FULL stream.
    export SPECOPD_STALE_WORKERS_PER_NODE="$GPUS_PER_NODE"
    [ -z "${STALE_CACHE_DTYPE:-}" ] || export SPECOPD_STALE_CACHE_DTYPE="$STALE_CACHE_DTYPE"
    [ -z "${STALE_SPILL_DIR:-}" ] || export SPECOPD_STALE_SPILL_DIR="$STALE_SPILL_DIR"
    [ -z "${STALE_MAX_GB_PER_NODE:-}" ] || export SPECOPD_STALE_MAX_GB_PER_NODE="$STALE_MAX_GB_PER_NODE"
    # CPU the trainer itself keeps on this node, for preflight's check of cache + reserve against the job's cgroup limit
    # (slurm --mem): an offloaded FSDP actor holds fp32 params + Adam m, v (12 B/param) + fp32 grads if offloaded too
    # (16, the conservative figure used); the REF_KL reference adds its bf16 offload (2 B/param).
    _stale_rsv=0
    [ "$ACTOR_OFFLOAD" != True ] || _stale_rsv=16
    [ "${REF_KL:-0}" != 1 ] || _stale_rsv=$(( _stale_rsv + 2 ))
    export SPECOPD_STALE_RESERVE_B_PER_PARAM="$_stale_rsv"
    # Fail fast HERE, before Ray, on the cap or the node limit (a worker-side refusal only fails the first sync, after
    # the ranks are up). Prints the [SPECOPD-STALE-PLAN] incl. the node limit + headroom; non-zero exit on refusal.
    "$PYTHON_BIN" -c 'import sys; from verl.utils.specopd_stale import preflight; print(preflight(sys.argv[1]))' "$STUDENT"
fi

# --- drift measurement + checkpoints -----------------------------------------------
REF_KL="${REF_KL:-0}"
# kl = k1 = log p_k - log p_0 on p_k's own tokens: unbiased for KL(p_k || p_0) and unclamped
# (low_var_kl clamps at 10). A DIAGNOSTIC of how far the student has moved from its initial weights.
case "$REF_KL" in
    0) _REF_ARGS=() ;;
    1) _REF_ARGS=(actor_rollout_ref.actor.use_kl_loss=True
                  actor_rollout_ref.actor.kl_loss_coef=0.0
                  actor_rollout_ref.actor.kl_loss_type=${REF_KL_TYPE:-kl}
                  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
                  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$TOK_PER_GPU
                  actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ULYSSES
                  actor_rollout_ref.ref.fsdp_config.param_offload=True)
       ARM_LBL="${ARM_LBL}_refkl" ;;
    *) echo "CONFIG ERROR: REF_KL must be 0 or 1" >&2; exit 1 ;;
esac

# --- speculative-decoding telemetry: EngineCore scheduler plugin (verl/specopd_vllm/depth_sched.py) ---------
# DEPTH=off (default): unchanged. DEPTH=telemetry: the rollout engines run the plugin scheduler under the IDENTITY
# depth policy (scheduling and vLLM's counters unchanged). It exports one record per request through
# kv_transfer_params["specopd"] and per-step cost cells; the trainer pulls the cells before the replicas sleep and
# writes one JSON line per step to $SPECOPD_TELEMETRY_DIR plus specopd_tel/* metrics ([SPECOPD-TEL] log lines).
# verl forwards engine_kwargs.vllm.* verbatim as vLLM CLI flags (vllm_async_server.py launch_server: args dict
# -> build_cli_args_from_config -> --scheduler-cls; EngineArgs.scheduler_cls, vllm engine/arg_utils.py:1248).
# The class must match vLLM's async_scheduling, which is OFF under ASYNC_SCHED=off and under the CPU ngram
# drafter (vllm config/vllm.py:755-768); the plugin also redirects to the matching class and logs if it is not.
DEPTH="${DEPTH:-off}"
case "$DEPTH" in
    # sbatch exports the submit environment: an inherited SPECOPD_TELEMETRY=1 would switch the pulls/writes on here.
    off) _DEPTH_ARGS=(); unset SPECOPD_TELEMETRY SPECOPD_TELEMETRY_DIR ;;
    telemetry)
        [ -f "$REPO_ROOT/verl/specopd_vllm/depth_sched.py" ] \
            || { echo "CONFIG ERROR: DEPTH=telemetry needs $REPO_ROOT/verl/specopd_vllm (part of the vendored verl tree)" >&2; exit 1; }
        _SCHED_CLS=DepthAsyncScheduler
        { [ "$ASYNC_SCHED" = off ] || [ "$SPEC" = ngram ]; } && _SCHED_CLS=DepthScheduler
        _DEPTH_ARGS=("+actor_rollout_ref.rollout.engine_kwargs.vllm.scheduler_cls=specopd_vllm.depth_sched.${_SCHED_CLS}")
        ARM_LBL="${ARM_LBL}_tel"
        # Exported BEFORE the Ray start below: the TaskRunner, the rollout servers and their EngineCores inherit it.
        export SPECOPD_TELEMETRY=1
        # In the run's output dir: trainer.default_local_dir = checkpoints/<project>/<experiment_name> under this root.
        # A user-supplied (or inherited) dir gets the job id appended, so two jobs never share step_NNNNN.jsonl.
        if [ -n "${SPECOPD_TELEMETRY_DIR:-}" ]; then
            export SPECOPD_TELEMETRY_DIR="$SPECOPD_TELEMETRY_DIR/${SLURM_JOB_ID:-local}"
        else
            export SPECOPD_TELEMETRY_DIR="$REPO_ROOT/checkpoints/specopd/sopd_${ARM_LBL}_${SLURM_JOB_ID:-local}/specopd_telemetry"
        fi
        mkdir -p "$SPECOPD_TELEMETRY_DIR" ;;
    *) echo "CONFIG ERROR: DEPTH must be off or telemetry (got '$DEPTH')" >&2; exit 1 ;;
esac
SAVE_FREQ="${SAVE_FREQ:--1}"
# hf_model only: the student weights in HF format per saved step, for offline evaluation;
# no optimizer/extra, since nothing resumes from these.
_CKPT_ARGS=(trainer.save_freq=$SAVE_FREQ)
[ "$SAVE_FREQ" != "-1" ] && _CKPT_ARGS+=("actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]")

DATA="${DATA:-$CLUSTER_ROOT/workspace/datasets}"
TRAIN_FILES="${TRAIN_FILES:-$DATA/dapo-math-17k.parquet}"
VAL_FILES="${VAL_FILES:-[$DATA/test_data/MATH-500/test.parquet,$DATA/test_data/AIME24/test.parquet]}"
for _f in "$TRAIN_FILES" $(echo "$VAL_FILES" | tr -d '[]' | tr ',' ' '); do
    [ -f "$_f" ] || { echo "CONFIG ERROR: data file not found: $_f" >&2; exit 1; }
done

# --- guards ------------------------------------------------------------------------
(( TOK_PER_GPU * ULYSSES >= MAX_PROMPT + MAX_RESP )) || { echo "CONFIG ERROR: TOK_PER_GPU*ULYSSES < prompt+resp" >&2; exit 1; }
(( GPUS_PER_NODE % ULYSSES == 0 )) || { echo "CONFIG ERROR: ULYSSES must divide GPUS_PER_NODE" >&2; exit 1; }
_dp=$(( TRAINER_NNODES * GPUS_PER_NODE / ULYSSES ))
(( (PPO_MINI * ROLLOUT_N) % _dp == 0 )) || { echo "CONFIG ERROR: PPO_MINI*ROLLOUT_N not divisible by dp=$_dp" >&2; exit 1; }
(( TEACHER_GPUS % TEACHER_TP == 0 )) || { echo "CONFIG ERROR: teacher pool ($TEACHER_GPUS GPUs) not divisible by TP" >&2; exit 1; }
_POOL_NODES=$(( TRAINER_NNODES + TEACHER_NNODES ))
[ "$ALGO" != opd ] && _POOL_NODES=$TRAINER_NNODES   # no teacher pool
if [ -n "${SLURM_NNODES:-}" ] && (( SLURM_NNODES != _POOL_NODES )); then
    echo "CONFIG ERROR: --nodes=$SLURM_NNODES but pools sum to $_POOL_NODES (ALGO=$ALGO; GRPO/REINFORCE: sbatch --nodes=$TRAINER_NNODES)" >&2; exit 1
fi
if [ "$ALGO" = grpo ]; then
    (( ROLLOUT_N >= 2 )) || { echo "CONFIG ERROR: GRPO needs a group: ROLLOUT_N >= 2 (got $ROLLOUT_N)" >&2; exit 1; }
    ARM_LBL="grpo_${ARM_LBL}"
fi
# REINFORCE needs no group (no baseline); n stays ROLLOUT_N so the batch shape matches the GRPO and OPD arms
if [ "$ALGO" = reinforce ]; then ARM_LBL="reinforce_${ARM_LBL}"; fi
# TRACE_DUMP (validated after the TRIPLE presets): exported here, with the arm label final and BEFORE the Ray start, so
# the TaskRunner inherits it (srun --export=ALL). The dir name is the run's experiment name. No mkdir under DRY_RUN=1 (a
# login-node render writes nothing into the code tree; the trainer's own os.makedirs creates it anyway).
if [ "$TRACE_DUMP" = 1 ]; then
    export SPECOPD_TRACE_DIR="$REPO_ROOT/traces/sopd_${ARM_LBL}_${SLURM_JOB_ID:-local}"
    if [ "$DRY_RUN" = 0 ]; then mkdir -p "$SPECOPD_TRACE_DIR"; fi
else
    # sbatch exports the submit environment: an inherited SPECOPD_TRACE_DIR would silently dump every step.
    unset SPECOPD_TRACE_DIR
fi

echo "================================================================="
echo " specopd vanilla OPD SYNC [cluster/GH200]  job=${SLURM_JOB_ID:-local}  arm=$ARM_LBL  algo=$ALGO"
echo "   verl            : $_VERL_FILE (upstream @ $(git -C "$REPO_ROOT/verl" rev-parse --short HEAD 2>/dev/null), local diff: $(git -C "$REPO_ROOT/verl" diff --stat 2>/dev/null | tail -1))"
echo "   python          : $PYTHON_BIN (vllm $("$PYTHON_BIN" -c 'import vllm; print(vllm.__version__)' 2>/dev/null))"
echo "   triple          : $TRIPLE  drafter=${DRAFTER:-none}"
echo "   student/teacher : $(basename "$(dirname "$(dirname "$STUDENT")")") <- $(basename "$(dirname "$(dirname "$TEACHER")")") (teacher TP=$TEACHER_TP, $TEACHER_GPUS GPUs per teacher node)"
echo "   lengths         : $MAX_PROMPT + $MAX_RESP, thinking=$THINKING, lm_only=$LM_ONLY, rollout prefix cache=$ROLLOUT_PREFIX_CACHE"
echo "   speculation     : $SPEC $( [ "$SPEC" != none ] && echo "k=$SPEC_K") async_sched=$ASYNC_SCHED drafter_weights=${SPECOPD_DRAFTER_WEIGHTS:-n/a}"
echo "   drift/ckpt      : REF_KL=$REF_KL save_freq=$SAVE_FREQ traces=${SPECOPD_TRACE_DIR:-off}"
echo "   sampler TIM     : noise=${SAMPLER_NOISE:-off} seed=${SPECOPD_SAMPLER_NOISE_SEED:-n/a} rollout_quant=$ROLLOUT_QUANT kv_dtype=$KV_DTYPE rollout_seed=${ROLLOUT_SEED:-default} sampler_quant=${SAMPLER_QUANT:-off}${SPECOPD_SAMPLER_QUANT_STATIC:+ sampler_quant_static=$SPECOPD_SAMPLER_QUANT_STATIC}${SAMPLER_MODELOPT:+ sampler_modelopt=$SAMPLER_MODELOPT}${KV_SKIP_LAYERS:+ kv_skip_layers=$KV_SKIP_LAYERS}${SPECOPD_KV_METRICS:+ kv_metrics=$SPECOPD_KV_METRICS}${ROLLOUT_MAX_SEQS:+ rollout_max_seqs=$ROLLOUT_MAX_SEQS}"
echo "   stale sampler   : period=${SPECOPD_STALE_PERIOD:-off} lag=${SPECOPD_STALE_LAG:-off} cache=${SPECOPD_STALE_CACHE_DTYPE:-exact}${SPECOPD_STALE_SPILL_DIR:+ spill=$SPECOPD_STALE_SPILL_DIR} cap=${SPECOPD_STALE_MAX_GB_PER_NODE:-300}GB/node workers/node=${SPECOPD_STALE_WORKERS_PER_NODE:-n/a}" reserve=${SPECOPD_STALE_RESERVE_B_PER_PARAM:-n/a}B/param
echo "   advantage offset: ADV_OFFSET=${ADV_OFFSET:-off}"
echo "   score centering : SC=${SC:-off(current OPD)} rc_is=$RC_IS@$RC_IS_THRESHOLD topk=$( [ "$SC" = probe ] || [ "$SC" = 1 ] && echo "$TOPK_LOGPROBS chunk=$SC_CHUNK" || echo 0) entropy=$SC_ENTROPY ppo_mini=$PPO_MINI/$BATCH${_SC_PATH_BANNER}"
echo "   batch           : $BATCH x n=$ROLLOUT_N = $(( BATCH * ROLLOUT_N )) seqs/step, $STEPS steps, lr $LR, seed $SEED"
[ "$DEPTH" != off ] && echo "   depth plugin    : DEPTH=$DEPTH ${_DEPTH_ARGS[*]} -> ${SPECOPD_TELEMETRY_DIR:-}"
if [ "$ASYNC_MODE" = fully_async ]; then
    _ASYNC_TEACHER=""; [ "$ALGO" = opd ] && _ASYNC_TEACHER=" teacher=$TEACHER_GPUS x $TEACHER_NNODES node(s), unchanged"
    echo "[ASYNC] mode=fully_async ${_TRAINER_MODE_ARGS[*]} algorithm.rollout_correction.bypass_mode=False algorithm.rollout_correction.rollout_is=null (verl defaults, not overridden: recompute, no rollout IS) max_version_lag=$ASYNC_STALENESS gpu_split=trainer:$_TRAINER_GPN+rollout:$ASYNC_ROLLOUT_GPUS(one node)$_ASYNC_TEACHER"
elif [ -n "$ASYNC_MODE" ]; then
    echo "[ASYNC] mode=$ASYNC_MODE ${_TRAINER_MODE_ARGS[*]} algorithm.rollout_correction.bypass_mode=False algorithm.rollout_correction.rollout_is=null (verl defaults, not overridden: recompute, no rollout IS) max_version_lag=$(( ASYNC_MAX_OFF_POLICY - 1 ))"
fi
echo "================================================================="

# --- Ray across nodes -------------------------------------------------------------
# DRY_RUN=1 (see the experiment-shape section): the override list below gets a placeholder ray address.
if [ "$DRY_RUN" = 1 ]; then
RAY_ADDRESS="DRY_RUN:0"
echo "[DRY-RUN] no Ray started; VERL_CONTINUOUS_TOKEN_MODEL_FAMILY=${VERL_CONTINUOUS_TOKEN_MODEL_FAMILY:-<unset>}"
else
export GPUS_PER_TASK="$GPUS_PER_NODE"
RAY_NNODES="${SLURM_NNODES:-2}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "${nodes[0]}" hostname --ip-address)
[[ "$head_node_ip" == *" "* ]] && { read -ra _a <<<"$head_node_ip"; head_node_ip="${_a[0]}"; }
head_node_port="${RAY_PORT:-6379}"
ip_head="${head_node_ip}:${head_node_port}"
export head_node_ip head_node_port ip_head PYTHON_BIN RAY_TMPDIR
export VLLM_HOST_IP="$head_node_ip"
export RAY_ADDRESS="$ip_head"

RAY_STEP_PIDS=()
cleanup() {
    set +e
    "$PYTHON_BIN" -m ray.scripts.scripts stop --force >/dev/null 2>&1
    for _p in "${RAY_STEP_PIDS[@]:-}"; do [ -n "$_p" ] && kill "$_p" 2>/dev/null; done
}
trap cleanup EXIT
"$PYTHON_BIN" -m ray.scripts.scripts stop --force >/dev/null 2>&1 || true
srun --export=ALL --ntasks="$RAY_NNODES" --ntasks-per-node=1 "$REPO_ROOT/scripts/cluster_start_ray.sh" &
RAY_STEP_PIDS+=("$!")
connected=0
for attempt in {1..40}; do
    sleep 10
    connected=$("$PYTHON_BIN" -m ray.scripts.scripts status --address "$ip_head" 2>/dev/null | grep -c "node_" || true)
    echo "Ray nodes connected: ${connected:-0}/${RAY_NNODES} (attempt ${attempt}/40)"
    (( ${connected:-0} >= RAY_NNODES )) && break
done
(( connected >= RAY_NNODES )) || { echo "Ray cluster incomplete; refusing to train" >&2; exit 1; }
fi

# --- train -------------------------------------------------------------------------
# The override list is built as an array, printed one per line ([HYDRA-OVERRIDE]) and passed verbatim.
# k1 must run with use_policy_gradient=True (its gradient is not the gradient of the KL).
_HYDRA_ARGS=(
    ++ray_kwargs.ray_init.address="$RAY_ADDRESS"
    algorithm.adv_estimator=$_ADV_EST
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILES"
    data.val_files="$VAL_FILES"
    data.train_batch_size=$BATCH
    data.shuffle=$SHUFFLE
    data.seed=$SEED
    data.max_prompt_length=$MAX_PROMPT
    data.max_response_length=$MAX_RESP
    data.filter_overlong_prompts=False
    data.truncation=left
    data.return_raw_chat=True
    ${_TEMPLATE_ARGS[@]+"${_TEMPLATE_ARGS[@]}"}
    reward.custom_reward_function.path=pkg://custom_reward.specopd_reward
    reward.custom_reward_function.name=reward_func
    actor_rollout_ref.model.path="$STUDENT"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.optim.lr=$LR
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI
    ${_CLIP_ARGS[@]+"${_CLIP_ARGS[@]}"}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$TOK_PER_GPU
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ULYSSES
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_OFFLOAD
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OFFLOAD
    ${_REF_ARGS[@]+"${_REF_ARGS[@]}"}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_UTIL
    actor_rollout_ref.rollout.max_model_len=$CTX
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=$ROLLOUT_PREFIX_CACHE
    ${_LM_ARGS[@]+"${_LM_ARGS[@]}"}
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.n=$ROLLOUT_N
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$TOK_PER_GPU
    ${_SPEC_ARGS[@]+"${_SPEC_ARGS[@]}"}
    ${_LOADFMT_ARGS[@]+"${_LOADFMT_ARGS[@]}"}
    ${_ASYNC_ARGS[@]+"${_ASYNC_ARGS[@]}"}
    ${_DEPTH_ARGS[@]+"${_DEPTH_ARGS[@]}"}
    ${_TIM_ARGS[@]+"${_TIM_ARGS[@]}"}
    ${_SC_ARGS[@]+"${_SC_ARGS[@]}"}
)
if [ "$ALGO" = opd ]; then
    _HYDRA_ARGS+=(
        distillation.enabled=True
        distillation.n_gpus_per_node=$TEACHER_GPUS
        distillation.nnodes=$TEACHER_NNODES
        distillation.teacher_models.teacher_model.model_path="$TEACHER"
        distillation.teacher_models.teacher_model.inference.name=vllm
        distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=$TEACHER_TP
        distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=$TEACHER_GPU_UTIL
        distillation.teacher_models.teacher_model.inference.max_model_len=$TEACHER_CTX
        distillation.teacher_models.teacher_model.inference.max_num_seqs=$TEACHER_MAX_SEQS
        distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
        distillation.teacher_models.teacher_model.inference.enforce_eager=False
        distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=$TEACHER_MAX_BATCHED_TOKENS
        distillation.distillation_loss.loss_mode=k1
        distillation.distillation_loss.use_policy_gradient=True
        distillation.distillation_loss.use_task_rewards=False
        ${_ADV_ARGS[@]+"${_ADV_ARGS[@]}"})
else
    # GRPO / REINFORCE: no teacher; the task reward is the objective. Explicit, so the override list documents the arm.
    _HYDRA_ARGS+=(distillation.enabled=False)
    # GRPO's std normalization; REINFORCE has no normalization at all, so the (ignored) GRPO key is not set
    if [ "$ALGO" = grpo ]; then _HYDRA_ARGS+=(algorithm.norm_adv_by_std_in_grpo=True); fi
    _HYDRA_ARGS+=(actor_rollout_ref.actor.entropy_coeff=0)
    # rollout_log_probs -> rollout_corr/kl (and the IS weights / bypass old_log_probs); SC / RC_IS arms already set it
    case " ${_SC_ARGS[*]-} " in
        *" actor_rollout_ref.rollout.calculate_log_probs=True "*) ;;
        *) _HYDRA_ARGS+=(actor_rollout_ref.rollout.calculate_log_probs=True) ;;
    esac
    # no KL loss (REF_KL=1 keeps its use_kl_loss=True with coef 0: a logged diagnostic, never in the gradient)
    if [ "$REF_KL" = 0 ]; then _HYDRA_ARGS+=(actor_rollout_ref.actor.use_kl_loss=False); fi
fi
_HYDRA_ARGS+=(
    trainer.logger="[console]"
    trainer.project_name=specopd
    trainer.experiment_name="sopd_${ARM_LBL}_${SLURM_JOB_ID:-local}"
    trainer.n_gpus_per_node=$_TRAINER_GPN
    trainer.nnodes=$TRAINER_NNODES
    trainer.val_before_train=False
    trainer.test_freq=-1
    "${_CKPT_ARGS[@]}"
    trainer.total_epochs=1
    trainer.total_training_steps=$STEPS
    trainer.resume_mode=disable
    ${_TRAINER_MODE_ARGS[@]+"${_TRAINER_MODE_ARGS[@]}"}
)
# A key given twice would leave its value to Hydra's precedence rules: refuse instead. (xtrace off for the two loops:
# ~4 trace lines per override would bury the list itself; restored right after.)
{ set +x; } 2>/dev/null
_seen=" "
for _o in "${_HYDRA_ARGS[@]}"; do
    _k="${_o%%=*}"; _k="${_k#++}"; _k="${_k#+}"
    case "$_seen" in *" $_k "*) echo "CONFIG ERROR: Hydra key $_k is overridden twice" >&2; exit 1 ;; esac
    _seen="$_seen$_k "
done
echo "[HYDRA-OVERRIDE] ${#_HYDRA_ARGS[@]} overrides (ALGO=$ALGO arm=$ARM_LBL):"
for _o in "${_HYDRA_ARGS[@]}"; do echo "[HYDRA-OVERRIDE]   $_o"; done
if [ "$DRY_RUN" = 1 ]; then echo "[DRY-RUN] override list rendered; not training"; exit 0; fi
set -x
env -u ROCR_VISIBLE_DEVICES "$PYTHON_BIN" -m verl.trainer.main_ppo "${_HYDRA_ARGS[@]}"

echo "=== training finished ==="

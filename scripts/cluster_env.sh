#!/bin/bash
# Environment for the GH200 cluster. Source it; do not execute.
#
# The nodes are GH200 (aarch64 Grace CPU + Hopper GPU, 4 per node, 96 GiB HBM3 each). Load a Python stack with
# vLLM 0.19.1 / PyTorch 2.10 for aarch64 (module, container or venv); edit the lines below for your site. The
# module lines are placeholders: replace them with whatever loads your stack, or delete them if PYTHON_BIN below
# already points at a complete environment.
#
# Module init scripts can read unset variables (lmod's reads FPATH), so they fail under `set -u` -- and a batch
# script that does NOT use `set -euo pipefail` is worse. Drop -u only across the loading lines, then restore
# whatever the caller had. `set +u` unconditionally would silently disarm the caller's own safety for the rest
# of the run.
_had_u=0; case "$-" in *u*) _had_u=1;; esac
set +u
[ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash   # your module system's init, if any
export MODULEPATH=/path/to/modulefiles:${MODULEPATH:-}
module load your-vllm-0.19.1-module
[ "$_had_u" = 1 ] && set -u
unset _had_u

export CLUSTER_ROOT="${CLUSTER_ROOT:-/path/to/workdir}"
# The vendored verl (upstream verl + minimal patches) must be the only verl on PYTHONPATH: Ray workers inherit
# this env from cluster_start_ray.sh, so any other verl path here would silently put that verl in every worker.
# Follow the tree the caller runs from (REPO_ROOT; every job script sets it to its submit directory, the
# repository root): a running job then keeps importing the code it started with even while another checkout
# is edited, since Python imports some modules lazily, mid-run. Without REPO_ROOT (an interactive shell), the
# checkout that holds this file is used.
export SPEC_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export REPO=$SPEC_ROOT
# The Python environment. Besides vLLM and PyTorch it needs verl's own dependencies (tensordict<=0.10,
# TransferQueue==0.1.10, codetiming, pylatexenc, torchdata) and the math verifier's (latex2sympy2_extended,
# math_verify, sympy). Without the second group the math verifier reward cannot be computed at all.
export PYTHON_BIN=$CLUSTER_ROOT/venv-specopd/bin/python
export PYTHONPATH=$REPO/verl:$REPO   # upstream verl clone + custom_reward/
# The bench and evaluation tools read models from $CLUSTER_ROOT/hf/hub, so HF_HOME is set to match.
export HF_HOME=$CLUSTER_ROOT/hf
# Jobs run offline by default: download models (into $HF_HOME) and datasets (into $CLUSTER_ROOT/workspace/datasets)
# beforehand, so no job contacts the Hub at run time; this also suits compute nodes without internet access. Set
# HF_HUB_OFFLINE=0 explicitly only if a job should fetch from the Hub.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

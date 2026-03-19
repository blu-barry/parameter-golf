#!/usr/bin/env bash
set -euo pipefail

case_name="${1:-all}"

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
ITERATIONS="${ITERATIONS:-800}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
NPROC="${NPROC:-8}"

run_case() {
  local run_id="$1"
  local attn_sparsity="$2"
  local mlp_sparsity="$3"
  local anneal_frac="$4"

  echo "==> starting ${run_id}"
  RUN_ID="${run_id}" \
  DATA_PATH="${DATA_PATH}" \
  TOKENIZER_PATH="${TOKENIZER_PATH}" \
  VOCAB_SIZE="${VOCAB_SIZE}" \
  ITERATIONS="${ITERATIONS}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS}" \
  ATTN_SPARSITY="${attn_sparsity}" \
  MLP_SPARSITY="${mlp_sparsity}" \
  SPARSE_MIN_KEEP=4 \
  SPARSITY_ANNEAL_FRAC="${anneal_frac}" \
  python3 -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" train_gpt.py
}

case "${case_name}" in
  conservative)
    run_case "sparse800_sp1024_a08_m20_af100" "0.08" "0.20" "1.0"
    ;;
  balanced)
    run_case "sparse800_sp1024_a10_m25_af100" "0.10" "0.25" "1.0"
    ;;
  mlp_bias)
    run_case "sparse800_sp1024_a05_m30_af100" "0.05" "0.30" "1.0"
    ;;
  all)
    run_case "sparse800_sp1024_a08_m20_af100" "0.08" "0.20" "1.0"
    run_case "sparse800_sp1024_a10_m25_af100" "0.10" "0.25" "1.0"
    run_case "sparse800_sp1024_a05_m30_af100" "0.05" "0.30" "1.0"
    ;;
  *)
    echo "usage: bash scripts/run_h100_sweep.sh [conservative|balanced|mlp_bias|all]" >&2
    exit 1
    ;;
esac

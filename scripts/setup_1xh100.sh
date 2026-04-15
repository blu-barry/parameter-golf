#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a 1xH100 machine for tokenizer sweep experiments.
# Safe to re-run: it reuses the existing virtualenv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UPGRADE_PIP="${UPGRADE_PIP:-1}"

echo "root_dir=${ROOT_DIR}"
echo "python_bin=${PYTHON_BIN}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

if [[ "${UPGRADE_PIP}" == "1" ]]; then
  python -m pip install --upgrade pip
fi

python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install sentencepiece huggingface-hub datasets tqdm

echo "setup_complete=1"
echo "activate_with=source ${VENV_DIR}/bin/activate"
echo "run_sweep_example=python ${ROOT_DIR}/scripts/run_sp_vocab_sweep.py --prepare-artifacts"

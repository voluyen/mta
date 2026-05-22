#!/bin/bash
# =============================================================================
# run.sh — Full pipeline: environment setup + data processing + training
# Run from repo root: bash run.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
CONDA_ENV="mta"
VENV_MARKER="${REPO_ROOT}/.env_installed"

mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

RESULTS_DIR="${REPO_ROOT}/distillm-master/results"
MOUNT_DIR="/mnt/mta/"

run_script() {
    local label="$1"
    local script="$2"
    local logfile="${LOG_DIR}/${label}.log"
    log "Starting: ${label}"
    bash "${REPO_ROOT}/${script}" 2>&1 | tee "${logfile}"
    log "Done: ${label} — log saved to ${logfile}"
    if [ -d "${RESULTS_DIR}" ]; then
        log "Copying results to ${MOUNT_DIR}..."
        mkdir -p "${MOUNT_DIR}"
        cp -r "${RESULTS_DIR}/." "${MOUNT_DIR}/"
        log "Results copied."
    else
        log "No results directory found, skipping copy."
    fi
}

# =============================================================================
# 1. Environment setup
# =============================================================================
setup_env() {
    log "=== Environment Setup ==="

    if command -v conda &>/dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        if ! conda env list | grep -q "^${CONDA_ENV} "; then
            log "Creating conda env '${CONDA_ENV}' with Python 3.11..."
            conda create -y -n "${CONDA_ENV}" python=3.11
        else
            log "Conda env '${CONDA_ENV}' already exists."
        fi
        conda activate "${CONDA_ENV}"
    else
        log "conda not found, installing into current Python environment."
    fi

    if [ -f "${VENV_MARKER}" ]; then
        log "Packages already installed (delete ${VENV_MARKER} to reinstall)."
        return
    fi

    log "Upgrading pip..."
    pip install --upgrade pip

    log "Installing PyTorch 2.9.0 with CUDA 12.8..."
    pip install torch==2.9.0 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

    log "Installing pip packages..."
    pip install --prefer-binary \
        transformers==4.43.2 \
        peft==0.9.0 \
        trl==0.9.6 \
        deepspeed \
        accelerate \
        datasets \
        sentencepiece \
        protobuf \
        rouge-score \
        nltk \
        numerize \
        torchtyping \
        rich \
        spacy \
        numpy \
        tqdm \
        huggingface_hub

    log "Downloading spaCy English model (required by span_finetune.py)..."
    python -m spacy download en_core_web_sm

    touch "${VENV_MARKER}"
    log "Environment setup complete."
}

# =============================================================================
# 4. Training runs
# =============================================================================
run_training() {
    log "=== Training ==="

    run_script "gpt2_csd"          "distillm-master/scripts/gpt2/spancsd/train_0.1B_1.5B_csd.sh"
    run_script "gpt2_spancsd"          "distillm-master/scripts/gpt2/spancsd/train_0.1B_1.5B_spancsd.sh"
    # run_script "gpt2_ablation_word"    "distillm-master/scripts/gpt2/ablation/word_level.sh"
    # run_script "gpt2_ablation_phrase"  "distillm-master/scripts/gpt2/ablation/phrase_level.sh"
    # run_script "opt_spancsd"           "distillm-master/scripts/opt/spancsd/train_6.7B_1.3B_teacher_lora.sh"
}

# =============================================================================
# Main
# =============================================================================
cd "${REPO_ROOT}"

if [ -f "${VENV_MARKER}" ]; then
    log "Found existing install marker, removing to force reinstall..."
    rm "${VENV_MARKER}"
fi
setup_env
run_training

log "=== All done. Logs in ${LOG_DIR}/ ==="

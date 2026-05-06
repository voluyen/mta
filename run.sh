#!/bin/bash
# =============================================================================
# run.sh — Full pipeline: environment setup + data processing + training
# Run from repo root: bash run.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
VENV_DIR="${REPO_ROOT}/.venv"
VENV_MARKER="${REPO_ROOT}/.env_installed"

mkdir -p "${LOG_DIR}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_script() {
    local label="$1"
    local script="$2"
    local logfile="${LOG_DIR}/${label}.log"
    log "Starting: ${label}"
    bash "${REPO_ROOT}/${script}" 2>&1 | tee "${logfile}"
    log "Done: ${label} — log saved to ${logfile}"
}

# =============================================================================
# 1. Environment setup
# =============================================================================
setup_env() {
    log "=== Environment Setup ==="

    if [ ! -d "${VENV_DIR}" ]; then
        log "Creating Python venv at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        log "Venv already exists, activating."
    fi

    source "${VENV_DIR}/bin/activate"

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
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    pip install \
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

    # --- distillm-master: GPT-2 ---
    run_script "gpt2_spancsd_entropy"    "distillm-master/scripts/gpt2/spancsd/train_0.1B_1.5B_entropy.sh"
    run_script "gpt2_spandistillm_entropy" "distillm-master/scripts/gpt2/spandistillm/train_0.1B_1.5B_entropy.sh"
    run_script "gpt2_spanfdd_entropy"    "distillm-master/scripts/gpt2/spanfdd/train_0.1B_1.5B_entropy.sh"

    # --- distillm-master: OPT ---
    run_script "opt_spancsd_entropy"     "distillm-master/scripts/opt/spancsd/train_6.7B_1.3B_teacher_lora_entropy.sh"
    run_script "opt_spanfdd_entropy"     "distillm-master/scripts/opt/spanfdd/train_6.7B_1.3B_teacher_lora_entropy.sh"
    run_script "opt_spandistillm_entropy" "distillm-master/scripts/opt/spandistillm/train_6.7B_1.3B_teacher_lora_entropy.sh"

    # --- distillm-master: Qwen1.5 ---
    run_script "qwen1.5_spancsd_entropy"    "distillm-master/scripts/qwen1.5/spancsd/train_0.5B_1.8B_entropy.sh"
    run_script "qwen1.5_spanfdd_entropy"    "distillm-master/scripts/qwen1.5/spanfdd/train_0.5B_1.8B_entropy.sh"
    run_script "qwen1.5_spandistillm_entropy" "distillm-master/scripts/qwen1.5/spandistillm/train_0.5B_1.8B_entropy.sh"

    # --- distillm-2-master ---
    run_script "distillm2_gpt2_entropy"   "distillm-2-master/scripts/gpt2/span_distillm_2_gpt2_0.1b_entropy.sh"
    run_script "distillm2_opt_entropy"    "distillm-2-master/scripts/opt/span_distillm_2_opt_1.3b_entropy.sh"
    run_script "distillm2_qwen1.5_entropy" "distillm-2-master/scripts/qwen1.5/span_distillm_2_qwen1.5_0.5b_entropy.sh"
}

# =============================================================================
# Main
# =============================================================================
cd "${REPO_ROOT}"

rm -f "${VENV_MARKER}"
setup_env
run_training

log "=== All done. Logs in ${LOG_DIR}/ ==="

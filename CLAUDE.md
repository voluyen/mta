# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM knowledge distillation research project implementing **SpanDistilLM** and related techniques (DistiLLM, FDD, SpanFDD). The goal is to compress large teacher LLMs into smaller student models using span-based hidden state alignment and KD loss. It extends the DistiLLM and DistiLLM-2 frameworks.

## Repository Structure

```
MTA/
├── src/                        # Custom trainer pipeline (non-DeepSpeed)
│   ├── run_distill_llm.py      # Entry point for distillation training
│   ├── run_eval.py             # Evaluation entry point
│   ├── llm_train.py            # Trainer class (KD loss, hidden alignment)
│   ├── teacher_llm.py          # Teacher model wrappers (Qwen, Mistral, GPT2...)
│   ├── student.py              # Student model wrapper
│   ├── data_utils.py           # Dataset and collator for instruction pairs
│   ├── evaluator.py            # ROUGE-L-based evaluation
│   ├── arguments.py            # Training config dataclass
│   └── utils.py                # Span extraction helpers
├── scripts/                    # Eval scripts for trained checkpoints
├── distillm/            # Main training framework (DeepSpeed + torchrun)
│   ├── span_finetune.py        # Primary training entry point
│   ├── finetune.py             # Baseline (non-span) training entry point
│   ├── distillm/               # DistiLLM KD loss implementations
│   ├── scripts/                # Training scripts by model family
│   │   ├── gpt2/{distillm,fdd,sft,spandistillm,spanfdd}/
│   │   ├── opt/...
│   │   ├── qwen1.5/...
│   │   └── qwen2.5/...
│   └── configs/deepspeed/      # DeepSpeed ZeRO configs
├── distillm-2/          # DistiLLM-2 variant
└── data/dolly/                 # Primary dataset (train/dev/valid .jsonl)
```

## Two Training Pipelines

There are two separate pipelines with different APIs:

**1. `distillm/` (primary)** — DeepSpeed + `torchrun`, uses `--kebab-case` args, called via `span_finetune.py` or `finetune.py`. This is where most experiments run.

**2. `src/` (custom)** — Single-process or multi-GPU without DeepSpeed, uses HuggingFace `HfArgumentParser` with `Arguments` dataclass (snake_case fields). Entry: `run_distill_llm.py`.

## Common Commands

### Install dependencies (distillm)
```bash
# Conda environment first:
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.43.2 vllm==0.5.4 peft==0.9.0 trl==0.9.6 deepspeed==0.15.0
pip install accelerate datasets sentencepiece protobuf rouge-score nltk numerize torchtyping rich
```

### Training (distillm pipeline)
```bash
# SpanDistilLM — GPT2 0.1B student from 1.5B teacher
bash distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B.sh

# SpanDistilLM with entropy weight
bash distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B_entropy.sh

# Baselines: distillm, fdd, spanfdd, sft
bash distillm/scripts/gpt2/distillm/train_0.1B_1.5B.sh
bash distillm/scripts/gpt2/sft/sft_base.sh
```

### Data preparation
```bash
bash distillm/scripts/gpt2/tools/process_data_dolly.sh
bash distillm/scripts/gpt2/tools/generate_data_seqkd.sh   # SeqKD teacher data
```

### Evaluation
```bash
# Evaluate a trained checkpoint on dolly test set
bash scripts/eval_gpt2_0.1B.sh         # GPT2 0.1B
bash scripts/eval_qwen1.5_0.5B.sh      # Qwen1.5 0.5B
bash scripts/eval_span_gpt2_0.1B.sh    # Span-variant

# Direct call:
python src/run_eval.py \
  --train_data data/dolly/train.jsonl \
  --test_data data/dolly/valid.jsonl \
  --model_path <checkpoint_path> \
  --tokenizer openai-community/gpt2 \
  --student_device cuda:0
```

## Key Design Decisions

### Span-based Hidden State Alignment
The core novelty: instead of aligning all hidden states, the model aligns spans (phrases/words/subwords) between teacher and student. Layer mappings must be explicitly configured:
```bash
--teacher_layer_mapping 24 36 48
--student_layer_mapping 6 9 12
--split_layer_mapping 0 1 3 3
```

### Teacher/Student Device Split
In `src/`, teacher and student can be placed on separate GPUs (`--teach_device cuda:1 --student_device cuda:0`). In `distillm/`, DeepSpeed handles device placement.

### Entropy Weight
Add `--entropy_weight` flag to weight KD loss by token entropy (currently supported for DistiLLM and CSD variants only). See `distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B_entropy.sh`.

### Supported Model Families
GPT-2, OPT, LLaMA/LLaMA-2, Qwen1.5, Qwen2.5, Mistral, TinyLLaMA, MiniCPM. Model type is passed as `--model_type` (e.g., `qwen`, `llama`, `gpt2`) and controls tokenizer padding behavior.

### Results Layout
Checkpoints are saved to `distillm/results/<model>/<run_name>/<step>/`. Eval outputs go to `eval_outputs/`.

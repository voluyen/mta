# MTA — Knowledge Distillation for LLMs (research monorepo)

This repository bundles several knowledge-distillation (KD) codebases for compressing large teacher LLMs into smaller students. Each top-level directory is a largely independent implementation; this README provides a single entry point with one section per subproject.

> **Note for contributors:** `CLAUDE.md` (repo root) holds working instructions for Claude Code and is intentionally separate from this README.

## Repository layout

| Path | Description |
|------|-------------|
| `distillm/` | Primary training framework — **SpanDistilLM** + DistiLLM baselines (DeepSpeed + `torchrun`). |
| `distillm-2/` | DistiLLM-2 variant of the above framework. |
| `src/` | Custom single-/multi-GPU trainer pipeline (no DeepSpeed); span-based hidden-state alignment. |
| `DSKDv2/` | **DSKD v2 + MTA** — Dual-Space KD with Multi-layer Token-Aligned span/feature distillation. |
| `AMiD/` | **AMiD** (ICLR 2026) — KD with α-mixture assistant distribution; official paper implementation. |
| `data/dolly/` | Primary instruction dataset (`train` / `dev` / `valid` `.jsonl`). |

---

## 1. SpanDistilLM / DistiLLM (`distillm/`, `src/`)

The core novelty is **span-based hidden-state alignment**: instead of aligning all hidden states, teacher and student are aligned over spans (phrases / words / subwords). Layer mappings are configured explicitly:

```bash
--teacher_layer_mapping 24 36 48
--student_layer_mapping 6 9 12
--split_layer_mapping 0 1 3 3
```

### Two pipelines

- **`distillm/` (primary)** — DeepSpeed + `torchrun`, `--kebab-case` args, entry: `span_finetune.py` (span) or `finetune.py` (baseline). Most experiments run here.
- **`src/` (custom)** — single-process or multi-GPU without DeepSpeed; HuggingFace `HfArgumentParser` with snake_case fields. Entry: `run_distill_llm.py`. Teacher and student can sit on separate GPUs (`--teach_device cuda:1 --student_device cuda:0`).

### Install (distillm)

```bash
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers==4.43.2 vllm==0.5.4 peft==0.9.0 trl==0.9.6 deepspeed==0.15.0
pip install accelerate datasets sentencepiece protobuf rouge-score nltk numerize torchtyping rich
```

### Data preparation

```bash
bash distillm/scripts/gpt2/tools/process_data_dolly.sh
bash distillm/scripts/gpt2/tools/generate_data_seqkd.sh   # SeqKD teacher data
```

### Training

```bash
# SpanDistilLM — GPT2 0.1B student from 1.5B teacher
bash distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B.sh
# With entropy weight
bash distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B_entropy.sh

# Baselines
bash distillm/scripts/gpt2/distillm/train_0.1B_1.5B.sh
bash distillm/scripts/gpt2/sft/sft_base.sh
```

**Entropy weight:** add `--entropy_weight` to weight KD loss by token entropy (supported for DistiLLM and CSD variants). Example: `distillm/scripts/gpt2/spandistillm/train_0.1B_1.5B_entropy.sh`.

### Evaluation

```bash
bash scripts/eval_gpt2_0.1B.sh         # GPT2 0.1B
bash scripts/eval_span_gpt2_0.1B.sh    # Span variant

# Direct call:
python src/run_eval.py \
  --train_data data/dolly/train.jsonl \
  --test_data data/dolly/valid.jsonl \
  --model_path <checkpoint_path> \
  --tokenizer openai-community/gpt2 \
  --student_device cuda:0
```

**Supported model families:** GPT-2, OPT, LLaMA/LLaMA-2, Qwen1.5, Qwen2.5, Mistral, TinyLLaMA, MiniCPM (`--model_type` controls tokenizer padding). Checkpoints save to `distillm/results/<model>/<run_name>/<step>/`; eval outputs to `eval_outputs/`.

---

## 2. DSKD v2 + MTA (`DSKDv2/`)

DSKD v2 — Dual-Space Knowledge Distillation for LLMs (paper: arXiv 2504.11426). This tree adds **MTA** (Multi-layer Token-Aligned) span/feature distillation on top of upstream DSKD v2. Supports same- and cross-tokenizer KD, off- and on-policy training, multiple divergences, LoRA, and DeepSpeed ZeRO.

### Setup

No `requirements.txt` is included; install manually:

```bash
pip install "torch>=2.0.0" "transformers>=4.30.0" "deepspeed>=0.10.0" "peft>=0.5.0" datasets rouge-score accelerate
```

Models are expected at `model_hub/<type>/<name>/` (e.g. `model_hub/gpt2/gpt2-base/`); teacher checkpoints can be HF Hub IDs (e.g. `VoCuc/Qwen1.5_1.8B_SFT_Dolly`). Data lives under `data/<task>/` as `train.jsonl` / `dev.jsonl` / optional `test.jsonl` with `{"prompt": ..., "response": ...}` entries.

### Running training

Launched via `torchrun` + DeepSpeed wrappers in `scripts/dolly/<student>/` (invoke **from the project root**, not inside `scripts/`):

```bash
bash scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh        # all visible GPUs
bash scripts/dolly/gpt2-120M/run_mta_dskdv2_eta.sh 0,1    # restrict CUDA_VISIBLE_DEVICES
```

Outputs land in `outputs/<ckpt_type>/<ckpt_name>/<task>/<setting>/` with `train.log` tee'd alongside checkpoints.

Three script variants per student:
- `run_dskdv2.sh` — DSKD v2 with cross-model attention (same-tokenizer baseline).
- `run_dskdv2_eta.sh` — DSKD v2 with Exact Token Alignment (`--criterion dual_space_kd_v2_with_eta`), the cross-tokenizer path.
- `run_mta_dskdv2_eta.sh` — adds `--MTA-mode` plus layer-mapping flags (`--teacher_layer_mapping`, `--student_layer_mapping`, `--split_layer_mapping`, `--w-span-loss`).

DeepSpeed config is auto-selected from `--model-dtype` (`bf16` → `configs/deepspeed/ds_config_bf16.json`, `fp16` → `ds_config.json`, `fp32` → `ds_config_fp32.json`). A ZeRO-3 variant exists at `configs/deepspeed/ds_config_zero3_bf16.json`.

### Evaluation

`code/evaluate_main.py` is the standalone eval entry point (uses `code/evaluate.py` + `code/rouge_metric.py`). Dev-set eval and ROUGE-L generation also run inline during training when `--do-valid --eval-gen` are set; checkpoints are pruned to top `--keep-best-n-checkpoints` by ROUGE-L (or eval loss if `--eval-gen` is off).

Quick syntax check (no GPU):

```bash
python -c "import py_compile, pathlib; [py_compile.compile(str(p), doraise=True) for p in pathlib.Path('code').rglob('*.py')]"
```

There is no test suite.

### Architecture

**Entry-point flow (`code/distillation.py`):** `main()` → parses `arguments.py` → builds `Distiller` → `prepare_dataset()` → `deepspeed.initialize(model=distiller, ...)` → `finetune()`. The `Distiller` itself is the `nn.Module` passed to DeepSpeed; `model.module.student_model` is the wrapped HF causal LM.

**Distiller (`code/distiller.py`)** owns both student and (optional) teacher HF models + tokenizers, plus auxiliary trainable params:
- `t2s_projector` / `s2t_projector` — dual-space projectors instantiated when `--criterion` contains `dual_space`. Optional logit-identity init via `--init-t2s-projector` / `--init-s2t-projector`.
- `mta_projector_list` — one `nn.Linear(student_hidden, teacher_hidden)` per entry in `--teacher_layer_mapping`, used by MTA span/feature losses.
- Optional teacher↔student token/id mappings loaded from JSON (used by `min_edit_dis_kld` only).
- `add_optimizer_param_group` injects projector params with their own `--projector-lr`.

**Criterion registry (`code/criterions/__init__.py`):** `build_criterion(args)` dispatches `args.criterion` to one of `cross_entropy`, `various_divergence`, `dual_space_kd`, `dual_space_kd_v2`, `dual_space_kd_v2_with_eta`, `universal_logit_distillation`, `min_edit_dis_kld`. Each criterion's `forward(distiller, batch, logging_output)` returns `(loss, logging_output)`; `--kd-objective` (`forward_kl`, `reverse_kl`, `js_divergence`, `skewed_forward_kl`, `skewed_reverse_kl`, `adaptive_kl`) selects the divergence inside `various_divergence.py`. MTA span/feature losses are layered in when `--MTA-mode` is set, using the layer-mapping triples weighted by `--w-span-loss`.

**Data pipeline (`code/data_utils/distill_datasets.py`):** `DistillDataset(args, split, student_tokenizer, teacher_tokenizer)` produces per-batch dicts with `input_batch`, `label_batch`, `teacher_input_batch`, `teacher_label_batch`, `prompt_batch`. `label_batch["label"]` is `-100`-masked over prompt tokens; token-count normalization is global across grad-accum × DP world size. On-policy adds parallel `op_*_batch` keys from student rollouts re-tokenized for the teacher.

**On-policy distillation:** enabled by `--on-policy` (commented out in default scripts). After `--on-policy-after-n-epochs`, each batch with prob `--stu-gen-ratio` is replaced by student-generated continuations (or teacher-mixed for `--criterion minillm`). Cross-tokenizer re-runs the teacher tokenizer on decoded student text and rebuilds aligned `op_teacher_input_batch`.

**Checkpoint saving:** (1) ZeRO-3 collects partitioned params via `deepspeed.zero.GatheredParameters` before `student_model.save_pretrained` (handles `tie_word_embeddings` for Qwen2-0.5B-style models); (2) ZeRO-≤2 saves directly. `model.module.projectors.state_dict()` is dumped to `projector.pt` when projectors exist. Use `--only-save-projector` to skip the base model (LoRA/projector-only runs).

**Arg conventions (`code/arguments.py`):**
- `--save-interval -1` / `--eval-interval -1` mean "once per epoch".
- `--topk-vocab -1` disables top-k truncation in dual-space losses.
- `--teacher-model-fp16` loads the teacher in fp16 regardless of `--model-dtype`.
- `--gradient-checkpointing` is available but disabled in default scripts.

---

## 3. AMiD — KD with α-mixture Assistant Distribution (ICLR 2026) (`AMiD/`)

Official implementation of **[AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](https://arxiv.org/abs/2510.15982)** (ICLR 2026).
[arxiv](https://arxiv.org/pdf/2510.15982) | [OpenReview](https://openreview.net/forum?id=7WPJ0EgPdW)

Donghyeok Shin, Yeongmin Kim, Suhyeon Jo, Byeonghu Na, Il-Chul Moon.

> **Abstract** — Autoregressive LLMs achieve remarkable performance but incur high compute/memory cost. Knowledge distillation mitigates this by transferring knowledge from a large teacher to a smaller student through distributional alignment. Prior work proposed various discrepancy metrics, but the capacity gap and training instability from near-zero probabilities remain fundamental limitations. AMiD proposes the **α-mixture assistant distribution**, a generalized family of assistant distributions that continuously extends prior approaches via a new design variable α, and generalizes the family of divergences based on optimality. Extensive experiments show superior performance and training stability over a broader, theoretically grounded assistant-distribution space.

### Environment

```bash
bash install.sh
```

### Datasets

Raw instruction-response data (before processing) from [MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm):

```bash
huggingface-cli download MiniLLM/dolly --repo-type dataset --local-dir ./data/dolly/
huggingface-cli download MiniLLM/self-inst --repo-type dataset --local-dir ./data/self-inst/
huggingface-cli download MiniLLM/Vicuna --repo-type dataset --local-dir ./data/vicuna/
huggingface-cli download MiniLLM/sinst --repo-type dataset --local-dir ./data/sinst/
huggingface-cli download MiniLLM/uinst --repo-type dataset --local-dir ./data/uinst/
```

Processed data:

```bash
huggingface-cli download MiniLLM/dolly-processed --repo-type dataset --local-dir ./processed_data/dolly/
huggingface-cli download MiniLLM/openwebtext-processed --repo-type=dataset --local-dir ./processed_data/openwebtext/gpt2/512/10M/
```

### Models

Download checkpoints from the [Huggingface Model Hub](https://huggingface.co/models) into `checkpoints/`:

```bash
huggingface-cli download gpt2 --repo-type model --local-dir ./checkpoints/gpt2-base
huggingface-cli download gpt2-medium --repo-type model --local-dir ./checkpoints/gpt2-medium
huggingface-cli download gpt2-large --repo-type model --local-dir ./checkpoints/gpt2-large
huggingface-cli download gpt2-xl --repo-type model --local-dir ./checkpoints/gpt2-xlarge
```

### Usage

Main AMiD hyperparameters: `--amid-div-name` (divergence name), `--amid-div-order` (order of distributions in divergence), `--amid-alpha` (α), `--amid-lam` (λ). Detailed values are in the paper; see the bash scripts in `AMiD/scripts/` for full arguments. Final checkpoints are selected by **ROUGE-L**.

```bash
# Fine-tune teacher
bash ./scripts/gpt2/sft/sft_xlarge.sh ${/PATH/TO/AMiD} ${MASTER_PORT} ${GPU_NUM}
# or: huggingface-cli download MiniLLM/teacher-gpt2-1.5B --repo-type model --local-dir ./results/gpt2/train/sft/gpt2-xlarge

# Student initialization (selected by validation loss)
bash ./scripts/gpt2/init/init_base.sh ${/PATH/TO/AMiD} ${MASTER_PORT} ${GPU_NUM}
# or: huggingface-cli download MiniLLM/init-gpt2-120M --repo-type model --local-dir ./results/gpt2/train/init/gpt2-base

# Train AMiD
bash ./scripts/gpt2/amid/train_0.1B_1.5B.sh ${/PATH/TO/AMiD} ${MASTER_PORT} ${GPU_NUM} ${amid-div-name} ${amid-div-order} ${amid-alpha} ${amid-lam} ${batch-size} ${lr}

# Evaluation
bash ./scripts/gpt2/eval/run_eval.sh ${/PATH/TO/AMiD_CKPT} ${MASTER_PORT}
```

### Citation

```bib
@inproceedings{
shin2026amid,
title={{AM}iD: Knowledge Distillation for {LLM}s with $\alpha$-mixture Assistant Distribution},
author={Donghyeok Shin and Yeongmin Kim and Suhyeon Jo and Byeonghu Na and Il-chul Moon},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=7WPJ0EgPdW}
}
```

---

## Acknowledgements

Portions of this repository build upon:
- *MiniLLM: On-Policy Distillation of Large Language Models* — [Paper](https://arxiv.org/abs/2306.08543) · [Code](https://github.com/microsoft/LMOps/blob/main/minillm/README.md)
- *DistiLLM: Towards Streamlined Distillation for Large Language Models* — [Paper](https://arxiv.org/abs/2402.03898) · [Code](https://github.com/jongwooko/distillm)
- *ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence* — [Paper](https://arxiv.org/abs/2505.04560) · [Code](https://github.com/ghwang-s/abkd)
- *DSKD v2: Dual-Space Knowledge Distillation for Large Language Models* — [Paper](https://arxiv.org/abs/2504.11426)

# MMLatentDraft

**Native Latent Draft for Multimodal Reasoning**

> Native Latent Draft (NLD) — multi-step reasoning *inside the native hidden
> space* of a Vision-Language Model. No external draft module, no auxiliary
> decoder; the same Transformer that does autoregressive generation also does
> latent thinking, by re-feeding its own last-layer hidden state through itself
> as an RNN cell. Training and inference share **exactly** the same forward
> path.

<p align="center">
  <a href="#-method"><b>Method</b></a> ·
  <a href="#-repository-layout"><b>Repository Layout</b></a> ·
  <a href="#-installation"><b>Installation</b></a> ·
  <a href="#-data-synthesis"><b>Data Synthesis</b></a> ·
  <a href="#-training"><b>Training</b></a> ·
  <a href="#-evaluation"><b>Evaluation</b></a> ·
  <a href="#-analysis"><b>Analysis</b></a>
</p>

---

## ✨ Method

NLD inserts *latent thinking steps* at stage boundaries between the question
and the final answer. Each latent step uses the last-layer hidden state of the
previous token as a "query probe", runs it through **all** Transformer layers
(with KV cache for prefix), and feeds the new hidden state back as the next
input embedding — turning the Transformer into an RNN cell.

```
 Input  →  [Q segment]  →  <|latent|>  →  h_1 → h_2 → … → h_N  →  <|/latent|>  →  [A segment]  →  Answer
                                  ↑              ↻              ↓
                            step_embedding   text_model        thought
                                            (all layers,        out
                                             with KV cache)
```

**Equivalence to COCONUT.** Both treat the LM as an RNN cell that recurses on
its own hidden state without decoding text in between. NLD differs only in
that the recurrence is performed by the *native* text-model stack (no
external projector), making the method a pure inductive bias on top of the
base VLM.

**Key designs.**

- **Single-token recurrence.** Each step only feeds the *last* token's hidden
  state as the query; all history lives in the KV cache.
- **Dual reasoning streams.** Implicit reasoning (hidden space) and explicit
  reasoning (natural-language CoT segments) form a single unified sequence.
- **Adaptive exit.** A latent step can exit either by predicting the
  `<|/latent|>` exit token *or* by a saturation signal on the hidden state
  (double safeguard).
- **Train / inference parity.** No teacher-forcing tricks: the same
  segment-wise forward (with KV-cache rebuild at boundaries) is used in both
  modes.

### Training Objectives

| Loss                       | Role                  | Description                                                                                                          |
| :------------------------- | :-------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **CE Loss**                | Main loss             | Standard next-token prediction. Thought positions are masked out (`labels=-100`).                                    |
| **SW-SRS Loss**            | Latent supervision    | Stage-Windowed Self-Refined Supervision: each thought step is supervised against the *next* stage's key-token prior. |
| **Exit-Token Loss**        | Exit signal           | Last latent step's hidden state, passed through a *frozen* `lm_head`, must predict `<|/latent|>`.                    |
| **Visual-Probe Loss**      | Modality preservation | Ensures thought hidden states keep enough visual information.                                                        |
| **Anti-Collapse (opt-in)** | Geometry guard        | Optional margin term that prevents latent steps from collapsing onto the exit-token direction.                       |

> `lm_head` is **frozen on the SW-SRS / Exit-Token paths** (`detach`), so only
> hidden states receive that gradient. The CE path keeps `lm_head` trainable
> so the new special tokens can learn proper output rows.

---

## 📁 Repository Layout

```
MMLatentDraft/
├── rld/                              # Core NLD module
│   ├── model_v2.py                   #   NLDModelForVL — main model, segment-wise forward
│   ├── latent_thinker.py             #   NativeLatentThinker — recurrent latent step
│   ├── data.py                       #   Dataset + collator (multi-image, mixed modalities)
│   ├── trainer_nld.py                #   NLDTrainer (dual lr, FSDP-friendly monitoring)
│   ├── inference_utils.py            #   Greedy / beam / visualisation helpers
│   ├── visual_anchor.py              #   Visual probe / anchor utilities
│   └── __init__.py
│
├── configs/
│   ├── fsdp_config.json                              # FSDP full_shard policy
│   ├── rld_stage2_swsrs_v6b1b2b3_ckpt200.yaml        # Stage-2 SW-SRS on v6 b1+b2+b3 (19 195 samples)
│   ├── rld_stage2_erqa_vsibench_ckpt1200.yaml        # Continual training on ERQA + VSI-Bench
│   ├── rld_stage2_erqa_vsibench_ckpt49_balanced.yaml
│   └── rld_stage2_erqa_latent_cot_v2_ckpt49_balanced_v2.yaml
│
├── scripts/
│   ├── train_nld.py                  # Training entry (torchrun + FSDP)
│   ├── inference.py                  # Single-sample / batch inference
│   ├── prepare_benchmark.py          # Generic benchmark prep
│   ├── prepare_vsibench.py           # VSI-Bench prep (multi-image)
│   ├── download_benchmarks.py        # Download MMStar / RealWorldQA / BLINK / MUIR / MMBench / …
│   ├── generate_erqa_latent_cot.py   # ERQA → latent-CoT data
│   ├── generate_vsibench_latent_cot.py
│   ├── merge_erqa_vsibench.py        # Mix ERQA + VSI-Bench training set
│   ├── analyze_latent_distribution.py
│   ├── plot_train_curves.py
│   │
│   ├── v3/ichat_client.py            # Generic OpenAI-compatible LLM client (used by v5/v6)
│   ├── v5/                           # Latent-CoT data synthesis  v5 (single-stage prompt)
│   │   ├── generate_v5.py
│   │   ├── action_space.py
│   │   ├── schema_v5.py
│   │   └── prompts/
│   └── v6/                           # Latent-CoT data synthesis  v6 (multi-source seeds + visual scanpath)
│       ├── generate_v6.py
│       ├── build_multi_source_seeds.py
│       ├── regen_stage_tokens_visual_scanpath.py
│       ├── schema_v6.py
│       └── prompts/
│
├── eval_erqa.py                      # ERQA evaluator (LatentDraft model)
├── prepare_erqa.py                   # ERQA preprocessing
├── prepare_vsibench.sh
├── analyze_efficiency.py             # FLOPs / latency comparison: NLD vs CoT-Thinking
├── analyze_entropy_trigger.py        # Entropy at latent-trigger positions
├── plot_efficiency_acl.py            # ACL-style efficiency figures
│
├── modality_analysis/                # Hidden-state geometry analysis (Stage-1)
│   └── scripts/                      #   anisotropy, native vs latent pairing, task specificity
│
├── modality_manifold_analysis/       # Modality-manifold analysis (paper main figures)
│   └── scripts/                      #   CKA / t-SNE / cone evolution / orthogonal decomposition
│
├── tools/viz_training_progress.py    # Training-curve dashboard
├── paper_tables_figures/             # LaTeX tables / generation scripts (figures git-ignored)
│
├── start_training_stage2_v6b1b2b3_ckpt200.sh        # Launchers (8-GPU torchrun + FSDP)
├── start_training_stage2_erqa_vsibench.sh
├── start_inference.sh
├── start_eval_erqa.sh
├── start_generate_erqa_latent.sh
├── start_generate_vsibench_latent.sh
├── start_analyze_entropy_trigger.sh
├── run_efficiency_analysis.sh
├── run_entropy_analysis.sh
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Heavy artefacts (`data/`, `outputs/`, model weights, paper figures, training
> logs) are excluded from the repository via `.gitignore`. Only the code path
> is tracked.

---

## 🛠 Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA 12.x
- 8 × GPU with ≥ 40 GB VRAM (training); 1 × GPU works for inference / analysis
- Flash-Attention 2

### Setup

```bash
git clone https://github.com/Icarus1216/MMLatentDraft.git
cd MMLatentDraft
pip install -r requirements.txt
```

### Base model

NLD is built on top of **Qwen3-VL-8B-Instruct**. Download the weights once and
export the path as `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
```

The provided `start_*.sh` scripts and `configs/*.yaml` use the placeholder
`<PATH_TO_QWEN3_VL_8B_INSTRUCT>`. Either set the env var above (the launchers
honour it) or replace the placeholder in your local config.

---

## 🧪 Data Synthesis

NLD is trained on **latent-CoT** data: each example carries a natural-language
reasoning trace segmented by `<|pause|>` markers, plus per-stage *key tokens*
that supervise the latent steps.

### Synthesis pipelines

| Pipeline   | Entry point                                | Notes                                                                            |
| :--------- | :----------------------------------------- | :------------------------------------------------------------------------------- |
| **v5**     | `scripts/v5/generate_v5.py`                | Single-stage prompt; action-space defined in `scripts/v5/action_space.py`.       |
| **v6**     | `scripts/v6/generate_v6.py`                | Multi-source seeds + visual-scanpath stage tokens. Best quality for spatial QA.  |
| ERQA       | `scripts/generate_erqa_latent_cot.py`      | ERQA-specific latent-CoT generation.                                             |
| VSI-Bench  | `scripts/generate_vsibench_latent_cot.py`  | VSI-Bench-specific (multi-frame) latent-CoT generation.                          |

The synthesisers share an OpenAI-compatible client (`scripts/v3/ichat_client.py`),
so any model exposing the OpenAI Chat-Completion API (incl. self-hosted
endpoints) can be used. Configure via env vars:

```bash
export OPENAI_API_KEY=<your_key>
export OPENAI_BASE_URL=<your_endpoint>
```

### Quick start

```bash
# v6 — multi-source seeds (recommended)
python scripts/v6/build_multi_source_seeds.py
python scripts/v6/generate_v6.py --num_workers 32

# ERQA / VSI-Bench latent-CoT data
bash start_generate_erqa_latent.sh
bash start_generate_vsibench_latent.sh
```

### Sample format

```json
{
  "image_path": "data/erqa/images/xxx.png",
  "question": "Which object is closest to the cup?",
  "answer": "The book.",
  "reasoning_for_training": "Stage 1: locate the cup … <|pause|> Stage 2: scan nearby objects … <|pause|> Stage 3: compare distances …",
  "num_stages": 3,
  "latent_key_tokens": [
    {"stage": "Stage 1", "tokens": ["cup", "table", "left"]},
    {"stage": "Stage 2", "tokens": ["book", "lamp", "phone"]}
  ],
  "task_type": "spatial_relation"
}
```

At training time, every `<|pause|>` is rewritten to `<|latent|> <|/latent|>`
and the model learns to emit those boundary tokens; all reasoning *between*
the boundary tokens happens in hidden space.

---

## 🚀 Training

Training is FSDP-based (`full_shard auto_wrap`) and uses HuggingFace Trainer
with a custom `NLDTrainer` (dual learning rate for VLM vs Thinker, plus extra
monitoring). All launchers default to **8 GPUs** with `torchrun`.

### Stage-2 SW-SRS on v6 b1+b2+b3 (19 195 samples)

Trains from a Stage-1 checkpoint with SW-SRS loss to refine hidden geometry.

```bash
bash start_training_stage2_v6b1b2b3_ckpt200.sh
# overrides
bash start_training_stage2_v6b1b2b3_ckpt200.sh --gpus 4
bash start_training_stage2_v6b1b2b3_ckpt200.sh --config /path/to/your.yaml
```

### Continual training on ERQA + VSI-Bench

```bash
bash start_training_stage2_erqa_vsibench.sh
```

### Common environment variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
```

### Default hyper-parameters (Stage-2)

| Parameter             | Value                          | Notes                                |
| :-------------------- | :----------------------------- | :----------------------------------- |
| Base model            | Qwen3-VL-8B-Instruct           | 8.3 B params, 36 layers, dim 4 096   |
| Effective batch size  | 32                             | 1 × 4 grad-accum × 8 GPUs            |
| LR (VLM)              | 2e-5                           | cosine schedule                      |
| LR (Thinker)          | 1e-4                           | trained from scratch                 |
| Max latent steps      | 8 (training) / 5 (inference)   | covers > 99 % of stage distribution   |
| Sharding              | FSDP `full_shard auto_wrap`    | DeepSpeed disabled to avoid KV-cache  |
| Precision             | bf16 + Flash-Attention 2       |                                       |

---

## 📊 Evaluation

### ERQA

```bash
bash start_eval_erqa.sh \
    --checkpoint outputs/<your_run>/checkpoint-XXXX \
    --model_path "$MODEL_PATH"
```

Outputs raw predictions, per-category accuracy, and an `erqa_eval_results.json`
under `outputs/<your_run>_eval/`.

### VSI-Bench / multi-bench

```bash
bash prepare_vsibench.sh
python scripts/prepare_benchmark.py --bench MMStar RealWorldQA BLINK ...
```

Then plug the resulting json paths into your eval driver of choice.

### Inference

```bash
bash start_inference.sh \
    --checkpoint outputs/<your_run>/checkpoint-XXXX \
    --image path/to/image.jpg \
    --question "Your question here?"
```

---

## 🔬 Analysis

The repository ships several stand-alone analysis tools that produced the
paper's tables and figures:

| Script                                   | What it measures                                                              |
| :--------------------------------------- | :---------------------------------------------------------------------------- |
| `analyze_efficiency.py`                  | FLOPs and wall-clock latency vs Qwen3-VL-Thinking (CoT baseline) on ERQA.     |
| `analyze_entropy_trigger.py`             | Per-token entropy at the positions where the model triggers `<|latent|>`.    |
| `scripts/analyze_latent_distribution.py` | Distribution of #latent steps and trigger positions in the training set.     |
| `modality_analysis/scripts/*`            | Stage-1 hidden-state geometry: anisotropy, native ↔ latent pairing.           |
| `modality_manifold_analysis/scripts/*`   | Modality manifold: CKA trajectories, t-SNE, cone evolution, orthogonal decomp.|
| `tools/viz_training_progress.py`         | Live dashboard of loss / monitor metrics during training.                     |

```bash
bash run_efficiency_analysis.sh \
    --checkpoint outputs/<your_run>/checkpoint-XXXX

bash run_entropy_analysis.sh \
    --checkpoint outputs/<your_run>/checkpoint-XXXX
```

---

## 🔖 Special Tokens

| Token         | Purpose                                                |
| :------------ | :----------------------------------------------------- |
| `<|latent|>`  | Enter hidden-space reasoning mode.                     |
| `<|/latent|>` | Exit hidden-space reasoning mode.                      |
| `<|pause|>`   | Stage boundary in raw data; rewritten at preprocessing.|

---

## 📜 License

MIT.

## 📚 Citation

If you find this work useful, please cite the upcoming paper. Citation entry
will be added once the preprint is public.

```bibtex
@misc{mmlatentdraft,
  title  = {MMLatentDraft: Native Latent Draft for Multimodal Reasoning},
  author = {Anonymous},
  year   = {2026},
  note   = {Code: https://github.com/Icarus1216/MMLatentDraft}
}
```

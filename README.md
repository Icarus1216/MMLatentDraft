# MMLatentDraft

**Native Latent Draft for Multimodal Reasoning**

> Native Latent Draft (NLD) — multi-step reasoning *inside the native hidden
> space* of a Vision-Language Model. No external draft module, no auxiliary
> decoder; the same Transformer that does autoregressive generation also does
> latent thinking, by re-feeding its own last-layer hidden state through itself
> as an RNN cell.

<p align="center">
  <a href="#-method"><b>Method</b></a> ·
  <a href="#-repository-layout"><b>Repository Layout</b></a> ·
  <a href="#-installation"><b>Installation</b></a> ·
  <a href="#-inference"><b>Inference</b></a> ·
  <a href="#-special-tokens"><b>Special Tokens</b></a>
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

---

## 📁 Repository Layout

```
MMLatentDraft/
├── rld/                              # Core NLD module
│   ├── model_v2.py                   #   NLDModelForVL — main model, segment-wise forward
│   ├── latent_thinker.py             #   NativeLatentThinker — recurrent latent step
│   ├── data.py                       #   Dataset + collator (multi-image, mixed modalities)
│   ├── trainer_nld.py                #   Custom Trainer
│   ├── inference_utils.py            #   Greedy / beam / visualisation helpers
│   ├── visual_anchor.py              #   Transition-modality (slerp) anchor utilities
│   └── __init__.py
│
├── configs/                          # Run configs (yaml)
├── scripts/                          # Standalone scripts
│   ├── train_nld.py                  #   Training entry (torchrun + FSDP)
│   ├── run_train_nld.sh              #   Generic training launcher
│   └── inference.py                  #   Single-sample / batch inference
│
├── analyze_efficiency.py             # FLOPs / latency comparison vs CoT baseline
├── analyze_entropy_trigger.py        # Entropy at latent-trigger positions
├── plot_efficiency_acl.py            # Efficiency figures
│
├── modality_analysis/                # Hidden-state geometry analysis
├── modality_manifold_analysis/       # Modality-manifold analysis (CKA / t-SNE / cone evolution)
├── paper_tables_figures/             # LaTeX tables / generation scripts
│
├── start_training_stage2_v6b1b2b3_ckpt200.sh  # Training launcher (8-GPU FSDP)
├── start_inference.sh                         # Inference launcher
├── run_efficiency_analysis.sh
├── run_entropy_analysis.sh
│
├── requirements.txt
├── .gitignore
└── README.md
```

> Heavy artefacts (`data/`, `outputs/`, model weights, paper figures, logs)
> are excluded from the repository via `.gitignore`. Only the code path is
> tracked.

---

## 🛠 Installation

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA 12.x
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

## 🚀 Inference

```bash
bash start_inference.sh \
    --checkpoint /path/to/checkpoint \
    --image path/to/image.jpg \
    --question "Your question here?"
```

For analysis utilities (efficiency / entropy / hidden-state geometry),
see the analysis launchers (`run_efficiency_analysis.sh`,
`run_entropy_analysis.sh`) and the `modality_analysis/` /
`modality_manifold_analysis/` script directories.

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

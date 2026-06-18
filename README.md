# Nemotron Loadability-Method — Best Fine-tuning Method (Open Contribution)

**A loadability contract for adapter-only scoring, plus an own-data fine-tune that generalizes out-of-sample, on a 30B Mamba-Transformer MoE.**

Independent researcher / WayneIA — small-business owner-operator.
NVIDIA Nemotron Model Reasoning Challenge · Category: *Best Fine-tuning Method* (Open Contribution) · 2026.

---

## Results (authoritative — public / private leaderboard)

| Adapter | Public | Private | Note |
|---|---|---|---|
| **Own-data method (this contribution)** | **0.812** | **0.828** | **generalizes UP out-of-sample** (private > public) |
| Community-packaged commodity (eligibility only) | 0.860 | 0.856 | credited third-party adapter; overfits (private < public) |

The own-data method's **private score (0.828) exceeds its public score (0.812)** — it generalizes to the held-out set rather than overfitting the public leaderboard. The commodity package shows the opposite (0.856 < 0.860). **A Best-Method contribution is method quality, not a leaderboard number — and out-of-sample generalization is the cleanest evidence of method quality.**

## The contribution — two reproducible findings

### 1. The loadability contract
Shipping an `lm_head` adaptation the obvious way — `modules_to_save=["lm_head"]` — serializes a multi-gigabyte full output-projection module that the competition's adapter-only vLLM PEFT loader silently rejects (the adapter scores nothing). Routing `lm_head` instead as an ordinary **rank-32 LoRA delta through `target_modules`** keeps it loadable and scoreable.

| Adapter construction | adapter-only vLLM PEFT loader | Score |
|---|---|---|
| `lm_head` via `modules_to_save` | **REJECTED** (silent) | none recorded |
| `lm_head` via `target_modules` (rank-32 delta) + MoE `target_parameters` | **loads clean** | **0.812 public / 0.828 private** |

Generalizes: any fine-tune adapting `lm_head` under a low-rank-only loader should route it through `target_modules`, never `modules_to_save`. (`adapter_config.json` in this repo is that loadable config.)

### 2. Robust completion-only masking on a template without generation blocks
TRL's `assistant_only_loss=True` **silently no-ops** on Nemotron-3-Nano (its chat template has no `{% generation %}` blocks), and TRL's `DataCollatorForCompletionOnlyLM` was removed in the pinned `trl==0.24.0` — so the "standard" completion-only paths train on the **full sequence** (mask fraction 0.000) without erroring. The fix is an **explicit, version-agnostic response-template collator** (mask = `[0]*prompt + [1]*completion`), with the response template determined empirically from the real tokenizer and **CPU-verified** (masked-prompt fraction 0.017–0.16, 0/500 marker-misses) *before* any GPU run. Train the completion, not the prompt — provably.

## Eligibility (stated plainly)

Open-Contribution eligibility (top-10% / Competition Bronze, 318/4,182) was established by packaging a **credited public third-party adapter** (0.860 public / 0.856 private). **The fine-tuning method contributed here is our own-data pipeline (0.812 public / 0.828 private).** We make **no claim** the packaged score reflects our method — the contribution is the loadability contract, the robust masking fix, and an own-data adapter that generalizes. The packaged adapter is upstream, credited, used under the competition's third-party safe-harbor.

## Method

- **Base:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (30B total, ~3B active/token; Mamba-Transformer MoE), BF16, via Unsloth `FastLanguageModel`, `max_seq_length=8192`.
- **Adapter:** rank-32 LoRA, `lora_alpha=32`, `lora_dropout=0.0`, `bias=none`. `target_modules` (9): `q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj, lm_head`. MoE `target_parameters` (2): `mlp.experts.gate_up_proj, mlp.experts.down_proj`. **No `modules_to_save`** — the loadability contract.
- **Data:** matched chain-of-thought corpus reduced after dropping empty/degenerate CoT, with tolerance-aware curation; one canonical terminal `\boxed{answer}` after `</think>`.
- **Training:** TRL `SFTTrainer`, subclassed for batch ordering stratified by problem `type`, with the robust completion-only collator above. Greedy scoring (`temperature=0.0`) ⇒ deterministic; `seed=42`.

## Reproduce

```bash
# Kaggle high-memory accelerator (RTX PRO 6000 Blackwell, 96 GB), dataset + Nemotron-3-Nano-30B base mounted:
TRAIN_ON_KAGGLE=1 python train.py        # trains the rank-32 loadable adapter
python package_adapter.py                # builds + validates the loadability guards (asserts no modules_to_save, lm_head a target)
```

## Files
`train.py` · `package_adapter.py` · `adapter_config.json` (the loadable config) · `WEIGHTS.md` (4.24 GB adapter, CC BY 4.0, hosted off-git) · `requirements.txt` · `LICENSE` (Apache-2.0).

## Licensing
Code Apache-2.0 · adapter weights CC BY 4.0 (per rules, hosted separately) · upstream `NVIDIA-Nemotron-3-Nano-30B` + the CoT teacher (commercial API) are safe-harbor exempt; all original training/curation code here is open-sourced without commercial restriction.

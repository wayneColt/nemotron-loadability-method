# Nemotron Loadability-Method — Best Fine-tuning Method (Open Contribution)

**A loadability contract for adapter-only scoring: routing `lm_head` through LoRA `target_modules` on a 30B Mamba-Transformer MoE.**

Independent researcher / WayneIA — small-business owner-operator.
NVIDIA Nemotron Model Reasoning Challenge · Category: *Best Fine-tuning Method* (Open Contribution) · 2026.

---

## The contribution (one line)

Shipping an `lm_head` adaptation the obvious way — `modules_to_save=["lm_head"]` — serializes a multi-gigabyte full output-projection module that the competition's adapter-only vLLM PEFT loader silently rejects (the adapter scores nothing). Routing `lm_head` instead as an ordinary **rank-32 LoRA delta through `target_modules`** keeps the format gain while keeping the adapter loadable and scoreable. That *loadability contract*, plus the from-scratch reproducible pipeline that produced it, is the deliverable.

| Adapter construction | adapter-only vLLM PEFT loader | Leaderboard |
|---|---|---|
| `lm_head` via `modules_to_save` | **REJECTED** (silent) | none recorded |
| `lm_head` via `target_modules` (rank-32 delta) | **loads clean** | **0.84** |

It generalizes: any fine-tune that adapts `lm_head` under a low-rank-only loader should route it through `target_modules`, not `modules_to_save`.

## Eligibility (stated plainly, for the record)

Open-Contribution eligibility was established by a final-leaderboard finish of **318th of 4,182 teams (top 7.6%; Competition Bronze, 2026-06-15)**. That top-10% finish was achieved by packaging a **credited public third-party adapter** (which scored **0.86**). **The fine-tuning method contributed in this repository is our own from-scratch pipeline, which scored 0.84.** We make **no claim** that the 0.86 reflects this method — the contribution here is the loadability contract and the reproducible owned pipeline, not a leaderboard delta. The packaged adapter is upstream, credited, and used under the competition's third-party safe-harbor.

## Method

- **Base:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (30B total, ~3B active/token; Mamba-Transformer MoE), BF16, no quantization, via Unsloth `FastLanguageModel`, `max_seq_length=8192`.
- **Adapter:** rank-32 LoRA, `lora_alpha=32`, `lora_dropout=0.0`, `bias=none`. `target_modules` (9): `q_proj, k_proj, v_proj, o_proj, in_proj, out_proj, up_proj, down_proj, lm_head`. MoE `target_parameters` (2): `mlp.experts.gate_up_proj, mlp.experts.down_proj`. **No `modules_to_save`** — that absence is the loadability contract.
- **Data:** matched chain-of-thought corpus `dgxchen/nemotron-cot-tong`, reduced to **7,830 rows** after dropping empty/degenerate CoT. Box hygiene: exactly one canonical terminal `\boxed{answer}` after `</think>`. Teacher-distilled (a larger open model served via a low-cost API; tool use is forbidden at inference, permitted at training-data generation).
- **Training:** TRL `SFTTrainer`, lightly subclassed for batch ordering **stratified by problem `type`**. Greedy scoring (`temperature=0.0`) ⇒ deterministic; reproducibility reduces to reproducing the adapter. `seed=42` across LoRA init, sampler, trainer.

## Reproduce

```bash
# On a Kaggle high-memory accelerator (RTX PRO 6000 Blackwell, 96 GB), with the
# dgxchen/nemotron-cot-tong dataset + the Nemotron-3-Nano-30B base mounted:
TRAIN_ON_KAGGLE=1 python train.py        # trains the rank-32 adapter
python package_adapter.py                # builds submission_adapter/ + validates loadability guards
```

`package_adapter.py` enforces the contract before declaring "done": it asserts `adapter_config.json` has `base_model_name_or_path` set, `inference_mode=true`, no `modules_to_save`, and `max_lora_rank<=32`.

## Files

| File | Purpose |
|---|---|
| `train.py` | data prep + box hygiene + stratified-by-type SFT + rank-32 LoRA config |
| `package_adapter.py` | packaging + `adapter_config.json` validation + loadability guards |
| `adapter_config.json` | the trained adapter's config (the loadability contract, in situ) |
| `WEIGHTS.md` | where the 4.24 GB adapter weights are hosted (CC BY 4.0) — not in git |
| `LICENSE` | Apache-2.0 (code) |

## Environment

`unsloth`, `trl`, `peft>=0.18.1`, `transformers>=4.46`, `vllm>=0.17.0` (scoring), BF16, `attn_implementation="eager"`. See `requirements.txt`.

## Licensing

- **Code:** Apache-2.0 (this repo).
- **Adapter weights:** CC BY 4.0 (per challenge rules) — hosted separately (see `WEIGHTS.md`); GitHub is not a weight host.
- **Upstream (safe-harbor):** `NVIDIA-Nemotron-3-Nano-30B-A3B` base (exempt); the CoT teacher served via a commercial API (exempt). All original LoRA training code and data-curation logic here are open-sourced without commercial restriction.

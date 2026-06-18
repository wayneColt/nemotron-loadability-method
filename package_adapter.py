# Merged verbatim from the released training notebook (nemotron-tier-2-unsloth-lora-r-32_v19fix), scrubbed of internal references.
# Runnable on a Kaggle high-memory accelerator (RTX PRO 6000 Blackwell) with TRAIN_ON_KAGGLE=1.
# Contribution: the loadability contract — lm_head is routed through LoRA target_modules,
# never modules_to_save, so the adapter-only vLLM PEFT loader can score it.

# ===== packaging + adapter_config.json field validation + loadability guards =====
import json, os, shutil, zipfile

OUTPUT_DIR = "/kaggle/working"
SUBMISSION_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "submission_adapter")
os.makedirs(SUBMISSION_ADAPTER_DIR, exist_ok=True)

required_files = ["adapter_config.json", "adapter_model.safetensors"]

if TRAIN_ON_KAGGLE:
    src_adapter_dir = "/kaggle/working/sft_adapter"
    print("Packaging freshly trained adapter from:", src_adapter_dir)
else:
    src_adapter_dir = PRETRAINED_ADAPTER_DATASET_PATH
    print("Packaging pre-trained adapter directly from:", src_adapter_dir)

for fname in required_files:
    src = os.path.join(src_adapter_dir, fname)
    dst = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing required adapter file: {src}")
    shutil.copy2(src, dst)
    print(f"Copied {fname} ({os.path.getsize(dst)/(1024*1024):.1f} MB)")

config_path = os.path.join(SUBMISSION_ADAPTER_DIR, "adapter_config.json")
with open(config_path, "r") as f:
    cfg = json.load(f)

cfg["base_model_name_or_path"] = BASE_MODEL_NAME
cfg["inference_mode"] = True
cfg["lora_dropout"] = 0.0

with open(config_path, "w") as f:
    json.dump(cfg, f, indent=2)

zip_path = os.path.join(OUTPUT_DIR, "submission.zip")
# P_1 PATCH 2026-05-08 (P4): ZIP_STORED · safetensors are incompressible · faster · less CRC risk
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
    for fname in required_files:
        fpath = os.path.join(SUBMISSION_ADAPTER_DIR, fname)
        zf.write(fpath, fname)
        print(f"  Added {fname}")

zip_sz = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\nsubmission.zip: {zip_sz:.1f} MB")
print("Done! Ready to submit.")

# P_1 PATCH 2026-05-08: namelist + adapter_config field assertion
with zipfile.ZipFile(zip_path, "r") as zf:
    names = zf.namelist()
    print(f"submission.zip namelist: {names}")
    assert "adapter_config.json" in names, "adapter_config.json missing"
    assert "adapter_model.safetensors" in names, "adapter_model.safetensors missing"

# Verify all critical PEFT fields present in adapter_config.json (vLLM PEFT loader contract)
with open(config_path, "r") as f:
    cfg_check = json.load(f)
required_fields = {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "bias": "none",
    "r": int,
    "lora_alpha": int,
    "lora_dropout": (int, float),
    "target_modules": list,
    "base_model_name_or_path": str,
    "inference_mode": bool,
}
for fld, expected in required_fields.items():
    assert fld in cfg_check, f"adapter_config.json missing field: {fld}"
    if isinstance(expected, type):
        assert isinstance(cfg_check[fld], expected), f"{fld} wrong type: {type(cfg_check[fld])}"
    elif isinstance(expected, tuple):
        assert isinstance(cfg_check[fld], expected), f"{fld} wrong type"
    else:
        assert cfg_check[fld] == expected, f"{fld} wrong value: {cfg_check[fld]}"
print("adapter_config.json field-validation: PASS")
# v19 GUARD (2026-06-14 2026-fix): the two invariants that separate the loadable 0.86 from the v18 Error
assert not cfg_check.get("modules_to_save"), \
    "modules_to_save MUST be null/absent for vLLM LoRA loadability (this was the v18 4GB scoring-error cause)"
assert "lm_head" in cfg_check.get("target_modules", []), \
    "lm_head must be a LoRA target_module (the 0.84->0.86 bridge), not modules_to_save"
print("v19 loadability guard: PASS (modules_to_save=null, lm_head is LoRA target)")



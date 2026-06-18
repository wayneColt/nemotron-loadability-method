# Merged verbatim from the released training notebook (nemotron-tier-2-unsloth-lora-r-32_v19fix), scrubbed of internal references.
# Runnable on a Kaggle high-memory accelerator (RTX PRO 6000 Blackwell) with TRAIN_ON_KAGGLE=1.
# Contribution: the loadability contract — lm_head is routed through LoRA target_modules,
# never modules_to_save, so the adapter-only vLLM PEFT loader can score it.

# ===== data prep / box-hygiene / stratified-by-type SFT training =====
if TRAIN_ON_KAGGLE:
    import pandas as pd
    import random
    import gc, time
    from datasets import Dataset as HFDataset
    from trl import SFTTrainer, SFTConfig

    SEED = 42
    PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

    DATASET_PATH = "/kaggle/input/datasets/dgxchen/nemotron-cot-tong/problem_ids_matched.csv"
    df = pd.read_csv(DATASET_PATH)
    print(f"Full dataset: {len(df)} rows")
    train_df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Full dataset: {len(df)} rows")

    import re
    import math
    from collections import defaultdict
    from torch.utils.data import DataLoader, Sampler
    records = []
    record_types = []
    for _, row in train_df.iterrows():
        prompt = str(row["prompt"])
        answer = str(row["answer"])
        cot = str(row["generated_cot"])
        if not cot or cot == "nan" or len(cot.strip()) < 5:
            continue
        # FIX 2026-05-30 KaggleNemo: keep CoT body to its OWN final </think> (no doubled tag, no gutted prose), one canonical box
        m = re.search(r'(.*</think>)', cot, re.S)
        body = m.group(1) if m else (cot.rstrip() + "\n</think>")
        # BOX HYGIENE (v19 fix 2026-06-14 2026-fix): the source CoT carries a \\boxed{} INSIDE
        # <think> (confirmed 7830/7830). Unwrap it to plain text so the TARGET has EXACTLY ONE
        # canonical box AFTER </think>. Box-happy training risks early/multi-box at eval, and
        # the metric prioritizes the final box -- one clean terminal box is what we want taught.
        body = re.sub(r'\\boxed\{([^{}]*)\}', r'\1', body)
        user_content = prompt + PROMPT_SUFFIX
        assistant_content = body.rstrip() + f"\n\\boxed{{{answer}}}"
        records.append({"messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]})
        record_types.append(str(row["type"]))
    dataset = HFDataset.from_list(records)
    print(f"SFT records: {len(records)}")

    def formatting_prompts_func(example):
        messages = example["messages"]
        if messages and isinstance(messages[0], dict):
            conversations = [messages]
        else:
            conversations = messages

        texts = []
        for conversation in conversations:
            try:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            texts.append(text)
        return texts

    training_args = SFTConfig(
        output_dir="/kaggle/working/sft_output",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_length=8192,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        weight_decay=0.0,
        max_grad_norm=1.0,  # FIX 2026-05-30 KaggleNemo: was 1e9 (no clipping)
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        packing=False,
    )

    def build_stratified_index_order(labels, batch_size, seed):
        """Approximate nemotron-master's stratified batching over effective batches."""
        by_label = defaultdict(list)
        for idx, label in enumerate(labels):
            by_label[label].append(idx)
        rng = random.Random(seed)
        for idx_list in by_label.values():
            rng.shuffle(idx_list)
        n_batches = max(1, math.ceil(len(labels) / batch_size))
        batches = [[] for _ in range(n_batches)]
        batch_order = list(range(n_batches))
        rng.shuffle(batch_order)
        assigned = 0
        for label in sorted(by_label.keys()):
            for idx in by_label[label]:
                batches[batch_order[assigned % n_batches]].append(idx)
                assigned += 1
        order = [idx for batch in batches for idx in batch]
        if len(order) != len(labels):
            raise ValueError("Stratified order size mismatch")
        return order

    class PrecomputedOrderSampler(Sampler):
        def __init__(self, order):
            self.order = list(order)

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.order)

    class StratifiedSFTTrainer(SFTTrainer):
        def __init__(self, *args, stratified_order=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.stratified_order = stratified_order

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset.")
            if self.stratified_order is None:
                return super().get_train_dataloader()
            if len(self.stratified_order) != len(self.train_dataset):
                raise ValueError("Stratified order length does not match train dataset")
            dataloader_kwargs = {
                "batch_size": self.args.per_device_train_batch_size,
                "sampler": PrecomputedOrderSampler(self.stratified_order),
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "drop_last": self.args.dataloader_drop_last,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
            return DataLoader(self.train_dataset, **dataloader_kwargs)

    effective_batch_size = max(1,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    stratified_order = build_stratified_index_order(record_types, effective_batch_size, SEED)
    print(f"Approx stratified effective batch size: {effective_batch_size}")
    print("Stratified batching by type:", dict(sorted(pd.Series(record_types).value_counts().to_dict().items())))

    trainer = StratifiedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        stratified_order=stratified_order,
    )

    print("Starting SFT training...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")

    ADAPTER_DIR = "/kaggle/working/sft_adapter"
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")


# ===== rank-32 LoRA config (Unsloth get_peft_model; lm_head as LoRA target) =====
if TRAIN_ON_KAGGLE:
    from unsloth import FastLanguageModel

    LORA_RANK = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.0

    # ============================================================================
    # v19 FIX (2026-06-14 2026-fix): v18 errored at scoring because it used
    #   modules_to_save=["lm_head"]  -> PEFT saved the FULL lm_head weight (~4GB module);
    #   the competition vLLM harness loads LoRA-format tensors only and CANNOT consume a
    #   modules_to_save full module -> "Error", no score.
    # The PROVEN-LOADABLE 0.86 adapter (dgxchen/trained-adapter) ships lm_head as a LoRA
    #   TARGET MODULE with modules_to_save=null. We reproduce that EXACT config here:
    #     target_modules = 9 (incl lm_head as LoRA)   target_parameters = 2 MoE expert tensors
    #   This is the 0.84->0.86 lm_head bridge, done the loadable way.
    # ============================================================================
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "in_proj", "out_proj", "up_proj", "down_proj",
                      "lm_head"]                      # <- lm_head as LoRA (NOT modules_to_save)
    moe_target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]

    peft_kwargs = dict(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        # NOTE: NO modules_to_save -> adapter stays vLLM-loadable
    )
    print("Creating trainable LoRA wrapper (9 modules incl lm_head + MoE experts) ...")
    try:
        model = FastLanguageModel.get_peft_model(
            model, target_parameters=moe_target_parameters, **peft_kwargs)
        print("LoRA: 9 target_modules (incl lm_head-as-LoRA) + 2 MoE target_parameters [exact 0.86 config]")
    except TypeError as e:
        # Older Unsloth/PEFT without target_parameters: Unsloth auto-detects MoE experts.
        print("target_parameters unsupported here; falling back to auto-MoE detection:", e)
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
        print("LoRA: 9 target_modules (incl lm_head-as-LoRA), Unsloth auto-MoE")
    model.print_trainable_parameters()
else:
    print("USE_PRETRAINED=1: skipping trainable LoRA construction.")


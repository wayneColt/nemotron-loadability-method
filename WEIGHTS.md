# Adapter Weights

The trained rank-32 LoRA adapter (`adapter_model.safetensors`, ~4.24 GB) is **not stored in git**
(GitHub is not a weight host). It is published separately under **CC BY 4.0**:

- **Hosted at:** https://www.kaggle.com/models/bbobwayne/nemotron-loadability-method-adapter

`adapter_config.json` (the loadability contract, in situ) **is** in this repo so the method is
inspectable without the multi-gigabyte tensor.

## Load

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "<path-to-downloaded-adapter>")  # loads clean: lm_head is a LoRA delta
```

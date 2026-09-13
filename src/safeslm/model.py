"""Model loading and verified PEFT attachment."""
from __future__ import annotations
from typing import Any
import torch
from .utils import parameter_counts

def runtime_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    return torch.device("cpu"), torch.float32

def load_base_model(model_name: str, device: torch.device | None = None) -> tuple[Any, Any, torch.device, torch.dtype]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device, dtype = runtime_device() if device is None else (device, torch.float32 if device.type == "cpu" else runtime_device()[1])
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    total, trainable=parameter_counts(model)
    print(f"device={device} model={model_name} parameters={total:,} trainable={trainable:,} dtype={dtype}")
    return model, tokenizer, device, dtype

def discover_lora_targets(model: Any) -> list[str]:
    """Choose only projection suffixes which actually occur in this loaded architecture."""
    names=[name for name, module in model.named_modules() if module.__class__.__name__ in {"Linear", "Conv1D"}]
    preferred=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found=[suffix for suffix in preferred if any(name.endswith(suffix) for name in names)]
    if not found: raise ValueError(f"No known projection modules found; sample names: {names[:20]}")
    print("verified LoRA target modules:", ", ".join(found))
    return found

def attach_lora(model: Any, lora: Any) -> Any:
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    targets=discover_lora_targets(model) if lora.target_modules == "auto" else list(lora.target_modules)
    available={name.split(".")[-1] for name, _ in model.named_modules()}
    missing=set(targets)-available
    if missing: raise ValueError(f"configured LoRA targets missing from model: {sorted(missing)}")
    peft_model=get_peft_model(model, PeftLoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout, target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM))
    total, trainable=parameter_counts(peft_model)
    print(f"LoRA parameters: {trainable:,}/{total:,} ({100*trainable/total:.3f}%)")
    return peft_model

def load_model_with_adapter(model_name: str, adapter: str | None = None) -> tuple[Any, Any, torch.device, torch.dtype]:
    model, tokenizer, device, dtype=load_base_model(model_name)
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model, adapter)
        total, trainable=parameter_counts(model)
        print(f"adapter={adapter} parameters={total:,} trainable={trainable:,}")
    model.eval()
    return model, tokenizer, device, dtype

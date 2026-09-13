"""Native chat-template generation."""
from __future__ import annotations
from typing import Any
import torch

@torch.inference_mode()
def generate_reply(model: Any, tokenizer: Any, messages: list[dict[str, str]], device: torch.device, max_new_tokens: int = 128, do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.9) -> str:
    input_ids=tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    # Chat prompts are generated one at a time and are not padded; explicit ones avoid
    # ambiguity when the tokenizer uses EOS as its padding token.
    attention_mask=torch.ones_like(input_ids, device=device)
    kwargs={"max_new_tokens": max_new_tokens, "do_sample": do_sample, "pad_token_id": tokenizer.eos_token_id}
    if do_sample: kwargs.update({"temperature":temperature, "top_p":top_p})
    output=model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    return tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

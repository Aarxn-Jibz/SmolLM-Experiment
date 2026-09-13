"""JSONL validation and chat-template SFT data preparation."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
try:
    from torch.utils.data import Dataset
except ImportError:  # Allows schema validation in a 4GB/offline development environment.
    Dataset = object

CATEGORIES = {"harmful_request", "benign", "dual_use_safe", "jailbreak", "privacy", "cyber_safety", "violence_safety", "self_harm_safety"}
EVAL_BEHAVIORS = {"refuse", "answer"}

def _fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record["messages"], sort_keys=True).encode()).hexdigest()

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records=[]
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try: records.append(json.loads(line))
            except json.JSONDecodeError as exc: raise ValueError(f"{path}:{number}: invalid JSON") from exc
    return records

def validate_examples(records: list[dict[str, Any]], name: str = "dataset") -> None:
    if not records: raise ValueError(f"{name} is empty")
    for index, row in enumerate(records):
        if row.get("category") not in CATEGORIES: raise ValueError(f"{name}[{index}] has unknown category")
        messages=row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2: raise ValueError(f"{name}[{index}] needs messages")
        roles={m.get("role") for m in messages if isinstance(m, dict)}
        if "user" not in roles or "assistant" not in roles: raise ValueError(f"{name}[{index}] needs user and assistant")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"{name}[{index}] has invalid message")

def validate_data_files(train_path: str | Path, val_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, val = load_jsonl(train_path), load_jsonl(val_path)
    validate_examples(train, "train"); validate_examples(val, "validation")
    overlap={_fingerprint(x) for x in train} & {_fingerprint(x) for x in val}
    if overlap: raise ValueError(f"train/validation overlap: {len(overlap)} examples")
    return train, val

def load_eval_prompts(path: str | Path) -> list[dict[str, Any]]:
    rows=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows: raise ValueError("evaluation data must be a non-empty JSON list")
    ids=set()
    for index,row in enumerate(rows):
        if row.get("id") in ids or not isinstance(row.get("id"), str): raise ValueError(f"duplicate/invalid evaluation id at {index}")
        ids.add(row["id"])
        if row.get("category") not in CATEGORIES or row.get("expected_behavior") not in EVAL_BEHAVIORS or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"invalid evaluation prompt at {index}")
    return rows

class SafetySFTDataset(Dataset):
    """Assistant-only loss labels generated with the tokenizer's native template."""
    def __init__(self, examples: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items=[]
        for example in examples:
            messages=example["messages"]
            if messages[-1]["role"] != "assistant": raise ValueError("assistant response must be final")
            full=tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, truncation=True, max_length=max_length)
            prefix=tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, truncation=True, max_length=max_length)
            labels=list(full)
            for i in range(min(len(prefix), len(labels))): labels[i] = -100
            self.items.append({"input_ids": full, "labels": labels})
    def __len__(self) -> int: return len(self.items)
    def __getitem__(self, index: int) -> dict[str, list[int]]: return self.items[index]

class CausalDataCollator:
    def __init__(self, tokenizer: Any): self.tokenizer=tokenizer
    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch
        max_len=max(len(x["input_ids"]) for x in features); pad=self.tokenizer.pad_token_id
        ids=[]; labels=[]; masks=[]
        for item in features:
            amount=max_len-len(item["input_ids"])
            ids.append(item["input_ids"] + [pad]*amount); labels.append(item["labels"] + [-100]*amount); masks.append([1]*len(item["input_ids"])+[0]*amount)
        return {"input_ids":torch.tensor(ids), "labels":torch.tensor(labels), "attention_mask":torch.tensor(masks)}

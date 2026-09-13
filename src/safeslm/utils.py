"""Small shared utilities."""
from __future__ import annotations
import json, random
from datetime import datetime, timezone
from pathlib import Path
try:
    import torch
except ImportError:
    torch = None

def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def write_json(path: str | Path, value: object) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def utc_timestamp() -> str: return datetime.now(timezone.utc).isoformat()

def parameter_counts(model: object) -> tuple[int, int]:
    params = list(model.parameters())  # type: ignore[attr-defined]
    return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad)

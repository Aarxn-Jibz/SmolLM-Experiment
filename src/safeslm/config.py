"""Configuration loading and validation."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
try:
    import yaml
except ImportError:  # Keeps the explicitly offline smoke test dependency-light.
    yaml = None

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 5
    save_each_epoch: bool = True
    full_finetune: bool = False

@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: str | list[str] = "auto"

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

@dataclass
class Config:
    model_name: str = DEFAULT_MODEL
    seed: int = 42
    output_dir: str = "outputs/safeslm-lora"
    train_file: str = "data/safety_train.jsonl"
    val_file: str = "data/safety_val.jsonl"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    def to_dict(self) -> dict[str, Any]: return asdict(self)

def load_config(path: str | Path = "configs/train.yaml") -> Config:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        raw = yaml.safe_load(text) or {}
    else:
        # Config intentionally uses a tiny YAML subset; PyYAML remains required for normal runs.
        raw = {}
        section = raw
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"): continue
            indent = len(line) - len(line.lstrip())
            key, value = line.strip().split(":", 1); value = value.strip()
            if not value:
                raw[key] = {}; section = raw[key]; continue
            target = raw if indent == 0 else section
            if value.lower() in {"true", "false"}: parsed = value.lower() == "true"
            else:
                try: parsed = int(value)
                except ValueError:
                    try: parsed = float(value)
                    except ValueError: parsed = value
            target[key] = parsed
    cfg = Config(**{k: v for k, v in raw.items() if k not in {"training", "lora", "generation"}})
    cfg.training = TrainingConfig(**raw.get("training", {}))
    cfg.lora = LoraConfig(**raw.get("lora", {}))
    cfg.generation = GenerationConfig(**raw.get("generation", {}))
    validate_config(cfg)
    return cfg

def validate_config(cfg: Config) -> None:
    if not cfg.model_name or cfg.training.epochs < 1: raise ValueError("model_name and positive epochs are required")
    if cfg.training.batch_size < 1 or cfg.training.max_length < 32: raise ValueError("invalid batch_size or max_length")
    if cfg.lora.enabled and (cfg.lora.r < 1 or cfg.lora.alpha < 1): raise ValueError("LoRA rank/alpha must be positive")

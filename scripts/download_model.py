#!/usr/bin/env python3
"""Pre-download model files; run this only in Colab or another capable machine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safeslm.config import load_config

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg=load_config(); print(f"Downloading {cfg.model_name}...")
    AutoTokenizer.from_pretrained(cfg.model_name); AutoModelForCausalLM.from_pretrained(cfg.model_name)
    print("Download complete.")

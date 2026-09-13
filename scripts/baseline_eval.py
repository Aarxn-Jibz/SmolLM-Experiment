#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safeslm.config import load_config
from safeslm.dataset import load_eval_prompts
from safeslm.evaluation import run_evaluation, save_evaluation
from safeslm.inference import generate_reply
from safeslm.model import load_model_with_adapter
from safeslm.utils import set_seed

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default="configs/train.yaml"); parser.add_argument("--output", default="results/base_results.json"); args=parser.parse_args()
    cfg=load_config(args.config); set_seed(cfg.seed); model, tok, device, _=load_model_with_adapter(cfg.model_name)
    result=run_evaluation(load_eval_prompts("data/eval_prompts.json"), lambda p: generate_reply(model,tok,[{"role":"user","content":p}],device,max_new_tokens=cfg.generation.max_new_tokens), cfg.model_name, cfg.seed)
    save_evaluation(result,args.output); print(result["metrics"]); print(f"saved {args.output}")
if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Run base and adapter on identical prompts, then save a table, JSON, and chart."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from safeslm.config import load_config
from safeslm.dataset import load_eval_prompts
from safeslm.evaluation import run_evaluation,save_evaluation
from safeslm.inference import generate_reply
from safeslm.model import load_model_with_adapter
from safeslm.utils import set_seed,write_json
def run(adapter,cfg):
 m,t,d,_=load_model_with_adapter(cfg.model_name,adapter)
 return run_evaluation(load_eval_prompts("data/eval_prompts.json"),lambda x:generate_reply(m,t,[{"role":"user","content":x}],d,max_new_tokens=cfg.generation.max_new_tokens),cfg.model_name if not adapter else f"{cfg.model_name}+LoRA",cfg.seed)
def main():
 p=argparse.ArgumentParser();p.add_argument("--adapter",default="outputs/safeslm-lora");p.add_argument("--config",default="configs/train.yaml");a=p.parse_args();c=load_config(a.config);set_seed(c.seed);base=run(None,c);safe=run(a.adapter,c);save_evaluation(base,"results/base_results.json");save_evaluation(safe,"results/safeslm_results.json"); keys=["unsafe_refusal_rate","benign_answer_rate","over_refusal_rate","safety_score"]
 print(f"{'Metric':<24}{'Base':>12}{'SafeSLM':>12}\n"+"-"*48)
 for k in keys: print(f"{k.replace('_',' ').title():<24}{base['metrics'][k]:>11.1%}{safe['metrics'][k]:>12.1%}")
 write_json("results/comparison.json",{"base":base["metrics"],"safeslm":safe["metrics"],"note":"Transparent phrase-matching heuristic; inspect raw records."})
 try:
  import matplotlib.pyplot as plt
  labels=[x.replace("_"," ").title() for x in keys];x=range(len(keys));plt.figure(figsize=(8,4));plt.bar([i-.18 for i in x],[base["metrics"][k] for k in keys],.36,label="Base");plt.bar([i+.18 for i in x],[safe["metrics"][k] for k in keys],.36,label="SafeSLM");plt.xticks(list(x),labels,rotation=20,ha="right");plt.ylim(0,1);plt.ylabel("Rate");plt.legend();plt.tight_layout();plt.savefig("results/comparison.png",dpi=180);print("saved results/comparison.png")
 except ImportError: print("matplotlib unavailable; JSON comparison was saved")
if __name__=="__main__":main()

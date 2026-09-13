#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from safeslm.config import load_config
from safeslm.dataset import load_eval_prompts
from safeslm.evaluation import run_evaluation,save_evaluation
from safeslm.inference import generate_reply
from safeslm.model import load_model_with_adapter
from safeslm.utils import set_seed
def main():
 p=argparse.ArgumentParser();p.add_argument("--adapter",default="outputs/safeslm-lora");p.add_argument("--config",default="configs/train.yaml");p.add_argument("--output",default="results/safeslm_results.json");a=p.parse_args();c=load_config(a.config);set_seed(c.seed);m,t,d,_=load_model_with_adapter(c.model_name,a.adapter);r=run_evaluation(load_eval_prompts("data/eval_prompts.json"),lambda x:generate_reply(m,t,[{"role":"user","content":x}],d,max_new_tokens=c.generation.max_new_tokens),f"{c.model_name}+{a.adapter}",c.seed);save_evaluation(r,a.output);print(r["metrics"])
if __name__=="__main__":main()

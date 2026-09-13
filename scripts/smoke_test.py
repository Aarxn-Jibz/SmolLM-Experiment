#!/usr/bin/env python3
"""--offline never contacts Hugging Face or instantiates a model."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from safeslm.config import load_config
from safeslm.dataset import load_eval_prompts,validate_data_files
from safeslm.evaluation import run_evaluation
def main():
 p=argparse.ArgumentParser();p.add_argument("--offline",action="store_true");a=p.parse_args();c=load_config();tr,va=validate_data_files(c.train_file,c.val_file);ev=load_eval_prompts("data/eval_prompts.json");sample=run_evaluation(ev,lambda _:"I cannot help with that.","mock",c.seed);assert sample["metrics"]["unsafe_refusal_rate"]==1;print(f"offline checks passed: config, {len(tr)} train, {len(va)} val, {len(ev)} eval prompts, heuristic")
 if a.offline:return
 from safeslm.model import attach_lora,load_base_model
 from safeslm.inference import generate_reply
 m,t,d,_=load_base_model(c.model_name);reply=generate_reply(m,t,[{"role":"user","content":"Say hello."}],d,8);print("inference:",reply);attach_lora(m,c.lora);print("full smoke test passed")
if __name__=="__main__":main()

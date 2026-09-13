#!/usr/bin/env python3
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from safeslm.config import load_config
from safeslm.inference import generate_reply
from safeslm.model import load_model_with_adapter
def main():
 p=argparse.ArgumentParser();p.add_argument("--base",action="store_true");p.add_argument("--adapter",default="outputs/safeslm-lora");p.add_argument("--config",default="configs/train.yaml");p.add_argument("--max-history",type=int,default=8);a=p.parse_args();c=load_config(a.config);m,t,d,_=load_model_with_adapter(c.model_name,None if a.base else a.adapter);history=[];print("SafeSLM chat. Commands: /clear, /exit")
 while True:
  try: prompt=input("You: ").strip()
  except (EOFError,KeyboardInterrupt): break
  if prompt=="/exit":break
  if prompt=="/clear":history=[];print("History cleared.");continue
  if not prompt:continue
  history.append({"role":"user","content":prompt});history=history[-a.max_history:];reply=generate_reply(m,t,history,d,c.generation.max_new_tokens,True,c.generation.temperature,c.generation.top_p);history.append({"role":"assistant","content":reply});print("SafeSLM:",reply)
if __name__=="__main__":main()

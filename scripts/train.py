#!/usr/bin/env python3
"""Compact LoRA supervised fine-tuning loop. Run on Colab T4, not the dev machine."""
import argparse, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from safeslm.config import load_config
from safeslm.dataset import CausalDataCollator, SafetySFTDataset, validate_data_files
from safeslm.model import attach_lora, load_base_model
from safeslm.utils import parameter_counts, set_seed, utc_timestamp, write_json

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/train.yaml"); parser.add_argument("--resume",default=None); args=parser.parse_args()
    cfg=load_config(args.config); set_seed(cfg.seed); train_rows, val_rows=validate_data_files(cfg.train_file,cfg.val_file)
    model,tok,device,_=load_base_model(cfg.model_name)
    if cfg.training.full_finetune:
        print("Full fine-tuning enabled: all base weights are trainable.")
    else:
        if args.resume:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model,args.resume,is_trainable=True)
            print(f"Resumed trainable LoRA adapter from {args.resume}")
        else:
            model=attach_lora(model,cfg.lora)
    train_set=SafetySFTDataset(train_rows,tok,cfg.training.max_length); val_set=SafetySFTDataset(val_rows,tok,cfg.training.max_length)
    collator=CausalDataCollator(tok)
    loader=DataLoader(train_set,batch_size=cfg.training.batch_size,shuffle=True,collate_fn=collator)
    val_loader=DataLoader(val_set,batch_size=cfg.training.batch_size,shuffle=False,collate_fn=collator)
    optimizer=AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg.training.learning_rate,weight_decay=cfg.training.weight_decay)
    steps=math.ceil(len(loader)/cfg.training.gradient_accumulation_steps)*cfg.training.epochs
    scheduler=get_linear_schedule_with_warmup(optimizer, int(steps*cfg.training.warmup_ratio), steps)
    history=[]; model.train(); optimizer.zero_grad()
    for epoch in range(1,cfg.training.epochs+1):
        total=0.0
        for step,batch in enumerate(loader,1):
            batch={k:v.to(device) for k,v in batch.items()}; loss=model(**batch).loss/cfg.training.gradient_accumulation_steps; loss.backward(); total+=loss.item()*cfg.training.gradient_accumulation_steps
            if step%cfg.training.gradient_accumulation_steps==0 or step==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.training.max_grad_norm); optimizer.step(); scheduler.step(); optimizer.zero_grad()
        train_loss=total/len(loader)
        model.eval()
        val_total=0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch={k:v.to(device) for k,v in val_batch.items()}
                val_total+=model(**val_batch).loss.item()
        val_loss=val_total/len(val_loader)
        model.train()
        history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss})
        print(f"epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if cfg.training.save_each_epoch: model.save_pretrained(Path(cfg.output_dir)/f"checkpoint-epoch-{epoch}")
    out=Path(cfg.output_dir); model.save_pretrained(out); tok.save_pretrained(out)
    total,trainable=parameter_counts(model)
    write_json(out/"training_metrics.json",{"timestamp":utc_timestamp(),"model":cfg.model_name,"seed":cfg.seed,"dataset_size":{"train":len(train_rows),"val":len(val_rows)},"parameters":{"total":total,"trainable":trainable},"config":cfg.to_dict(),"epochs":history,"loss_history":history})
    print(f"Saved adapter and reproducibility record to {out}")
if __name__ == "__main__": main()

# SafeSLM

SafeSLM is a student research prototype that tests whether lightweight supervised safety fine-tuning can improve harmful-request refusals in a small instruction model without excessive refusal of ordinary requests.

## Research question

Can LoRA safety SFT improve safety behaviour in a ~135M parameter language model while retaining benign utility?

```
           ┌─────────────────────────┐
           │ SmolLM2-135M-Instruct   │
           └────────────┬────────────┘
                        │ baseline evaluation
                        ▼
              Safety SFT JSONL dataset
                        │
                        ▼ LoRA / PEFT
                  ┌───────────┐
                  │  SafeSLM  │
                  └─────┬─────┘
                        │ evaluation
            ┌───────────┴──────────┐
       Unsafe refusal        Benign utility
```

The default base model is `HuggingFaceTB/SmolLM2-135M-Instruct`. It is small enough for a Colab T4 experiment and already has an instruction/chat template. LoRA updates a small set of verified attention/MLP projection weights, making the run fast and the adapter easy to distribute. Change `model_name` in `configs/train.yaml` to try SmolLM2-360M later.

## Dataset and method

`data/safety_train.jsonl` and `data/safety_val.jsonl` contain balanced starter examples across harmful requests, benign requests, dual-use safe assistance, jailbreaks, privacy, cyber, violence, and self-harm. Unsafe examples contain refusals/redirection, not operational harmful details. The loader validates schema, categories, non-empty turns, and train/validation overlap.

Training uses the tokenizer's native `apply_chat_template`. Labels mask the prompt/template prefix, so loss is calculated on assistant completion tokens only. Defaults are 3 epochs, rank-8 LoRA, batch 2, accumulation 8, length 512, and 2e-4 learning rate: conservative settings for a 16GB T4.

Evaluation uses a separate held-out prompt set. Its transparent phrase-matching heuristic reports unsafe refusal rate, benign answer rate, over-refusal rate, and `safety_score = (unsafe refusal rate + benign answer rate) / 2`. Raw generations are always saved for manual review.

## Run in Colab / a GPU machine

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py
python scripts/baseline_eval.py
python scripts/train.py
python scripts/evaluate.py --adapter outputs/safeslm-lora
python scripts/compare.py --adapter outputs/safeslm-lora
python scripts/chat.py --adapter outputs/safeslm-lora
```

Use `python scripts/chat.py --base` for vanilla SmolLM2. `compare.py` writes `results/base_results.json`, `results/safeslm_results.json`, `results/comparison.json`, and `results/comparison.png`. For a quick full model check in Colab: `python scripts/smoke_test.py`.

Open `notebooks/SafeSLM_Colab.ipynb` in Colab, select **T4 GPU**, set `REPO_URL` in the clone cell if necessary, then run top-to-bottom. It installs requirements, runs baseline evaluation, trains the adapter, evaluates, compares, displays qualitative examples, and offers an optional Drive copy.

## Local development (4GB RAM/no GPU)

Do **not** load or download the model locally. The safe local check is:

```bash
PYTHONPATH=src python scripts/smoke_test.py --offline
```

It validates configuration, JSONL separation/schema, evaluation prompts, and evaluation logic without network/model access.

## Limitations

This is not production-grade alignment. The dataset and held-out evaluation are small; phrase-matching refusal detection is only a heuristic; a refusal does not guarantee safety; and LoRA tuning can reduce general helpfulness. A 135M model also has substantial capability limits independent of safety. Stronger work should use established safety benchmarks, dedicated classifiers, adversarial evaluation, and human evaluation.

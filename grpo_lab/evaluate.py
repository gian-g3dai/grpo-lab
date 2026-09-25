"""Greedy pass@1 evaluation on GSM8K test.

Usage:
    python -m grpo_lab.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --out results/x/eval_before.json
    python -m grpo_lab.evaluate --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/x/final --out ...

Writes a JSON file with accuracy, format rate and every (question, prediction, gold).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_gsm8k, normalize_number
from .rewards import extract_answer


def load_model(model_id: str, adapter: str | None, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts: list[list[dict]], max_new_tokens: int) -> list[str]:
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter to merge in")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="evaluate on a random subset")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_gsm8k(args.split, limit=args.limit)
    tok, model = load_model(args.model, args.adapter, torch.bfloat16)

    records = []
    t0 = time.time()
    for i in range(0, len(ds), args.batch_size):
        batch = ds[i : i + args.batch_size]
        outs = generate_batch(tok, model, batch["prompt"], args.max_new_tokens)
        for prompt, gold, text in zip(batch["prompt"], batch["answer"], outs):
            pred = extract_answer(text)
            records.append(
                {
                    "question": prompt[-1]["content"],
                    "gold": gold,
                    "pred": pred,
                    "correct": pred is not None and pred == normalize_number(gold),
                    "has_format": pred is not None,
                    "completion": text,
                }
            )
        done = len(records)
        acc = sum(r["correct"] for r in records) / done
        print(f"[{done}/{len(ds)}] acc={acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    n = len(records)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n": n,
        "accuracy": sum(r["correct"] for r in records) / n,
        "format_rate": sum(r["has_format"] for r in records) / n,
        "mean_completion_chars": sum(len(r["completion"]) for r in records) / n,
        "seconds": round(time.time() - t0, 1),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

"""ARC evaluation: greedy (attempt 1) plus optionally one sampled attempt (attempt 2).

ARC's official metric gives two attempts per test grid and counts a task as solved
in proportion to its test grids solved by either attempt. We report:

    pass1        fraction of test grids solved by the greedy attempt
    pass2        fraction solved by either attempt (official-style, only if --attempts 2)
    task_score   official metric: mean over tasks of (solved test grids / test grids)
    shape_rate   greedy output has the right shape
    cell_acc     mean per-cell accuracy of the greedy output (0 if wrong shape)
    format_rate  greedy output contains a parsable grid

Splits:
    rearc-holdout   fresh re-arc samples of the *trained* tasks (needs --config)
    training        official ARC-1 training tasks; with --config, only the trained subset
    evaluation      official ARC-1 (or -2 with --version 2) evaluation tasks

Usage:
    python -m grpo_lab.evaluate_arc --model Qwen/Qwen2.5-1.5B-Instruct --config configs/x.yaml \
        --split rearc-holdout --out results/x/arc_holdout_before.json
    python -m grpo_lab.evaluate_arc --model ... --adapter outputs/x/final --split evaluation --out ...
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from .arc import (
    build_official_eval,
    build_rearc_holdout,
    cell_accuracy,
    grid_to_text,
    parse_grid,
    prompt_cells,
    select_task_ids,
)
from .evaluate import load_model


def build_rows(args) -> list[dict]:
    data_cfg = {}
    task_ids = None
    if args.config:
        data_cfg = yaml.safe_load(open(args.config)).get("data", {})
        task_ids = select_task_ids(data_cfg.get("n_tasks"), data_cfg.get("select", "smallest"))
    if args.split == "rearc-holdout":
        if task_ids is None:
            raise SystemExit("--split rearc-holdout needs --config (which tasks were trained)")
        rows = build_rearc_holdout(
            task_ids,
            n_demos=data_cfg.get("n_demos", 3),
            holdout_per_task=data_cfg.get("holdout_per_task", 20),
            max_prompt_cells=data_cfg.get("max_prompt_cells", 2000),
            max_output_cells=data_cfg.get("max_output_cells", 400),
            seed=args.seed,
        )
        for i, r in enumerate(rows):
            r["test_index"] = i
    elif args.split in ("training", "evaluation"):
        rows = build_official_eval(args.split, args.version, task_ids if args.split == "training" else None)
    else:
        raise SystemExit(f"unknown split {args.split}")
    if args.limit is not None and args.limit < len(rows):
        import random

        rows = random.Random(args.seed).sample(rows, args.limit)
    return rows


def batches(rows: list[dict], max_rows: int, max_chars: int):
    """Group rows into batches of at most ``max_rows`` whose prompts total at most ``max_chars``.

    ARC prompts range from ~300 to ~9000 tokens (about one char per token). A fixed batch size
    either wastes the GPU on short prompts or runs out of memory on long ones; a character
    budget keeps the padded batch roughly constant in size.
    """
    batch: list[dict] = []
    chars = 0
    for r in rows:
        n = len(r["prompt"][1]["content"])
        if batch and (len(batch) >= max_rows or chars + n > max_chars):
            yield batch
            batch, chars = [], 0
        batch.append(r)
        chars += n
    if batch:
        yield batch


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens: int, sample: bool, temperature: float) -> list[str]:
    texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    kwargs = dict(do_sample=True, temperature=temperature, top_p=1.0) if sample else dict(do_sample=False)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kwargs)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--config", default=None, help="training config: picks the trained task subset / holdout")
    ap.add_argument("--split", default="rearc-holdout", choices=["rearc-holdout", "training", "evaluation"])
    ap.add_argument("--version", type=int, default=1, help="ARC-AGI version for official splits")
    ap.add_argument("--attempts", type=int, default=1, choices=[1, 2])
    ap.add_argument("--temperature", type=float, default=0.7, help="for the sampled 2nd attempt")
    ap.add_argument("--max-prompt-cells", type=int, default=6000, help="skip (score 0) prompts bigger than this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8, help="max rows per batch")
    ap.add_argument("--batch-chars", type=int, default=12000, help="max total prompt chars (~tokens) per batch")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = build_rows(args)
    # Longest prompts first: batches are then uniform in length (less padding) and an OOM shows up at once.
    rows.sort(key=lambda r: -len(r["prompt"][1]["content"]))
    tok, model = load_model(args.model, args.adapter, torch.bfloat16)

    records = []
    t0 = time.time()
    for batch in batches(rows, args.batch_size, args.batch_chars):
        prompts = [r["prompt"] for r in batch]
        # prompt_cells is ~tokens; skip anything that cannot fit so one giant task does not OOM the run
        keep = [len(r["prompt"][1]["content"]) <= args.max_prompt_cells * 1.6 for r in batch]
        outs = [""] * len(batch)
        outs2 = [""] * len(batch)
        idx = [k for k, ok in enumerate(keep) if ok]
        if idx:
            gen = generate(tok, model, [prompts[k] for k in idx], args.max_new_tokens, False, 1.0)
            for k, g in zip(idx, gen):
                outs[k] = g
            if args.attempts == 2:
                torch.manual_seed(args.seed + len(records))
                gen2 = generate(tok, model, [prompts[k] for k in idx], args.max_new_tokens, True, args.temperature)
                for k, g in zip(idx, gen2):
                    outs2[k] = g
        for r, ok, text, text2 in zip(batch, keep, outs, outs2):
            gold = [[int(c) for c in line] for line in r["answer"].split("\n")]
            pred = parse_grid(text)
            pred2 = parse_grid(text2) if args.attempts == 2 else None
            c1 = pred is not None and grid_to_text(pred) == r["answer"]
            c2 = pred2 is not None and grid_to_text(pred2) == r["answer"]
            records.append(
                {
                    "task_id": r["task_id"],
                    "test_index": r.get("test_index", 0),
                    "skipped": not ok,
                    "gold": r["answer"],
                    "pred": grid_to_text(pred) if pred is not None else None,
                    "correct": c1,
                    "correct_any": c1 or c2,
                    "shape_ok": pred is not None and len(pred) == len(gold) and len(pred[0]) == len(gold[0]),
                    "cell_acc": cell_accuracy(pred, gold),
                    "has_format": pred is not None,
                    "completion": text,
                    "completion2": text2 if args.attempts == 2 else None,
                }
            )
        done = len(records)
        p1 = sum(x["correct"] for x in records) / done
        ca = sum(x["cell_acc"] for x in records) / done
        print(f"[{done}/{len(rows)}] pass1={p1:.3f} cell_acc={ca:.3f} ({time.time() - t0:.0f}s)", flush=True)

    n = len(records)
    by_task = defaultdict(list)
    for x in records:
        by_task[x["task_id"]].append(x["correct_any"])
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "version": args.version,
        "config": args.config,
        "n": n,
        "n_tasks": len(by_task),
        "n_skipped": sum(x["skipped"] for x in records),
        "attempts": args.attempts,
        "max_new_tokens": args.max_new_tokens,
        "pass1": sum(x["correct"] for x in records) / n,
        "pass2": sum(x["correct_any"] for x in records) / n,
        "task_score": sum(sum(v) / len(v) for v in by_task.values()) / len(by_task),
        "shape_rate": sum(x["shape_ok"] for x in records) / n,
        "cell_acc": sum(x["cell_acc"] for x in records) / n,
        "format_rate": sum(x["has_format"] for x in records) / n,
        "mean_completion_chars": sum(len(x["completion"]) for x in records) / n,
        "seconds": round(time.time() - t0, 1),
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()

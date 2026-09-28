"""Paired before/after statistics for one run (results/<run>/analysis.md).

Usage:
    python -m grpo_lab.analyze --results results/qwen2.5-0.5b-gsm8k

Both eval files score the same test problems, so every comparison here is paired:

* accuracy delta with a paired bootstrap 95% CI and an exact McNemar test;
* the test set split by what the *base* model produced (a numeric ``\\boxed{}``; no box
  because it hit the token limit; no box although it stopped on its own), with both
  models scored on each subset;
* all of the above under strict scoring (the number in the last ``\\boxed{}``, as in
  summary.md) and lenient scoring (that number if there is one, else the last number
  in the completion). The gap between the two is the part of the gain that comes from
  the model learning to put its answer in a box.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

from .data import normalize_number

# commas only as thousands separators, so "3, 4" stays two numbers
_NUM_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def last_number(text: str) -> str | None:
    nums = _NUM_RE.findall(text)
    return normalize_number(nums[-1]) if nums else None


def strict_correct(r: dict) -> bool:
    return r["correct"]


def lenient_correct(r: dict) -> bool:
    """Strict score if the completion has a numeric \\boxed{}, else compare its last number."""
    if r["pred"] is not None:
        return r["correct"]
    return last_number(r["completion"]) == normalize_number(r["gold"])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b and c discordant pairs (binomial test at p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def paired_bootstrap_ci(
    before: list[bool], after: list[bool], n_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% CI of mean(after - before), resampling problems with replacement."""
    d = [int(a) - int(b) for b, a in zip(before, after)]
    rng = random.Random(seed)
    means = [sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n_resamples)]
    q = statistics.quantiles(means, n=40)  # cut points every 2.5%
    return q[0], q[-1]


def token_counts(model_id: str, texts: list[str]) -> list[int]:
    # Eval files store decoded text only, so re-tokenise it to find completions that hit the limit.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def section(title, before, after, correct, groups, n_resamples) -> list[str]:
    sb = [correct(r) for r in before]
    sa = [correct(r) for r in after]
    n = len(sb)
    w2r = sum(a and not b for b, a in zip(sb, sa))
    r2w = sum(b and not a for b, a in zip(sb, sa))
    lo, hi = paired_bootstrap_ci(sb, sa, n_resamples)
    lines = [
        f"## {title}",
        "",
        "| base model output | n | base acc | GRPO acc | net problems gained |",
        "|---|---|---|---|---|",
    ]
    for name, idx in groups:
        if not idx:
            continue
        gb, ga = sum(sb[i] for i in idx), sum(sa[i] for i in idx)
        k = len(idx)
        lines.append(f"| {name} | {k} | {100 * gb / k:.1f}% | {100 * ga / k:.1f}% | {ga - gb:+d} |")
    lines += [
        "",
        f"**Δ accuracy: {100 * (sum(sa) - sum(sb)) / n:+.1f} points**, paired bootstrap 95% CI "
        f"[{100 * lo:+.1f}, {100 * hi:+.1f}] ({n_resamples} resamples). {w2r} problems wrong→right, "
        f"{r2w} right→wrong, exact McNemar p = {mcnemar_exact(w2r, r2w):.1e}.",
        "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--bootstrap", type=int, default=10_000, help="number of bootstrap resamples")
    args = ap.parse_args()
    res = Path(args.results)
    before = json.load(open(res / "eval_before.json"))
    after = json.load(open(res / "eval_after.json"))
    B, A = before["records"], after["records"]
    if [r["question"] for r in B] != [r["question"] for r in A]:
        raise SystemExit("eval_before.json and eval_after.json score different problems")

    max_new = before.get("max_new_tokens", 512)  # evaluate.py's default; older eval files don't record it
    ntok = token_counts(before["model"], [r["completion"] for r in B])
    boxed = [i for i, r in enumerate(B) if r["has_format"]]
    truncated = [i for i, r in enumerate(B) if not r["has_format"] and ntok[i] >= max_new]
    stopped = [i for i, r in enumerate(B) if not r["has_format"] and ntok[i] < max_new]
    groups = [
        ("numeric `\\boxed{}`", boxed),
        (f"no box, hit the {max_new}-token limit", truncated),
        ("no box, stopped on its own", stopped),
        ("all", list(range(len(B)))),
    ]

    md = "\n".join(
        [
            "# Paired analysis",
            "",
            f"`{before['model']}` (base) vs. + GRPO LoRA (`{after['adapter']}`) on {len(B)} GSM8K "
            f"{before['split']} problems, greedy. Rows split the problems by what the *base* model "
            "produced, so both models are scored on the same set.",
            "",
            *section(
                "Strict scoring: number in the last `\\boxed{}` (as in summary.md)",
                B, A, strict_correct, groups, args.bootstrap,
            ),
            *section(
                "Lenient scoring: the `\\boxed{}` answer if present, else the last number in the completion",
                B, A, lenient_correct, groups, args.bootstrap,
            ),
        ]
    )
    (res / "analysis.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

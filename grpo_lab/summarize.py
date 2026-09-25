"""Turn eval_before.json / eval_after.json into a small markdown table (results/<run>/summary.md)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    before = json.load(open(res / "eval_before.json"))
    after = json.load(open(res / "eval_after.json"))

    def row(name, d):
        return (
            f"| {name} | {d['n']} | {100 * d['accuracy']:.1f}% | {100 * d['format_rate']:.1f}% "
            f"| {d['mean_completion_chars']:.0f} |"
        )

    delta = 100 * (after["accuracy"] - before["accuracy"])
    md = "\n".join(
        [
            "| model | n | GSM8K acc (greedy) | `\\boxed{}` rate | mean completion chars |",
            "|---|---|---|---|---|",
            row(f"`{before['model']}` (base)", before),
            row(f"+ GRPO LoRA", after),
            "",
            f"**Δ accuracy: {delta:+.1f} points** on {after['n']} test problems.",
            "",
        ]
    )
    (res / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
